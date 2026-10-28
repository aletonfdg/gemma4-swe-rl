# train/build_pref_short.py -- пары (chosen, rejected) для обучения думать короче перед разведкой
"""
По логам MiMo ~70% текста рассуждений приходится на ходы перед read_file/run_command, и в 121 из 149
эпизодов кончилось время (во всех 52 no_patch). Идея как у ThinkingCap: модель должна останавливаться,
как только информации хватает на следующее действие.

Пара строится из одного хода модели с длинным рассуждением перед разведочным вызовом:
  * rejected -- ход как есть (длинное рассуждение + вызов);
  * chosen   -- то же рассуждение, обрезанное на границе абзаца (или предложения), + тот же вызов.
Точку обрезки берём не раньше, чем в рассуждении появилась цель вызова (путь файла или слово из команды):
вызов уже был обоснован, дальше модель думала лишнее. Варианты различаются длиной, поэтому обучать только
с reference (train.pref_lora --ref), иначе длинный вариант проигрывает уже из-за числа токенов.

Записи как у build_pref_data: {"chosen", "rejected", "tools", "thinking": True, "kind": "short_think", "task"}.

Запуск:
  python -m train.build_pref_short --runs mimo_p1_full eval_mimo50/v5 --system submission_v5/prompts/system.md \
      --out data/pref_short.jsonl
"""
import argparse
import copy
import json
import os
import re
from collections import Counter

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from train.build_pref_data import tool_status, trim_history
from train.build_sft_data import ALLOWED_TOOLS, load_run

EXPLORE = ("read_file", "run_command")
MIN_LONG = 2500         # короче -- не трогаем
MAX_LONG = 10000        # длиннее -- пример не влезет в 8192 токена вместе с историей
SHORT = 700             # желаемая длина обрезанного рассуждения
MAX_KEEP = 0.5          # обрезанное -- не больше этой доли исходного
MIN_SHORT = 150

_WORD = re.compile(r"[A-Za-z_][\w./-]{3,}")
_SHELL = {"grep", "find", "head", "tail", "python", "python3", "pytest", "echo", "sed", "awk", "less", "true",
          "xargs", "sort", "uniq", "wc", "-rn", "-name", "-type", "--include", "-maxdepth"}


def single_call(message):
    """-> (name, args) единственного вызова хода или None."""
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None
    fn = calls[0]["function"]
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    return (fn["name"], args) if isinstance(args, dict) else None


def call_targets(name, args):
    """Слова, по которым видно, что рассуждение уже дошло до этого вызова."""
    if name == "read_file":
        path = args.get("filepath") or args.get("path") or ""
        return {os.path.basename(path)} - {""}
    words = set(_WORD.findall(args.get("command") or ""))
    return {w for w in words if w not in _SHELL and not w.startswith("-")}


def cut_points(thought):
    """Позиции концов абзацев, а если их нет -- предложений; не внутри блока ```."""
    ends = [m.end() for m in re.finditer(r"\n\s*\n", thought)]
    if not ends:
        ends = [m.end() for m in re.finditer(r"[.!?:]\s", thought)]
    return [e for e in ends if thought[:e].count("```") % 2 == 0]


def shorten(thought, targets, short=SHORT, max_keep=MAX_KEEP, min_short=MIN_SHORT):
    """-> обрезанное рассуждение или None. Режем в ближайшей к `short` точке, где цель вызова уже названа."""
    first = min((thought.find(t) + len(t) for t in targets if t in thought), default=None)
    if first is None:
        return None
    limit = len(thought) * max_keep
    points = [e for e in cut_points(thought) if e >= max(first, min_short) and e <= limit]
    if not points:
        return None
    best = min(points, key=lambda e: abs(e - short))
    return thought[:best].rstrip()


def with_thought(message, thought):
    out = copy.deepcopy(message)
    out["thought"] = thought
    out.pop("reasoning_content", None)
    return out


def episode_pairs(dialog, system, tools, max_per_episode=4, min_long=MIN_LONG, max_long=MAX_LONG,
                  short=SHORT, keep_exchanges=3, max_tool_chars=3000):
    messages = dialog["messages"]
    status = tool_status(messages)
    pairs, stats = [], Counter()
    for k, m in enumerate(messages):
        if m.get("role") != "assistant" or len(pairs) >= max_per_episode:
            continue
        call = single_call(m)
        thought = (m.get("thought") or m.get("reasoning_content") or "").strip()
        if call is None or call[0] not in EXPLORE or len(thought) < min_long:
            continue
        if len(thought) > max_long:
            stats["too_long"] += 1
            continue
        if status.get(m["tool_calls"][0].get("id")) == "error":
            stats["call_failed"] += 1
            continue
        cut = shorten(thought, call_targets(*call), short)
        if cut is None:
            stats["no_cut"] += 1
            continue
        prefix = [{"role": "system", "content": system}] + trim_history(messages[:k], keep_exchanges, max_tool_chars)
        pairs.append({"chosen": prefix + [with_thought(m, cut)], "rejected": prefix + [with_thought(m, thought)],
                      "tools": tools, "thinking": True, "kind": "short_think",
                      "chars": [len(cut), len(thought)]})
        stats["pairs"] += 1
    return pairs, stats


def build(runs, system, exclude=(), max_per_episode=4, short=SHORT):
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    out, stats = [], Counter()
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            if row["instance_id"] in exclude or row.get("error") or row.get("crashed"):
                continue
            pairs, st = episode_pairs(dialog, system, tools, max_per_episode, short=short)
            stats.update(st)
            for p in pairs:
                p["task"] = row["instance_id"]
            out += pairs
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="data/pref_short.jsonl")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--max-per-episode", type=int, default=4)
    ap.add_argument("--short", type=int, default=SHORT, help="желаемая длина обрезанного рассуждения, символов")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    pairs, stats = build(args.runs, system, set(args.exclude), args.max_per_episode, args.short)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if pairs:
        kept = sum(p["chars"][0] for p in pairs) / sum(p["chars"][1] for p in pairs)
        print(f"пар записано: {len(pairs)} -> {args.out}; задач {len({p['task'] for p in pairs})}; "
              f"остаётся {kept:.0%} текста рассуждения; {dict(stats)}")
    else:
        print("пар нет;", dict(stats))


if __name__ == "__main__":
    main()
