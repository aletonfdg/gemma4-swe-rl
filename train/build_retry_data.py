# train/build_retry_data.py -- пары "после неудачной правки не повторять её"
"""
В прогоне v5 + pref_edit_v5_2 из 569 вызовов edit_file 170 -- дословный повтор правки, которая только что
вернула FileEditError (новых неудачных правок 85). Каждый повтор снова падает и съедает вызов и ~16 с.
Пары строим в точке повтора (rejected -- сам повтор, как модель его сделала):

  retry_read  -- chosen: read_file того же файла вокруг места правки (строки берём из прежнего read_file,
                 где первая строка old_string уже встречалась; иначе grep -n -F по этой строке), чтобы
                 модель увидела точный текст;
  retry_fixed -- chosen: правка того же файла, которая позже в этом эпизоде применилась (модель в итоге
                 нашла верный old_string), если такая есть.

Рассуждение хода в chosen и rejected одно и то же (то, что модель написала перед повтором), поэтому в
разности правдоподобий оно сокращается и учится только выбор вызова. История -- как в build_pref_data.

  python -m train.build_retry_data --runs <логи> --system submission_v3/prompts/system.md \
      --exclude-file eval/two_phase_small.json --out data/pref_retry.jsonl
"""
import argparse
import copy
import json
import os
import random
import shlex
from collections import Counter

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from train.build_pref_data import MAX_THOUGHT_CHARS, edit_args, tool_status, trim_history
from train.build_sft_data import ALLOWED_TOOLS, load_run

CONTEXT_LINES = 12


def call_of(message):
    calls = message.get("tool_calls") or []
    if len(calls) != 1:
        return None, None
    fn = calls[0]["function"]
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    except (TypeError, ValueError):
        return None, None
    return fn["name"], args if isinstance(args, dict) else None


def is_source(path):
    return bool(path) and not str(path).startswith(("/tmp", "tmp/"))


def first_line(text):
    """Первая непустая строка; литеральные \\n и \\" (дефект экранирования) сначала снимаем."""
    text = (text or "").replace("\\n", "\n").replace('\\"', '"')
    for line in text.splitlines():
        if line.strip():
            return line
    return ""


def locate(messages, filepath, old_string):
    """Номер строки первой строки old_string в файле по прежним read_file этого файла (или None)."""
    needle = first_line(old_string).strip()
    if not needle:
        return None
    outputs = {m.get("tool_call_id"): m for m in messages if m.get("role") == "tool"}
    for m in reversed(messages):
        name, args = call_of(m) if m.get("role") == "assistant" else (None, None)
        if name != "read_file" or args.get("filepath") != filepath:
            continue
        out = outputs.get(m["tool_calls"][0]["id"])
        try:
            content = json.loads(out["content"]).get("content", "") if out else ""
        except (ValueError, AttributeError, TypeError):
            continue
        for i, line in enumerate(content.splitlines()):
            if line.strip() == needle:
                return int(args.get("start_line") or 1) + i
    return None


def with_call(message, name, args):
    out = copy.deepcopy(message)
    fn = out["tool_calls"][0]["function"]
    fn["name"], fn["arguments"] = name, json.dumps(args, ensure_ascii=False)
    return out


def read_call(messages, filepath, old_string):
    line = locate(messages, filepath, old_string)
    if line is not None:
        span = len(old_string.replace("\\n", "\n").splitlines())
        return "read_file", {"filepath": filepath, "start_line": max(1, line - CONTEXT_LINES),
                             "end_line": line + span + CONTEXT_LINES}
    needle = first_line(old_string).strip()
    return "run_command", {"command": f"grep -n -F -- {shlex.quote(needle)} {shlex.quote(filepath)}"}


def episode_pairs(dialog, system, tools, keep_exchanges=3, max_tool_chars=3000, max_per_episode=6,
                  max_thought_chars=MAX_THOUGHT_CHARS):
    messages = dialog["messages"]
    status = tool_status(messages)
    failed, pairs = {}, []          # (файл, old_string) -> индекс последней неудачи
    for k, m in enumerate(messages):
        if m.get("role") != "assistant" or len(pairs) >= max_per_episode:
            continue
        args = edit_args(m)
        if args is None or not is_source(args.get("filepath")):
            continue
        key = (args.get("filepath"), args.get("old_string"))
        ok = status.get(m["tool_calls"][0]["id"]) == "ok"
        thought = (m.get("thought") or "").strip()
        # рассуждение может отсутствовать (в части прогонов оно не сохранялось): оно одно на обе стороны
        if key in failed and not ok and len(thought) <= max_thought_chars:
            prefix = [{"role": "system", "content": system}] + trim_history(messages[:k], keep_exchanges,
                                                                             max_tool_chars)
            base = {"tools": tools, "thinking": True}
            # если после неудачи модель уже перечитала этот файл, "перечитай" -- не тот ход (это новый цикл)
            between = [call_of(n) for n in messages[failed[key] + 1:k] if n.get("role") == "assistant"]
            reread = any(name == "read_file" and a.get("filepath") == args["filepath"] for name, a in between if a)
            if not reread:
                name, rargs = read_call(messages[:k], args["filepath"], args.get("old_string", ""))
                pairs.append(dict(base, kind="retry_read", chosen=prefix + [with_call(m, name, rargs)],
                                  rejected=prefix + [copy.deepcopy(m)]))
            later = next((edit_args(n) for n in messages[k + 1:] if n.get("role") == "assistant"
                          and edit_args(n) and edit_args(n).get("filepath") == args["filepath"]
                          and status.get(n["tool_calls"][0]["id"]) == "ok"), None)
            if later is not None and later.get("old_string") != args.get("old_string"):
                pairs.append(dict(base, kind="retry_fixed", chosen=prefix + [with_call(m, "edit_file", later)],
                                  rejected=prefix + [copy.deepcopy(m)]))
        if not ok:
            failed[key] = k
    return pairs


def build(runs, system, exclude=(), keep_exchanges=3, max_tool_chars=3000, max_per_episode=6, max_per_task=12,
          seed=0):
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    by_task, stats = {}, Counter()
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            if row["instance_id"] in exclude:
                stats["excluded"] += 1
                continue
            if row.get("error"):
                continue
            for p in episode_pairs(dialog, system, tools, keep_exchanges, max_tool_chars, max_per_episode):
                p["task"] = row["instance_id"]
                by_task.setdefault(p["task"], []).append(p)
    rng, pairs = random.Random(seed), []
    for task, ps in sorted(by_task.items()):
        rng.shuffle(ps)
        pairs += ps[:max_per_task]
    rng.shuffle(pairs)
    stats.update(p["kind"] for p in pairs)
    stats.update("chosen:" + p["chosen"][-1]["tool_calls"][0]["function"]["name"] for p in pairs)
    return pairs, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="data/pref_retry.jsonl")
    ap.add_argument("--exclude-file", default=None, help="json с 'all_ids' или 'ids' (оценочный набор)")
    ap.add_argument("--keep-exchanges", type=int, default=3)
    ap.add_argument("--max-tool-chars", type=int, default=3000)
    ap.add_argument("--max-per-episode", type=int, default=6)
    ap.add_argument("--max-per-task", type=int, default=12)
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    exclude = set()
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8") as f:
            d = json.load(f)
        exclude = set(d.get("all_ids") or d["ids"])
    pairs, stats = build(args.runs, system, exclude, args.keep_exchanges, args.max_tool_chars,
                         args.max_per_episode, args.max_per_task)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"пар записано: {len(pairs)} -> {args.out}; {dict(stats)}")
    print("задач в данных:", len({p["task"] for p in pairs}))


if __name__ == "__main__":
    main()
