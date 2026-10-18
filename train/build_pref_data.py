# train/build_pref_data.py -- пары (chosen, rejected) для обучения формату вызовов edit_file
"""
Модель в ~28% вызовов edit_file печатает в old_string/new_string литеральные символы "\\n" вместо
перевода строки; такой old_string не находится в файле (успешны 6 из 247 вызовов), а в run_command и
write_file того же дефекта нет. Награда проверяема: вызов применился или нет. Пары строим из траекторий:

  * real_fix  -- вызов модели с литеральным "\\n" (rejected) и тот же вызов с настоящими переводами
                 строк (chosen); контекст -- ровно тот, в котором модель ошиблась;
  * synthetic -- успешный вызов с переводами строк (chosen) и он же с "\\n" вместо них (rejected).

Записи: {"chosen": messages, "rejected": messages, "tools", "thinking": True, "kind", "task"}; оба списка
различаются только аргументами последнего вызова (рассуждение хода одинаковое, поэтому в разности
логарифмов правдоподобия оно сокращается). Системный промпт добавляем сами, как в build_sft_data.

Запуск:
  python -m train.build_pref_data --runs results/traj_think --system submission_v3/prompts/system.md \
      --out data/pref_edit.jsonl --exclude fastapi_14303 ...
"""
import argparse
import copy
import json
import os
from collections import Counter

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from train.build_sft_data import ALLOWED_TOOLS, load_run

BACKSLASH_N = chr(92) + "n"
FIELDS = ("old_string", "new_string")
MAX_THOUGHT_CHARS = 6000


def escape_newlines(text):
    return text.replace("\n", BACKSLASH_N)


def is_escaped(text):
    """Поле, в котором все переводы строк заменены на литеральный \\n. Одиночный \\n встречается и в
    настоящем коде (print("a\\n")), поэтому требуем не меньше двух и ни одного настоящего перевода."""
    return text.count(BACKSLASH_N) >= 2 and "\n" not in text


def unescape_newlines(text):
    return text.replace(BACKSLASH_N, "\n")


def edit_args(message):
    """Аргументы единственного вызова edit_file в сообщении ассистента или None."""
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "edit_file":
        return None
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return None
    return args if isinstance(args, dict) else None


def with_args(message, args):
    out = copy.deepcopy(message)
    out["tool_calls"][0]["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
    return out


def trim_history(messages, keep_exchanges, max_tool_chars):
    """Условие задачи (messages[0]) и последние keep_exchanges обменов (ход ассистента + ответы инструментов)
    перед целью. Потолок обучения 8192 токена, а к первой правке история обычно длиннее; формат вызова от
    дальнего прошлого почти не зависит, нужен лишь код, который модель только что прочитала. Длинные ответы
    инструментов обрезаем, оставляя начало."""
    starts = [i for i, m in enumerate(messages) if i > 0 and m.get("role") == "assistant"]
    first_kept = starts[-keep_exchanges] if len(starts) > keep_exchanges else (starts[0] if starts else len(messages))
    out = [messages[0]] + [dict(m) for m in messages[first_kept:]]
    for m in out:
        if m.get("role") == "tool" and len(m.get("content") or "") > max_tool_chars:
            m["content"] = m["content"][:max_tool_chars] + " ...[truncated]"
    return out


def tool_status(messages):
    out = {}
    for m in messages:
        if m.get("role") == "tool":
            try:
                out[m.get("tool_call_id")] = json.loads(m.get("content") or "{}").get("status")
            except ValueError:
                out[m.get("tool_call_id")] = None
    return out


def make_pair(args, ok):
    """-> (kind, chosen_args, rejected_args) или None."""
    texts = {f: args.get(f) or "" for f in FIELDS}
    if any(is_escaped(t) for t in texts.values()):
        if ok:                                   # удачный вызов с литеральным \n -- не учим на нём
            return None
        fixed = dict(args, **{f: unescape_newlines(t) if is_escaped(t) else t for f, t in texts.items()})
        return ("real_fix", fixed, args) if fixed != args else None
    if ok and any("\n" in t for t in texts.values()):
        broken = dict(args, **{f: escape_newlines(t) for f, t in texts.items()})
        return ("synthetic", args, broken) if broken != args else None
    return None


def episode_pairs(dialog, system, tools, max_per_episode, max_thought_chars=MAX_THOUGHT_CHARS,
                  keep_exchanges=3, max_tool_chars=3000):
    messages = dialog["messages"]
    status = tool_status(messages)
    pairs = []
    for k, m in enumerate(messages):
        if m.get("role") != "assistant" or len(pairs) >= max_per_episode:
            continue
        args = edit_args(m)
        thought = (m.get("thought") or "").strip()
        if args is None or not thought or len(thought) > max_thought_chars:
            continue
        ok = status.get(m["tool_calls"][0]["id"]) == "ok"
        made = make_pair(args, ok)
        if made is None:
            continue
        kind, good, bad = made
        prefix = [{"role": "system", "content": system}] + trim_history(messages[:k], keep_exchanges, max_tool_chars)
        pairs.append({"chosen": prefix + [with_args(m, good)], "rejected": prefix + [with_args(m, bad)],
                      "tools": tools, "thinking": True, "kind": kind})
    return pairs


def build(runs, system, exclude=(), max_per_episode=20, synthetic_per_real=1, max_thought_chars=MAX_THOUGHT_CHARS,
          keep_exchanges=3, max_tool_chars=3000):
    """synthetic_per_real=0 -- брать все доступные синтетические пары."""
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    real, synthetic = [], []
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            if row["instance_id"] in exclude or row.get("error"):
                continue
            for pair in episode_pairs(dialog, system, tools, max_per_episode, max_thought_chars,
                                       keep_exchanges, max_tool_chars):
                pair["task"] = row["instance_id"]
                (real if pair["kind"] == "real_fix" else synthetic).append(pair)
    used = synthetic[:len(real) * synthetic_per_real] if synthetic_per_real else synthetic
    return real + used, Counter({"real_fix": len(real), "synthetic_available": len(synthetic),
                                 "synthetic_used": len(used)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="data/pref_edit.jsonl")
    ap.add_argument("--exclude", nargs="*", default=[], help="задачи, которые не берём (оценочный набор)")
    ap.add_argument("--max-per-episode", type=int, default=20)
    ap.add_argument("--max-thought-chars", type=int, default=MAX_THOUGHT_CHARS)
    ap.add_argument("--keep-exchanges", type=int, default=3, help="сколько последних обменов оставлять в истории")
    ap.add_argument("--max-tool-chars", type=int, default=3000, help="длиннее ответы инструментов обрезаются")
    ap.add_argument("--synthetic-per-real", type=int, default=1,
                    help="синтетических пар на одну реальную (0 = все доступные)")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    pairs, stats = build(args.runs, system, set(args.exclude), args.max_per_episode, args.synthetic_per_real,
                         args.max_thought_chars, args.keep_exchanges, args.max_tool_chars)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"пар записано: {len(pairs)} -> {args.out}; {dict(stats)}")
    print("задач в данных:", len({p["task"] for p in pairs}))


if __name__ == "__main__":
    main()
