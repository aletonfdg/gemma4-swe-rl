# train/build_inv_data.py -- пары для адаптера следователя: "нужный код уже прочитан -> отвечай диагнозом"
"""
Двухфазный агент решает 86% эпизодов, если следователь отдаёт диагноз до ~210 с, и 0%, если не отдаёт
(37% эпизодов): модель находит нужный код к 3-му вызову, но не решает остановиться. Адаптер учит именно
это решение. Пары строим из старых эпизодов базы, где известен эталонный патч задачи:

  stop -- история до момента, когда модель увидела эталонный кусок кода (ответ инструмента содержит его
          удаляемые строки); chosen: короткое рассуждение + диагноз FILE/LINES/CAUSE/FIX/CHECK, rejected:
          что модель сделала на самом деле (ещё grep/read). Точек на эпизод -- до --stop-per-episode.
  go   -- ход ДО того, как модель увидела нужный код; chosen: её настоящий вызов (тот, что привёл к коду),
          rejected: преждевременный диагноз. Без них адаптер научится отвечать, ничего не найдя.

FILE/LINES/FIX/CHECK собираются из эталонного патча (точные). Рассуждение и CAUSE для stop и текст
преждевременного диагноза для go пишет сама Gemma на Colab (train/fill_inv_data.py): здесь в записях
поле "gen" с тем, что ей нужно, а последний ход chosen/rejected -- заглушка.

История приводится к инструментам следователя: write_file('/tmp/x') -> run_command с heredoc (у
следователя нет write_file); эпизод обрывается на первой правке исходника.

  python -m train.build_inv_data --runs <папки прогонов> --tasks data/comp_full/tasks.jsonl \
      --exclude-file eval/two_phase_small.json --out data/inv_pairs.jsonl
"""
import argparse
import copy
import json
import os
import random
import re
from collections import Counter

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from sim.two_phase import INVESTIGATOR_TOOLS
from train.build_pref_data import as_run_command, heredoc_command, tmp_write_args, trim_history
from train.build_sft_data import load_run

INVESTIGATOR_PROMPT = "prompts/two_phase/investigator.md"
NON_SOURCE = re.compile(r"(^|/)(tests?|docs?|docs_src)/|(^|/)test_[^/]*$|_test\.py$|\.(md|rst|txt|ya?ml|toml|cfg|ini)$")
PLACEHOLDER = "<<generated in fill_inv_data>>"


def parse_hunks(patch):
    """-> [{"file", "old_start", "old_len", "lines", "minus", "ctx"}] по исходным файлам (тесты и документацию
    пропускаем). lines -- строки хунка как в диффе (с префиксами ' ', '-', '+')."""
    out, path, cur = [], None, None
    for line in patch.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            cur = None
            continue
        if line.startswith(("--- ", "diff ", "index ")):
            cur = None
            continue
        m = re.match(r"@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
        if m:
            cur = None
            if path and not NON_SOURCE.search(path):
                cur = {"file": path, "old_start": int(m.group(1)),
                       "old_len": int(m.group(2)) if m.group(2) is not None else 1,
                       "lines": [], "minus": [], "ctx": []}
                out.append(cur)
            continue
        if cur is not None and line[:1] in (" ", "-", "+"):
            cur["lines"].append(line)
            if line[:1] == "-":
                cur["minus"].append(line[1:].strip())
            elif line[:1] == " ":
                cur["ctx"].append(line[1:].strip())
    return out


def key_lines(hunk):
    """Строки хунка (удаляемые и контекст) не короче 8 символов -- короткие (else:, ) встречаются везде."""
    return [x for x in hunk["minus"] + hunk["ctx"] if len(x) >= 8]


def hunk_seen(hunk, text):
    """Модель видела сам код, а не строку из grep: в тексте не меньше половины строк хунка (минимум 2) и
    все его удаляемые строки (если они длиннее 8 символов)."""
    keys = key_lines(hunk)
    need = min(len(keys), max(2, len(keys) // 2))
    if need < 2 or sum(k in text for k in keys) < need:
        return False
    return all(m in text for m in hunk["minus"] if len(m) >= 8)


def primary_file(hunks):
    size = Counter()
    for h in hunks:
        size[h["file"]] += sum(1 for x in h["lines"] if x[:1] in "+-")
    return size.most_common(1)[0][0]


def changed_lines(hunks):
    return sum(1 for h in hunks for x in h["lines"] if x[:1] in "+-")


def fix_text(hunks, max_changed=40):
    """Изменение в виде мини-диффа по файлам: строки '-' и '+' с одной строкой контекста вокруг. Большие
    патчи: целиком показываем первые хунки до max_changed строк, про остальные -- одна строка (файл, место),
    чтобы исправитель знал, что правок больше."""
    parts, shown, rest = [], 0, []
    for h in hunks:
        n = sum(1 for x in h["lines"] if x[:1] in "+-")
        if parts and shown + n > max_changed:
            rest.append(f"{h['file']} around line {h['old_start']} ({n} changed lines)")
            continue
        shown += n
        lines = h["lines"]
        keep = set()
        for i, x in enumerate(lines):
            if x[:1] in "+-":
                keep.update((i - 1, i, i + 1))
        body = [lines[i] for i in range(len(lines)) if i in keep]
        parts.append(f"{h['file']} (around line {h['old_start']}):\n" + "\n".join(body))
    if rest:
        parts.append("Same kind of change is also needed at: " + "; ".join(rest))
    return "\n\n".join(parts)


def lines_text(hunks):
    by_file = {}
    for h in hunks:
        end = h["old_start"] + max(h["old_len"], 1) - 1
        by_file.setdefault(h["file"], []).append(f"{h['old_start']}-{end}")
    return "; ".join(f"{f}: {', '.join(r)}" for f, r in by_file.items())


def check_command(task, hunks):
    """Тесты, которые уже есть в репозитории (файлы, которые test_patch меняет, а не создаёт); иначе --
    импорт изменённого модуля. Имена новых тестов не берём: следователь их не видит."""
    existing = re.findall(r"^--- a/(\S+)\n\+\+\+ b/\1", task.get("test_patch") or "", re.M)
    tests = [p for p in existing if p.endswith(".py")]
    if tests:
        return "python -m pytest " + " ".join(tests[:2]) + " -q -x"
    prim = primary_file(hunks)
    if not prim.endswith(".py"):
        return "python -m pytest -q -x"
    module = prim[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        module = module[: -len(".__init__")]
    return f'python -c "import {module}"'


def diagnosis(hunks, cause, check):
    files = list(dict.fromkeys(h["file"] for h in hunks))
    return (f"FILE: {', '.join(files)}\nLINES: {lines_text(hunks)}\nCAUSE: {cause.strip()}\n"
            f"FIX:\n{fix_text(hunks)}\nCHECK: {check}")


def is_source_edit(call):
    name = call["function"]["name"]
    if name not in ("edit_file", "write_file"):
        return False
    try:
        path = str(json.loads(call["function"]["arguments"]).get("filepath", ""))
    except (TypeError, ValueError, AttributeError):
        return True
    return not (path.startswith("/tmp/") or path.startswith("tmp/"))


def to_investigator(message):
    """Ход ассистента в инструментах следователя или None (правка исходника, чужой инструмент, write_file
    среди нескольких вызовов)."""
    calls = message.get("tool_calls") or []
    if not calls:
        return None
    tmp = tmp_write_args(message)
    if tmp is not None:
        return as_run_command(message, heredoc_command(tmp["filepath"], tmp["content"]))
    if any(c["function"]["name"] not in INVESTIGATOR_TOOLS for c in calls):
        return None
    return copy.deepcopy(message)


def convert_history(messages):
    """История в инструментах следователя или None, если в ней есть неприводимый ход."""
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            m = to_investigator(m)
            if m is None:
                return None
        out.append(m)
    return out


def find_cut(messages, hunks):
    """-> индекс ответа инструмента, где модель впервые увидела эталонный кусок главного файла, или None
    (не увидела до первой правки исходника)."""
    prim = primary_file(hunks)
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and any(is_source_edit(c) for c in m.get("tool_calls") or []):
            return None
        if m.get("role") == "tool" and any(hunk_seen(h, m.get("content") or "") for h in hunks if h["file"] == prim):
            return i
    return None


def _prefix(system, history, keep_exchanges, max_tool_chars):
    return [{"role": "system", "content": system}] + trim_history(history, keep_exchanges, max_tool_chars)


def _context_chars(msgs):
    return sum(len(m.get("content") or "") + sum(len(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
               for m in msgs)


def _generated_turn():
    return {"role": "assistant", "content": PLACEHOLDER, "thought": PLACEHOLDER}


def episode_records(task, messages, system, tools, stop_per_episode=2, keep_exchanges=6, max_tool_chars=4000,
                    max_context_chars=24000):
    """-> [запись] для одного эпизода: stop-пары после точки среза и одна go-пара перед ней."""
    hunks = parse_hunks(task["patch"])
    if not hunks:
        return []
    cut = find_cut(messages, hunks)
    if cut is None:
        return []
    prim = primary_file(hunks)
    keys = [k for h in hunks if h["file"] == prim for k in key_lines(h)]
    check = check_command(task, hunks)
    gen_stop = {"mode": "stop", "problem": messages[0].get("content") or "", "fix": fix_text(hunks),
                "check": check, "hunks": hunks}
    records = []
    stops = []
    for k in range(cut + 1, len(messages)):
        if messages[k].get("role") != "assistant":
            continue
        if to_investigator(messages[k]) is None:
            break
        stops.append(k)
        if len(stops) >= stop_per_episode:
            break
    for k in stops:
        history = convert_history(messages[:k])
        if history is None:
            break
        prefix = _prefix(system, history, keep_exchanges, max_tool_chars)
        if not any(key in (x.get("content") or "") for x in prefix if x.get("role") == "tool" for key in keys):
            continue          # обрезка истории съела нужный код: диагноз был бы не из чего
        if _context_chars(prefix) > max_context_chars:
            continue
        records.append({"kind": "stop", "task": task["instance_id"], "tools": tools, "thinking": True,
                        "chosen": prefix + [_generated_turn()],
                        "rejected": prefix + [to_investigator(messages[k])], "gen": gen_stop})
    # go: последний ход перед точкой среза -- вызов, который и показал нужный код. Самый первый ход не
    # берём: до него модель ещё ничего не видела, преждевременный диагноз там слишком очевидно плох.
    before = [k for k in range(1, cut) if messages[k].get("role") == "assistant"]
    if len(before) >= 2:
        k = before[-1]
        history = convert_history(messages[:k])
        conv = to_investigator(messages[k])
        if history is not None and conv is not None:
            prefix = _prefix(system, history, keep_exchanges, max_tool_chars)
            if _context_chars(prefix) <= max_context_chars:
                records.append({"kind": "go", "task": task["instance_id"], "tools": tools, "thinking": True,
                                "chosen": prefix + [conv], "rejected": prefix + [_generated_turn()],
                                "gen": {"mode": "premature"}})
    return records


def build(runs, tasks, system, exclude=(), stop_per_episode=2, max_changed=120, max_per_task=24, seed=0):
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in INVESTIGATOR_TOOLS]
    stats, by_task = Counter(), {}
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            iid = row["instance_id"]
            if iid not in tasks or row.get("error"):
                stats["no_task_or_error"] += 1
                continue
            if iid in exclude:
                stats["excluded"] += 1
                continue
            hunks = parse_hunks(tasks[iid]["patch"])
            if not hunks or changed_lines(hunks) > max_changed:
                stats["patch_too_big_or_no_source"] += 1
                continue
            recs = episode_records(tasks[iid], dialog["messages"], system, tools, stop_per_episode)
            stats["episodes_used" if recs else "episodes_no_cut"] += 1
            by_task.setdefault(iid, []).extend(recs)
    rng = random.Random(seed)
    out = []
    for iid, recs in sorted(by_task.items()):
        rng.shuffle(recs)
        out.extend(recs[:max_per_task])        # одна задача не должна заполнить всё
    rng.shuffle(out)
    stats.update(r["kind"] for r in out)
    stats["tasks"] = len({r["task"] for r in out})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--tasks", nargs="+", default=["data/comp_full/tasks.jsonl"], help="tasks.jsonl с полем patch")
    ap.add_argument("--system", default=INVESTIGATOR_PROMPT)
    ap.add_argument("--exclude-file", default=None, help="json с ключом all_ids или ids (оценочные задачи)")
    ap.add_argument("--out", default="data/inv_pairs.jsonl")
    ap.add_argument("--stop-per-episode", type=int, default=2)
    ap.add_argument("--max-changed", type=int, default=120, help="патчи длиннее (строк +/-) не берём")
    ap.add_argument("--max-per-task", type=int, default=24)
    args = ap.parse_args()
    tasks = {}
    for path in args.tasks:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    tasks.setdefault(t["instance_id"], t)
    exclude = set()
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8") as f:
            ex = json.load(f)
        exclude = set(ex.get("all_ids") or ex.get("ids") or [])
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    records, stats = build(args.runs, tasks, system, exclude, args.stop_per_episode, args.max_changed,
                           args.max_per_task)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"записей: {len(records)} -> {args.out}; {dict(stats)}")


if __name__ == "__main__":
    main()
