# tools/analyze_edits.py -- как модель пользуется edit_file и как быстро доходит до первой правки
"""
Читает папки результатов sim.run_mimo_baseline (results.jsonl + messages/) и печатает:
  * вызовы edit_file по исходу (ok / FileEditError / ...) и по виду old_string: настоящие переводы
    строк или литеральные символы "\\n" (модель печатает их как в JSON -- такой текст не находится в файле);
  * решённость эпизодов в группах: без edit_file / только чистые вызовы / 1-2 или 3+ вызова с литеральным "\\n";
  * причины завершения (tests_failed, no_patch, ...) и ход первой правки у решённых и нет.

Запуск (в Colab после git pull):
  python -m tools.analyze_edits results/sft_think_v1_eval
  python -m tools.analyze_edits results/traj_think results/v3_prompt     # несколько папок суммируются
"""
import argparse
import collections
import json
import os
import statistics

BACKSLASH_N = chr(92) + "n"      # два символа: обратная косая и n
EDIT_TOOLS = ("edit_file", "write_file")


def dialog_name(row):
    rep = row.get("rep", 0)
    return f"{row['instance_id']}.json" if rep == 0 else f"{row['instance_id']}__r{rep}.json"


def load_episodes(run_dir):
    """-> [(row, messages)]: последняя строка на (задача, повтор), как в раннере; эпизоды с ошибкой пропускаем."""
    last = {}
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                last[(row["instance_id"], row.get("rep", 0))] = row
    out = []
    for row in last.values():
        path = os.path.join(run_dir, "messages", dialog_name(row))
        if row.get("error") or not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            out.append((row, json.load(f)["messages"]))
    return out


def tool_result(messages):
    """tool_call_id -> разобранный JSON ответа инструмента ({} если не JSON)."""
    out = {}
    for m in messages:
        if m.get("role") == "tool":
            try:
                out[m.get("tool_call_id")] = json.loads(m.get("content") or "{}")
            except ValueError:
                out[m.get("tool_call_id")] = {}
    return out


def old_string_kind(old):
    if BACKSLASH_N in old:
        return "literal_backslash_n"
    return "real_newline" if "\n" in old else "single_line"


def analyze_episode(row, messages):
    """-> dict: вызовы edit_file [(исход, вид)], номер хода первой правки, число ходов."""
    results = tool_result(messages)
    calls, turn, first_edit = [], 0, None
    for m in messages:
        if m.get("role") != "assistant":
            continue
        turn += 1
        for tc in m.get("tool_calls") or []:
            name = tc["function"]["name"]
            if name in EDIT_TOOLS and first_edit is None:
                first_edit = turn
            if name != "edit_file":
                continue
            try:
                args = json.loads(tc["function"]["arguments"])
            except ValueError:
                calls.append(("bad_json_args", "single_line"))
                continue
            res = results.get(tc["id"], {})
            outcome = res.get("error_type") if res.get("status") == "error" else res.get("status", "?")
            calls.append((outcome, old_string_kind(args.get("old_string") or "")))
    return {"calls": calls, "first_edit": first_edit, "turns": turn}


def group_of(calls):
    if not calls:
        return "no edit_file calls"
    literal = sum(1 for _, kind in calls if kind == "literal_backslash_n")
    if literal == 0:
        return "edit_file, no literal \\n"
    return "literal \\n in 1-2 calls" if literal <= 2 else "literal \\n in 3+ calls"


def summarize(episodes):
    calls = collections.Counter()
    groups = collections.defaultdict(lambda: [0, 0])
    reasons = collections.Counter()
    first = {True: [], False: []}
    for row, messages in episodes:
        info = analyze_episode(row, messages)
        calls.update(info["calls"])
        solved = bool(row.get("passed"))
        g = groups[group_of(info["calls"])]
        g[0] += 1
        g[1] += solved
        reasons[(row.get("result") or {}).get("reason")] += 1
        if info["first_edit"]:
            first[solved].append(info["first_edit"])
    return calls, groups, reasons, first


def report(episodes):
    calls, groups, reasons, first = summarize(episodes)
    total = sum(calls.values())
    literal = sum(n for (_, kind), n in calls.items() if kind == "literal_backslash_n")
    print(f"эпизодов: {len(episodes)}, вызовов edit_file: {total}")
    for (outcome, kind), n in sorted(calls.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {outcome:15s} {kind}")
    if total:
        ok_literal = calls.get(("ok", "literal_backslash_n"), 0)
        print(f"с литеральным \\n в old_string: {literal} ({100 * literal / total:.0f}% вызовов), удачных из них {ok_literal}")
    print("решённость по группам эпизодов:")
    for name, (n, solved) in sorted(groups.items()):
        print(f"  {name:28s} эпизодов {n:4d}, решено {solved:4d} ({100 * solved / n:.0f}%)")
    print("причины завершения:", dict(reasons))
    for solved, label in ((True, "решённые"), (False, "нерешённые")):
        if first[solved]:
            print(f"ход первой правки, медиана ({label}): {statistics.median(first[solved])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="папки с results.jsonl и messages/")
    args = ap.parse_args()
    episodes = []
    for run_dir in args.runs:
        episodes.extend(load_episodes(run_dir))
    report(episodes)


if __name__ == "__main__":
    main()
