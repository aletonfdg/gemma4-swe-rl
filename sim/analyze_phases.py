# sim/analyze_phases.py -- where an episode spends its calls and time: search -> first edit -> after it
"""
For every episode of a run (results.jsonl + messages/*.json):
  first_try / first_ok -- call number and second of the first attempt to change source and of the first
                          successful change (edit_file, write_file not to /tmp, run_command with sed -i,
                          perl -pi, a python heredoc, open(..., 'w'), patch, git apply)
  fails / repeats      -- failed change attempts and exact repeats of a failed one
  after                -- calls after the last successful change and their kinds (test / read / search / edit)
  end                  -- submit (submit_patch was called) or timeout (calls/time ran out)

  python -m sim.analyze_phases RUN_DIR [RUN_DIR2]     # with two runs, also a per-task table A | B
"""
import json
import re
import sys
from collections import Counter

from train.build_sft_data import load_run

WRITE_CMD = re.compile(r"sed\s+-i|perl\s+-pi|python3?\s+-\s*<<|open\([^)]*['\"]w['\"]|\bpatch\s+-p|git\s+apply")
TEST_CMD = re.compile(r"pytest|python3?\s+-c|unittest")


def calls_of(messages):
    """-> [(name, args, status)] in call order."""
    status = {}
    for m in messages:
        if m.get("role") == "tool":
            try:
                c = json.loads(m.get("content") or "{}")
            except ValueError:
                c = {}
            status[m.get("tool_call_id")] = c.get("status") if isinstance(c, dict) else None
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                args = {}
            out.append((fn.get("name"), args if isinstance(args, dict) else {}, status.get(tc.get("id"))))
    return out


def kind(name, args):
    if name == "edit_file":
        return "edit"
    if name == "write_file":
        return "tmp" if str(args.get("filepath", "")).startswith("/tmp") else "edit"
    if name == "run_command":
        cmd = str(args.get("command", ""))
        if WRITE_CMD.search(cmd) and not re.search(r"open\(['\"]/tmp/|>\s*/tmp/|cat\s*>\s*/tmp", cmd):
            return "edit"
        if TEST_CMD.search(cmd):
            return "test"
        return "search"
    if name == "read_file":
        return "read"
    return name or "?"


def analyze(row, dialog):
    calls = calls_of(dialog.get("messages") or [])
    kinds = [kind(n, a) for n, a, _ in calls]
    tool_at = [e["at"] for e in (row.get("timing") or {}).get("events", []) if e.get("k") == "tool"]

    def at(i):  # 1-based call number -> seconds since episode start
        return round(tool_at[i - 1]) if i and i <= len(tool_at) else None

    tries = [i + 1 for i, k in enumerate(kinds) if k == "edit"]
    oks = [i for i in tries if calls[i - 1][2] == "ok"]
    fails = [json.dumps(calls[i - 1][1], sort_keys=True) for i in tries if calls[i - 1][2] != "ok"]
    last_ok = oks[-1] if oks else None
    after = Counter(kinds[last_ok:]) if last_ok else Counter()
    after.pop("submit_patch", None)
    submit = any(n == "submit_patch" for n, *_ in calls)
    first_try = tries[0] if tries else None
    return {
        "id": row["instance_id"], "passed": bool(row.get("passed")), "reason": row.get("reason"),
        "calls": len(calls), "first_try": first_try, "first_ok": oks[0] if oks else None,
        "t_try": at(first_try), "t_ok": at(oks[0] if oks else None), "t_last_ok": at(last_ok),
        "n_ok": len(oks), "fails": len(fails), "repeats": len(fails) - len(set(fails)),
        "after": sum(after.values()), "after_kinds": dict(after), "submit": submit,
        "end": "submit" if submit else "timeout",
        "before_kinds": dict(Counter(kinds[:(first_try or len(kinds) + 1) - 1])),
    }


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else "-"


def summary(name, eps):
    solved = [e for e in eps if e["passed"]]
    print(f"\n=== {name}: {len(eps)} episodes, solved {len(solved)}")
    print("  end:", dict(Counter(e["end"] for e in eps)),
          f"| submit_patch among solved: {sum(e['submit'] for e in solved)}/{len(solved)}")
    print("  no change attempt at all:", sum(e["first_try"] is None for e in eps),
          "| attempted, none succeeded:", sum(bool(e["first_try"]) and not e["first_ok"] for e in eps))
    for label, sel in (("solved", solved), ("failed", [e for e in eps if not e["passed"]])):
        print(f"  {label:6} median: first attempt call {med([e['first_try'] for e in sel])} "
              f"({med([e['t_try'] for e in sel])} s), first success call {med([e['first_ok'] for e in sel])} "
              f"({med([e['t_ok'] for e in sel])} s), last success {med([e['t_last_ok'] for e in sel])} s, "
              f"calls {med([e['calls'] for e in sel])}, calls after last success "
              f"{med([e['after'] for e in sel if e['first_ok']])}")
    ft = [e["first_try"] or 999 for e in eps]
    print("  first change attempt at call: <=8", sum(x <= 8 for x in ft), "| 9-15", sum(9 <= x <= 15 for x in ft),
          "| 16+", sum(16 <= x < 999 for x in ft), "| never", sum(x == 999 for x in ft))
    tt = [e["t_try"] for e in eps if e["t_try"] is not None]
    print("  first change attempt at second: <=90", sum(x <= 90 for x in tt), "| 91-180",
          sum(90 < x <= 180 for x in tt), "| 181+", sum(x > 180 for x in tt))
    early = [e for e in eps if e["t_try"] is not None and e["t_try"] <= 120]
    late = [e for e in eps if e["t_try"] is not None and e["t_try"] > 120]
    print(f"  solved when first attempt <=120 s: {sum(e['passed'] for e in early)}/{len(early)}, "
          f">120 s: {sum(e['passed'] for e in late)}/{len(late)}")
    print("  failed change attempts", sum(e["fails"] for e in eps), "| exact repeats", sum(e["repeats"] for e in eps))
    for title, key in (("before the first change attempt", "before_kinds"), ("after the last success", "after_kinds")):
        c = Counter()
        for e in eps:
            c.update(e[key])
        print(f"  calls {title}:", dict(c.most_common()))
    return {e["id"]: e for e in eps}


def main():
    tables = [summary(r, [analyze(row, d) for row, d in load_run(r)]) for r in sys.argv[1:]]
    if len(tables) == 2:
        a, b = tables
        print("\nper task  (solved, first attempt call/second, first success call, calls, end)   A | B")

        def f(e):
            if not e:
                return "n/a".ljust(26)
            return (f"{'+' if e['passed'] else '-'} {e['first_try'] or '-':>2}/{e['t_try'] or '-':>3}s "
                    f"{e['first_ok'] or '-':>2} {e['calls']:>2} {e['end'][:3]}")
        for tid in sorted(set(a) | set(b)):
            print(f"  {tid:16} {f(a.get(tid))} | {f(b.get(tid))}")


if __name__ == "__main__":
    main()
