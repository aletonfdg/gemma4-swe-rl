"""Анализ результатов `swegemma eval` (папка --results-dir) и сравнение двух прогонов.

    python evaluation/analyze_results.py report  results/run_01 [--out DIR]
    python evaluation/analyze_results.py compare results/run_A results/run_B

Ожидаемая раскладка (HARNESS_README.md, раздел 9.2):
    summary.json, task_results.jsonl, patches/<id>.patch, test_outputs/<id>.log, traces/...
Имена полей внутри JSON в README не расписаны, поэтому парсер перебирает варианты
(см. KEYS_*) и ничего не додумывает: чего нет в данных -- помечается как None/unknown.
Только стандартная библиотека.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

KEYS_ID = ("instance_id", "task_id", "id")
KEYS_RESOLVED = ("resolved", "passed", "success")
KEYS_ERROR = ("error", "agent_error", "error_message")
KEYS_TOOL_CALLS = ("tool_calls_used", "tool_calls", "n_tool_calls")
KEYS_ELAPSED = ("agent_elapsed_seconds", "elapsed_seconds", "duration_seconds", "duration")

PROTECTED_NAMES = {"conftest.py", "pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg",
                   ".pytest.ini", "sitecustomize.py", "usercustomize.py", "_swegemma_stubs.py"}
TEST_DIRS = {"tests", "test", "testing"}
STRAY_RE = re.compile(r"^(repro|reproduce|reproduction|scratch|tmp|temp|debug|demo|try|check)\w*\.py$"
                      r"|\.(orig|rej|bak)$")

CATEGORIES = ("resolved", "no_patch", "patch_apply_failed", "timeout_or_budget",
              "tests_failed", "other_error")


def _first(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def is_protected(path):
    parts = path.split("/")
    base = parts[-1]
    if base in PROTECTED_NAMES or base.endswith(".pth"):
        return True
    if (base.startswith("test_") and base.endswith(".py")) or base.endswith("_test.py"):
        return True
    return base.endswith(".py") and any(p in TEST_DIRS for p in parts[:-1])


def parse_patch(text):
    """-> список {'path', 'new_file'} для каждого файла в unified diff."""
    files = []
    for chunk in re.split(r"(?m)^diff --git ", text):
        m = re.match(r"a/(\S+) b/(\S+)", chunk)
        if m:
            files.append({"path": m.group(2),
                          "new_file": bool(re.search(r"(?m)^new file mode", chunk))})
    return files


def patch_flags(text):
    files = parse_patch(text)
    return {
        "files_changed": len(files),
        "touched_tests_or_config": any(is_protected(f["path"]) for f in files),
        "stray_files": any(f["new_file"] and STRAY_RE.search(os.path.basename(f["path"]))
                           for f in files),
    }


def classify(resolved, patch_text, error_text, log_text):
    if resolved:
        return "resolved"
    if not patch_text.strip() or patch_text.strip() == "NO_PATCH":
        return "no_patch"
    blob = f"{error_text}\n{log_text}"
    if re.search(r"Failed to apply agent patch", blob, re.I):
        return "patch_apply_failed"
    if re.search(r"timeout|timed out|budget|TimeoutExceeded|max_turns", blob, re.I):
        return "timeout_or_budget"
    if re.search(r"\b\d+ failed\b|FAILED|AssertionError|exit_code\W+[1-9]|errors? during collection",
                 blob):
        return "tests_failed"
    return "other_error"


def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def load_run(run_dir):
    rows = {}
    jl = os.path.join(run_dir, "task_results.jsonl")
    if os.path.exists(jl):
        for line in _read(jl).splitlines():
            if line.strip():
                d = json.loads(line)
                tid = _first(d, KEYS_ID)
                if tid:
                    rows[tid] = d
    pdir = os.path.join(run_dir, "patches")
    if os.path.isdir(pdir):                       # задачи, которых нет в jsonl, но есть патч
        for fn in os.listdir(pdir):
            if fn.endswith(".patch"):
                rows.setdefault(fn[:-6], {"instance_id": fn[:-6]})

    tasks = []
    for tid, d in sorted(rows.items()):
        patch = _read(os.path.join(run_dir, "patches", f"{tid}.patch"))
        log = _read(os.path.join(run_dir, "test_outputs", f"{tid}.log"))
        resolved = bool(_first(d, KEYS_RESOLVED))
        err = str(_first(d, KEYS_ERROR) or "")
        t = {"task_id": tid, "repo": tid.rsplit("_", 1)[0], "resolved": resolved,
             "category": classify(resolved, patch, err, log),
             "patch_bytes": len(patch.encode("utf-8")),
             "tool_calls": _first(d, KEYS_TOOL_CALLS),
             "elapsed_s": _first(d, KEYS_ELAPSED)}
        t.update(patch_flags(patch))
        tasks.append(t)
    return tasks


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def sign_test_p(only_a, only_b):
    """Точный двусторонний тест знаков по расходящимся парам."""
    n, k = only_a + only_b, min(only_a, only_b)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def _mean(vals):
    vals = [float(v) for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def render_report(tasks):
    n = len(tasks)
    k = sum(t["resolved"] for t in tasks)
    lo, hi = wilson(k, n)
    out = ["# Результаты прогона\n",
           f"Решено: **{k}/{n}** = {100 * k / max(n, 1):.1f}%  "
           f"(95% доверительный интервал Уилсона: {100 * lo:.0f}%..{100 * hi:.0f}%)\n"]
    if n < 60:
        out.append(f"> Выборка мала ({n} задач): одна задача = {100 / max(n, 1):.1f} п.п., "
                   f"интервал широкий. Не делайте выводов о малых различиях.\n")

    cats = Counter(t["category"] for t in tasks)
    out.append("## Причины\n\n| категория | задач |\n|---|---|")
    out += [f"| {c} | {cats.get(c, 0)} |" for c in CATEGORIES]

    out.append("\n## Флаги (могут совпадать с любой категорией)\n")
    out.append(f"- патч трогает тесты/конфиги (будут сброшены харнессом): "
               f"{sum(t['touched_tests_or_config'] for t in tasks)}")
    out.append(f"- в патче лишние файлы (repro/scratch/.orig): {sum(t['stray_files'] for t in tasks)}")

    by_repo = defaultdict(list)
    for t in tasks:
        by_repo[t["repo"]].append(t)
    out.append("\n## По репозиториям\n\n| репозиторий | решено | всего |\n|---|---|---|")
    for repo, ts in sorted(by_repo.items()):
        out.append(f"| {repo} | {sum(t['resolved'] for t in ts)} | {len(ts)} |")

    tc, el = _mean(t["tool_calls"] for t in tasks), _mean(t["elapsed_s"] for t in tasks)
    out.append("\n## Ресурсы\n")
    out.append(f"- среднее число вызовов инструментов: {'н/д' if tc is None else f'{tc:.1f}'}")
    out.append(f"- среднее время агента, с: {'н/д' if el is None else f'{el:.0f}'}")
    if tc is None and el is None:
        out.append("- (в task_results.jsonl не найдено ни одного из полей "
                   f"{KEYS_TOOL_CALLS + KEYS_ELAPSED}; смотрите traces/ вручную)")

    bad = [t["task_id"] for t in tasks if not t["resolved"]]
    out.append(f"\n## Нерешённые ({len(bad)})\n\n" + (", ".join(bad) if bad else "нет"))
    return "\n".join(out) + "\n"


def cmd_report(args):
    tasks = load_run(args.run)
    if not tasks:
        sys.exit(f"В {args.run} не найдено ни task_results.jsonl, ни patches/*.patch")
    out_dir = args.out or args.run
    os.makedirs(out_dir, exist_ok=True)
    text = render_report(tasks)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(out_dir, "tasks.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tasks[0].keys()))
        w.writeheader()
        w.writerows(tasks)
    print(text)
    print(f"-> {out_dir}/report.md, {out_dir}/tasks.csv")


def cmd_compare(args):
    a = {t["task_id"]: t for t in load_run(args.run_a)}
    b = {t["task_id"]: t for t in load_run(args.run_b)}
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("У прогонов нет общих задач")
    only_a = [i for i in common if a[i]["resolved"] and not b[i]["resolved"]]
    only_b = [i for i in common if b[i]["resolved"] and not a[i]["resolved"]]
    ka, kb = sum(a[i]["resolved"] for i in common), sum(b[i]["resolved"] for i in common)
    p = sign_test_p(len(only_a), len(only_b))
    print(f"Общих задач: {len(common)} (A: {len(a)}, B: {len(b)})")
    print(f"A решил {ka}, B решил {kb}")
    print(f"только A: {len(only_a)} {only_a}")
    print(f"только B: {len(only_b)} {only_b}")
    print(f"двусторонний тест знаков: p = {p:.3f}"
          + ("  (различие НЕ значимо)" if p >= 0.05 else "  (различие значимо при 0.05)"))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # консоль Windows по умолчанию не UTF-8
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("run")
    r.add_argument("--out")
    r.set_defaults(fn=cmd_report)
    c = sub.add_parser("compare")
    c.add_argument("run_a")
    c.add_argument("run_b")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
