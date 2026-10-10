import importlib.util
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "analyze_results", os.path.join(ROOT, "evaluation", "analyze_results.py"))
ar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ar)

SRC_PATCH = "diff --git a/pkg/mod.py b/pkg/mod.py\n--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1 +1 @@\n-a\n+b\n"
TEST_PATCH = "diff --git a/tests/test_x.py b/tests/test_x.py\n--- a/tests/test_x.py\n+++ b/tests/test_x.py\n@@ -1 +1 @@\n-a\n+b\n"
STRAY_PATCH = SRC_PATCH + ("diff --git a/repro.py b/repro.py\nnew file mode 100644\n"
                           "--- /dev/null\n+++ b/repro.py\n@@ -0,0 +1 @@\n+print(1)\n")


def make_run(tmp_path, name, items):
    """items: {task_id: dict(resolved, patch, log, error, extra)}"""
    run = tmp_path / name
    (run / "patches").mkdir(parents=True)
    (run / "test_outputs").mkdir()
    lines = []
    for tid, it in items.items():
        (run / "patches" / f"{tid}.patch").write_text(it.get("patch", ""), encoding="utf-8")
        (run / "test_outputs" / f"{tid}.log").write_text(it.get("log", ""), encoding="utf-8")
        row = {"instance_id": tid, "resolved": it.get("resolved", False),
               "error": it.get("error")}
        row.update(it.get("extra", {}))
        lines.append(json.dumps(row))
    (run / "task_results.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return str(run)


def test_is_protected_matches_harness_rules():
    for p in ("tests/test_a.py", "pkg/test_b.py", "pkg/b_test.py", "conftest.py", "pytest.ini",
              "pyproject.toml", "tox.ini", "x/sitecustomize.py", "a/b.pth", "tests/helpers.py"):
        assert ar.is_protected(p), p
    for p in ("pkg/mod.py", "docs_src/app.py", "README.md", "pkg/latest.py"):
        assert not ar.is_protected(p), p


def test_patch_flags():
    assert ar.patch_flags(SRC_PATCH) == {"files_changed": 1, "touched_tests_or_config": False,
                                         "stray_files": False}
    assert ar.patch_flags(TEST_PATCH)["touched_tests_or_config"]
    f = ar.patch_flags(STRAY_PATCH)
    assert f["files_changed"] == 2 and f["stray_files"]


def test_classification_categories(tmp_path):
    run = make_run(tmp_path, "r", {
        "repo_1": {"resolved": True, "patch": SRC_PATCH},
        "repo_2": {"patch": ""},
        "repo_3": {"patch": "NO_PATCH"},
        "repo_4": {"patch": SRC_PATCH, "error": "Failed to apply agent patch: x"},
        "repo_5": {"patch": SRC_PATCH, "error": "TimeoutExceeded after 300s"},
        "repo_6": {"patch": SRC_PATCH, "log": "== 2 failed, 3 passed in 1.2s =="},
        "repo_7": {"patch": SRC_PATCH, "log": "something unrecognised"},
    })
    cats = {t["task_id"]: t["category"] for t in ar.load_run(run)}
    assert cats == {"repo_1": "resolved", "repo_2": "no_patch", "repo_3": "no_patch",
                    "repo_4": "patch_apply_failed", "repo_5": "timeout_or_budget",
                    "repo_6": "tests_failed", "repo_7": "other_error"}


def test_unknown_field_names_do_not_crash_and_report_is_honest(tmp_path):
    run = make_run(tmp_path, "r", {"repo_1": {"patch": SRC_PATCH}, "repo_2": {"patch": SRC_PATCH}})
    tasks = ar.load_run(run)
    assert all(t["tool_calls"] is None for t in tasks)
    text = ar.render_report(tasks)
    assert "н/д" in text and "Выборка мала" in text


def test_tool_calls_and_elapsed_are_averaged_when_present(tmp_path):
    run = make_run(tmp_path, "r", {
        "repo_1": {"patch": SRC_PATCH, "extra": {"tool_calls_used": 10, "agent_elapsed_seconds": 100}},
        "repo_2": {"patch": SRC_PATCH, "extra": {"tool_calls_used": 20, "agent_elapsed_seconds": 200}},
    })
    text = ar.render_report(ar.load_run(run))
    assert "15.0" in text and "150" in text


def test_wilson_interval_contains_point_estimate_and_is_wide_for_small_n():
    lo, hi = ar.wilson(3, 13)
    assert lo < 3 / 13 < hi
    assert hi - lo > 0.3
    assert ar.wilson(0, 0) == (0.0, 0.0)


def test_sign_test():
    assert ar.sign_test_p(0, 0) == 1.0
    assert ar.sign_test_p(5, 5) == 1.0
    assert math.isclose(ar.sign_test_p(0, 5), 2 / 32)
    assert ar.sign_test_p(0, 10) < 0.01


def test_compare_cli(tmp_path, capsys):
    a = make_run(tmp_path, "a", {"r_1": {"resolved": True, "patch": SRC_PATCH},
                                 "r_2": {"patch": SRC_PATCH}, "r_3": {"patch": ""}})
    b = make_run(tmp_path, "b", {"r_1": {"resolved": True, "patch": SRC_PATCH},
                                 "r_2": {"resolved": True, "patch": SRC_PATCH}, "r_3": {"patch": ""}})

    class A:
        run_a, run_b = a, b
    ar.cmd_compare(A)
    out = capsys.readouterr().out
    assert "A решил 1, B решил 2" in out and "только B: 1" in out and "НЕ значимо" in out
