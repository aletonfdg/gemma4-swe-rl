import json

from sim.comp_sandbox import (CompSession, CompSessionManager, IMAGES, _junit_ok, _patch_text,
                              _test_files, load_comp_tasks, verify_comp_patch)
from sim.mimo_sandbox import MimoSession, SessionManager


class Recorder:
    """Поддельный docker: запоминает вызовы, на всё отвечает успехом (код 0, пустой вывод)."""

    def __init__(self, replies=None):
        self.calls, self.replies = [], replies or {}

    def run(self, args, input=None, timeout=None):
        self.calls.append((list(args), input))
        for key, reply in self.replies.items():
            if key in " ".join(args):
                return reply
        return 0, b"", b""


def task(repo="fastapi/fastapi", iid="fastapi_1"):
    return {"instance_id": iid, "repo": repo, "base_commit": "c" * 40, "problem_statement": "bug",
            "test_patch": "diff --git a/tests/test_a.py b/tests/test_a.py\n+++ b/tests/test_a.py\n+x\n",
            "patch": "p"}


def test_load_comp_tasks_filters_by_repo_and_valid_list(tmp_path):
    rows = [task(iid="fastapi_1"), task(iid="fastapi_2"), task("encode/httpx", "httpx_1")]
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps({"valid": ["fastapi_2", "httpx_1"], "invalid": []}))
    assert sorted(load_comp_tasks(str(tmp_path))) == ["fastapi_1", "fastapi_2"]   # httpx без образа
    assert sorted(load_comp_tasks(str(tmp_path), str(valid))) == ["fastapi_2"]


def test_session_uses_workspace_repo_image_and_mounts(tmp_path):
    runner = Recorder()
    sess = CompSession(task("Textualize/rich", "rich_1"), runner, comp_dir=str(tmp_path))
    args = sess.docker_run_args()
    assert sess.workdir == "/workspace" and sess.image == IMAGES["Textualize/rich"]
    assert args[-3:] == ["swebench-rich:latest", "sleep", "infinity"]
    assert f"{tmp_path.as_posix()}/wheels:/wheels:ro" in "".join(args).replace("\\", "/")
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert sess.resolve("/workspace/a/b.py") == "/workspace/a/b.py"
    assert sess.resolve("a/b.py") == "/workspace/a/b.py"
    assert MimoSession.workdir == "/testbed"          # MiMo не затронут рефакторингом


def test_graph_tools_report_unavailable_without_graph_files(tmp_path):
    sess = CompSession(task(), Recorder(), comp_dir=str(tmp_path))
    assert not sess.has_graph() and sess.repo_name() == "fastapi/fastapi"
    out = json.loads(sess.call("get_code_neighbors", {"node": "x"}))
    assert out["status"] == "error" and out["error_type"] == "GraphUnavailable"


def test_snapshot_is_streamed_into_container_and_baseline_tagged(tmp_path):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "fastapi_1.tgz").write_bytes(b"TGZ")
    runner = Recorder()
    sess = CompSession(task(), runner, comp_dir=str(tmp_path), graphs=False)
    sess.start()
    streamed = [c for c in runner.calls if c[1] == b"TGZ"]
    assert streamed and "tar xzf - -C /workspace" in " ".join(streamed[0][0])
    script = " ".join(" ".join(c[0]) for c in runner.calls)
    assert "setup.py --fast-path fastapi/fastapi" in script and "git tag -f _swegemma_baseline" in script


def test_patch_text_adds_missing_final_newline_and_junit_check():
    assert _patch_text("a") == "a\n" and _patch_text("a\n") == "a\n"
    assert _test_files(task()["test_patch"]) == ["tests/test_a.py"]
    ok = '<testsuite tests="3" failures="0" errors="0" skipped="1"></testsuite>'
    assert _junit_ok(ok)
    assert not _junit_ok('<testsuite tests="3" failures="1" errors="0" skipped="0"></testsuite>')
    assert not _junit_ok('<testsuite tests="2" failures="0" errors="0" skipped="2"></testsuite>')
    assert not _junit_ok("not xml")


def test_verify_rejects_empty_patch_without_touching_docker():
    runner = Recorder()
    res = verify_comp_patch(runner, task(), "  ", ".")
    assert res["reason"] == "no_patch" and not res["passed"] and runner.calls == []


def test_verify_resets_test_files_from_patch_and_test_patch(tmp_path):
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "fastapi_1.tgz").write_bytes(b"TGZ")
    runner = Recorder()
    verify_comp_patch(runner, task(), "diff --git a/src/m.py b/src/m.py\n+++ b/src/m.py\n+1\n", str(tmp_path))
    resets = [" ".join(c[0]) for c in runner.calls if "git checkout _swegemma_baseline --" in " ".join(c[0])]
    assert resets and "tests/test_a.py" in resets[0] and "src/m.py" not in resets[0]


def test_manager_hooks_build_comp_sessions(tmp_path):
    mgr = CompSessionManager({"fastapi_1": task()}, str(tmp_path), graphs=False, runner=Recorder())
    sess = mgr._make_session(task(), None)
    assert isinstance(sess, CompSession) and sess.image == "swebench-fastapi:latest"
    assert isinstance(SessionManager({}, runner=Recorder())._make_session(
        {"instance_id": "x", "cwd": "/testbed"}, None), MimoSession)
