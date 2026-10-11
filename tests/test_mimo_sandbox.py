"""Тесты песочницы MiMo без Docker: контейнеры эмулирует tests/fake_docker.py (реальные bash/git)."""
import json
import shutil
import threading
import urllib.error
import urllib.request

import pytest

from sim.mimo_env import MimoRemoteEnv
from sim.mimo_sandbox import Budget, Busy, MimoSession, SessionManager
from sim.mimo_server import App, serve
from tests.fake_docker import FakeDocker, find_bash

pytestmark = pytest.mark.skipif(
    not (find_bash() and shutil.which("git") and shutil.which("python")),
    reason="нужны bash, git и python в PATH")

TEST_PATCH = (
    "diff --git a/mimo_test_command.sh b/mimo_test_command.sh\n"
    "new file mode 100755\n--- /dev/null\n+++ b/mimo_test_command.sh\n@@ -0,0 +1,3 @@\n"
    "+#!/usr/bin/env bash\n"
    "+python -c \"from pkg.mod import add; assert add(2, 3) == 5\"\n"
    "+exit $?\n")
# этот скрипт ещё и требует НЕизменённый tests/test_helper.py
TEST_PATCH_PROTECTED = (
    "diff --git a/mimo_test_command.sh b/mimo_test_command.sh\n"
    "new file mode 100755\n--- /dev/null\n+++ b/mimo_test_command.sh\n@@ -0,0 +1,4 @@\n"
    "+#!/usr/bin/env bash\n"
    "+python -c \"from pkg.mod import add; assert add(2, 3) == 5\" || exit 1\n"
    "+python -c \"import runpy; assert runpy.run_path('tests/test_helper.py')['X'] == 1\" || exit 1\n"
    "+exit 0\n")
FIX = {"filepath": "pkg/mod.py", "old_string": "return a - b", "new_string": "return a + b"}


def make_task(iid="t-1", test_patch=TEST_PATCH):
    return {"instance_id": iid, "cwd": "/testbed", "problem_statement": "add() is wrong",
            "test_command": "bash mimo_test_command.sh", "test_patch": test_patch,
            "verifier_timeout_sec": 60}


@pytest.fixture
def docker(tmp_path):
    return FakeDocker(tmp_path)


def call(session, name, **args):
    return json.loads(session.call(name, args))


# ---------- чистая логика ----------

def test_resolve_maps_workspace_aliases_and_blocks_traversal():
    s = MimoSession(make_task(), runner=None)
    assert s.resolve("pkg/mod.py") == "/testbed/pkg/mod.py"
    assert s.resolve("/workspace/pkg/mod.py") == "/testbed/pkg/mod.py"
    assert s.resolve("/testbed/pkg/mod.py") == "/testbed/pkg/mod.py"
    assert s.resolve("/workspace") == "/testbed"
    assert s.resolve("a/../b.py") == "/testbed/b.py"
    for bad in ("../x", "a/../../x", "", "a\x00b"):
        with pytest.raises(ValueError):
            s.resolve(bad)


def test_budget_from_dict_casts_and_ignores_unknown():
    b = Budget.from_dict({"tool_calls": "12", "time_minutes": 2, "evil": 1})
    assert b.tool_calls == 12 and b.time_minutes == 2.0 and b.turns == 60


def test_unsupported_cwd_is_rejected():
    from sim.mimo_sandbox import SandboxError
    with pytest.raises(SandboxError):
        MimoSession(dict(make_task(), cwd="/workspace/repo"), runner=None)


# ---------- инструменты внутри «контейнера» ----------

def test_tools_end_to_end(docker):
    s = MimoSession(make_task(), docker)
    s.start()
    try:
        assert docker.images == ["xiaomimimo/mimo-v2.6-rl-oss:t-1"]
        r = call(s, "read_file", filepath="/workspace/pkg/mod.py")
        assert r["status"] == "ok" and "a - b" in r["content"] and r["total_lines"] == 2
        assert call(s, "read_file", filepath="nope.py")["error_type"] == "FileNotFoundError"
        assert call(s, "read_file", filepath="../x")["error_type"] == "ValidationError"

        r = call(s, "run_command", command="echo hi && ls")
        assert r["status"] == "ok" and "hi" in r["stdout"] and "pkg" in r["stdout"]
        r = call(s, "run_command", command="exit 3")
        assert r["error_type"] == "CommandError" and r["details"]["exit_code"] == 3

        r = call(s, "edit_file", **FIX)
        assert r["status"] == "ok" and r["occurrences"] == 1 and "+    return a + b" in r["diff"]
        assert call(s, "edit_file", filepath="pkg/mod.py", old_string="zzz",
                    new_string="y")["error_type"] == "FileEditError"
        assert call(s, "write_file", filepath="pkg/new.py", content="Y = 1\n")["size"] == 6

        r = call(s, "submit_patch")
        assert r["status"] == "ok" and r["files_changed"] == 2
        assert "+    return a + b" in s.submitted_patch and "pkg/new.py" in s.submitted_patch
        assert call(s, "get_code_neighbors", node="x")["error_type"] == "GraphUnavailable"
        assert call(s, "no_such_tool")["error_type"] == "UnknownTool"
        assert call(s, "edit_file", wrong_arg=1)["error_type"] == "ValidationError"
    finally:
        s.stop()


def test_tool_call_budget_is_enforced_and_submit_is_free(docker):
    s = MimoSession(make_task(), docker, Budget(tool_calls=2))
    s.start()
    try:
        assert call(s, "run_command", command="true")["status"] == "ok"
        assert call(s, "run_command", command="true")["status"] == "ok"
        assert call(s, "run_command", command="true")["error_type"] == "BudgetExceeded"
        assert call(s, "submit_patch")["status"] == "ok"
        assert call(s, "get_status")["tool_calls_remaining"] == 0
    finally:
        s.stop()


def test_agent_preexisting_changes_are_excluded_from_patch(docker):
    """baseline снимается ПОСЛЕ старта: то, что уже было в образе, в патч не попадает."""
    s = MimoSession(make_task(), docker)
    s.start()
    try:
        assert s.compute_patch().strip() == ""
    finally:
        s.stop()


# ---------- награда ----------

def finish_with(docker, edits, test_patch=TEST_PATCH):
    mgr = SessionManager({"t-1": make_task(test_patch=test_patch)}, runner=docker, max_sessions=2)
    info = mgr.start("t-1", {"tool_calls": 30})
    for name, args in edits:
        json.loads(mgr.tool(info["session_id"], name, args))
    res = mgr.finish(info["session_id"])
    assert mgr.active() == 0 and docker.containers == {}   # контейнеры убраны
    return res


def test_correct_fix_passes_in_fresh_container(docker):
    res = finish_with(docker, [("edit_file", FIX)])
    assert res["passed"] is True and res["reason"] == "tests_passed"
    assert "+    return a + b" in res["patch"]


def test_no_change_and_wrong_change_fail(docker):
    res = finish_with(docker, [])
    assert res["passed"] is False and res["reason"] == "no_patch"
    res = finish_with(docker, [("write_file", {"filepath": "README.md", "content": "changed\n"})])
    assert res["passed"] is False and res["reason"] == "tests_failed"


def test_side_effects_outside_patch_do_not_count(docker):
    """Файл в /tmp не попадает в патч, значит, проверка в свежем контейнере его не увидит."""
    res = finish_with(docker, [("run_command", {"command": "echo 'x' > /tmp/marker"})])
    assert res["passed"] is False


def test_protected_files_are_reset_before_tests(docker):
    edits = [("edit_file", FIX),
             ("write_file", {"filepath": "tests/test_helper.py", "content": "X = 2\n"})]
    res = finish_with(docker, edits, test_patch=TEST_PATCH_PROTECTED)
    assert "tests/test_helper.py" in res["patch"]       # агент его менял
    assert res["passed"] is True                          # но перед тестами файл откатили
    # контроль: без отката X == 2 провалил бы скрипт, а значит, проверка реально работает
    res = finish_with(docker, [("write_file", {"filepath": "tests/test_helper.py",
                                                "content": "X = 2\n"})],
                      test_patch=TEST_PATCH_PROTECTED)
    assert res["passed"] is False


def test_unappliable_test_patch_is_flagged_as_infra_error(docker):
    bad = TEST_PATCH.replace("new file mode 100755", "deleted file mode 100755")
    res = finish_with(docker, [("edit_file", FIX)], test_patch=bad)
    assert res["passed"] is False and res["infra_error"] is True


def test_slots_busy_then_freed_on_abort(docker):
    mgr = SessionManager({"t-1": make_task()}, runner=docker, max_sessions=1)
    info = mgr.start("t-1")
    with pytest.raises(Busy):
        mgr.start("t-1")
    with pytest.raises(KeyError):
        mgr.start("unknown")
    assert mgr.abort(info["session_id"]) is True
    assert mgr.abort(info["session_id"]) is False
    mgr.abort(mgr.start("t-1")["session_id"])


def test_abort_all_frees_every_slot_and_container(docker):
    mgr = SessionManager({"t-1": make_task()}, runner=docker, max_sessions=2)
    mgr.start("t-1")
    mgr.start("t-1")
    with pytest.raises(Busy):
        mgr.start("t-1")
    assert mgr.abort_all() == 2 and mgr.active() == 0 and docker.containers == {}
    assert mgr.abort_all() == 0
    mgr.abort(mgr.start("t-1")["session_id"])      # слоты снова свободны


def test_remove_stale_containers_removes_only_labelled_ones():
    class Docker:
        calls = []

        def run(self, args, input=None, timeout=None):
            self.calls.append(args)
            return (0, b"aaa\nbbb\n", b"") if args[0] == "ps" else (0, b"", b"")

    d = Docker()
    mgr = SessionManager({}, runner=d)
    assert mgr.remove_stale_containers() == 2
    assert d.calls[0] == ["ps", "-aq", "--filter", "label=swegemma_mimo=1"]
    assert d.calls[1] == ["rm", "-f", "aaa", "bbb"]


def test_reaper_closes_expired_sessions(docker):
    mgr = SessionManager({"t-1": make_task()}, runner=docker, max_sessions=1, ttl_seconds=-1)
    mgr.start("t-1")
    assert mgr.reap() == 1 and mgr.active() == 0 and docker.containers == {}


# ---------- HTTP + клиент + run_episode ----------

@pytest.fixture
def server(docker, monkeypatch):
    import os
    monkeypatch.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # config.yaml
    mgr = SessionManager({"t-1": make_task()}, runner=docker, max_sessions=2)
    # wait_seconds очень мал: заставляем клиент пройти через ветку pending/poll
    app = App(mgr, mgr.tasks, token="secret", wait_seconds=0.01)
    srv = serve(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", docker
    srv.shutdown()
    srv.server_close()


def test_http_rejects_bad_token_but_health_is_open(server):
    url, _ = server
    assert json.load(urllib.request.urlopen(url + "/health"))["ok"] is True
    req = urllib.request.Request(url + "/tasks", headers={"Authorization": "Bearer wrong"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 401


def test_remote_env_full_episode_through_run_episode(server):
    from rl.rollout import run_episode
    url, docker = server
    env = MimoRemoteEnv(url, "secret", poll_sleep=0.05)
    assert [t["instance_id"] for t in env.tasks] == ["t-1"]

    class ScriptedLLM:
        def __init__(self):
            self.script = [("edit_file", FIX), ("submit_patch", {})]

        def chat(self, messages, schemas, temperature=0.0, lora_name=None):
            if not self.script:
                return dict(content="done", tool_calls=[], token_logprobs=[], finish_reason="stop")
            name, args = self.script.pop(0)
            return dict(content="", token_logprobs=[("t", -0.1)], finish_reason="tool_calls",
                        tool_calls=[{"id": f"c{len(self.script)}", "name": name,
                                     "arguments": json.dumps(args)}])

    steps, messages, passed = run_episode(env, 0, ScriptedLLM(), None, None, temperature=0.0)
    assert passed is True
    assert env.last_result["reason"] == "tests_passed" and not env.crashed
    assert "Workspace Layout" in messages[0]["content"] and "pkg/mod.py" in messages[0]["content"]
    assert "add() is wrong" in messages[0]["content"]
    assert steps and docker.containers == {}


def test_baseline_runner_records_where_the_time_goes(server, tmp_path):
    import argparse
    from sim import run_mimo_baseline as rb
    url, docker = server

    class TimedLLM:
        def __init__(self):
            self.script = [("edit_file", FIX), ("submit_patch", {})]

        def chat(self, messages, schemas, temperature=0.0, lora_name=None):
            name, args = self.script.pop(0)
            return dict(content="", token_logprobs=[], finish_reason="tool_calls",
                        usage={"prompt_tokens": 100, "completion_tokens": 40},
                        tool_calls=[{"id": f"c{len(self.script)}", "name": name,
                                     "arguments": json.dumps(args)}])

    args = argparse.Namespace(sandbox_url=url, token="secret", out=str(tmp_path),
                              temperature=0.0, poll_sleep=0.05)
    rb.run_one("t-1", args, TimedLLM())
    row = rb.load_rows(str(tmp_path / "results.jsonl"))[0]
    tm = row["timing"]
    assert row["passed"] is True and tm["turns"] == 2 and tm["ctok"] == 80 and tm["max_ptok"] == 100
    kinds = [e["k"] for e in tm["events"]]
    assert kinds.count("llm") == 2 and kinds.count("tool") == 2     # edit_file, submit_patch
    assert "reset" in kinds and "finalize" in kinds and "verify" in kinds
    assert tm["events"][0]["k"] == "reset" and all(e["s"] >= 0 for e in tm["events"])
    assert [e["name"] for e in tm["events"] if e["k"] == "tool"] == ["edit_file", "submit_patch"]
    assert all(e["par"] == 1 for e in tm["events"] if e["k"] == "llm")   # один эпизод шёл
    rb.print_timing([row])                                          # не падает на реальных данных


def test_baseline_client_prepends_system_prompt_only_when_given(monkeypatch):
    from llm.llm_client import LLMClient
    from sim.run_mimo_baseline import BaselineClient
    seen = []
    monkeypatch.setattr(LLMClient, "chat", lambda self, messages, tools, **kw: seen.append(messages) or {})
    user = [{"role": "user", "content": "task"}]
    BaselineClient(base_url="http://x").chat(user, [])
    BaselineClient(base_url="http://x", system="RULES").chat(user, [])
    assert seen[0] == user
    assert seen[1] == [{"role": "system", "content": "RULES"}] + user
    assert user == [{"role": "user", "content": "task"}]      # исходный список не изменён


def test_clean_start_closes_abandoned_sessions_over_http(server):
    url, docker = server
    env = MimoRemoteEnv(url, "secret", poll_sleep=0.05)
    env.reset(0)                                    # «брошенный» эпизод: его никто не закроет
    assert json.load(urllib.request.urlopen(url + "/health"))["active"] == 1
    other = MimoRemoteEnv(url, "secret", task_ids=[])
    assert other._http("POST", "/abort_all", {})["result"] == 1
    assert json.load(urllib.request.urlopen(url + "/health"))["active"] == 0
    assert docker.containers == {}


def test_remote_env_marks_crash_when_server_is_gone(server):
    url, _ = server
    env = MimoRemoteEnv(url, "secret", poll_sleep=0.05, retries=1)
    env.reset(0)
    env.base_url = "http://127.0.0.1:1"          # сервер «пропал»
    obs = json.loads(env.call_tool("run_command", {"command": "true"}))
    assert obs["error_type"] == "SandboxUnavailable"
    assert env.crashed and env.done
    assert env.finalize() == "" and env.verify() is False
