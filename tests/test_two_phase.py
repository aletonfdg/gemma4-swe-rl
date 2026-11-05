import json
from types import SimpleNamespace

from sim.two_phase import FIXER_TOOLS, INVESTIGATOR_TOOLS, run_episode_two_phase


class FakeCtx:
    def __init__(self, calls):
        self.tool_calls_used, self.turns_used, self.patch_submitted, self.calls = 0, 0, False, calls

    def budget_exhausted(self):
        return self.tool_calls_used >= self.calls or self.turns_used >= 50


class FakeEnv:
    def __init__(self, calls=20):
        self.task = {"instance_id": "t1", "repo": "r", "problem_statement": "bug"}
        self.budget = SimpleNamespace(turns=50)
        self.calls_limit = calls
        self.done, self.crashed, self.log = False, False, []

    def reset(self, idx):
        self.ctx = FakeCtx(self.calls_limit)
        return "TASK PROMPT"

    def call_tool(self, name, args):
        self.ctx.tool_calls_used += 1
        self.log.append(name)
        if name == "submit_patch":
            self.ctx.patch_submitted = True
        return json.dumps({"status": "ok"})

    def nudge_message(self, content, finish_reason, had_tool_call):
        return None if had_tool_call else "continue"

    def finalize(self):
        return "diff" if "edit_file" in self.log else ""

    def verify(self, patch):
        return bool(patch)

    def cleanup(self):
        pass


def call(name, args, i):
    return {"id": f"c{i}", "name": name, "arguments": json.dumps(args)}


class ScriptedClient:
    """Отвечает по сценарию; запоминает системный промпт и инструменты каждого запроса."""

    def __init__(self, script, seen):
        self.script, self.seen = script, seen

    def chat(self, messages, tools, **kw):
        self.seen.append((messages[0]["content"] if messages[0]["role"] == "system" else None,
                          sorted(t["function"]["name"] for t in tools), len(messages)))
        content, calls = self.script.pop(0)
        return {"content": content, "tool_calls": calls, "finish_reason": "stop", "reasoning": None}


def make_factory(scripts, seen, phases=None):
    from sim.two_phase import SystemClient
    it = iter(scripts)

    def factory(system, phase):
        if phases is not None:
            phases.append(phase)
        return SystemClient(ScriptedClient(next(it), seen), system)
    return factory


def test_investigator_hands_diagnosis_to_fixer_with_fresh_history():
    seen = []
    inv = [("", [call("read_file", {"filepath": "a.py"}, 1)]),
           ("", [call("edit_file", {"filepath": "a.py"}, 2)]),          # запрещён следователю
           ("FILE: a.py\nFIX: x", [])]
    fix = [("", [call("edit_file", {"filepath": "a.py", "old_string": "a", "new_string": "b"}, 3)]),
           ("", [call("submit_patch", {}, 4)])]
    env = FakeEnv()
    phases = []
    steps, messages, passed, stats = run_episode_two_phase(env, make_factory([inv, fix], seen, phases), "INV", "BASE")
    assert phases == ["investigator", "fixer"]
    assert passed and env.log == ["read_file", "edit_file", "submit_patch"]   # чужой вызов не дошёл до песочницы
    assert stats["inv_stopped"] and stats["inv_calls"] == 1 and stats["inv_forbidden"] == 1
    assert stats["fix_first_edit_call"] == 1 and stats["fix_calls"] == 2
    inv_sys, inv_tools, _ = seen[0]
    assert inv_sys == "INV" and inv_tools == sorted(INVESTIGATOR_TOOLS)
    fix_sys, fix_tools, n_msgs = seen[3]
    assert fix_sys.startswith("BASE") and "FILE: a.py" in fix_sys and fix_tools == sorted(FIXER_TOOLS)
    assert n_msgs == 2                                  # system + условие задачи: история следователя не тянется


def test_investigator_that_never_stops_leaves_no_fix_phase():
    seen = []
    inv = [("", [call("run_command", {"command": "ls"}, i)]) for i in range(5)]
    env = FakeEnv(calls=5)
    _, _, passed, stats = run_episode_two_phase(env, make_factory([inv, []], seen), "INV", "BASE")
    assert not passed and not stats["inv_stopped"] and stats["fix_calls"] == 0
