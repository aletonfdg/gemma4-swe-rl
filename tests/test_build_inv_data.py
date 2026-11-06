import json

from train.build_inv_data import (PLACEHOLDER, check_command, episode_records, find_cut, fix_text, hunk_seen,
                                  parse_hunks, to_investigator)
from train.fill_inv_data import fill_one, is_filled, parse_stop_reply

PATCH = """diff --git a/pkg/core.py b/pkg/core.py
--- a/pkg/core.py
+++ b/pkg/core.py
@@ -10,4 +10,4 @@ def area(w, h):
     if w is None:
         raise ValueError("width is required")
-    result = width_value * height_value
+    result = float(width_value) * float(height_value)
     return result
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,3 @@
 import pkg
+def test_area(): pass
"""
CODE = ('    if w is None:\n        raise ValueError("width is required")\n'
        '    result = width_value * height_value\n    return result\n')
TASK = {"instance_id": "t1", "patch": PATCH,
        "test_patch": "--- a/tests/test_core.py\n+++ b/tests/test_core.py\n@@ -1 +1,2 @@\n x\n+y\n"}


def call(name, args, i):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def asst(name, args, i):
    return {"role": "assistant", "content": "", "tool_calls": [call(name, args, i)]}


def tool(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": content}


def episode(after):
    return [{"role": "user", "content": "issue: area breaks"},
            asst("run_command", {"command": "git grep -n area"}, 1),
            tool(1, "pkg/core.py:12:    result = width_value * height_value"),     # строка из grep -- ещё не код
            asst("write_file", {"filepath": "/tmp/r.py", "content": "print(1)\n"}, 2),
            tool(2, "ok"),
            asst("read_file", {"filepath": "pkg/core.py", "start_line": 1, "end_line": 40}, 3),
            tool(3, CODE)] + after


def test_hunks_skip_tests_and_seen_needs_real_code():
    hunks = parse_hunks(PATCH)
    assert [h["file"] for h in hunks] == ["pkg/core.py"]
    assert not hunk_seen(hunks[0], "pkg/core.py:12:    result = width_value * height_value")
    assert hunk_seen(hunks[0], CODE)
    assert "-    result = width_value * height_value" in fix_text(hunks)
    assert check_command(TASK, hunks) == "python -m pytest tests/test_core.py -q -x"


def test_cut_is_after_reading_code_and_none_after_source_edit():
    msgs = episode([])
    assert find_cut(msgs, parse_hunks(PATCH)) == 6
    edited = msgs[:3] + [asst("edit_file", {"filepath": "pkg/core.py"}, 9), tool(9, "ok")] + msgs[3:]
    assert find_cut(edited, parse_hunks(PATCH)) is None


def test_tmp_write_becomes_heredoc_and_edit_is_not_investigator():
    conv = to_investigator(asst("write_file", {"filepath": "/tmp/r.py", "content": "print(1)\n"}, 2))
    fn = conv["tool_calls"][0]["function"]
    assert fn["name"] == "run_command" and "cat > /tmp/r.py <<'EOF'" in json.loads(fn["arguments"])["command"]
    assert to_investigator(asst("edit_file", {"filepath": "pkg/core.py"}, 3)) is None


def test_episode_records_stop_and_go():
    after = [asst("run_command", {"command": "git log -3"}, 4), tool(4, "log"),
             asst("read_file", {"filepath": "pkg/other.py"}, 5), tool(5, "x"),
             asst("edit_file", {"filepath": "pkg/core.py"}, 6), tool(6, "ok")]
    recs = episode_records(TASK, episode(after), "INV", [{"function": {"name": "read_file"}}], stop_per_episode=3)
    stops = [r for r in recs if r["kind"] == "stop"]
    gos = [r for r in recs if r["kind"] == "go"]
    assert len(stops) == 2                              # третий ход -- правка: точек остановки дальше нет
    assert stops[0]["chosen"][-1]["content"] == PLACEHOLDER
    assert stops[0]["rejected"][-1]["tool_calls"][0]["function"]["name"] == "run_command"
    assert stops[0]["chosen"][0] == {"role": "system", "content": "INV"}
    history_calls = [c["function"]["name"] for m in stops[0]["chosen"] for c in m.get("tool_calls") or []]
    assert "write_file" not in history_calls            # история в инструментах следователя
    assert len(gos) == 1 and gos[0]["chosen"][-1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert gos[0]["rejected"][-1]["content"] == PLACEHOLDER


class FakeClient:
    def __init__(self, replies):
        self.replies = replies

    def chat(self, messages, tools, **kw):
        return self.replies.pop(0)


def test_fill_stop_validates_and_builds_diagnosis():
    hunks = parse_hunks(PATCH)
    assert parse_stop_reply('{"thought": "The patch multiplies width_value directly.", "cause": "x" }', hunks) is None
    bad = {"content": '{"thought": "Looks fine to me, nothing found.", "cause": "Something is wrong somewhere."}'}
    good = {"content": '{"thought": "area multiplies width_value as a string, I can hand over now.", '
                       '"cause": "area multiplies width_value and height_value without converting to float."}'}
    recs = episode_records(TASK, episode([asst("run_command", {"command": "ls"}, 4), tool(4, "x")]), "INV", [])
    stop = next(r for r in recs if r["kind"] == "stop")
    out = fill_one(stop, FakeClient([bad, good]))
    assert out is not None and is_filled(out)
    text = out["chosen"][-1]["content"]
    assert text.startswith("FILE: pkg/core.py") and "CAUSE: area multiplies" in text and "CHECK: python -m pytest" in text


def test_fill_premature_needs_text_diagnosis():
    recs = episode_records(TASK, episode([asst("run_command", {"command": "ls"}, 4), tool(4, "x")]), "INV", [])
    go = next(r for r in recs if r["kind"] == "go")
    out = fill_one(go, FakeClient([{"content": "", "tool_calls": [{"name": "read_file"}]},
                                   {"content": "FILE: pkg/core.py\nCAUSE: guess", "reasoning": "r", "tool_calls": []}]))
    assert out["rejected"][-1] == {"role": "assistant", "content": "FILE: pkg/core.py\nCAUSE: guess", "thought": "r"}
    assert is_filled(out)
