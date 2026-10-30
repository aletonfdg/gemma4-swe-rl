import json
import random

from train.build_pref_fix import episode_pair, is_detour, patch_hunks, unique_hunks

PATCH = """--- a/pkg/core.py
+++ b/pkg/core.py
@@ -1,4 +1,4 @@
 def parse(x):
-    return x
+    return x.strip()


"""
SOURCE = "def parse(x):\n    return x\n\n\ndef other():\n    pass\n"


def call_msg(call_id, name, args, thought="thinking " * 50):
    return {"role": "assistant", "content": "", "thought": thought, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool_msg(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(content)}


def dialog(*middle):
    return {"messages": [{"role": "user", "content": "task"}, *middle]}


def test_patch_hunks_split_old_and_new():
    assert patch_hunks(PATCH) == [("pkg/core.py", "def parse(x):\n    return x\n\n\n",
                                   "def parse(x):\n    return x.strip()\n\n\n")]


def test_unique_hunks_require_single_occurrence():
    hunks = patch_hunks(PATCH)
    assert unique_hunks(hunks, {"pkg/core.py": SOURCE}) == hunks
    assert unique_hunks(hunks, {"pkg/core.py": SOURCE + SOURCE}) == []
    assert unique_hunks(hunks, {}) == []


def test_is_detour():
    gold = {"pkg/core.py"}
    assert is_detour(("write_file", {"filepath": "/tmp/repro.py"}), gold)
    assert is_detour(("run_command", {"command": "python /tmp/repro.py"}), gold)
    assert not is_detour(("run_command", {"command": "grep -rn parse ."}), gold)
    assert not is_detour(("write_file", {"filepath": "pkg/core.py"}), gold)


def read_then(after):
    return dialog(call_msg("c1", "read_file", {"filepath": "pkg/core.py"}),
                  tool_msg("c1", {"filepath": "pkg/core.py", "content": SOURCE, "status": "ok"}),
                  after, tool_msg("c2", {"status": "ok"}))


def test_pair_replaces_repro_script_with_gold_edit():
    detour = call_msg("c2", "write_file", {"filepath": "/tmp/repro.py", "content": "print(1)"})
    pair, why = episode_pair(read_then(detour), PATCH, {"pkg/core.py": SOURCE}, "SYS", [], random.Random(0))
    assert why == "pair" and pair["kind"] == "fix_first"
    chosen, rejected = pair["chosen"], pair["rejected"]
    assert chosen[:-1] == rejected[:-1] and chosen[0] == {"role": "system", "content": "SYS"}
    assert rejected[-1]["tool_calls"] == detour["tool_calls"]
    args = json.loads(chosen[-1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"filepath": "pkg/core.py", "old_string": "def parse(x):\n    return x\n\n\n",
                    "new_string": "def parse(x):\n    return x.strip()\n\n\n"}
    assert len(chosen[-1]["thought"]) < 120


def test_no_pair_when_gold_was_edited_or_never_read():
    edit = call_msg("c2", "edit_file", {"filepath": "pkg/core.py", "old_string": "x", "new_string": "y"})
    assert episode_pair(read_then(edit), PATCH, {"pkg/core.py": SOURCE}, "S", [], random.Random(0))[1] == "edited_gold"
    detour = call_msg("c2", "write_file", {"filepath": "/tmp/repro.py", "content": "print(1)"})
    unread = dialog(detour, tool_msg("c2", {"status": "ok"}))
    assert episode_pair(unread, PATCH, {"pkg/core.py": SOURCE}, "S", [], random.Random(0))[1] == "gold_not_read"
