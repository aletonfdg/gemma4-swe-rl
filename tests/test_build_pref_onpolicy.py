import json
import random

from train.build_pref_onpolicy import build, episode_pair, is_source, winning_edit

SOURCE = "def parse(x):\n    return x\n\n\ndef other():\n    pass\n"
OLD, GOOD, BAD = "    return x\n", "    return x.strip()\n", "    return str(x)\n"
PATCH = "--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1,2 +1,2 @@\n def parse(x):\n-    return x\n+    return x.strip()\n"


def call_msg(call_id, name, args, thought="thinking " * 20):
    return {"role": "assistant", "content": "", "thought": thought, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool_msg(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(content)}


def read_then(after, status="ok"):
    return {"messages": [{"role": "user", "content": "task"},
                         call_msg("c1", "read_file", {"filepath": "pkg/core.py"}),
                         tool_msg("c1", {"filepath": "pkg/core.py", "content": SOURCE, "status": "ok"}),
                         after, tool_msg("c2", {"status": status})]}


def edit(new, call_id="c2", thought="thinking " * 20):
    return call_msg(call_id, "edit_file", {"filepath": "pkg/core.py", "old_string": OLD, "new_string": new}, thought)


def solved():
    d = read_then(edit(GOOD, thought="strip it"))
    d["patch"] = PATCH
    return d


def test_is_source():
    assert is_source("pkg/core.py") and is_source("/workspace/src/_pytest/main.py")
    assert not is_source("tests/test_core.py") and not is_source("/tmp/repro.py")
    assert not is_source("repro.py") and not is_source("pkg/core_test.py") and not is_source("README.md")


def test_winning_edit_takes_first_edit_kept_in_patch():
    msg, path, old, new = winning_edit(solved())
    assert (path, old, new) == ("pkg/core.py", OLD, GOOD) and msg["thought"] == "strip it"
    reworked = solved()
    reworked["patch"] = PATCH.replace("x.strip()", "x.lstrip()")
    assert winning_edit(reworked) is None


def test_pair_replaces_wrong_edit_with_own_correct_one():
    pair, why = episode_pair(read_then(edit(BAD)), winning_edit(solved()), "SYS", [], random.Random(0))
    assert why == "wrong_edit" and pair["kind"] == "onpolicy"
    chosen, rejected = pair["chosen"], pair["rejected"]
    assert chosen[:-1] == rejected[:-1] and chosen[0] == {"role": "system", "content": "SYS"}
    assert json.loads(rejected[-1]["tool_calls"][0]["function"]["arguments"])["new_string"] == BAD
    args = json.loads(chosen[-1]["tool_calls"][0]["function"]["arguments"])
    assert args == {"filepath": "pkg/core.py", "old_string": OLD, "new_string": GOOD}
    assert chosen[-1]["thought"] == "strip it"


def test_pair_replaces_repro_detour_and_skips_same_edit():
    detour = call_msg("c2", "write_file", {"filepath": "/tmp/repro.py", "content": "print(1)"})
    pair, why = episode_pair(read_then(detour), winning_edit(solved()), "S", [], random.Random(0))
    assert why == "detour" and pair["rejected"][-1]["tool_calls"] == detour["tool_calls"]
    assert episode_pair(read_then(edit(GOOD)), winning_edit(solved()), "S", [], random.Random(0))[1] == "same_edit"


def test_no_thought_on_rejected_means_none_on_chosen():
    pair, _ = episode_pair(read_then(edit(BAD, thought="")), winning_edit(solved()), "S", [], random.Random(0))
    assert pair["chosen"][-1]["thought"] == ""


def write_run(root, episodes):
    (root / "messages").mkdir(parents=True)
    with open(root / "results.jsonl", "w", encoding="utf-8") as f:
        for rep, (iid, passed, dialog) in enumerate(episodes):
            reason = "tests_passed" if passed else "tests_failed"
            f.write(json.dumps({"instance_id": iid, "rep": rep, "passed": passed, "result": {"reason": reason}}) + "\n")
            name = f"{iid}.json" if rep == 0 else f"{iid}__r{rep}.json"
            (root / "messages" / name).write_text(json.dumps(dialog), encoding="utf-8")


def test_build_pairs_only_mixed_tasks_and_honours_exclude(tmp_path):
    write_run(tmp_path / "run", [("t1", True, solved()), ("t1", False, read_then(edit(BAD))),
                                 ("t2", False, read_then(edit(BAD))),
                                 ("t3", True, solved()), ("t3", False, read_then(edit(BAD)))])
    pairs, stats = build([str(tmp_path / "run")], "S", exclude={"t3"})
    assert [p["task"] for p in pairs] == ["t1"] and stats["tasks_mixed"] == 1
