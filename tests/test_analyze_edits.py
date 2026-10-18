import json

from tools.analyze_edits import analyze_episode, group_of, load_episodes, old_string_kind, summarize


def edit_call(call_id, old, new="y"):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": "edit_file", "arguments": json.dumps({"filepath": "a.py", "old_string": old, "new_string": new})}}]}


def reply(call_id, status="ok", error_type=None):
    body = {"status": status}
    if error_type:
        body["error_type"] = error_type
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(body)}


def messages():
    return [{"role": "user", "content": "task"},
            edit_call("c1", "a" + chr(92) + "nb"), reply("c1", "error", "FileEditError"),    # литеральный \n
            edit_call("c2", "a\nb"), reply("c2")]                                              # настоящий перевод строки


def test_old_string_kinds():
    assert old_string_kind("x") == "single_line"
    assert old_string_kind("x\ny") == "real_newline"
    assert old_string_kind("x" + chr(92) + "ny") == "literal_backslash_n"


def test_analyze_episode_counts_outcome_and_kind_and_first_edit_turn():
    info = analyze_episode({}, messages())
    assert info["calls"] == [("FileEditError", "literal_backslash_n"), ("ok", "real_newline")]
    assert info["first_edit"] == 1 and info["turns"] == 2


def test_groups_and_summary():
    assert group_of([]) == "no edit_file calls"
    assert group_of([("ok", "real_newline")]) == "edit_file, no literal \\n"
    assert group_of([("x", "literal_backslash_n")] * 3) == "literal \\n in 3+ calls"
    calls, groups, reasons, first = summarize([({"passed": True, "result": {"reason": "tests_passed"}}, messages())])
    assert calls[("ok", "real_newline")] == 1 and groups["literal \\n in 1-2 calls"] == [1, 1]
    assert reasons["tests_passed"] == 1 and first[True] == [1]


def test_load_episodes_takes_last_row_and_skips_errors(tmp_path):
    run = tmp_path / "run"
    (run / "messages").mkdir(parents=True)
    (run / "messages" / "t.json").write_text(json.dumps({"messages": messages()}), encoding="utf-8")
    failed = {"instance_id": "t", "rep": 0, "passed": False, "error": "SandboxUnavailable"}
    good = {"instance_id": "t", "rep": 0, "passed": True, "result": {"reason": "tests_passed"}}
    broken = {"instance_id": "u", "rep": 0, "passed": False, "error": "x"}
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in (failed, good, broken)), encoding="utf-8")
    episodes = load_episodes(str(run))
    assert len(episodes) == 1 and episodes[0][0]["passed"] is True
