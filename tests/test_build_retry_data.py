import json

from train.build_retry_data import episode_pairs

FILE = "a = 1\nb = 2\ndef f():\n    return a + b\n"
ERR = json.dumps({"status": "error", "error_type": "FileEditError", "error_message": "old_string not matched"})
OK = json.dumps({"status": "ok", "diff": "..."})


def asst(i, name, args, thought="t"):
    return {"role": "assistant", "content": "", "thought": thought,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool(i, content):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": content}


BAD = {"filepath": "pkg/m.py", "old_string": "def f():\\n    return a+b", "new_string": "x"}
GOOD = {"filepath": "pkg/m.py", "old_string": "def f():\n    return a + b", "new_string": "x"}
READ = asst(1, "read_file", {"filepath": "pkg/m.py", "start_line": 40, "end_line": 80})
READ_OUT = tool(1, json.dumps({"filepath": "pkg/m.py", "content": FILE}))


def pairs_of(messages):
    return episode_pairs({"messages": messages}, "SYS", [])


def test_repeat_gives_read_around_the_place_and_later_fix():
    msgs = [{"role": "user", "content": "issue"}, READ, READ_OUT,
            asst(2, "edit_file", BAD), tool(2, ERR),
            asst(3, "edit_file", BAD, thought="try again"), tool(3, ERR),
            asst(4, "edit_file", GOOD), tool(4, OK)]
    ps = pairs_of(msgs)
    assert [p["kind"] for p in ps] == ["retry_read", "retry_fixed"]
    read = ps[0]["chosen"][-1]
    assert read["thought"] == "try again" == ps[0]["rejected"][-1]["thought"]
    args = json.loads(read["tool_calls"][0]["function"]["arguments"])
    assert read["tool_calls"][0]["function"]["name"] == "read_file"
    assert args["filepath"] == "pkg/m.py" and args["start_line"] <= 42 <= args["end_line"]   # def f() -- 3-я строка от 40
    assert json.loads(ps[1]["chosen"][-1]["tool_calls"][0]["function"]["arguments"]) == GOOD
    assert ps[0]["chosen"][0] == {"role": "system", "content": "SYS"}
    assert ps[0]["rejected"][-1]["tool_calls"][0]["function"]["arguments"] == json.dumps(BAD)


def test_no_read_pair_if_model_already_reread_and_no_pair_for_first_failure():
    msgs = [{"role": "user", "content": "issue"}, asst(2, "edit_file", BAD), tool(2, ERR),
            READ, READ_OUT, asst(3, "edit_file", BAD), tool(3, ERR)]
    assert pairs_of(msgs) == []
    assert pairs_of(msgs[:3]) == []


def test_unknown_place_falls_back_to_grep():
    msgs = [{"role": "user", "content": "issue"}, asst(2, "edit_file", BAD), tool(2, ERR),
            asst(3, "edit_file", BAD), tool(3, ERR)]
    (p,) = pairs_of(msgs)
    call = p["chosen"][-1]["tool_calls"][0]["function"]
    assert call["name"] == "run_command" and json.loads(call["arguments"])["command"] == \
        "grep -n -F -- 'def f():' pkg/m.py"
