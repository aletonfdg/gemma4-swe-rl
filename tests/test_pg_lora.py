import json

from train.pg_lora import broken_reason, build_records, drop_oldest_turns, episode_reward, group_advantages


def call(name, **args):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def turn(name, thought="short plan", **args):
    return {"role": "assistant", "content": "", "thought": thought, "tool_calls": [call(name, **args)]}


def tool():
    return {"role": "tool", "tool_call_id": "x", "content": "{}"}


def episode(edit=True, thought="short plan"):
    msgs = [{"role": "user", "content": "task"}, turn("read_file", thought, filepath="a.py"), tool()]
    if edit:
        msgs += [turn("edit_file", thought, filepath="a.py", old_string="x", new_string="y"), tool()]
    msgs += [turn("submit_patch", thought), tool()]
    return {"messages": msgs}


def write_run(path, episodes):
    """episodes: [(task, rep, passed, dialog)] -> папка в формате sim.run_mimo_baseline."""
    (path / "messages").mkdir(parents=True)
    rows = []
    for task, rep, passed, dlg in episodes:
        name = f"{task}.json" if rep == 0 else f"{task}__r{rep}.json"
        (path / "messages" / name).write_text(json.dumps(dlg), encoding="utf-8")
        rows.append({"instance_id": task, "rep": rep, "passed": passed, "timing": {}})
    (path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return str(path)


def test_reward_penalizes_missing_edit_and_long_thinking():
    assert episode_reward(True, episode()) == 1.0
    assert episode_reward(False, episode(edit=False)) == -0.2
    long = episode(thought="x" * 3000)
    assert abs(episode_reward(True, long, length_penalty=0.2) - 0.8) < 1e-9      # все ходы длинные
    assert episode_reward(True, long, long_thought_chars=5000) == 1.0


def test_group_advantages_sign_and_no_spread():
    adv = group_advantages([1.0, 0.0, 0.0, 0.0])
    assert adv[0] > 0 and all(a < 0 for a in adv[1:]) and abs(sum(adv)) < 1e-9
    assert group_advantages([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    assert group_advantages([1.0]) == [0.0]


def test_drop_oldest_turns_keeps_task_and_target():
    msgs = [{"role": "system", "content": "SYS"}] + episode()["messages"]   # read, edit, submit
    target = msgs[-2]                                                       # ход submit_patch как цель
    msgs = msgs[:-1]
    assert drop_oldest_turns(msgs, 0) == msgs
    one = drop_oldest_turns(msgs, 1)
    assert one[:2] == msgs[:2] and one[-1] is target
    names = [m["tool_calls"][0]["function"]["name"] for m in one if m.get("tool_calls")]
    assert names == ["edit_file", "submit_patch"]                           # read_file с ответом ушёл
    assert [m["role"] for m in drop_oldest_turns(msgs, 2)] == ["system", "user", "assistant"]
    assert drop_oldest_turns(msgs, 3) is None


def test_broken_episode_is_detected():
    d = episode()
    d["messages"][1]["tool_calls"][0]["function"]["name"] = 'grep -n "x"<|"|>'
    assert broken_reason(d) == "bad_tool_name"
    d = episode()
    d["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{not json"
    assert broken_reason(d) == "bad_json_args"
    assert broken_reason(episode()) is None


def test_build_records_uses_only_mixed_groups_and_merges_runs(tmp_path):
    a = write_run(tmp_path / "a", [("mixed", 0, True, episode()), ("mixed", 1, False, episode(edit=False)),
                                   ("all_fail", 0, False, episode()), ("all_fail", 1, False, episode()),
                                   ("eval_task", 0, True, episode()), ("eval_task", 1, False, episode())])
    b = write_run(tmp_path / "b", [("mixed", 2, False, episode()), ("single", 0, True, episode())])
    records, stats = build_records([a, b], "SYS", exclude={"eval_task"}, turns=3)
    assert {r["task"] for r in records} == {"mixed"}
    assert stats["groups_used"] == 1 and stats["group_no_spread"] == 1 and stats["group_too_small"] == 1
    assert stats["excluded"] == 2
    pos = [r for r in records if r["adv"] > 0]
    neg = [r for r in records if r["adv"] < 0]
    assert pos and neg and all(r["passed"] for r in pos) and not any(r["passed"] for r in neg)
    for r in records:
        assert r["messages"][0] == {"role": "system", "content": "SYS"}
        assert r["target"] == "last" and r["thinking"] is True and r["messages"][-1]["role"] == "assistant"


def test_failed_episode_with_no_edit_is_ranked_below_failed_episode_with_edit(tmp_path):
    run = write_run(tmp_path / "r", [("t", 0, False, episode(edit=True)), ("t", 1, False, episode(edit=False))])
    records, stats = build_records([run], "SYS", turns=3)
    # последний ход эпизода (submit_patch) видит всю историю: по нему и узнаём, была ли правка
    last_turns = [r for r in records if r["messages"][-1]["tool_calls"][0]["function"]["name"] == "submit_patch"]
    by_edit = {any(tc["function"]["name"] == "edit_file" for m in r["messages"] for tc in m.get("tool_calls") or []):
               r["adv"] for r in last_turns}
    assert stats["groups_used"] == 1 and by_edit[True] > 0 > by_edit[False]
