import json

from train.build_sft_data import (build, reject_reason, select_turns, split_tasks, to_turn_records)


def call(name, **args):
    return {"id": f"c-{name}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def assistant(name, thought="short plan", **args):
    msg = {"role": "assistant", "content": "", "tool_calls": [call(name, **args)]}
    if thought is not None:
        msg["thought"] = thought
    return msg


def tool():
    return {"role": "tool", "tool_call_id": "x", "content": "{}"}


def dialog(n_reads=5):
    msgs = [{"role": "user", "content": "task"}]
    for i in range(n_reads):
        msgs += [assistant("read_file", filepath=f"f{i}.py"), tool()]
    msgs += [assistant("edit_file", filepath="a.py", old_string="x", new_string="y"), tool(),
             assistant("submit_patch", thought="submit now"), tool()]       # как в настоящих диалогах
    return {"messages": msgs}


def last_assistant(d):
    return max(k for k, m in enumerate(d["messages"]) if m["role"] == "assistant")


def row(task, rep=0, passed=True, compactions=0):
    return {"instance_id": task, "rep": rep, "passed": passed, "timing": {"compactions": compactions}}


def test_select_turns_always_has_first_edit_and_last_turn():
    d = dialog()
    chosen = select_turns(d["messages"], 4)
    roles = [d["messages"][k] for k in chosen]
    assert chosen[0] == 1 and chosen[-1] == last_assistant(d)
    assert any(m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == "edit_file" for m in roles)
    assert chosen == sorted(set(chosen))


def test_turn_records_use_history_without_cut_after_the_target_turn():
    d = dialog()
    recs = to_turn_records(d, "SYS", [], 4, 3000)
    assert recs and all(r["target"] == "last" and r["thinking"] is True for r in recs)
    for r in recs:
        assert r["messages"][0] == {"role": "system", "content": "SYS"}
        assert r["messages"][-1]["role"] == "assistant"          # обрезано сразу после цели


def test_turns_without_thought_or_with_too_long_thought_are_skipped():
    d = dialog()
    d["messages"][1]["thought"] = None                            # первый ход без рассуждения
    d["messages"][last_assistant(d)]["thought"] = "x" * 5000      # последний слишком длинный
    recs = to_turn_records(d, "SYS", [], 6, 3000)
    lengths = {len(r["messages"]) - 1 for r in recs}              # индекс цели в сообщениях без system
    assert 1 not in lengths and last_assistant(d) not in lengths and recs


def test_compacted_episodes_are_only_allowed_in_turn_mode():
    d = dialog()
    assert reject_reason(row("t", compactions=2), d) == "compacted"
    assert reject_reason(row("t", compactions=2), d, allow_compacted=True) is None


def test_garbled_tool_names_and_runaway_text_are_rejected():
    d = dialog()
    d["messages"][1]["tool_calls"][0]["function"]["name"] = 'grep -n "x"<|"|>'
    assert reject_reason(row("t"), d) == "bad_tool_name"
    d = dialog()
    d["messages"][last_assistant(d)]["content"] = "y" * 5000
    assert reject_reason(row("t"), d) == "runaway_text"


def test_final_plain_answer_after_submit_is_allowed_but_not_in_the_middle():
    d = dialog()
    d["messages"] += [{"role": "assistant", "content": "Done."}]       # промпт коллеги: «Done.» после сдачи
    assert reject_reason(row("t"), d) is None
    d["messages"].insert(1, {"role": "assistant", "content": "thinking out loud"})
    assert reject_reason(row("t"), d) == "turn_without_call"


def test_holdout_split_is_deterministic_and_disjoint():
    tasks = [f"task_{i}" for i in range(10)]
    a, b = split_tasks(tasks, 0.3)
    assert (a, b) == split_tasks(list(reversed(tasks)), 0.3) and not (a & b) and len(b) == 3


def test_rerun_rows_replace_failed_ones_like_in_the_runner(tmp_path):
    from train.build_sft_data import load_run
    run = tmp_path / "run"
    (run / "messages").mkdir(parents=True)
    (run / "messages" / "t.json").write_text(json.dumps(dialog()), encoding="utf-8")
    failed = {"instance_id": "t", "rep": 0, "passed": False, "error": "SandboxUnavailable: docker"}
    (run / "results.jsonl").write_text(json.dumps(failed) + "\n" + json.dumps(row("t")) + "\n", encoding="utf-8")
    loaded = load_run(str(run))
    assert len(loaded) == 1 and loaded[0][0]["passed"] is True


def test_build_end_to_end_turn_mode(tmp_path):
    run = tmp_path / "run"
    (run / "messages").mkdir(parents=True)
    rows = []
    for i in range(4):
        task = f"repo_{i}"
        rows.append(row(task))
        (run / "messages" / f"{task}.json").write_text(json.dumps(dialog()), encoding="utf-8")
    rows.append(row("repo_failed", passed=False))
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records, train_tasks, held_tasks, rejected, avail = build([str(run)], "SYS", per_task=1,
                                                              holdout_frac=0.25, turns=3)
    assert len(avail) == 4 and len(train_tasks) == 3 and len(held_tasks) == 1
    assert records and all(r["target"] == "last" for r in records)
    assert len(records) <= 3 * 3


def test_excluded_tasks_never_reach_train_or_heldout_sets(tmp_path):
    run = tmp_path / "run"
    (run / "messages").mkdir(parents=True)
    rows = []
    for i in range(4):
        rows.append(row(f"repo_{i}"))
        (run / "messages" / f"repo_{i}.json").write_text(json.dumps(dialog()), encoding="utf-8")
    (run / "results.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    records, train_tasks, held_tasks, rejected, avail = build([str(run)], "SYS", per_task=1, holdout_frac=0.25,
                                                              turns=3, exclude={"repo_0", "repo_2"})
    assert sorted(avail) == ["repo_1", "repo_3"]
    assert not ({"repo_0", "repo_2"} & (set(train_tasks) | set(held_tasks)))
    assert build([str(run)], "SYS", turns=3, exclude=None)[4].keys() == {f"repo_{i}" for i in range(4)}
