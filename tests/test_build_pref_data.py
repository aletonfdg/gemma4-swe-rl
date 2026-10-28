import json

from train.build_pref_data import (BACKSLASH_N, build, edit_args, escape_newlines, is_escaped, make_pair,
                                   unescape_newlines)
from train.pref_lora import pref_weight, split_by_task

NEW = "def f():\n    return 1\n"
OLD = "def f():\n    return 0\n"


def edit_msg(call_id, args, thought="plan"):
    return {"role": "assistant", "content": "", "thought": thought, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": "edit_file", "arguments": json.dumps(args)}}]}


def tool_msg(call_id, status):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"status": status})}


def dialog(bad, good):
    return {"messages": [{"role": "user", "content": "task"},
                         edit_msg("c1", bad), tool_msg("c1", "error"),
                         edit_msg("c2", good), tool_msg("c2", "ok")]}


def test_escape_helpers_roundtrip_and_ignore_single_backslash_n_in_code():
    assert unescape_newlines(escape_newlines(NEW)) == NEW
    assert is_escaped(escape_newlines(NEW)) and not is_escaped(NEW)
    assert not is_escaped('print("a' + BACKSLASH_N + '")')        # одиночный \n -- обычный код


def test_real_failure_gets_fixed_chosen_and_successful_call_gets_synthetic_reject():
    bad = {"filepath": "a.py", "old_string": escape_newlines(OLD), "new_string": escape_newlines(NEW)}
    kind, good, rejected = make_pair(bad, ok=False)
    assert kind == "real_fix" and good["old_string"] == OLD and rejected == bad
    kind, chosen, rejected = make_pair({"filepath": "a.py", "old_string": OLD, "new_string": NEW}, ok=True)
    assert kind == "synthetic" and chosen["new_string"] == NEW and BACKSLASH_N in rejected["new_string"]
    assert make_pair({"filepath": "a.py", "old_string": "x", "new_string": "y"}, ok=True) is None
    assert make_pair(bad, ok=True) is None                         # удачный вызов с литеральным \n пропускаем


def test_build_makes_pairs_differing_only_in_the_last_call(tmp_path):
    run = tmp_path / "run"
    (run / "messages").mkdir(parents=True)
    bad = {"filepath": "a.py", "old_string": escape_newlines(OLD), "new_string": escape_newlines(NEW)}
    good = {"filepath": "a.py", "old_string": OLD, "new_string": NEW}
    (run / "messages" / "t.json").write_text(json.dumps(dialog(bad, good)), encoding="utf-8")
    (run / "results.jsonl").write_text(json.dumps({"instance_id": "t", "rep": 0, "passed": False}), encoding="utf-8")
    pairs, stats = build([str(run)], "SYS", synthetic_per_real=0)
    assert stats["real_fix"] == 1 and stats["synthetic_used"] == 1 and len(pairs) == 2
    for p in pairs:
        assert p["chosen"][:-1] == p["rejected"][:-1] and p["chosen"][0]["content"] == "SYS"
        assert p["chosen"][-1]["thought"] == p["rejected"][-1]["thought"]
        assert edit_args(p["chosen"][-1]) != edit_args(p["rejected"][-1])
    assert build([str(run)], "SYS", exclude={"t"})[0] == []         # исключённые задачи не берём


def test_pref_weight_is_half_beta_at_zero_margin_and_shrinks_with_margin():
    w0, z0 = pref_weight(-10.0, -10.0, 0.1)
    assert abs(w0 - 0.05) < 1e-9 and z0 == 0.0
    w1, _ = pref_weight(-10.0, -60.0, 0.1)
    w2, _ = pref_weight(-60.0, -10.0, 0.1)
    assert w1 < w0 < w2 <= 0.1


def test_split_by_task_is_disjoint_and_deterministic():
    rows = [{"task": f"t{i}", "i": j} for i in range(10) for j in range(2)]
    train, ev = split_by_task(rows, 0.2)
    assert not ({r["task"] for r in train} & {r["task"] for r in ev}) and len({r["task"] for r in ev}) == 2
    again = split_by_task(list(reversed(rows)), 0.2)[1]
    assert sorted(r["task"] for r in again) == sorted(r["task"] for r in ev)


def test_trim_history_keeps_task_and_last_exchanges_and_cuts_long_tool_output():
    from train.build_pref_data import trim_history
    msgs = [{"role": "user", "content": "task"}]
    for i in range(5):
        msgs += [{"role": "assistant", "content": f"a{i}", "tool_calls": []},
                 {"role": "tool", "tool_call_id": str(i), "content": "x" * 100}]
    out = trim_history(msgs, 2, 10)
    assert out[0]["content"] == "task" and [m["content"] for m in out if m["role"] == "assistant"] == ["a3", "a4"]
    assert all(len(m["content"]) < 40 for m in out if m["role"] == "tool")
    assert msgs[2]["content"] == "x" * 100                         # исходные сообщения не меняем
    assert trim_history(msgs[:1], 2, 10) == msgs[:1]


def test_repeat_only_named_kinds_and_reject_bad_spec():
    from train.pref_lora import parse_repeat, repeat_kinds
    rows = [{"kind": "real_fix"}, {"kind": "short_think"}, {"kind": "synthetic"}]
    out = repeat_kinds(rows, parse_repeat(["real_fix=2", "synthetic=2"]))
    assert [r["kind"] for r in out] == ["real_fix", "real_fix", "short_think", "synthetic", "synthetic"]
    for bad in (["real_fix"], ["real_fix=0"], ["=2"]):
        try:
            parse_repeat(bad)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_eval_summary_reports_each_kind_and_total():
    from train.pref_lora import eval_summary
    out = eval_summary([("real_fix", 2.0), ("real_fix", -1.0), ("short_think", 3.0)])
    assert out["real_fix"] == (2, 0.5, 0.5) and out["short_think"] == (1, 1.0, 3.0)
    assert out["все"] == (3, 2 / 3, 4 / 3)
    assert eval_summary([("real_fix", 1.0)]) == {"real_fix": (1, 1.0, 1.0)} and eval_summary([]) == {}
