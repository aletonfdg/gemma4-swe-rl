import json

from train.build_pref_short import call_targets, episode_pairs, shorten

PARA = "Looking at the code, the bug is in parse(). " * 4          # ~180 символов на абзац


def call_msg(call_id, name, args, thought):
    return {"role": "assistant", "content": "", "thought": thought, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def tool_msg(call_id, status):
    return {"role": "tool", "tool_call_id": call_id, "content": json.dumps({"status": status})}


def long_thought(target):
    return "\n\n".join([PARA, f"So I should read {target} next.", *[PARA] * 20])


def test_call_targets_take_file_name_and_command_words():
    assert call_targets("read_file", {"filepath": "src/pkg/core.py"}) == {"core.py"}
    assert call_targets("run_command", {"command": "grep -rn secret_key src"}) == {"secret_key"}


def test_shorten_cuts_at_paragraph_after_target_is_named():
    thought = long_thought("core.py")
    cut = shorten(thought, {"core.py"}, short=100)
    assert cut.endswith("So I should read core.py next.")
    assert thought.startswith(cut) and len(cut) <= len(thought) * 0.5


def test_shorten_gives_up_when_target_never_named_or_named_too_late():
    assert shorten(long_thought("core.py"), {"other.py"}) is None
    late = "\n\n".join([PARA] * 20 + ["read core.py", PARA])
    assert shorten(late, {"core.py"}) is None


def test_shorten_does_not_cut_inside_code_block():
    thought = "Read core.py.\n\n```python\ndef f():\n\n    pass\n```\n\n" + "\n\n".join([PARA] * 20)
    cut = shorten(thought, {"core.py"}, short=30, min_short=1)
    assert cut.count("```") % 2 == 0


def test_episode_pairs_keep_call_and_shorten_only_exploration_turns():
    explore = call_msg("c1", "read_file", {"filepath": "core.py"}, long_thought("core.py"))
    edit = call_msg("c2", "edit_file", {"filepath": "core.py", "old_string": "a", "new_string": "b"},
                    long_thought("core.py"))
    dialog = {"messages": [{"role": "user", "content": "task"}, explore, tool_msg("c1", "ok"),
                           edit, tool_msg("c2", "ok")]}
    pairs, stats = episode_pairs(dialog, "SYSTEM", tools=[])
    assert len(pairs) == 1 and stats["pairs"] == 1
    chosen, rejected = pairs[0]["chosen"], pairs[0]["rejected"]
    assert chosen[0] == {"role": "system", "content": "SYSTEM"} and chosen[:-1] == rejected[:-1]
    assert chosen[-1]["tool_calls"] == rejected[-1]["tool_calls"]
    assert len(chosen[-1]["thought"]) < len(rejected[-1]["thought"])
    assert rejected[-1]["thought"] == explore["thought"].strip()


def test_episode_pairs_skip_failed_calls():
    explore = call_msg("c1", "read_file", {"filepath": "core.py"}, long_thought("core.py"))
    dialog = {"messages": [{"role": "user", "content": "task"}, explore, tool_msg("c1", "error")]}
    pairs, stats = episode_pairs(dialog, "SYSTEM", tools=[])
    assert pairs == [] and stats["call_failed"] == 1
