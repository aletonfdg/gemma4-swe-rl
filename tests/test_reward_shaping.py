import json

from reward.reward_shaping import RewardTracker


def pytest_obs(stdout, ok):
    """Формат ответов sim/tools.py: успех -- поля наверху, падение -- внутри details."""
    if ok:
        return json.dumps({"status": "ok", "stdout": stdout, "stderr": "", "exit_code": 0})
    return json.dumps({"status": "error", "error_type": "CommandError", "error_message": "exit 1",
                       "details": {"stdout": stdout, "stderr": "", "exit_code": 1}})


def run(tracker, obs, command="pytest -q"):
    return tracker.on_step("run_command", {"command": command}, obs)


def test_progress_is_rewarded_for_passing_run_with_top_level_stdout():
    t = RewardTracker({})
    assert run(t, pytest_obs("5 passed in 0.1s", ok=True)) == 5.0


def test_failing_run_reads_stdout_from_details():
    t = RewardTracker({})
    # 19 новых прошедших (+19), 0 падений было -> прирост по failed 0, штраф за падающие -2
    assert run(t, pytest_obs("3 failed, 19 passed in 1.1s", ok=False)) == 19.0 - 2.0
    # стало 21 прошедших (+2) и 1 падение вместо 3 (+2), штраф -2 остаётся
    # другая команда, чтобы не сработал loop_penalty (он проверяется отдельно)
    assert run(t, pytest_obs("1 failed, 21 passed in 1.0s", ok=False),
               command="pytest -q tests/") == 2.0 + 2.0 - 2.0


EDIT_OK = json.dumps({"status": "ok", "filepath": "a.py", "occurrences": 1})
EDIT_ARGS = {"filepath": "a.py", "old_string": "x", "new_string": "y"}


def test_rerunning_same_test_command_after_an_edit_is_not_a_loop():
    t = RewardTracker({})
    assert run(t, pytest_obs("5 passed in 0.1s", ok=True)) == 5.0
    assert t.on_step("edit_file", EDIT_ARGS, EDIT_OK) == 0.0
    # тот же pytest после правки: без loop_penalty; только бонус за запуск после правки (+2)
    assert run(t, pytest_obs("5 passed in 0.1s", ok=True)) == 2.0


def test_rerunning_same_command_without_edit_is_still_a_loop():
    t = RewardTracker({})
    run(t, pytest_obs("5 passed in 0.1s", ok=True))
    assert run(t, pytest_obs("5 passed in 0.1s", ok=True)) == RewardTracker.R["loop_penalty"]


def test_repeating_an_identical_edit_is_still_a_loop():
    t = RewardTracker({})
    assert t.on_step("edit_file", EDIT_ARGS, EDIT_OK) == 0.0
    assert t.on_step("edit_file", EDIT_ARGS, EDIT_OK) == RewardTracker.R["loop_penalty"]


def test_error_without_details_does_not_crash():
    t = RewardTracker({})
    obs = json.dumps({"status": "error", "error_type": "TimeoutExceeded",
                      "error_message": "Timed out after 300s"})
    assert run(t, obs) == 0.0
