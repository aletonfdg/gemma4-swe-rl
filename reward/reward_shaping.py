# reward/reward_shaping.py
import json
import re
from collections import deque


def extract_changed_files(patch):
    return {m.group(1).strip() for m in re.finditer(r'^\+\+\+ b/(.+)$', patch, re.MULTILINE)}


def extract_changed_symbols(patch):
    return {m.group(1) for m in re.finditer(
        r'^@@ .+ @@\s*(?:class|def)\s+(\w+)', patch, re.MULTILINE)}


class RewardTracker:
    R = {
        "terminal_success": 100.0, "terminal_budget_fail": -50.0, "loop_penalty": -20.0,
        "tool_syntax_error": -10.0, "edit_file_error": -5.0, "graph_tool_miss": -1.0,
        "test_run_fail": -2.0, "explore_bug_file": 5.0, "explore_bug_symbol": 5.0,
        "test_run_bonus": 2.0, "test_progress_per_unit": 1.0,
        "scratch_file_leak": -5.0,
    }
    MAX_TEST_RUN_BONUSES = 5
    LOOP_WINDOW = 4

    def __init__(self, task):
        self.task = task
        self.changed_files = extract_changed_files(task.get("patch", ""))
        self.changed_symbols = extract_changed_symbols(task.get("patch", ""))
        self.visited_correct_files = set()
        self.visited_correct_symbols = set()
        self.action_history = deque(maxlen=self.LOOP_WINDOW)
        self.test_run_count = 0
        self.code_changed_since_last_test = False
        self.last_test_state = {"passed": 0, "failed": 0}

    def on_malformed_tool_call(self):
        return self.R["tool_syntax_error"]

    def on_step(self, tool_name, tool_args, tool_result):
        reward = 0.0
        sig = (tool_name, json.dumps(tool_args, sort_keys=True))
        if sig in self.action_history:
            reward += self.R["loop_penalty"]
        self.action_history.append(sig)

        try:
            parsed = json.loads(tool_result)
        except Exception:
            parsed = {}
        status = parsed.get("status", "unknown")

        if tool_name == "read_file" and status == "ok":
            fp = tool_args.get("filepath", "")
            if fp in self.changed_files and fp not in self.visited_correct_files:
                reward += self.R["explore_bug_file"]
                self.visited_correct_files.add(fp)

        if tool_name in ("get_code_neighbors", "search_similar_code", "get_code_subgraph"):
            reward += self._graph_reward(tool_name, parsed, status)

        if tool_name == "edit_file":
            reward += self.R["edit_file_error"] if status == "error" else 0.0
            if status == "ok":
                self.code_changed_since_last_test = True
        if tool_name == "write_file" and status == "ok":
            self.code_changed_since_last_test = True
            fp = tool_args.get("filepath", "")
            if fp.startswith("/workspace/") is False and not fp.startswith("/tmp"):
                pass  # оставлено как хук: see scratch-file leak check at submit_patch time

        if tool_name == "run_command" and re.search(r"\bpytest\b|\bnpm\s+test\b", tool_args.get("command", "")):
            reward += self._test_reward(parsed)
        return reward

    def _graph_reward(self, tool_name, parsed, status):
        if status == "error":
            if parsed.get("error_type") in ("GraphUnavailable", "EmbeddingsUnavailable"):
                return 0.0
            return self.R["graph_tool_miss"]
        found = set()
        if tool_name == "get_code_neighbors":
            found.add(parsed.get("node", ""))
            found.update(parsed.get("neighbors", []))
        elif tool_name == "search_similar_code":
            found.update(r.get("node_name", "") for r in parsed.get("results", []))
        elif tool_name == "get_code_subgraph":
            found.update(parsed.get("nodes", []))
        new_hits = {n for n in found if any(s in n.split(".") for s in self.changed_symbols)} \
            - self.visited_correct_symbols
        if new_hits:
            self.visited_correct_symbols.update(new_hits)
            return self.R["explore_bug_symbol"]
        return 0.0

    def _test_reward(self, parsed):
        reward = 0.0
        out = (parsed.get("stdout", "") or "") + (parsed.get("stderr", "") or "")
        if self.code_changed_since_last_test and self.test_run_count < self.MAX_TEST_RUN_BONUSES:
            reward += self.R["test_run_bonus"]
        self.test_run_count += 1
        self.code_changed_since_last_test = False

        m_pass = re.search(r"(\d+) passed", out)
        m_fail = re.search(r"(\d+) failed", out)
        passed = int(m_pass.group(1)) if m_pass else 0
        failed = int(m_fail.group(1)) if m_fail else 0
        prev = self.last_test_state
        reward += self.R["test_progress_per_unit"] * max(0, passed - prev["passed"])
        reward += self.R["test_progress_per_unit"] * max(0, prev["failed"] - failed)
        if failed > 0:
            reward += self.R["test_run_fail"]
        self.last_test_state = {"passed": passed, "failed": failed}
        return reward

    def terminal_reward(self, verifier_passed, hit_budget_limit, sandbox_crashed):
        if verifier_passed:
            return self.R["terminal_success"]
        if hit_budget_limit or sandbox_crashed:
            return self.R["terminal_budget_fail"] + min(10.0, 2.0 * self.last_test_state["passed"])
        return 0.0
