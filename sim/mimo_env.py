# sim/mimo_env.py -- клиент удалённой песочницы MiMo (запускается там, где обучается модель).
"""
Тот же интерфейс, что у sim.swe_env.SWEEnv (reset / call_tool / nudge_message / finalize /
cleanup, поля task, budget, ctx, done), поэтому rl.rollout.run_episode работает без переделки.
Отличие: инструменты исполняются на сервере sim/mimo_server.py внутри контейнера, а награда
считается там же (env.verify), потому что в Colab нет Docker.

    env = MimoRemoteEnv("https://xxx.trycloudflare.com", token="...")
    steps, messages, passed = run_episode(env, 0, llm_client, None, None, ...)
"""
import json
import os
import time
import urllib.error
import urllib.request

from sim.config import default_budget_harness, tight_budget_harness
from sim.prompts import build_agent_prompt
from sim.swe_env import SWEEnv

RETRY_STATUS = {502, 503, 504, 520, 522, 524}


class SandboxUnavailable(RuntimeError):
    pass


class _RemoteCtx:
    """Минимальный аналог SwegemmaContext: счётчики бюджета на стороне клиента."""

    def __init__(self, budget, harness):
        self.budget, self.harness = budget, harness
        self.tool_calls_used = 0
        self.turns_used = 0
        self.patch_submitted = False
        self.start_time = time.time()

    def remaining_time_seconds(self):
        return max(0.0, self.budget.time_minutes * 60.0 - (time.time() - self.start_time))

    def remaining_tool_calls(self):
        return max(0, self.budget.tool_calls - self.tool_calls_used)

    def budget_exhausted(self):
        return (self.remaining_tool_calls() <= 0 or self.remaining_time_seconds() <= 0
                or self.turns_used >= self.budget.turns)


class MimoRemoteEnv:
    nudge_message = SWEEnv.nudge_message   # та же логика подсказок, что в локальном окружении

    def __init__(self, base_url=None, token=None, use_tight_budget=True,
                 eval_config_path="eval_config.yaml", task_ids=None, request_timeout=75.0,
                 retries=12, start_wait_seconds=1800.0, poll_sleep=5.0, budget_override=None):
        self.budget_override = budget_override or {}   # {"tool_calls": N, "time_minutes": M, "turns": K}
        self.base_url = (base_url or os.environ["SWE_SANDBOX_URL"]).rstrip("/")
        self.token = token or os.environ["SWE_SANDBOX_TOKEN"]
        self.use_tight_budget = use_tight_budget
        self.eval_config_path = eval_config_path
        self.request_timeout, self.retries = request_timeout, retries
        self.start_wait_seconds, self.poll_sleep = start_wait_seconds, poll_sleep
        ids = task_ids if task_ids is not None else self._http("GET", "/tasks")["result"]
        self.tasks = [{"instance_id": i} for i in ids]
        self.session_id = None
        self.crashed = False
        self.done = False
        self._final = None

    # ---------- HTTP ----------
    def _http(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        last = None
        for attempt in range(self.retries):
            req = urllib.request.Request(
                self.base_url + path, data=data, method=method,
                headers={"Authorization": f"Bearer {self.token}",
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.request_timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                payload = e.read().decode("utf-8", errors="replace")
                if e.code not in RETRY_STATUS:
                    try:
                        return json.loads(payload)
                    except ValueError:
                        raise SandboxUnavailable(f"HTTP {e.code}: {payload[:200]}")
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
                last = repr(e)
            time.sleep(min(2 ** attempt, 20))
        raise SandboxUnavailable(f"{method} {path}: {last}")

    def _call(self, path, body):
        env = self._http("POST", path, body)
        while env.get("ok") and env.get("pending"):
            env = self._http("POST", "/poll", {"job": env["job"]})
        return env

    # ---------- интерфейс SWEEnv ----------
    def reset(self, task_idx, force_tight_budget=None):
        self.cleanup()
        self.crashed = self.done = False
        self._final = None
        instance_id = self.tasks[task_idx]["instance_id"]
        use_tight = self.use_tight_budget if force_tight_budget is None else force_tight_budget
        self.budget, self.harness = (tight_budget_harness(self.eval_config_path) if use_tight
                                     else default_budget_harness())
        for key, value in self.budget_override.items():
            if value is not None:
                setattr(self.budget, key, value)
        body = {"instance_id": instance_id, "budget": {
            "tool_calls": self.budget.tool_calls, "time_minutes": self.budget.time_minutes,
            "turns": self.budget.turns,
            "command_timeout_seconds": self.harness.command_timeout_seconds,
            "max_stdout_chars": self.harness.max_stdout_chars,
            "max_file_lines": self.harness.max_file_lines,
            "max_file_chars": self.harness.max_file_chars}}
        deadline = time.time() + self.start_wait_seconds
        while True:
            env = self._call("/start", body)
            if env.get("ok"):
                break
            if env.get("code") == "busy" and time.time() < deadline:
                time.sleep(self.poll_sleep)
                continue
            raise SandboxUnavailable(f"/start {instance_id}: {env}")
        info = env["result"]
        self.session_id = info["session_id"]
        self.task = {"instance_id": instance_id, "repo": info["repo"],
                     "problem_statement": info["problem_statement"]}
        self.ctx = _RemoteCtx(self.budget, self.harness)
        self.consecutive_nudges = 0
        return build_agent_prompt(self.task, "/workspace", self.budget, self.harness,
                                  has_graph=bool(info.get("has_graph")), enable_sandbox_testing=True,
                                  layout=info["layout"])

    def call_tool(self, tool_name, tool_args):
        if tool_name not in ("submit_patch", "get_status"):
            self.ctx.tool_calls_used += 1
        try:
            env = self._call("/tool", {"session_id": self.session_id, "name": tool_name,
                                       "args": tool_args})
        except SandboxUnavailable as e:
            env = {"ok": False, "code": "unavailable", "error": str(e)}
        if not env.get("ok"):
            self.crashed = self.done = True
            return json.dumps({"status": "error", "error_type": "SandboxUnavailable",
                               "error_message": str(env.get("error"))}, ensure_ascii=False)
        if tool_name == "submit_patch":
            self.ctx.patch_submitted = True
        return env["result"]

    def finalize(self):
        """Патч + награда одним запросом: сервер считает diff и проверяет его в свежем контейнере."""
        if self.crashed or self.session_id is None:
            self._final = {"passed": False, "patch": "", "reason": "sandbox_crashed"}
            return ""
        try:
            env = self._call("/finish", {"session_id": self.session_id})
        except SandboxUnavailable as e:
            env = {"ok": False, "error": str(e)}
        self.session_id = None   # сервер закрывает сессию в любом случае
        if not env.get("ok"):
            self.crashed = True
            self._final = {"passed": False, "patch": "", "reason": f"finish_failed: {env.get('error')}"}
            return ""
        self._final = env["result"]
        if self._final.get("infra_error"):
            self.crashed = True   # сбой песочницы, а не вина агента
        return self._final.get("patch", "")

    def verify(self, patch=None):
        return bool(self._final and self._final.get("passed"))

    @property
    def last_result(self):
        return self._final

    def cleanup(self):
        if getattr(self, "session_id", None):
            try:
                self._http("POST", "/abort", {"session_id": self.session_id})
            except SandboxUnavailable:
                pass
            self.session_id = None
