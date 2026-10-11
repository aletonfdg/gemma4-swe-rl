# sim/mimo_sandbox.py -- песочница для задач MiMo: ВСЁ выполняется внутри контейнера.
"""
Отличие от sim/container.py + sim/tools.py: там run_command идёт subprocess'ом на ХОСТЕ,
а репозиторий приходит снимком. Здесь репозиторий уже лежит в образе (/testbed), поэтому
каждый инструмент (run_command/read_file/edit_file/write_file/submit_patch) исполняется через
`docker exec`. Формат ответов инструментов совпадает с sim/tools.py, чтобы prompts и
RewardTracker работали без изменений.

Награда: патч агента применяется в СВЕЖЕМ контейнере (как у настоящего грейдера, чтобы побочные
эффекты вроде установленных пакетов не засчитывались), затем сбрасываются защищённые файлы,
применяется test_patch и запускается `bash mimo_test_command.sh`; passed == (код возврата 0).
Эталонных патчей в датасете нет, поэтому «после верного патча тесты проходят» эта схема
доказать не может -- только эмпирически, когда агент впервые решит задачу.

Docker вызывается через `DockerRunner`; в тестах он подменяется фейком (Docker не нужен).
"""
import io
import json
import posixpath
import re
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, fields

from sim.tools import _apply_replacement, _err, _maybe_add_budget_warning, _ok, _truncate
from sim.verifier import _extract_touched_files, _is_protected

REGISTRY = "xiaomimimo/mimo-v2.6-rl-oss"
WORKDIR = "/testbed"
BASELINE_TAG = "_swegemma_baseline"
APPLY_PASSES = (
    "git apply --unsafe-paths -p1 {p}",
    "git apply --unsafe-paths -p1 -3 {p}",
    "git apply --unsafe-paths -p1 --ignore-space-change --ignore-whitespace {p}",
    "git apply --unsafe-paths -p1 --recount {p}",
    "git apply --unsafe-paths -p0 {p}",
    "patch -p1 --batch --forward < {p}",
    "patch -p0 --batch --forward < {p}",
)
SETUP_SCRIPT = (
    "git config --global --add safe.directory '*' 2>/dev/null; "
    f"cd {WORKDIR} && (git rev-parse --git-dir >/dev/null 2>&1 || git init -q) && "
    "mkdir -p .git/info && printf '__pycache__/\\n*.pyc\\n.pytest_cache/\\n*.egg-info/\\n' "
    ">> .git/info/exclude && git add -A >/dev/null 2>&1; "
    "git -c user.email=sim@local -c user.name=sim commit -q --allow-empty -m baseline && "
    f"git tag -f {BASELINE_TAG} >/dev/null"
)
NO_COUNT_TOOLS = {"submit_patch", "get_status"}
TOOLS = {"run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status",
         "get_code_neighbors", "search_similar_code", "get_code_subgraph"}


class SandboxError(RuntimeError):
    pass


class Busy(RuntimeError):
    """Все слоты заняты -- клиент должен повторить позже."""


@dataclass
class Budget:
    tool_calls: int = 30
    time_minutes: float = 5.0
    turns: int = 60
    command_timeout_seconds: int = 300
    max_stdout_chars: int = 5000
    max_file_lines: int = 150
    max_file_chars: int = 10000

    @classmethod
    def from_dict(cls, d):
        kwargs = {}
        for f in fields(cls):
            if d and f.name in d:
                kwargs[f.name] = type(f.default)(d[f.name])
        return cls(**kwargs)


class DockerRunner:
    """Тонкая обёртка над docker CLI. run() -> (returncode, stdout_bytes, stderr_bytes)."""

    def __init__(self, docker="docker"):
        self.docker = docker

    def run(self, args, input=None, timeout=None):
        try:
            p = subprocess.run([self.docker, *args], input=input if input is not None else b"",
                               capture_output=True, timeout=timeout)
            return p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            return -1, b"", b"TimeoutExpired"


def _text(b):
    return b.decode("utf-8", errors="surrogateescape")


def _lossy(b):
    return b.decode("utf-8", errors="replace")


class MimoSession:
    def __init__(self, task, runner, budget=None, mem="4g", cpus=2.0, image_prefix=REGISTRY):
        if task.get("cwd", WORKDIR) != WORKDIR:
            raise SandboxError(f"поддерживается только cwd={WORKDIR}, у задачи {task.get('cwd')}")
        self.task = task
        self.runner = runner
        self.budget = budget or Budget()
        self.mem, self.cpus = mem, cpus
        self.image = f"{image_prefix}:{task['instance_id']}"
        self.name = f"swegemma_mimo_{uuid.uuid4().hex[:12]}"
        self.tool_calls_used = 0
        self.submitted_patch = ""
        self.patch_submitted = False
        self.has_timeout = False
        self.workspace_link = False
        self.agent_start = time.time()
        self._lock = threading.Lock()
        self._container = False

    # ---------- контейнер ----------
    def exec_bytes(self, argv, input=None, timeout=None):
        return self.runner.run(["exec", "-i", "-w", WORKDIR, self.name, *argv], input=input,
                               timeout=timeout)

    def start(self):
        rc, _, err = self.runner.run([
            "run", "-d", "--name", self.name, "--label", "swegemma_mimo=1", "--network", "none",
            "--memory", self.mem, "--cpus", str(self.cpus), self.image, "sleep", "infinity"])
        if rc:
            raise SandboxError(f"docker run не удался: {_lossy(err)[-400:]}")
        self._container = True
        rc, _, err = self.exec_bytes(["bash", "-c", SETUP_SCRIPT], timeout=600)
        if rc:
            self.stop()
            raise SandboxError(f"подготовка baseline не удалась: {_lossy(err)[-400:]}")
        self.has_timeout = self.exec_bytes(["bash", "-c", "command -v timeout"])[0] == 0
        # /workspace -> /testbed, чтобы пути из промпта совпадали с боевой песочницей
        link = self.exec_bytes(["bash", "-c", f"[ -e /workspace ] || ln -s {WORKDIR} /workspace"])
        self.workspace_link = link[0] == 0
        self.agent_start = time.time()

    def stop(self):
        if self._container:
            self.runner.run(["rm", "-f", self.name])
            self._container = False

    # ---------- информация для промпта ----------
    def repo_name(self):
        rc, out, _ = self.exec_bytes(["bash", "-c", "git config --get remote.origin.url"])
        url = _lossy(out).strip() if rc == 0 else ""
        m = re.search(r"([^/:]+/[^/]+?)(?:\.git)?/?$", url)
        return m.group(1) if m else "this repository"

    def layout(self, max_entries=150):
        rc, out, _ = self.exec_bytes(["bash", "-c", "find . -maxdepth 3 | head -n 600"])
        lines = [l for l in _lossy(out).splitlines()
                 if l.strip() and ".git" not in l and "__pycache__" not in l and not l.endswith(".pyc")]
        return "\n".join(lines[:max_entries]) or "(unavailable)"

    # ---------- бюджет ----------
    def remaining_time_seconds(self):
        return max(0.0, self.budget.time_minutes * 60.0 - (time.time() - self.agent_start))

    def remaining_tool_calls(self):
        return max(0, self.budget.tool_calls - self.tool_calls_used)

    def agent_elapsed_seconds(self):
        return time.time() - self.agent_start

    # ---------- файлы ----------
    def resolve(self, filepath):
        if not filepath or "\x00" in filepath:
            raise ValueError("Invalid path")
        p = filepath
        for prefix in ("/workspace", WORKDIR):
            if p == prefix or p.startswith(prefix + "/"):
                p = p[len(prefix):]
                break
        p = p.lstrip("/")
        norm = posixpath.normpath(p) if p else "."
        if norm == ".." or norm.startswith("../"):
            raise ValueError("Path traversal forbidden")
        return WORKDIR if norm == "." else posixpath.join(WORKDIR, norm)

    def read_bytes(self, path):
        rc, out, _ = self.exec_bytes(["bash", "-c", 'test -f "$1" && cat -- "$1"', "_", path])
        return out if rc == 0 else None

    def write_bytes(self, path, data):
        rc, _, err = self.exec_bytes(
            ["bash", "-c", 'mkdir -p -- "$(dirname -- "$1")" && cat > "$1"', "_", path], input=data)
        if rc:
            raise SandboxError(_lossy(err)[-300:] or "write failed")

    # ---------- патч ----------
    def compute_patch(self):
        self.exec_bytes(["bash", "-c", "git add -N . >/dev/null 2>&1"])
        rc, out, _ = self.exec_bytes(["bash", "-c", f"git diff --binary {BASELINE_TAG}"])
        patch = _lossy(out) if rc == 0 else ""
        self.submitted_patch, self.patch_submitted = patch, True
        return patch

    # ---------- инструменты ----------
    def call(self, name, args):
        if name not in TOOLS:
            return json.dumps(_err("UnknownTool", f"Unknown tool {name}"), ensure_ascii=False)
        with self._lock:
            if name not in NO_COUNT_TOOLS:
                if self.remaining_tool_calls() <= 0 or self.remaining_time_seconds() <= 0:
                    return json.dumps(_err("BudgetExceeded", "Budget exhausted."), ensure_ascii=False)
                self.tool_calls_used += 1
            try:
                result = getattr(self, "_tool_" + name)(**(args or {}))
            except TypeError as e:
                result = _err("ValidationError", str(e))
            except ValueError as e:
                result = _err("ValidationError", str(e))
            except SandboxError as e:
                result = _err("ExecutionError", str(e))
            result = _maybe_add_budget_warning(self, result)
            return json.dumps(result, ensure_ascii=False)

    def _tool_run_command(self, command):
        timeout = min(self.budget.command_timeout_seconds, max(1, int(self.remaining_time_seconds())))
        argv = ["bash", "-lc", command]
        if self.has_timeout:
            argv = ["timeout", "-k", "5", str(timeout), *argv]
        rc, out, err = self.exec_bytes(argv, timeout=timeout + 30)
        if rc == -1 or (self.has_timeout and rc == 124):
            return _err("TimeoutExceeded", f"Timed out after {timeout}s")
        limit = self.budget.max_stdout_chars
        stdout, _ = _truncate(_lossy(out), limit)
        stderr, _ = _truncate(_lossy(err), limit)
        if rc == 0:
            return _ok(stdout=stdout, stderr=stderr, exit_code=0)
        return _err("CommandError", f"exit {rc}", stdout=stdout, stderr=stderr, exit_code=rc)

    def _tool_read_file(self, filepath, start_line=None, end_line=None):
        full = self.resolve(filepath)
        data = self.read_bytes(full)
        if data is None:
            return _err("FileNotFoundError", f"'{filepath}' does not exist.")
        lines = io.StringIO(_lossy(data), newline="\n").readlines()
        total = len(lines)
        start = max(0, (start_line - 1)) if start_line else 0
        stop = min(end_line, total) if end_line else total
        sel = lines[start:stop]
        truncated = False
        if len(sel) > self.budget.max_file_lines:
            sel = sel[:self.budget.max_file_lines]
            truncated = True
        content, char_trunc = _truncate("".join(sel), self.budget.max_file_chars)
        return _ok(filepath=filepath, content=content or "File is empty.", start_line=start + 1,
                   end_line=start + len(sel), total_lines=total, is_truncated=truncated or char_trunc)

    def _tool_edit_file(self, filepath, old_string, new_string, allow_multiple=False):
        import difflib
        full = self.resolve(filepath)
        data = self.read_bytes(full)
        if not data:
            return _err("FileEditError", "File missing or empty.")
        if not old_string:
            return _err("FileEditError", "old_string must not be empty.")
        original = _text(data)
        new_content, occ, strategy = _apply_replacement(original, old_string, new_string,
                                                        allow_multiple)
        if new_content is None:
            return _err("FileEditError",
                        f"old_string not matched (tried exact/flexible/regex, best={occ}).")
        self.write_bytes(full, new_content.encode("utf-8", errors="surrogateescape"))
        diff = "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                            new_content.splitlines(keepends=True),
                                            fromfile=f"a/{filepath}", tofile=f"b/{filepath}"))
        diff, trunc = _truncate(diff, self.budget.max_stdout_chars)
        return _ok(filepath=filepath, occurrences=occ, strategy=strategy, diff=diff,
                   is_truncated=trunc)

    def _tool_write_file(self, filepath, content):
        full = self.resolve(filepath)
        data = content.encode("utf-8")
        self.write_bytes(full, data)
        return _ok(filepath=filepath, size=len(data))

    def _tool_submit_patch(self):
        patch = self.compute_patch()
        return _ok(patch_size=len(patch), files_changed=len(re.findall(r"(?m)^diff --git ", patch)))

    def _tool_get_status(self):
        return _ok(
            tool_calls_used=self.tool_calls_used, patch_submitted=self.patch_submitted,
            patch_size=len(self.submitted_patch), tool_calls_remaining=self.remaining_tool_calls(),
            max_tool_calls=self.budget.tool_calls,
            time_seconds_remaining=self.remaining_time_seconds(),
            max_time_minutes=self.budget.time_minutes,
            agent_elapsed_seconds=self.agent_elapsed_seconds(), max_turns=self.budget.turns,
            command_timeout_seconds=self.budget.command_timeout_seconds)

    def _tool_get_code_neighbors(self, node=None, edge_type=None, max_neighbors=50):
        return _err("GraphUnavailable", "No graph for this repo.")

    def _tool_search_similar_code(self, query=None, k=10):
        return _err("EmbeddingsUnavailable", "No embeddings for this repo.")

    def _tool_get_code_subgraph(self, nodes=None):
        return _err("GraphUnavailable", "No graph for this repo.")


def verify_patch(runner, task, patch, mem="4g", cpus=2.0, image_prefix=REGISTRY, timeout_cap=900):
    """Проверка патча в свежем контейнере. -> dict(passed, reason, infra_error, tail)."""
    if not patch or not patch.strip() or patch.strip() == "NO_PATCH":
        return {"passed": False, "reason": "no_patch", "infra_error": False, "tail": []}
    sess = MimoSession(task, runner, mem=mem, cpus=cpus, image_prefix=image_prefix)
    try:
        sess.start()
        sess.write_bytes("/tmp/_agent.patch", patch.encode("utf-8", errors="surrogateescape"))
        if not any(sess.exec_bytes(["bash", "-c", c.format(p="/tmp/_agent.patch")])[0] == 0
                   for c in APPLY_PASSES):
            return {"passed": False, "reason": "patch_apply_failed", "infra_error": False,
                    "tail": []}
        protected = sorted(f for f in _extract_touched_files(patch) if _is_protected(f))
        if protected:
            files = " ".join(shlex.quote(f) for f in protected)
            sess.exec_bytes(["bash", "-c",
                             f"git checkout {BASELINE_TAG} -- {files} 2>/dev/null; "
                             f"git clean -f -- {files} 2>/dev/null; true"])
        sess.write_bytes("/tmp/_test.patch", task["test_patch"].encode("utf-8"))
        rc, _, err = sess.exec_bytes(["bash", "-c", "git apply --whitespace=nowarn /tmp/_test.patch"])
        if rc:
            return {"passed": False, "reason": "test_patch_failed", "infra_error": True,
                    "tail": _lossy(err).splitlines()[-8:]}
        limit = min(int(task.get("verifier_timeout_sec", timeout_cap)), timeout_cap)
        argv = ["bash", "mimo_test_command.sh"]
        if sess.has_timeout:
            argv = ["timeout", "-k", "5", str(limit), *argv]
        rc, out, err = sess.exec_bytes(argv, timeout=limit + 30)
        tail = (_lossy(out) + _lossy(err)).replace("\r", "").strip().splitlines()[-15:]
        if rc == -1 or (sess.has_timeout and rc == 124):
            return {"passed": False, "reason": "tests_timeout", "infra_error": False, "tail": tail}
        return {"passed": rc == 0, "reason": "tests_passed" if rc == 0 else "tests_failed",
                "infra_error": False, "tail": tail}
    except SandboxError as e:
        return {"passed": False, "reason": f"sandbox_error: {e}", "infra_error": True, "tail": []}
    finally:
        sess.stop()


class SessionManager:
    def __init__(self, tasks, runner=None, max_sessions=3, ttl_seconds=3600, mem="4g", cpus=2.0,
                 image_prefix=REGISTRY, verify_timeout_cap=900):
        self.tasks = tasks
        self.runner = runner or DockerRunner()
        self.mem, self.cpus, self.image_prefix = mem, cpus, image_prefix
        self.ttl, self.verify_timeout_cap = ttl_seconds, verify_timeout_cap
        self._sessions = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max_sessions)

    def _release(self, sid):
        with self._lock:
            entry = self._sessions.pop(sid, None)
        if entry:
            entry["session"].stop()
            self._slots.release()
        return entry

    def start(self, instance_id, budget_dict=None):
        task = self.tasks.get(instance_id)
        if task is None:
            raise KeyError(instance_id)
        if not self._slots.acquire(blocking=False):
            raise Busy("все слоты заняты")
        sid = uuid.uuid4().hex
        try:
            sess = MimoSession(task, self.runner, Budget.from_dict(budget_dict), self.mem, self.cpus,
                               self.image_prefix)
            sess.start()
        except BaseException:
            self._slots.release()
            raise
        with self._lock:
            self._sessions[sid] = {"session": sess, "created": time.time()}
        return {"session_id": sid, "instance_id": instance_id,
                "problem_statement": task["problem_statement"], "repo": sess.repo_name(),
                "layout": sess.layout(), "workspace_link": sess.workspace_link}

    def _get(self, sid):
        with self._lock:
            entry = self._sessions.get(sid)
        if entry is None:
            raise KeyError(sid)
        return entry["session"]

    def tool(self, sid, name, args):
        return self._get(sid).call(name, args)

    def finish(self, sid):
        sess = self._get(sid)
        try:
            patch = sess.submitted_patch if sess.patch_submitted else sess.compute_patch()
            sess.stop()   # освободить ресурсы до проверки
            res = verify_patch(self.runner, sess.task, patch, self.mem, self.cpus,
                               self.image_prefix, self.verify_timeout_cap)
        finally:
            self._release(sid)
        res["patch"] = patch
        return res

    def abort(self, sid):
        return self._release(sid) is not None

    def abort_all(self):
        """Закрыть все сессии и освободить слоты (например, клиент упал и бросил эпизоды)."""
        with self._lock:
            sids = list(self._sessions)
        return sum(self._release(sid) is not None for sid in sids)

    def remove_stale_containers(self):
        """Удалить контейнеры песочницы, оставшиеся от прошлого запуска сервера (метка swegemma_mimo)."""
        rc, out, _ = self.runner.run(["ps", "-aq", "--filter", "label=swegemma_mimo=1"], timeout=60)
        ids = out.decode("utf-8", "replace").split() if rc == 0 else []
        if ids:
            self.runner.run(["rm", "-f", *ids], timeout=120)
        return len(ids)

    def reap(self):
        now = time.time()
        with self._lock:
            old = [s for s, e in self._sessions.items() if now - e["created"] > self.ttl]
        for sid in old:
            self._release(sid)
        return len(old)

    def active(self):
        with self._lock:
            return len(self._sessions)
