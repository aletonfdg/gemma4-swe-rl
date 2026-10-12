# sim/comp_sandbox.py -- песочница для публичных задач конкурса (fastapi, rich, requests).
"""
Повторяет двухконтейнерный жизненный цикл харнесса (HARNESS_README, разделы 4 и 8):
  * контейнер A: снимок репозитория -> /workspace, editable-установка, setup.py --fast-path,
    baseline-коммит с тегом `_swegemma_baseline`; агент работает через `docker exec`;
  * контейнер B (verify_comp_patch): свежий снимок, патч агента (7 проходов применения), сброс
    защищённых файлов и файлов из test_patch, применение test_patch, pytest + JUnit-проверка.

Отличия от боевой среды (честно):
  * зависимости не из /wheels (там нет, например, typing-inspection), а запечены в образах
    swebench-fastapi / swebench-rich / swebench-requests при их сборке (в песочнице сети нет);
  * pytest.ini и conftest.py пишет настоящий sandbox/setup.py, но не весь harness.setup_workspace_test_config;
  * проверка идёт по тестовым файлам из test_patch (как в наших прогонах заземления), а не по
    FAIL_TO_PASS/PASS_TO_PASS, которых в публичных данных нет.
Годными считаются задачи из data/benchmark_valid.json: на них эталонный патч проходит, пустой падает.

Графовые инструменты читают graphs/<id>.json и embeddings/<id>.npz на хосте (networkx, numpy);
если их нет, инструменты отвечают GraphUnavailable, а has_graph() == False.
"""
import json
import os
import re
import shlex
import xml.etree.ElementTree as ET

from sim.mimo_sandbox import (APPLY_PASSES, BASELINE_TAG, MimoSession, SandboxError, SessionManager,
                              _lossy)
from sim.verifier import _extract_touched_files, _is_protected

IMAGES = {
    "fastapi/fastapi": "swebench-fastapi:latest",
    "Textualize/rich": "swebench-rich:latest",
    "psf/requests": "swebench-requests:latest",
}
WORKDIR = "/workspace"
PYTEST_FLAGS = "-p no:anyio -o timeout=0 -o python_classes='Test* *Test' -q"


def load_comp_tasks(comp_dir, valid_json=None):
    """{instance_id: task} для задач с готовым образом; valid_json ограничивает список годными."""
    allowed = None
    if valid_json:
        allowed = set(json.load(open(valid_json, encoding="utf-8"))["valid"])
    tasks = {}
    with open(os.path.join(comp_dir, "tasks.jsonl"), encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["repo"] in IMAGES and (allowed is None or t["instance_id"] in allowed):
                tasks[t["instance_id"]] = t
    return tasks


def _test_files(test_patch):
    return sorted({f for f in re.findall(r"^\+\+\+ b/(\S+)", test_patch, re.M) if f.endswith(".py")})


def _patch_text(patch):
    return patch if patch.endswith("\n") else patch + "\n"   # в tasks.jsonl патчи без последнего \n


class CompSession(MimoSession):
    workdir = WORKDIR

    def __init__(self, task, runner, budget=None, mem="4g", cpus=2.0, comp_dir=".", graphs=True):
        super().__init__(task, runner, budget, mem, cpus)
        self.image = IMAGES[task["repo"]]
        self.comp_dir = os.path.abspath(comp_dir)
        self._graph_ctx = self._load_graph(graphs)

    def _load_graph(self, enabled):
        if not enabled:
            return None
        # в архиве файлы названы по коммиту (<repo>_<base_commit>); копии по instance_id как запасной вариант
        stems = [f"{self.task['repo'].split('/')[1].lower()}_{self.task['base_commit']}",
                 self.task["instance_id"]]
        graph = next((p for s in stems if os.path.exists(p := os.path.join(self.comp_dir, "graphs", s + ".json"))),
                     os.path.join(self.comp_dir, "graphs", stems[0] + ".json"))
        emb = next((p for s in stems if os.path.exists(p := os.path.join(self.comp_dir, "embeddings", s + ".npz"))),
                   os.path.join(self.comp_dir, "embeddings", stems[0] + ".npz"))
        if not (os.path.exists(graph) or os.path.exists(emb)):
            return None
        try:
            from sim.context import SwegemmaContext
            ctx = SwegemmaContext(WORKDIR, None, None, graph if os.path.exists(graph) else None,
                                  emb if os.path.exists(emb) else None)
        except Exception:   # нет networkx/numpy или битый файл: работаем без графа
            return None
        return ctx if ctx.has_code_intel() else None

    def has_graph(self):
        return self._graph_ctx is not None

    def repo_name(self):
        return self.task["repo"]

    def docker_run_args(self):
        args = super().docker_run_args()
        mounts = ["-v", f"{self.comp_dir}/wheels:/wheels:ro", "-v", f"{self.comp_dir}/sandbox:/sandbox:ro"]
        i = args.index(self.image)
        return args[:i] + mounts + args[i:]

    def prepare(self):
        iid, repo = self.task["instance_id"], self.task["repo"]
        with open(os.path.join(self.comp_dir, "snapshots", iid + ".tgz"), "rb") as f:
            data = f.read()
        rc, _, err = self.exec_bytes(["bash", "-c", f"tar xzf - -C {WORKDIR}"], input=data, timeout=600)
        if rc:
            raise SandboxError(f"снимок не распакован: {_lossy(err)[-300:]}")
        script = (
            "git config --global --add safe.directory '*' 2>/dev/null; "
            "pip install --no-index --find-links=/wheels --no-build-isolation --no-deps -e /workspace "
            f">/dev/null 2>&1; python3 /sandbox/setup.py --fast-path {shlex.quote(repo)} >/dev/null 2>&1; "
            "git add -A >/dev/null 2>&1; git commit -q -m eval_baseline --allow-empty && "
            f"git tag -f {BASELINE_TAG} >/dev/null")
        rc, _, err = self.exec_bytes(["bash", "-c", script], timeout=900)
        if rc:
            raise SandboxError(f"подготовка baseline не удалась: {_lossy(err)[-300:]}")

    # ---------- графовые инструменты (на хосте) ----------
    def _graph_tool(self, fn, **kwargs):
        from sim.tools import _err
        if self._graph_ctx is None:
            return _err("GraphUnavailable", "No graph for this repo.")
        return fn.__wrapped__(self._graph_ctx, **kwargs)   # без повторного учёта бюджета

    def _tool_get_code_neighbors(self, node=None, edge_type=None, max_neighbors=50):
        from sim.tools import get_code_neighbors
        return self._graph_tool(get_code_neighbors, node=node, edge_type=edge_type,
                                max_neighbors=max_neighbors)

    def _tool_search_similar_code(self, query=None, k=10):
        from sim.tools import search_similar_code
        return self._graph_tool(search_similar_code, query=query, k=k)

    def _tool_get_code_subgraph(self, nodes=None):
        from sim.tools import get_code_subgraph
        return self._graph_tool(get_code_subgraph, nodes=nodes or [])


def _junit_ok(xml_text):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return False
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = sum(int(s.get("tests", 0)) for s in suites)
    bad = sum(int(s.get(k, 0)) for s in suites for k in ("failures", "errors"))
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    return total - bad - skipped > 0 and bad == 0


def verify_comp_patch(runner, task, patch, comp_dir, mem="4g", cpus=2.0, timeout_cap=900):
    """Проверка в свежем контейнере -> dict(passed, reason, infra_error, tail)."""
    if not patch or not patch.strip() or patch.strip() == "NO_PATCH":
        return {"passed": False, "reason": "no_patch", "infra_error": False, "tail": []}
    sess = CompSession(task, runner, mem=mem, cpus=cpus, comp_dir=comp_dir, graphs=False)
    try:
        sess.start()
        sess.write_bytes("/tmp/_agent.patch", _patch_text(patch).encode("utf-8", errors="surrogateescape"))
        if not any(sess.exec_bytes(["bash", "-c", c.format(p="/tmp/_agent.patch")])[0] == 0
                   for c in APPLY_PASSES):
            return {"passed": False, "reason": "patch_apply_failed", "infra_error": False, "tail": []}
        test_files = _test_files(task["test_patch"])
        reset = sorted({f for f in _extract_touched_files(patch) if _is_protected(f)} | set(test_files))
        if reset:
            files = " ".join(shlex.quote(f) for f in reset)
            sess.exec_bytes(["bash", "-c",
                             f"git checkout {BASELINE_TAG} -- {files} 2>/dev/null; "
                             f"git clean -f -- {files} 2>/dev/null; true"])
        sess.write_bytes("/tmp/_test.patch", _patch_text(task["test_patch"]).encode("utf-8"))
        rc, _, err = sess.exec_bytes(["bash", "-c", "git apply --unsafe-paths -p1 /tmp/_test.patch"])
        if rc:
            return {"passed": False, "reason": "test_patch_failed", "infra_error": True,
                    "tail": _lossy(err).splitlines()[-8:]}
        limit = min(int(task.get("verifier_timeout_sec", timeout_cap)), timeout_cap)
        cmd = ("PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest "
               + " ".join(shlex.quote(f) for f in test_files)
               + f" --junitxml=/tmp/_junit.xml {PYTEST_FLAGS} 2>&1")
        argv = ["bash", "-lc", cmd]
        if sess.has_timeout:
            argv = ["timeout", "-k", "5", str(limit), *argv]
        rc, out, _ = sess.exec_bytes(argv, timeout=limit + 30)
        tail = _lossy(out).replace("\r", "").strip().splitlines()[-15:]
        if rc == -1 or (sess.has_timeout and rc == 124):
            return {"passed": False, "reason": "tests_timeout", "infra_error": False, "tail": tail}
        junit = sess.read_bytes("/tmp/_junit.xml")
        passed = rc == 0 and junit is not None and _junit_ok(_lossy(junit))
        return {"passed": passed, "reason": "tests_passed" if passed else "tests_failed",
                "infra_error": False, "tail": tail}
    except SandboxError as e:
        return {"passed": False, "reason": f"sandbox_error: {e}", "infra_error": True, "tail": []}
    finally:
        sess.stop()


class CompSessionManager(SessionManager):
    def __init__(self, tasks, comp_dir, graphs=True, **kw):
        super().__init__(tasks, **kw)
        self.comp_dir, self.graphs = comp_dir, graphs

    def _make_session(self, task, budget):
        return CompSession(task, self.runner, budget, self.mem, self.cpus, self.comp_dir, self.graphs)

    def _verify(self, task, patch):
        return verify_comp_patch(self.runner, task, patch, self.comp_dir, self.mem, self.cpus,
                                 self.verify_timeout_cap)
