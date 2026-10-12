# sim/mimo_server.py -- HTTP-сервер песочницы MiMo (запускается на машине с Docker).
"""
    python -m sim.mimo_server --data C:/Users/vishn/Desktop/data/mimo_v2_6_code --port 8765
    cloudflared tunnel --url http://127.0.0.1:8765      # получите https://....trycloudflare.com

Протокол: JSON по HTTP, заголовок `Authorization: Bearer <token>` обязателен везде, кроме /health.
Ответ всегда {"ok": true, "result": ...} или {"ok": false, "code": ..., "error": ...}.
Долгие операции (start/tool/finish) идут фоновыми задачами: если за WAIT_SECONDS не успели,
сервер отвечает {"ok": true, "pending": true, "job": id}, и клиент опрашивает POST /poll.
Это нужно из-за туннелей, которые обрывают запрос, висящий дольше ~100 с.

    GET  /health                                      -> {"ok": true}
    GET  /tasks                                       -> {"ok": true, "result": [instance_id, ...]}
    POST /start   {"instance_id", "budget": {...}}    -> session_id, problem_statement, repo, layout
    POST /tool    {"session_id", "name", "args"}      -> строка-наблюдение (как sim/tools.py)
    POST /finish  {"session_id"}                      -> patch, passed, reason, tail
    POST /abort   {"session_id"}
    POST /poll    {"job"}
По умолчанию слушает только 127.0.0.1: наружу выставляйте через туннель, не через 0.0.0.0.
"""
import argparse
import hmac
import json
import os
import re
import secrets
import sys
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sim.mimo_sandbox import Busy, SandboxError, SessionManager

MAX_BODY = 4_000_000
WAIT_SECONDS = 50.0
META_FILES = {"test_commands.json", "mimo_test_command.sh"}


def load_tasks(data_dir, limit=None):
    """Python-задачи с pytest и cwd=/testbed -> {instance_id: instance_json}."""
    from datasets import load_from_disk
    tasks = {}
    for row in load_from_disk(data_dir)["train"]:
        ij = json.loads(row["extra_info"]["instance_json"])
        files = [f for f in re.findall(r"^diff --git a/(\S+)", ij["test_patch"], re.M)
                 if f not in META_FILES]
        if (ij.get("cwd") == "/testbed" and files and all(f.endswith(".py") for f in files)
                and "pytest" in ij["test_command"] + ij["test_patch"]):
            tasks[ij["instance_id"]] = ij
            if limit and len(tasks) >= limit:
                break
    return tasks


def _wrap(fn):
    def run():
        try:
            return {"ok": True, "result": fn()}
        except Busy as e:
            return {"ok": False, "code": "busy", "error": str(e)}
        except KeyError as e:
            return {"ok": False, "code": "unknown", "error": f"unknown id: {e}"}
        except SandboxError as e:
            return {"ok": False, "code": "sandbox", "error": str(e)}
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "code": "internal", "error": f"{type(e).__name__}: {e}"}
    return run


class App:
    def __init__(self, manager, tasks, token, wait_seconds=WAIT_SECONDS, workers=16):
        self.mgr, self.tasks, self.token = manager, tasks, token
        self.wait_seconds = wait_seconds
        self._pool = ThreadPoolExecutor(workers)
        self._jobs = {}
        self._lock = threading.Lock()

    def authorized(self, header):
        got = header or ""
        want = f"Bearer {self.token}"
        return hmac.compare_digest(got.encode("utf-8"), want.encode("utf-8"))

    def _submit(self, fn):
        job = uuid.uuid4().hex
        with self._lock:
            self._jobs[job] = self._pool.submit(_wrap(fn))
        return job

    def _await(self, job):
        with self._lock:
            fut = self._jobs.get(job)
        if fut is None:
            return 404, {"ok": False, "code": "unknown", "error": "unknown job"}
        try:
            env = fut.result(timeout=self.wait_seconds)
        except FutureTimeout:
            return 200, {"ok": True, "pending": True, "job": job}
        with self._lock:
            self._jobs.pop(job, None)
        return 200, env

    def handle_get(self, path):
        if path == "/health":
            return 200, {"ok": True, "active": self.mgr.active()}
        if path == "/tasks":
            return 200, {"ok": True, "result": sorted(self.tasks)}
        return 404, {"ok": False, "code": "not_found", "error": path}

    def handle_post(self, path, body):
        try:
            if path == "/start":
                iid, budget = str(body["instance_id"]), body.get("budget")
                return self._await(self._submit(lambda: self.mgr.start(iid, budget)))
            if path == "/tool":
                sid, name = str(body["session_id"]), str(body["name"])
                args = body.get("args") or {}
                if not isinstance(args, dict):
                    raise ValueError("args must be an object")
                return self._await(self._submit(lambda: self.mgr.tool(sid, name, args)))
            if path == "/finish":
                sid = str(body["session_id"])
                return self._await(self._submit(lambda: self.mgr.finish(sid)))
            if path == "/abort":
                return 200, {"ok": True, "result": self.mgr.abort(str(body["session_id"]))}
            if path == "/abort_all":
                return 200, {"ok": True, "result": self.mgr.abort_all()}
            if path == "/poll":
                return self._await(str(body["job"]))
        except (KeyError, ValueError, TypeError) as e:
            return 400, {"ok": False, "code": "bad_request", "error": f"{type(e).__name__}: {e}"}
        return 404, {"ok": False, "code": "not_found", "error": path}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, code, obj):
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/health" and not app.authorized(self.headers.get("Authorization")):
                return self._send(401, {"ok": False, "code": "unauthorized", "error": "bad token"})
            self._send(*app.handle_get(self.path))

        def do_POST(self):
            if not app.authorized(self.headers.get("Authorization")):
                return self._send(401, {"ok": False, "code": "unauthorized", "error": "bad token"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._send(413, {"ok": False, "code": "too_large", "error": "body size"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("body must be an object")
            except ValueError as e:
                return self._send(400, {"ok": False, "code": "bad_request", "error": str(e)})
            self._send(*app.handle_post(self.path, body))

    return Handler


def serve(app, host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), make_handler(app))
    server.daemon_threads = True

    def reaper():
        stop = threading.Event()
        while not stop.wait(60):
            n = app.mgr.reap()
            if n:
                print(f"[reaper] закрыто просроченных сессий: {n}", file=sys.stderr)
    threading.Thread(target=reaper, daemon=True).start()
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", default=None, help="папка load_from_disk с датасетом MiMo")
    ap.add_argument("--comp", default=None,
                    help="папка с распакованными данными конкурса (tasks.jsonl, snapshots/, graphs/, ...) "
                         "вместо MiMo; образы swebench-fastapi/rich/requests должны быть собраны")
    ap.add_argument("--valid", default=None,
                    help="data/benchmark_valid.json: брать только задачи, где эталонный патч проходит")
    ap.add_argument("--no-graph", action="store_true", help="--comp: отключить графовые инструменты")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=None, help="по умолчанию: $SWE_SANDBOX_TOKEN или случайный")
    ap.add_argument("--max-sessions", type=int, default=3)
    ap.add_argument("--ttl", type=int, default=3600, help="секунд до принудительного закрытия сессии")
    ap.add_argument("--mem", default="4g")
    ap.add_argument("--cpus", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None, help="взять только первые N задач")
    args = ap.parse_args()

    token = args.token or os.environ.get("SWE_SANDBOX_TOKEN") or secrets.token_urlsafe(24)
    if bool(args.data) == bool(args.comp):
        ap.error("укажите ровно один из --data (MiMo) и --comp (задачи конкурса)")
    if args.comp:
        from sim.comp_sandbox import CompSessionManager, load_comp_tasks
        tasks = load_comp_tasks(args.comp, args.valid)
        if args.limit:
            tasks = dict(list(tasks.items())[:args.limit])
        if not args.no_graph:
            try:
                import networkx, numpy  # noqa: F401
            except ImportError as e:
                print(f"ВНИМАНИЕ: графовые инструменты отключены ({e}); установите: pip install networkx numpy",
                      file=sys.stderr)
        manager = CompSessionManager(tasks, args.comp, graphs=not args.no_graph,
                                     max_sessions=args.max_sessions, ttl_seconds=args.ttl,
                                     mem=args.mem, cpus=args.cpus)
    else:
        tasks = load_tasks(args.data, args.limit)
        manager = SessionManager(tasks, max_sessions=args.max_sessions, ttl_seconds=args.ttl,
                                 mem=args.mem, cpus=args.cpus)
    try:
        stale = manager.remove_stale_containers()
        if stale:
            print(f"Удалено брошенных контейнеров от прошлого запуска: {stale}")
    except Exception as e:   # Docker недоступен -- сервер всё равно запустится, ошибка вылезет на /start
        print(f"Не удалось проверить старые контейнеры: {e}", file=sys.stderr)
    server = serve(App(manager, tasks, token), args.host, args.port)
    print(f"Задач (python, pytest): {len(tasks)}; слотов: {args.max_sessions}")
    print(f"Слушаю http://{args.host}:{args.port}")
    print(f"Токен: {token}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("остановка")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
