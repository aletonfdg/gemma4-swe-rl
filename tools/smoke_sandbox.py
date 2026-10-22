# tools/smoke_sandbox.py -- быстрая проверка сервера песочницы без модели
"""
    python -m tools.smoke_sandbox --url http://127.0.0.1:8765 --token <токен> [--id <instance_id>]

Запускает одну задачу, делает несколько вызовов инструментов (run_command, неудачный edit_file),
завершает эпизод без правки и печатает время каждого шага и количество образов Docker до и после.
Ожидание на неисправленном репозитории: passed=False (тесты падают), образ задачи удалён, если сервер
запущен с --prune-images. Адрес и токен можно взять из $SWE_SANDBOX_URL и $SWE_SANDBOX_TOKEN.
"""
import argparse
import os
import random
import subprocess
import time

from sim.mimo_env import MimoRemoteEnv


def docker_images():
    p = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True)
    return [x for x in p.stdout.split() if x]


def timed(label, fn):
    t0 = time.time()
    out = fn()
    print(f"[{time.time() - t0:6.1f}s] {label}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("SWE_SANDBOX_URL"))
    ap.add_argument("--token", default=os.environ.get("SWE_SANDBOX_TOKEN"))
    ap.add_argument("--id", default=None, help="по умолчанию случайная задача")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = MimoRemoteEnv(args.url.strip(), args.token.strip())
    ids = [t["instance_id"] for t in env.tasks]
    print("задач на сервере:", len(ids))
    iid = args.id or random.Random(args.seed).choice(ids)
    env.tasks = [{"instance_id": iid}]
    before = docker_images()
    print("задача:", iid, "| образов Docker до:", len(before))

    prompt = timed("start (образ + контейнер)", lambda: env.reset(0))
    print("   условие задачи, символов:", len(env.task["problem_statement"]), "| промпт:", len(prompt))
    out = timed("run_command pwd; git status", lambda: env.call_tool("run_command", {"command": "pwd; git status --short | head"}))
    print("   ", str(out)[:300].replace("\n", " | "))
    out = timed("run_command ls", lambda: env.call_tool("run_command", {"command": "ls"}))
    print("   ", str(out)[:300].replace("\n", " | "))
    out = timed("edit_file на несуществующем файле (должна быть ошибка)",
                lambda: env.call_tool("edit_file", {"filepath": "no_such_file.py", "old_string": "a", "new_string": "b"}))
    print("   ", str(out)[:300])
    timed("finalize (проверка пустого патча)", env.finalize)
    res = env.last_result or {}
    print("   passed:", res.get("passed"), "| reason:", res.get("reason"), "| infra_error:", res.get("infra_error"))
    env.cleanup()
    time.sleep(3)
    after = docker_images()
    gone = [x for x in before if x not in after]
    print("образов Docker после:", len(after), "| исчезло:", gone, "| появилось:", [x for x in after if x not in before])
    ok = res.get("passed") is False and not res.get("infra_error")
    print("ИТОГ:", "сервер работает" if ok else "ПРОВЕРЬТЕ вывод выше")


if __name__ == "__main__":
    main()
