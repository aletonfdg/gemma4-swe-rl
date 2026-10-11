# sim/run_mimo_baseline.py -- прогон модели по задачам MiMo через удалённую песочницу (для Colab).
"""
    python -m sim.run_mimo_baseline --first 15 --workers 2 --out results/mimo_baseline

Каждая задача = один эпизод `rl.rollout.run_episode`; строка с итогом дописывается в
<out>/results.jsonl сразу после эпизода (прерванный прогон можно продолжить: уже готовые
задачи пропускаются), полная переписка -- в <out>/messages/<id>.json. Параметры сервера
берутся из SWE_SANDBOX_URL / SWE_SANDBOX_TOKEN, модель -- из --llm-url.
"""
import argparse
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from llm.llm_client import LLMClient
from rl.rollout import run_episode
from sim.mimo_env import MimoRemoteEnv

_LOCK = threading.Lock()


class BaselineClient(LLMClient):
    """Тот же клиент, но с лимитом вывода ближе к боевому sampling.yaml (там 16384 токенов;
    с растущим контекстом 32768 столько не помещается вместе с промптом, берём 8192)."""

    def __init__(self, *a, max_tokens=8192, **kw):
        super().__init__(*a, **kw)
        self._max_tokens = max_tokens

    def chat(self, messages, tools, temperature=0.2, max_tokens=None, lora_name=None):
        return super().chat(messages, tools, temperature=temperature,
                            max_tokens=max_tokens or self._max_tokens, lora_name=lora_name)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def load_rows(path):
    """Последняя строка по каждой задаче: повторный запуск дописывает новую вместо упавшей."""
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        last = {}
        for line in f:
            if line.strip():
                row = json.loads(line)
                last[row["instance_id"]] = row
    return list(last.values())


def done_ids(path):
    """Готовы только задачи без сбоя инфраструктуры: упавшие (туннель, песочница) перезапускаются."""
    return {r["instance_id"] for r in load_rows(path) if not r.get("error") and not r.get("crashed")}


def run_one(task_id, args, client):
    t0 = time.time()
    env = MimoRemoteEnv(args.sandbox_url, args.token, task_ids=[task_id])
    row = {"instance_id": task_id, "passed": False, "error": None}
    try:
        steps, messages, passed = run_episode(env, 0, client, None, None, temperature=args.temperature)
        row.update(passed=bool(passed), steps=len(steps), turns=env.ctx.turns_used,
                   tool_calls=env.ctx.tool_calls_used, crashed=bool(env.crashed),
                   reward=sum(s["reward"] for s in steps),
                   result={k: v for k, v in (env.last_result or {}).items() if k != "patch"},
                   patch_chars=len((env.last_result or {}).get("patch", "")))
        os.makedirs(os.path.join(args.out, "messages"), exist_ok=True)
        with open(os.path.join(args.out, "messages", f"{task_id}.json"), "w", encoding="utf-8") as f:
            json.dump({"messages": messages, "patch": (env.last_result or {}).get("patch", "")},
                      f, ensure_ascii=False)
    except Exception as e:   # один упавший эпизод не должен ронять весь прогон
        row["error"] = f"{type(e).__name__}: {e}"
        try:
            env.cleanup()
        except Exception:
            pass
    row["seconds"] = round(time.time() - t0)
    with _LOCK:
        with open(os.path.join(args.out, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{task_id}: passed={row['passed']} reason={row.get('result', {}).get('reason')} "
          f"calls={row.get('tool_calls')} {row['seconds']}s err={row['error']}", flush=True)


def summarize(path):
    rows = load_rows(path)
    ok = [r for r in rows if not r.get("crashed") and not r.get("error")]   # сбои инфраструктуры не в счёт
    k = sum(r["passed"] for r in ok)
    lo, hi = wilson(k, len(ok))
    print(f"\nрешено {k}/{len(ok)} (сбоев инфраструктуры: {len(rows) - len(ok)}); "
          f"95% интервал Уилсона {lo:.0%}-{hi:.0%}")
    reasons = {}
    for r in ok:
        key = r.get("result", {}).get("reason")
        reasons[key] = reasons.get(key, 0) + 1
    print("причины:", reasons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=15, help="первые N задач из списка сервера")
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=2, help="не больше --max-sessions сервера")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--sandbox-url", default=os.environ.get("SWE_SANDBOX_URL"))
    ap.add_argument("--token", default=os.environ.get("SWE_SANDBOX_TOKEN"))
    ap.add_argument("--out", default="results/mimo_baseline")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ids = args.ids or [t["instance_id"] for t in MimoRemoteEnv(args.sandbox_url, args.token).tasks][:args.first]
    path = os.path.join(args.out, "results.jsonl")
    todo = [i for i in ids if i not in done_ids(path)]
    print(f"задач: {len(ids)}, осталось: {len(todo)}, потоков: {args.workers}", flush=True)

    client = BaselineClient(base_url=args.llm_url, model=args.model, max_tokens=args.max_tokens)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda i: run_one(i, args, client), todo))
    if os.path.exists(path):
        summarize(path)


if __name__ == "__main__":
    main()
