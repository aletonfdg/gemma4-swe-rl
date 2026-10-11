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
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import BadRequestError

from llm.llm_client import LLMClient
from rl.rollout import run_episode
from sim.mimo_env import MimoRemoteEnv

_LOCK = threading.Lock()
MAX_MODEL_LEN = 32768
MIN_OUTPUT_TOKENS = 512


class ContextFull(RuntimeError):
    """Промпт занял почти весь контекст: модель уже не может ответить. Это провал агента, а не сбой."""


_ACTIVE = [0]   # сколько эпизодов идёт прямо сейчас: от этого зависит скорость модели


class EpisodeTrace:
    """Хронология эпизода: куда уходит бюджет времени (модель, инструменты, остальное)."""

    def __init__(self):
        self.t0 = time.time()
        self.events = []

    def add(self, kind, start, dur, **kw):
        self.events.append({"k": kind, "at": round(start - self.t0, 2), "s": round(dur, 2), **kw})

    def summary(self):
        ev = self.events
        llm = [e for e in ev if e["k"] == "llm"]

        def tot(*kinds):
            return round(sum(e["s"] for e in ev if e["k"] in kinds), 1)
        return {"reset_s": tot("reset"), "llm_s": tot("llm"), "tool_s": tot("tool"),
                "finish_s": tot("finalize", "verify"), "turns": len(llm),
                "ctok": sum(e.get("ctok") or 0 for e in llm),
                "max_ptok": max([e.get("ptok") or 0 for e in llm], default=0), "events": ev}


class TracedClient:
    """Клиент модели одного эпизода: меряет задержку и токены каждого хода."""

    def __init__(self, base, trace):
        self.base, self.trace = base, trace

    def chat(self, *a, **kw):
        t, out = time.time(), None
        try:
            out = self.base.chat(*a, **kw)
            return out
        finally:
            u = (out or {}).get("usage") or {}
            self.trace.add("llm", t, time.time() - t, ptok=u.get("prompt_tokens"),
                           ctok=u.get("completion_tokens"), par=_ACTIVE[0])


class TimedEnv(MimoRemoteEnv):
    """Окружение MiMo, которое пишет время сброса, каждого инструмента и проверки в трассу."""

    def __init__(self, *a, trace, **kw):
        super().__init__(*a, **kw)
        self.trace = trace

    def _timed(self, kind, fn, *a, **kw):
        t = time.time()
        try:
            return fn(*a, **kw)
        finally:
            self.trace.add(kind, t, time.time() - t)

    def reset(self, *a, **kw):
        return self._timed("reset", super().reset, *a, **kw)

    def call_tool(self, tool_name, tool_args):
        t = time.time()
        try:
            return super().call_tool(tool_name, tool_args)
        finally:
            self.trace.add("tool", t, time.time() - t, name=tool_name)

    def finalize(self):
        return self._timed("finalize", super().finalize)

    def verify(self, *a, **kw):
        return self._timed("verify", super().verify, *a, **kw)


class BaselineClient(LLMClient):
    """Тот же клиент, но с лимитом вывода ближе к боевому sampling.yaml (там 16384 токенов;
    с растущим контекстом 32768 столько не помещается вместе с промптом, берём 8192)."""

    def __init__(self, *a, max_tokens=8192, system=None, thinking=True, **kw):
        super().__init__(*a, **kw)
        self._max_tokens = max_tokens
        self._thinking = thinking     # False = enable_thinking=false (в правилах thinking_budget: 0)
        self._system = system     # как `instruction` в agent.yaml боевой отправки (prompts/system.md)

    def chat(self, messages, tools, temperature=0.2, max_tokens=None, lora_name=None):
        limit = max_tokens or self._max_tokens
        if self._system:
            messages = [{"role": "system", "content": self._system}] + list(messages)
        for _ in range(2):
            try:
                return super().chat(messages, tools, temperature=temperature,
                                    max_tokens=limit, lora_name=lora_name, thinking=self._thinking)
            except BadRequestError as e:
                m = re.search(r"at least (\d+) input tokens", str(e))
                if not m:
                    raise
                room = MAX_MODEL_LEN - int(m.group(1)) - 64   # промпт вырос: ужимаем только вывод
                if room < MIN_OUTPUT_TOKENS:
                    raise ContextFull(str(e)) from e
                limit = min(limit, room)
        raise ContextFull("не удалось уместить запрос в контекст")


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
    trace = EpisodeTrace()
    env = TimedEnv(args.sandbox_url, args.token, task_ids=[task_id], trace=trace,
                   poll_sleep=getattr(args, "poll_sleep", 5.0))
    client = TracedClient(client, trace)
    row = {"instance_id": task_id, "passed": False, "error": None}
    with _LOCK:
        _ACTIVE[0] += 1
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
    except ContextFull:      # контекст переполнен: патч не сдан, это провал агента (не сбой)
        row.update(result={"reason": "context_full", "passed": False}, steps=0,
                   turns=env.ctx.turns_used, tool_calls=env.ctx.tool_calls_used, crashed=False)
        env.cleanup()
    except Exception as e:   # один упавший эпизод не должен ронять весь прогон
        row["error"] = f"{type(e).__name__}: {e}"
        try:
            env.cleanup()
        except Exception:
            pass
    row["seconds"] = round(time.time() - t0)
    row["timing"] = trace.summary()
    with _LOCK:
        _ACTIVE[0] -= 1
        with open(os.path.join(args.out, "results.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tm = row["timing"]
    print(f"{task_id}: passed={row['passed']} reason={row.get('result', {}).get('reason')} "
          f"calls={row.get('tool_calls')} {row['seconds']}s [модель {tm['llm_s']}s/{tm['turns']} ходов/"
          f"{tm['ctok']} ток, инструменты {tm['tool_s']}s, старт {tm['reset_s']}s, "
          f"проверка {tm['finish_s']}s] err={row['error']}", flush=True)


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
    print_timing(ok)


def print_timing(rows):
    """Куда уходит бюджет времени агента: доля модели и инструментов, скорость генерации."""
    tm = [r["timing"] for r in rows if r.get("timing") and r["timing"]["turns"]]
    if not tm:
        return
    agent = [t["llm_s"] + t["tool_s"] for t in tm]
    llm, tool = sum(t["llm_s"] for t in tm), sum(t["tool_s"] for t in tm)
    turns = sum(t["turns"] for t in tm)
    ctok = sum(t["ctok"] for t in tm)
    print(f"\nвремя агента (модель + инструменты), среднее на эпизод: {sum(agent) / len(tm):.0f} с")
    print(f"  модель {llm / (llm + tool):.0%}, инструменты {tool / (llm + tool):.0%}")
    print(f"  ход модели: {llm / turns:.1f} с в среднем, {ctok / turns:.0f} токенов вывода на ход, "
          f"{ctok / max(llm, 1e-9):.1f} ток/с")
    tools = [e["s"] for r in rows for e in (r.get("timing") or {}).get("events", []) if e["k"] == "tool"]
    if tools:
        tools.sort()
        print(f"  вызов инструмента: медиана {tools[len(tools) // 2]:.1f} с, "
              f"максимум {tools[-1]:.1f} с, всего вызовов {len(tools)}")
    by_par = {}
    for r in rows:
        for e in (r.get("timing") or {}).get("events", []):
            if e["k"] == "llm" and e.get("ctok") and e["s"] > 0:
                by_par.setdefault(e.get("par"), []).append(e["ctok"] / e["s"])
    for par in sorted(by_par, key=lambda x: (x is None, x)):
        v = sorted(by_par[par])
        print(f"  скорость при {par} параллельных эпизодах: медиана {v[len(v) // 2]:.1f} ток/с ({len(v)} ходов)")


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
    ap.add_argument("--thinking", choices=("on", "off"), default="on",
                    help="off = без размышлений модели (enable_thinking=false): ходы короче, их больше за бюджет")
    ap.add_argument("--clean-start", action="store_true",
                    help="перед запуском закрыть все сессии на сервере (брошенные прошлым прогоном); "
                         "нельзя, если на этот сервер одновременно работает другой прогон")
    ap.add_argument("--system-file", default=None,
                    help="системный промпт агента, например submission_v1_noadapter/prompts/system.md")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.clean_start:
        closed = MimoRemoteEnv(args.sandbox_url, args.token, task_ids=[])._http("POST", "/abort_all", {})
        print("закрыто брошенных сессий на сервере:", closed.get("result"), flush=True)
    ids = args.ids or [t["instance_id"] for t in MimoRemoteEnv(args.sandbox_url, args.token).tasks][:args.first]
    path = os.path.join(args.out, "results.jsonl")
    todo = [i for i in ids if i not in done_ids(path)]
    print(f"задач: {len(ids)}, осталось: {len(todo)}, потоков: {args.workers}", flush=True)

    system = open(args.system_file, encoding="utf-8").read() if args.system_file else None
    print("системный промпт:", args.system_file or "нет (голая модель)", flush=True)
    print("размышления модели:", args.thinking, flush=True)
    client = BaselineClient(base_url=args.llm_url, model=args.model, max_tokens=args.max_tokens,
                            system=system, thinking=(args.thinking == "on"))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda i: run_one(i, args, client), todo))
    if os.path.exists(path):
        summarize(path)


if __name__ == "__main__":
    main()
