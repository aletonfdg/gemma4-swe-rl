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
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from openai import BadRequestError

from llm.llm_client import LLMClient
from rl.rollout import run_episode
from sim.compaction import CompactingClient
from sim.mimo_env import MimoRemoteEnv
from sim.two_phase import SystemClient, run_episode_two_phase

_LOCK = threading.Lock()
MAX_MODEL_LEN = 32768
MIN_OUTPUT_TOKENS = 512
GRAPH_TOOLS = ("code_analyzer_agent", "get_code_neighbors", "search_similar_code", "get_code_subgraph")


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
                "compact_s": tot("compact"), "compactions": sum(e["k"] == "compact" for e in ev),
                "finish_s": tot("finalize", "verify"), "turns": len(llm),
                "ctok": sum(e.get("ctok") or 0 for e in llm),
                "max_ptok": max([e.get("ptok") or 0 for e in llm], default=0), "events": ev}


class TracedClient:
    """Клиент модели одного эпизода: меряет задержку и токены каждого хода."""

    def __init__(self, base, trace):
        self.base, self.trace = base, trace

    def chat(self, *a, kind="llm", **kw):      # kind="compact" -- запрос сводки при сжатии контекста
        t, out = time.time(), None
        try:
            out = self.base.chat(*a, **kw)
            return out
        finally:
            u = (out or {}).get("usage") or {}
            self.trace.add(kind, t, time.time() - t, ptok=u.get("prompt_tokens"),
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

    def __init__(self, *a, max_tokens=8192, system=None, thinking=True, hide_tools=(), thinking_budget=None, **kw):
        super().__init__(*a, **kw)
        self._thinking_budget = thinking_budget   # None = без лимита (как было); Kaggle: thinking_budget из sampling.yaml
        self._hide_tools = set(hide_tools)    # инструменты, которых в этой среде нет (граф, аналитик)
        self._max_tokens = max_tokens
        self._thinking = thinking     # False = enable_thinking=false (в правилах thinking_budget: 0)
        self._system = system     # как `instruction` в agent.yaml боевой отправки (prompts/system.md)

    def chat(self, messages, tools, temperature=0.2, max_tokens=None, lora_name=None):
        limit = max_tokens or self._max_tokens
        if self._hide_tools:
            tools = [t for t in tools if t["function"]["name"] not in self._hide_tools]
        if self._system:
            messages = [{"role": "system", "content": self._system}] + list(messages)
        for _ in range(2):
            try:
                return super().chat(messages, tools, temperature=temperature,
                                    max_tokens=limit, lora_name=lora_name, thinking=self._thinking,
                                    thinking_budget=self._thinking_budget)
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
                last[(row["instance_id"], row.get("rep", 0))] = row
    return list(last.values())


def done_ids(path):
    """Готовы пары (задача, повтор) без сбоя инфраструктуры: упавшие (туннель, песочница) перезапускаются."""
    return {(r["instance_id"], r.get("rep", 0)) for r in load_rows(path)
            if not r.get("error") and not r.get("crashed")}


def run_one(task_id, args, client, rep=0):
    t0 = time.time()
    trace = EpisodeTrace()
    env = TimedEnv(args.sandbox_url, args.token, task_ids=[task_id], trace=trace,
                   poll_sleep=getattr(args, "poll_sleep", 5.0),
                   budget_override={"tool_calls": getattr(args, "tool_calls", None),
                                    "time_minutes": getattr(args, "time_minutes", None),
                                    "turns": (args.tool_calls * 2 if getattr(args, "tool_calls", None) else None)})
    two_phase = getattr(args, "two_phase", False)
    base = client
    client = TracedClient(client, trace)
    if getattr(args, "compaction", "off") == "on":   # сжатие контекста, как в боевом харнессе
        client = CompactingClient(client)
    row = {"instance_id": task_id, "rep": rep, "passed": False, "error": None}
    with _LOCK:
        _ACTIVE[0] += 1
    try:
        if two_phase:   # следователь -> исправитель (sim/two_phase.py); base -- клиент без системного промпта
            def make_client(system, phase):
                model = args.investigator_client if phase == "investigator" else base
                c = SystemClient(TracedClient(model, trace), system)
                return CompactingClient(c) if getattr(args, "compaction", "off") == "on" else c
            steps, messages, passed, stats = run_episode_two_phase(
                env, make_client, args.investigator_system, args.fixer_system, temperature=args.temperature)
            row["two_phase"] = stats
        else:
            steps, messages, passed = run_episode(env, 0, client, None, None, temperature=args.temperature)
        row.update(passed=bool(passed), steps=len(steps), turns=env.ctx.turns_used,
                   tool_calls=env.ctx.tool_calls_used, crashed=bool(env.crashed),
                   reward=sum(s["reward"] for s in steps),
                   result={k: v for k, v in (env.last_result or {}).items() if k != "patch"},
                   patch_chars=len((env.last_result or {}).get("patch", "")))
        os.makedirs(os.path.join(args.out, "messages"), exist_ok=True)
        name = f"{task_id}.json" if rep == 0 else f"{task_id}__r{rep}.json"
        with open(os.path.join(args.out, "messages", name), "w", encoding="utf-8") as f:
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
    comp = f", сжатий {tm['compactions']}/{tm['compact_s']}s" if tm["compactions"] else ""
    print(f"{task_id}#{rep}: passed={row['passed']} reason={row.get('result', {}).get('reason')} "
          f"calls={row.get('tool_calls')} {row['seconds']}s [модель {tm['llm_s']}s/{tm['turns']} ходов/"
          f"{tm['ctok']} ток, инструменты {tm['tool_s']}s, старт {tm['reset_s']}s, "
          f"проверка {tm['finish_s']}s{comp}] err={row['error']}", flush=True)
    tp = row.get("two_phase")
    if tp:
        print(f"    следователь: {tp['inv_calls']} вызовов, диагноз {'есть' if tp['inv_stopped'] else 'НЕТ'} "
              f"({tp['diagnosis_chars']} симв.), запрещённых вызовов {tp['inv_forbidden']} | исправитель: "
              f"{tp['fix_calls']} вызовов, первая правка на {tp['fix_first_edit_call']}-м", flush=True)


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
    by_task = {}
    for r in ok:
        by_task.setdefault(r["instance_id"], []).append(bool(r["passed"]))
    if any(len(v) > 1 for v in by_task.values()):   # были повторы: показываем разброс по задачам
        solved = sorted(t for t, v in by_task.items() if any(v))
        mean = sum(sum(v) / len(v) for v in by_task.values()) / len(by_task)
        print(f"задач решено хотя бы раз: {len(solved)} из {len(by_task)}; "
              f"средняя доля решения по задачам: {mean:.1%}")
        print("(интервал выше считается по эпизодам; попытки одной задачи зависимы, честный интервал шире)")
        for t in solved:
            print(f"  {t[-6:]}: {sum(by_task[t])}/{len(by_task[t])}")
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
    ap.add_argument("--shuffle-seed", type=int, default=None,
                    help="перемешать список задач сервера этим зерном перед --first (одно и то же зерно = один порядок)")
    ap.add_argument("--workers", type=int, default=2, help="не больше --max-sessions сервера")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--llm-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--sandbox-url", default=os.environ.get("SWE_SANDBOX_URL"))
    ap.add_argument("--token", default=os.environ.get("SWE_SANDBOX_TOKEN"))
    ap.add_argument("--out", default="results/mimo_baseline")
    ap.add_argument("--no-analyzer", action="store_true",
                    help="скрыть code_analyzer_agent и графовые инструменты (в MiMo графов нет: "
                         "аналитик впустую съедает общий бюджет вызовов)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="сколько раз прогнать каждую задачу (агент недетерминирован: одна попытка = шум)")
    ap.add_argument("--compaction", choices=("on", "off"), default="on",
                    help="сжатие контекста как в боевом харнессе (порог 14336 токенов); off = как раньше")
    ap.add_argument("--thinking", choices=("on", "off"), default="on",
                    help="off = без размышлений модели (enable_thinking=false): ходы короче, их больше за бюджет")
    ap.add_argument("--thinking-budget", type=int, default=None,
                    help="лимит токенов рассуждения на ход, как thinking_budget в sampling.yaml (на Kaggle соблюдается); "
                         "серверу vLLM нужен llm.llm_client.THINKING_BUDGET_SERVER_ARGS")
    ap.add_argument("--tool-calls", type=int, default=None,
                    help="лимит вызовов инструментов на задачу (по умолчанию из eval_config.yaml; в боевом харнессе 100)")
    ap.add_argument("--time-minutes", type=float, default=None,
                    help="лимит времени агента на задачу в минутах (по умолчанию из eval_config.yaml; в боевом 60)")
    ap.add_argument("--clean-start", action="store_true",
                    help="перед запуском закрыть все сессии на сервере (брошенные прошлым прогоном); "
                         "нельзя, если на этот сервер одновременно работает другой прогон")
    ap.add_argument("--system-file", default=None,
                    help="системный промпт агента, например submission_v1_noadapter/prompts/system.md")
    ap.add_argument("--two-phase", action="store_true",
                    help="следователь (только чтение, ответ -- диагноз) -> исправитель (--system-file + диагноз); "
                         "см. sim/two_phase.py")
    ap.add_argument("--investigator-file", default="prompts/two_phase/investigator.md",
                    help="--two-phase: системный промпт следователя")
    ap.add_argument("--investigator-model", default=None,
                    help="--two-phase: модель/адаптер следователя (по умолчанию -- как --model). Адаптер правок "
                         "следователю не нужен: он не правит; базовая модель -- gemma-4-31b-it-qat-w4a16-ct")
    args = ap.parse_args()
    if args.two_phase:
        assert args.system_file, "--two-phase: нужен --system-file (промпт исправителя)"
        args.investigator_system = open(args.investigator_file, encoding="utf-8").read()
        args.fixer_system = open(args.system_file, encoding="utf-8").read()

    os.makedirs(args.out, exist_ok=True)
    if args.clean_start:
        closed = MimoRemoteEnv(args.sandbox_url, args.token, task_ids=[])._http("POST", "/abort_all", {})
        print("закрыто брошенных сессий на сервере:", closed.get("result"), flush=True)
    ids = args.ids
    if not ids:
        ids = [t["instance_id"] for t in MimoRemoteEnv(args.sandbox_url, args.token).tasks]
        if args.shuffle_seed is not None:   # список сервера отсортирован: без перемешивания --first берёт соседние репозитории
            random.Random(args.shuffle_seed).shuffle(ids)
        ids = ids[:args.first]
    path = os.path.join(args.out, "results.jsonl")
    done = done_ids(path)
    todo = [(i, r) for r in range(args.repeats) for i in ids if (i, r) not in done]   # сначала все задачи, потом повторы
    print(f"задач: {len(ids)}, повторов: {args.repeats}, эпизодов осталось: {len(todo)}, "
          f"потоков: {args.workers}", flush=True)
    print("сжатие контекста:", args.compaction, flush=True)
    print("бюджет: вызовов", args.tool_calls or "из eval_config.yaml",
          "| минут", args.time_minutes or "из eval_config.yaml", flush=True)

    system = open(args.system_file, encoding="utf-8").read() if args.system_file else None
    print("системный промпт:", args.system_file or "нет (голая модель)", flush=True)
    print("размышления модели:", args.thinking, "| лимит рассуждения:", args.thinking_budget or "нет", flush=True)
    hidden = GRAPH_TOOLS if args.no_analyzer else ()
    print("скрытые инструменты:", sorted(hidden) or "нет", flush=True)
    client = BaselineClient(base_url=args.llm_url, model=args.model, max_tokens=args.max_tokens,
                            system=None if args.two_phase else system,   # в двух фазах промпт задаёт sim/two_phase
                            thinking=(args.thinking == "on"), hide_tools=hidden,
                            thinking_budget=args.thinking_budget)
    if args.two_phase:
        inv_model = args.investigator_model or args.model
        args.investigator_client = BaselineClient(base_url=args.llm_url, model=inv_model, max_tokens=args.max_tokens,
                                                  thinking=(args.thinking == "on"), hide_tools=hidden,
                                                  thinking_budget=args.thinking_budget)
        print(f"двухфазный режим: следователь {inv_model} ({args.investigator_file}), исправитель {args.model}",
              flush=True)
    else:
        print("двухфазный режим: нет", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(lambda t: run_one(t[0], args, client, t[1]), todo))
    if os.path.exists(path):
        summarize(path)


if __name__ == "__main__":
    main()
