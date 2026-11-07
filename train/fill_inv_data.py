# train/fill_inv_data.py -- дописывает пары следователя (train/build_inv_data.py) ответами самой Gemma
"""
build_inv_data оставляет заглушки в последнем ходе. Здесь их заполняет та же модель, что потом учится
(базовая Gemma в vLLM), чтобы текст был в её собственной манере:

  CAUSE (stop)        -- модели дают условие, прочитанный код и правильное изменение; она пишет 1-2
                         предложения причины. FILE/LINES/FIX/CHECK уже точные (из патча), диагноз собирает
                         build_inv_data.diagnosis. Причина должна назвать идентификатор из изменения.
  преждевременный     -- (go) история следователя + просьба ответить сейчас; текст с FILE: становится
  диагноз                rejected (без этой просьбы).
  рассуждение         -- у ПОСЛЕДНЕГО хода обеих сторон, одним генератором (THOUGHT_INSTRUCTIONS): 1-2
                         предложения, ведущие ровно к этому действию (вызову или диагнозу). В inv_v1
                         рассуждение было только у сгенерированных диагнозов (у настоящих вызовов оно в
                         2/3 прогонов не сохранилось), и адаптер выучил "подумал -> отвечай".

Ответ, не прошедший проверку, -- ещё попытка (температура выше), потом пара выбрасывается.

  python -m train.fill_inv_data --data data/inv_pairs_v2.jsonl --out data/inv_pairs_v2_filled.jsonl --workers 8
"""
import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from train.build_inv_data import PLACEHOLDER, diagnosis

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
FORBIDDEN = re.compile(r"\b(patch|reference|gold|diff|ground truth|the solution|provided change|known change)\b",
                       re.I)
MAX_SEEN_CHARS = 6000
MAX_THOUGHT_CHARS = 400
MAX_PREMATURE_CHARS = 2500

CAUSE_INSTRUCTIONS = """You are helping to write training examples for a code-investigation agent.
The agent read a GitHub issue and some code of the repository. The correct change is known (shown below),
but the text must sound as if the agent found the cause by itself from the code it read.

Write a JSON object with one string field:
  "cause": 1-2 sentences: why the current code produces the reported behavior. Name the concrete
           function/variable/condition involved. Do not mention any patch, diff, reference or known solution.
Reply with the JSON object only."""

THOUGHT_INSTRUCTIONS = """You write the private reasoning of a code-investigation agent at one moment of its work.
You are given the issue, the latest tool outputs the agent has seen, and the NEXT ACTION the agent takes:
either a tool call, or its final diagnosis (which ends the investigation).

Write a JSON object with one string field:
  "thought": 1-2 short sentences in the first person that lead to exactly this action: what the agent
             knows at this point or still lacks, and why it takes this action now. Use only what the
             tool outputs show. Do not mention any patch, diff, reference or known solution.
Reply with the JSON object only."""

PREMATURE_NUDGE = ("Stop investigating now and answer with your diagnosis in the required format "
                   "(FILE / LINES / CAUSE / FIX / CHECK), using only what you have seen so far.")


def changed_identifiers(hunks):
    words = set()
    for h in hunks:
        for line in h["lines"]:
            if line[:1] in "+-":
                words.update(IDENT.findall(line[1:]))
    return words


def seen_code(messages, limit=MAX_SEEN_CHARS):
    """Последние ответы инструментов в истории (то, что модель только что прочитала)."""
    parts, total = [], 0
    for m in reversed(messages):
        if m.get("role") != "tool":
            continue
        parts.append((m.get("content") or "")[: limit - total])
        total += len(parts[-1])
        if total >= limit:
            break
    return "\n---\n".join(reversed(parts))


def parse_json_field(text, field):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    value = obj.get(field) if isinstance(obj, dict) else None
    return value.strip() if isinstance(value, str) else None


def cause_request(rec):
    gen = rec["gen"]
    body = (f"ISSUE:\n{gen['problem'][:4000]}\n\nCODE THE AGENT HAS READ:\n{seen_code(rec['chosen'][:-1])}\n\n"
            f"CORRECT CHANGE (known to you, not to be mentioned):\n{gen['fix']}")
    return [{"role": "system", "content": CAUSE_INSTRUCTIONS}, {"role": "user", "content": body}]


def parse_cause(text, hunks):
    cause = parse_json_field(text, "cause")
    if not cause or not 20 <= len(cause) <= 600 or FORBIDDEN.search(cause):
        return None
    if not changed_identifiers(hunks) & set(IDENT.findall(cause)):
        return None
    return cause


def action_text(turn):
    calls = turn.get("tool_calls") or []
    if calls:
        fn = calls[0]["function"]
        args = fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"])
        return f"tool call {fn['name']}({args[:1500]})"
    return "final diagnosis:\n" + (turn.get("content") or "")[:1500]


def thought_request(problem, history, turn):
    body = (f"ISSUE:\n{problem[:3000]}\n\nLATEST TOOL OUTPUTS:\n{seen_code(history, 4000)}\n\n"
            f"NEXT ACTION:\n{action_text(turn)}")
    return [{"role": "system", "content": THOUGHT_INSTRUCTIONS}, {"role": "user", "content": body}]


def parse_thought(text):
    thought = parse_json_field(text, "thought")
    if not thought or not 20 <= len(thought) <= MAX_THOUGHT_CHARS or FORBIDDEN.search(thought):
        return None
    if "FILE:" in thought:
        return None
    return thought


def parse_premature(out):
    """-> текст или None: нужен текст с FILE: и без вызова инструмента, не длиннее MAX_PREMATURE_CHARS."""
    if out.get("tool_calls"):
        return None
    text = (out.get("content") or "").strip()
    if "FILE:" not in text or len(text) > MAX_PREMATURE_CHARS:
        return None
    return text


def _retry(attempts, ask, parse):
    for i in range(attempts):
        got = parse(ask(0.3 if i == 0 else 0.8))
        if got:
            return got
    return None


def fill_one(rec, client, attempts=3):
    """-> заполненная запись или None."""
    problem = rec["gen"].get("problem") or ""
    history = rec["chosen"][:-1]
    out = dict(rec)
    if rec["kind"] == "stop":
        hunks = rec["gen"]["hunks"]
        cause = _retry(attempts, lambda t: client.chat(cause_request(rec), [], temperature=t, max_tokens=1024,
                                                      thinking=False).get("content"),
                       lambda text: parse_cause(text, hunks))
        if not cause:
            return None
        out["chosen"] = history + [{"role": "assistant", "content": diagnosis(
            hunks, cause, rec["gen"]["check"], rec["gen"].get("unseen", []))}]
    else:
        # без инструментов: с ними модель на просьбу "ответь сейчас" почти всегда делает ещё вызов
        # (41% go-пар не заполнились); история с вызовами при этом рендерится как обычно
        nudged = history + [{"role": "user", "content": PREMATURE_NUDGE}]
        text = _retry(attempts, lambda t: client.chat(nudged, [], temperature=t, max_tokens=2048,
                                                     thinking=False), parse_premature)
        if not text:
            return None
        out["rejected"] = history + [{"role": "assistant", "content": text}]
    # рассуждение последнего хода -- одним генератором у обеих сторон
    for side in ("chosen", "rejected"):
        turn = {k: v for k, v in out[side][-1].items() if k != "thought"}
        thought = _retry(attempts, lambda t: client.chat(thought_request(problem, history, turn), [], temperature=t,
                                                        max_tokens=512, thinking=False).get("content"),
                         parse_thought)
        if not thought:
            return None
        out[side] = out[side][:-1] + [dict(turn, thought=thought)]
    return out


def is_filled(rec):
    return PLACEHOLDER not in json.dumps(rec["chosen"][-1]) + json.dumps(rec["rejected"][-1])


def record_key(rec):
    """Одна и та же пара до и после заполнения: общий префикс chosen не меняется."""
    return hashlib.sha1(json.dumps([rec["task"], rec["kind"], rec["chosen"][:-1]], sort_keys=True).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inv_pairs_v2.jsonl")
    ap.add_argument("--out", default="data/inv_pairs_v2_filled.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="оставить уже заполненные пары из --out, заполнить остальные")
    args = ap.parse_args()
    from llm.llm_client import LLMClient
    client = LLMClient(base_url=args.base_url, model=args.model)
    with open(args.data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    done, failed = [], {"stop": 0, "go": 0}
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = [json.loads(line) for line in f if line.strip()]
        have = {record_key(r) for r in done}
        records = [r for r in records if record_key(r) not in have]
        print(f"уже заполнено {len(done)}, осталось {len(records)}", flush=True)

    def work(rec):
        try:
            return rec, fill_one(rec, client)
        except Exception as e:          # одна сбойная пара не должна ронять весь прогон
            print("ошибка:", type(e).__name__, str(e)[:200], flush=True)
            return rec, None

    with ThreadPoolExecutor(args.workers) as pool:
        for i, (rec, out) in enumerate(pool.map(work, records), 1):
            if out is None or not is_filled(out):
                failed[rec["kind"]] += 1
            else:
                out.pop("gen", None)
                done.append(out)
            if i % 50 == 0:
                print(f"   {i}/{len(records)}: готово {len(done)}, не удалось {failed}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = {k: sum(r["kind"] == k for r in done) for k in ("stop", "go")}
    print(f"записано {len(done)} -> {args.out}; по видам {kinds}; не удалось {failed}")


if __name__ == "__main__":
    main()
