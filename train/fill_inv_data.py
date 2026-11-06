# train/fill_inv_data.py -- дописывает пары следователя (train/build_inv_data.py) ответами самой Gemma
"""
build_inv_data оставляет в последнем ходе chosen (stop) или rejected (go) заглушку. Здесь её заполняет
та же модель, что потом учится (базовая Gemma в vLLM), чтобы текст был в её собственной манере:

  stop -- модели дают условие задачи, код, который она уже прочитала, и эталонное изменение; она пишет
          короткое рассуждение от первого лица ("вижу причину, передаю диагноз") и CAUSE (1-2 предложения).
          FILE/LINES/FIX/CHECK уже точные (из патча); диагноз собирает build_inv_data.diagnosis.
          Проверка: JSON разобран, рассуждение не упоминает патч/эталон, причина называет идентификатор
          из изменения. Иначе -- ещё попытка, потом пара выбрасывается.
  go   -- преждевременный диагноз: та же история следователя + просьба ответить сейчас; ответ (рассуждение
          и текст с FILE:) становится rejected в исходной истории (без этой просьбы).

  python -m train.fill_inv_data --data data/inv_pairs.jsonl --out data/inv_pairs_filled.jsonl --workers 8
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from train.build_inv_data import PLACEHOLDER, diagnosis

IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
FORBIDDEN = re.compile(r"\b(patch|reference|gold|diff|ground truth|the solution|provided change)\b", re.I)
MAX_SEEN_CHARS = 6000
MAX_THOUGHT_CHARS = 1500

STOP_INSTRUCTIONS = """You are helping to write training examples for a code-investigation agent.
The agent read a GitHub issue and some code of the repository. The correct change is known (shown below),
but the agent must sound as if it found the cause by itself from the code it read.

Write a JSON object with two string fields:
  "thought": 1-3 short sentences in the first person, as the agent's private reasoning at the moment it
             stops searching: what in the code it just read causes the issue, and that it now has enough
             to hand over the diagnosis. Do not mention any patch, diff, reference or known solution.
  "cause":   1-2 sentences: why the current code produces the reported behavior. Name the concrete
             function/variable/condition involved.
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


def stop_request(rec):
    gen = rec["gen"]
    body = (f"ISSUE:\n{gen['problem'][:4000]}\n\nCODE THE AGENT HAS READ:\n{seen_code(rec['chosen'][:-1])}\n\n"
            f"CORRECT CHANGE (known to you, not to be mentioned):\n{gen['fix']}")
    return [{"role": "system", "content": STOP_INSTRUCTIONS}, {"role": "user", "content": body}]


def parse_stop_reply(text, hunks):
    """-> (thought, cause) или None, если ответ не годится."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except ValueError:
        return None
    thought, cause = str(obj.get("thought", "")).strip(), str(obj.get("cause", "")).strip()
    if not (20 <= len(thought) <= MAX_THOUGHT_CHARS and 20 <= len(cause) <= 600):
        return None
    if FORBIDDEN.search(thought) or FORBIDDEN.search(cause):
        return None
    if not (changed_identifiers(hunks) & set(IDENT.findall(thought + " " + cause))):
        return None
    return thought, cause


def apply_stop(rec, thought, cause):
    out = dict(rec)
    turn = {"role": "assistant", "content": diagnosis(rec["gen"]["hunks"], cause, rec["gen"]["check"]),
            "thought": thought}
    out["chosen"] = rec["chosen"][:-1] + [turn]
    return out


def parse_premature(out):
    """-> (thought, text) или None: нужен текст с FILE: и без вызова инструмента."""
    if out.get("tool_calls"):
        return None
    text = (out.get("content") or "").strip()
    if "FILE:" not in text:
        return None
    return (out.get("reasoning") or "").strip()[:MAX_THOUGHT_CHARS], text


def apply_premature(rec, thought, text):
    out = dict(rec)
    out["rejected"] = rec["rejected"][:-1] + [{"role": "assistant", "content": text, "thought": thought}]
    return out


def fill_one(rec, client, attempts=3):
    """-> заполненная запись или None."""
    for i in range(attempts):
        temp = 0.3 if i == 0 else 0.8
        if rec["kind"] == "stop":
            out = client.chat(stop_request(rec), [], temperature=temp, max_tokens=2048, thinking=False)
            got = parse_stop_reply(out.get("content"), rec["gen"]["hunks"])
            if got:
                return apply_stop(rec, *got)
        else:
            history = rec["rejected"][:-1] + [{"role": "user", "content": PREMATURE_NUDGE}]
            out = client.chat(history, rec["tools"], temperature=temp, max_tokens=4096, thinking=True)
            got = parse_premature(out)
            if got:
                return apply_premature(rec, *got)
    return None


def is_filled(rec):
    return PLACEHOLDER not in json.dumps(rec["chosen"][-1]) + json.dumps(rec["rejected"][-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/inv_pairs.jsonl")
    ap.add_argument("--out", default="data/inv_pairs_filled.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="gemma-4-31b-it-qat-w4a16-ct")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from llm.llm_client import LLMClient
    client = LLMClient(base_url=args.base_url, model=args.model)
    with open(args.data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    done, failed = [], {"stop": 0, "go": 0}

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
