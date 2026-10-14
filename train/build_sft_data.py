# train/build_sft_data.py -- траектории локальных прогонов -> jsonl для train.manual_lora_qat
"""
Берёт решённые эпизоды из папок результатов sim.run_mimo_baseline (results.jsonl + messages/) и
собирает примеры для SFT в формате `{"messages": [...], "tools": [...]}`.

Что учитываем:
  * в сохранённых диалогах нет системного сообщения (его добавляет клиент прогона) -- добавляем
    prompts/system.md отправки, чтобы обучение шло на том же контексте, что и бой;
  * `tools` -- ровно те инструменты, что видела модель (без графовых и аналитика);
  * эпизоды со сжатием контекста пропускаем: модель видела сводку вместо начала диалога, а в
    сохранённом файле лежит полная история, и обучение на ней шло бы на другом контексте;
  * бракуем битые вызовы (имя инструмента с обрывками разметки, невалидный JSON аргументов) и
    ходы с зацикливанием (огромный текст вместо вызова);
  * на задачу берём не больше --per-task самых коротких решений, чтобы лёгкие задачи не доминировали;
  * часть решённых задач откладываем целиком (--holdout-frac), чтобы проверять адаптер на задачах,
    которых он не видел.

Запуск:
  python -m train.build_sft_data --runs results/v3_prompt --system submission_v3/prompts/system.md \
      --out data/sft_trajectories.jsonl
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

from sim.tool_schemas import ROOT_TOOL_SCHEMAS

ALLOWED_TOOLS = ("run_command", "read_file", "edit_file", "write_file", "submit_patch", "get_status")
MAX_ASSISTANT_TEXT = 2000


def dialog_name(row):
    rep = row.get("rep", 0)
    return f"{row['instance_id']}.json" if rep == 0 else f"{row['instance_id']}__r{rep}.json"


def load_run(run_dir):
    """-> [(row, dialog)] для всех эпизодов папки, у которых есть сохранённый диалог."""
    out = []
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        path = os.path.join(run_dir, "messages", dialog_name(row))
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                out.append((row, json.load(f)))
    return out


def reject_reason(row, dialog):
    """None, если эпизод годится для обучения, иначе короткая причина отказа."""
    if not row.get("passed"):
        return "not_passed"
    if (row.get("timing") or {}).get("compactions", 0) > 0:
        return "compacted"
    messages = dialog.get("messages") or []
    called = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if len(m.get("content") or "") > MAX_ASSISTANT_TEXT:
            return "runaway_text"
        calls = m.get("tool_calls") or []
        if not calls:
            return "turn_without_call"
        for tc in calls:
            fn = tc.get("function") or {}
            if fn.get("name") not in ALLOWED_TOOLS:
                return "bad_tool_name"
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except ValueError:
                return "bad_json_args"
            if not isinstance(args, dict):
                return "bad_json_args"
            called.append(fn["name"])
    if "submit_patch" not in called:
        return "no_submit"
    return None


def chars(dialog):
    return sum(len(json.dumps(m, ensure_ascii=False)) for m in dialog["messages"])


def to_record(dialog, system, tools):
    return {"messages": [{"role": "system", "content": system}] + dialog["messages"], "tools": tools}


def split_tasks(task_ids, holdout_frac):
    """Детерминированно: сортируем по sha1 идентификатора, первые доли -- отложенные."""
    ordered = sorted(set(task_ids), key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    n_hold = int(round(len(ordered) * holdout_frac))
    return set(ordered[n_hold:]), set(ordered[:n_hold])      # (обучающие, отложенные)


def build(runs, system, per_task=3, holdout_frac=0.3):
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    rejected = Counter()
    by_task = defaultdict(list)
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            reason = reject_reason(row, dialog)
            if reason:
                if row.get("passed"):
                    rejected[reason] += 1
                continue
            by_task[row["instance_id"]].append((chars(dialog), dialog))
    train_tasks, held_tasks = split_tasks(by_task.keys(), holdout_frac)
    records = []
    for task in sorted(train_tasks):
        for _, dialog in sorted(by_task[task], key=lambda x: x[0])[:per_task]:
            records.append(to_record(dialog, system, tools))
    return records, sorted(train_tasks), sorted(held_tasks), rejected, {t: len(v) for t, v in by_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True, help="prompts/system.md отправки")
    ap.add_argument("--out", default="data/sft_trajectories.jsonl")
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    records, train_tasks, held_tasks, rejected, avail = build(args.runs, system, args.per_task,
                                                              args.holdout_frac)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta = {"train_tasks": train_tasks, "heldout_tasks": held_tasks}
    with open(args.out + ".tasks.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"решённых задач с годными траекториями: {len(avail)} "
          f"(обучение {len(train_tasks)}, отложено {len(held_tasks)})")
    print(f"записано примеров: {len(records)} -> {args.out}")
    print(f"отбраковано решённых эпизодов по причинам: {dict(rejected)}")


if __name__ == "__main__":
    main()
