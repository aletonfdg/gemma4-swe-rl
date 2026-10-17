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
    last = {}      # как в раннере: повторный запуск дописывает новую строку вместо упавшей, берём последнюю
    with open(os.path.join(run_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                last[(row["instance_id"], row.get("rep", 0))] = row
    for row in last.values():
        path = os.path.join(run_dir, "messages", dialog_name(row))
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                out.append((row, json.load(f)))
    return out


def reject_reason(row, dialog, allow_compacted=False):
    """None, если эпизод годится для обучения, иначе короткая причина отказа. allow_compacted: режим по
    ходам берёт короткие префиксы (до ~8k токенов), до которых сжатие контекста (~14k) не доходило."""
    if not row.get("passed"):
        return "not_passed"
    if not allow_compacted and (row.get("timing") or {}).get("compactions", 0) > 0:
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


def select_turns(messages, count):
    """Индексы ходов ассистента для обучения: первый (поиск места), первая правка, последний, остальные
    равномерно. Ходы без вызова инструмента (итоговый текст) считаются обычными."""
    idx = [k for k, m in enumerate(messages) if m.get("role") == "assistant"]
    if not idx or count <= 0:
        return []
    chosen = {idx[0], idx[-1]}
    for k in idx:
        if any(tc["function"]["name"] in ("edit_file", "write_file") for tc in messages[k].get("tool_calls") or []):
            chosen.add(k)
            break
    rest = [k for k in idx if k not in chosen]
    need = count - len(chosen)
    if need > 0 and rest:
        step = len(rest) / need
        chosen.update(rest[int(i * step)] for i in range(min(need, len(rest))))
    return sorted(chosen)


def to_turn_records(dialog, system, tools, count, max_thought_chars):
    """По записи на выбранный ход: история без рассуждений, цель -- рассуждение и вызов этого хода.
    Ходы без записанного рассуждения или со слишком длинным пропускаем: учим коротко думать."""
    messages = dialog["messages"]
    records = []
    for k in select_turns(messages, count):
        thought = (messages[k].get("thought") or "").strip()
        if not thought or len(thought) > max_thought_chars:
            continue
        records.append({"messages": [{"role": "system", "content": system}] + messages[:k + 1], "tools": tools,
                        "target": "last", "thinking": True})
    return records


def split_tasks(task_ids, holdout_frac):
    """Детерминированно: сортируем по sha1 идентификатора, первые доли -- отложенные."""
    ordered = sorted(set(task_ids), key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    n_hold = int(round(len(ordered) * holdout_frac))
    return set(ordered[n_hold:]), set(ordered[:n_hold])      # (обучающие, отложенные)


def build(runs, system, per_task=3, holdout_frac=0.3, turns=0, max_thought_chars=3000):
    """turns=0: по примеру на весь эпизод (рассуждений нет). turns>0: по примеру на каждый из `turns`
    выбранных ходов эпизода, с рассуждением модели в цели обучения."""
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    rejected = Counter()
    by_task = defaultdict(list)
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            reason = reject_reason(row, dialog, allow_compacted=bool(turns))
            if reason:
                if row.get("passed"):
                    rejected[reason] += 1
                continue
            by_task[row["instance_id"]].append((chars(dialog), dialog))
    train_tasks, held_tasks = split_tasks(by_task.keys(), holdout_frac)
    records = []
    for task in sorted(train_tasks):
        for _, dialog in sorted(by_task[task], key=lambda x: x[0])[:per_task]:
            if turns:
                records.extend(to_turn_records(dialog, system, tools, turns, max_thought_chars))
            else:
                records.append(to_record(dialog, system, tools))
    return records, sorted(train_tasks), sorted(held_tasks), rejected, {t: len(v) for t, v in by_task.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True, help="prompts/system.md отправки")
    ap.add_argument("--out", default="data/sft_trajectories.jsonl")
    ap.add_argument("--per-task", type=int, default=3)
    ap.add_argument("--holdout-frac", type=float, default=0.3)
    ap.add_argument("--turns", type=int, default=0,
                    help="0: пример на весь эпизод; N>0: пример на каждый из N ходов с рассуждением")
    ap.add_argument("--max-thought-chars", type=int, default=3000,
                    help="ходы с более длинным рассуждением пропускаются (учим коротко думать)")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    records, train_tasks, held_tasks, rejected, avail = build(args.runs, system, args.per_task,
                                                              args.holdout_frac, args.turns,
                                                              args.max_thought_chars)
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
