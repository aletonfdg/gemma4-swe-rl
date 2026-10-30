# train/build_pref_fix.py -- пары (chosen, rejected): править исходник сразу, а не писать repro-скрипты
"""
Перепись логов traj_think на 129 публичных задачах: в 74 из 146 эпизодов tests_failed модель не правила ни
одного файла эталонного патча, хотя нужный файл почти всегда читала; правки уходили в /tmp/repro*.py и тесты.
Вместе с no_patch почти половина эпизодов кончается без правки исходника.

Пара строится из неудачного эпизода (tests_failed / no_patch), где эталонный файл так и не правили:
  * rejected -- первый ход после чтения эталонного файла, который пишет/запускает скрипт (write_file не в
    эталонный файл или run_command с python /tmp/...), как есть;
  * chosen   -- на том же месте короткое рассуждение + edit_file по первому подходящему куску эталонного патча
    (old_string = контекст + удаляемые строки, new_string = контекст + добавленные). old_string проверяется по
    снимку репозитория: ровно одно вхождение в исходном файле.
Учим поведению "нашёл место -- правь", поэтому история должна содержать прочитанный кусок: ответ read_file
эталонного файла оставляем окном вокруг old_string. Обучать с reference (train.pref_lora --ref).

Записи как у build_pref_data: {"chosen", "rejected", "tools", "thinking": True, "kind": "fix_first", "task"}.

Запуск (локально: снимки только в data/comp_full/snapshots):
  python -m train.build_pref_fix --runs <traj_think> --tasks data/comp/tasks.jsonl \
      --snapshots data/comp_full/snapshots --system submission_v5/prompts/system.md --out data/pref_fix.jsonl
"""
import argparse
import copy
import json
import os
import random
import re
import tarfile
from collections import Counter

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from train.build_pref_data import tool_status, trim_history
from train.build_pref_short import single_call
from train.build_sft_data import ALLOWED_TOOLS, load_run

FAILED = ("tests_failed", "no_patch")
WINDOW = 1500           # символов ответа read_file вокруг old_string
THOUGHTS = (
    "I've located the problem in `{path}`. Fixing it directly in the source now.",
    "The root cause is in `{path}`; I'll edit it now and verify afterwards.",
    "I have enough context from `{path}` -- the fix belongs there. Applying the edit.",
    "The code I just read in `{path}` is where the bug is. Editing it now.",
)


def patch_hunks(patch):
    """Unified diff -> [(path, old, new)] по кускам; old/new -- контекст + удалённые/добавленные строки.
    Длину куска берём из заголовка @@ -a,b +c,d @@: пустая строка контекста иногда приходит без пробела."""
    out, path = [], None
    lines = patch.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if line.startswith("--- "):
            m = re.match(r"--- a/(\S+)", line)
            path = m.group(1) if m else None
            continue
        m = re.match(r"@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@", line)
        if not m:
            continue
        left_old, left_new = int(m.group(1) or 1), int(m.group(2) or 1)
        old, new = [], []
        while i < len(lines) and (left_old > 0 or left_new > 0):
            body = lines[i]
            i += 1
            if body.startswith("\\"):
                continue
            tag, text = (body[0], body[1:]) if body[:1] in (" ", "-", "+") else (" ", body)
            if tag != "+":
                old.append(text)
                left_old -= 1
            if tag != "-":
                new.append(text)
                left_new -= 1
        if path and old != new:
            out.append((path, "".join(old), "".join(new)))
    return out


def snapshot_files(path, wanted):
    """Содержимое файлов `wanted` из tgz-снимка репозитория (пути относительно корня)."""
    out = {}
    with tarfile.open(path) as tar:
        for member in tar:
            name = member.name[2:] if member.name.startswith("./") else member.name
            if name in wanted and member.isfile():
                out[name] = tar.extractfile(member).read().decode("utf-8", errors="replace")
    return out


def unique_hunks(hunks, files):
    """Куски, чей old_string встречается в исходном файле ровно один раз."""
    return [(path, old, new) for path, old, new in hunks
            if files.get(path) is not None and old.strip() and files[path].count(old) == 1]


def norm(path):
    return (path or "").replace("\\", "/").removeprefix("/workspace/").removeprefix("./")


def is_detour(call, gold):
    """Скрипт вместо правки: write_file не в эталонный файл или запуск python-скрипта из /tmp."""
    name, args = call
    if name == "write_file":
        return norm(args.get("filepath") or args.get("path")) not in gold
    if name == "run_command":
        return bool(re.search(r"python\d?\s+(-\S+\s+)*/tmp/\S+\.py", args.get("command") or ""))
    return False


def probe_line(old):
    """Строка old_string, по которой ищем её в истории (первая не короче 12 символов)."""
    return next((ln.strip() for ln in old.splitlines() if len(ln.strip()) >= 12), old.strip())


def window_read(messages, old):
    """Копия истории, где последний ответ инструмента с old_string сокращён до окна вокруг него;
    -> (копия, индекс этого ответа или None)."""
    out = copy.deepcopy(messages)
    probe = probe_line(old)
    for i in range(len(out) - 1, -1, -1):
        content = out[i].get("content") or ""
        if out[i].get("role") != "tool" or probe not in content:
            continue
        at = content.find(probe)
        if len(content) > 2 * WINDOW:
            lo, hi = max(0, at - WINDOW), at + WINDOW
            out[i]["content"] = (("...[truncated] " if lo else "") + content[lo:hi]
                                 + (" ...[truncated]" if hi < len(content) else ""))
        return out, i
    return out, None


def chosen_turn(message, path, old, new, rng):
    out = copy.deepcopy(message)
    out["thought"] = rng.choice(THOUGHTS).format(path=path)
    out.pop("reasoning_content", None)
    out["tool_calls"] = [{"id": message["tool_calls"][0].get("id", "call_fix"), "type": "function", "function": {
        "name": "edit_file",
        "arguments": json.dumps({"filepath": path, "old_string": old, "new_string": new}, ensure_ascii=False)}}]
    return out


def episode_pair(dialog, gold_patch, files, system, tools, rng, keep_exchanges=2, max_keep=6, max_tool_chars=1500):
    """-> (пара или None, причина)."""
    messages = dialog["messages"]
    hunks = patch_hunks(gold_patch)
    gold = {p for p, _, _ in hunks}
    candidates = unique_hunks(hunks, files)
    if not candidates:
        return None, "no_unique_hunk"
    status = tool_status(messages)
    seen_gold = False
    for k, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        call = single_call(m)
        if call is None:
            continue
        name, args = call
        target = norm(args.get("filepath") or args.get("path"))
        if name in ("edit_file", "write_file") and target in gold:
            return None, "edited_gold"
        if name == "read_file" and target in gold and status.get(m["tool_calls"][0].get("id")) != "error":
            seen_gold = True
            continue
        if not (seen_gold and is_detour(call, gold)):
            continue
        # кусок, который модель видела ближе всего к этому ходу
        found = [(at, (path, old, new), history) for path, old, new in candidates
                 for history, at in [window_read(messages[:k], old)] if at is not None]
        if not found:
            return None, "old_not_in_history"
        at, (path, old, new), history = max(found, key=lambda x: x[0])
        keep = sum(1 for x in history[at - 1:] if x.get("role") == "assistant")
        if keep > max_keep:
            return None, "read_too_far"
        # max_tool_chars с запасом: окно вокруг old_string уже сокращено до 2*WINDOW
        prefix = [{"role": "system", "content": system}] + trim_history(
            history, max(keep_exchanges, keep), max(max_tool_chars, 2 * WINDOW + 40))
        rejected = copy.deepcopy(m)
        rejected.pop("reasoning_content", None)
        return {"chosen": prefix + [chosen_turn(m, path, old, new, rng)], "rejected": prefix + [rejected],
                "tools": tools, "thinking": True, "kind": "fix_first"}, "pair"
    return None, "no_detour_after_read" if seen_gold else "gold_not_read"


def build(runs, tasks_path, snapshots, system, seed=0):
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    with open(tasks_path, encoding="utf-8") as f:
        tasks = {t["instance_id"]: t for t in map(json.loads, f)}
    rng = random.Random(seed)
    out, stats, cache = [], Counter(), {}
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            iid = row["instance_id"]
            reason = (row.get("result") or {}).get("reason")
            if iid not in tasks or row.get("error") or row.get("crashed") or reason not in FAILED:
                continue
            if iid not in cache:
                gold = {p for p, _, _ in patch_hunks(tasks[iid]["patch"])}
                snap = os.path.join(snapshots, iid + ".tgz")
                cache[iid] = snapshot_files(snap, gold) if os.path.exists(snap) else {}
            pair, why = episode_pair(dialog, tasks[iid]["patch"], cache[iid], system, tools, rng)
            stats[why] += 1
            if pair:
                pair["task"] = iid
                out.append(pair)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--tasks", default="data/comp/tasks.jsonl")
    ap.add_argument("--snapshots", default="data/comp_full/snapshots")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="data/pref_fix.jsonl")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    pairs, stats = build(args.runs, args.tasks, args.snapshots, system)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"пар записано: {len(pairs)} -> {args.out}; задач {len({p['task'] for p in pairs})}; {dict(stats)}")


if __name__ == "__main__":
    main()
