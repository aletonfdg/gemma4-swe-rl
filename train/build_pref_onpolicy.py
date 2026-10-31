# train/build_pref_onpolicy.py -- пары (chosen, rejected) из своих же эпизодов: верная первая правка vs неверная
"""
Перепись времени (v3 / traj_think, ~620 эпизодов): первая правка исходника в решённых эпизодах на ~93 с, в
проваленных на ~114-129 с -- место модель находит почти одинаково. Разница в том, верна ли первая правка: в
решённых после неё ~1 минута, в проваленных 3-4 минуты новых правок до лимита времени. Задачи, которые модель
решает через раз, дают пары на одном и том же месте:
  * rejected -- первая правка исходника в проваленном эпизоде (tests_failed), как есть; если правок исходника не
    было -- первый repro-скрипт после чтения нужного файла (как в build_pref_fix);
  * chosen   -- на том же ходу первая удачная правка того же файла из решённого эпизода той же задачи, с её
    размышлением. Правка взята с исходного файла (первая в этот файл, инструмент ответил не error), поэтому
    old_string однозначно есть и в исходном файле проваленного эпизода. Правку, которую решённый эпизод потом
    переделал (её добавленные строки не дошли до итогового патча), не берём.
История проваленного эпизода должна содержать прочитанный кусок с old_string (окно вокруг него, как в
build_pref_fix). Обучать с reference (train.pref_lora --ref).

Записи как у build_pref_data: {"chosen", "rejected", "tools", "thinking": True, "kind": "onpolicy", "task"}.

Запуск:
  python -m train.build_pref_onpolicy --runs <папки с results.jsonl и messages/> \
      --system submission_v5/prompts/system.md --exclude eval/mimo50.json --out data/pref_onpolicy.jsonl
"""
import argparse
import copy
import json
import os
import random
from collections import Counter, defaultdict

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from train.build_pref_data import is_escaped, tool_status, trim_history
from train.build_pref_fix import THOUGHTS, WINDOW, is_detour, norm, window_read
from train.build_pref_short import single_call
from train.build_sft_data import ALLOWED_TOOLS, load_run

MAX_THOUGHT_CHARS = 1500     # chosen берёт размышление решённого эпизода; длинное режем по абзацу


def is_source(path):
    p = norm(path)
    parts = p.split("/")
    name = parts[-1]
    return (bool(p) and not p.startswith("/tmp") and name.endswith(".py") and not name.startswith(("test_", "repro"))
            and not name.endswith("_test.py") and name != "conftest.py"
            and not any(d in ("test", "tests", "testing", "tmp") for d in parts[:-1]))


def source_edits(messages):
    """-> [(k, path, old, new, ok)] для ходов с единственным вызовом edit_file в исходник."""
    status = tool_status(messages)
    out = []
    for k, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        call = single_call(m)
        if call is None or call[0] != "edit_file":
            continue
        args = call[1]
        path = norm(args.get("filepath") or args.get("path"))
        if is_source(path):
            ok = status.get(m["tool_calls"][0].get("id")) != "error"
            out.append((k, path, args.get("old_string") or "", args.get("new_string") or "", ok))
    return out


def added_lines(text):
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def patch_added(patch):
    return {ln[1:].strip() for ln in (patch or "").splitlines() if ln.startswith("+") and not ln.startswith("+++")}


def winning_edit(dialog):
    """Первая удачная правка решённого эпизода, которая дошла до итогового патча -> (msg, path, old, new) или None."""
    messages = dialog["messages"]
    seen = set()
    final = patch_added(dialog.get("patch"))
    for k, path, old, new, ok in source_edits(messages):
        first_in_file = path not in seen
        seen.add(path)
        if not (ok and first_in_file and old.strip()) or is_escaped(old) or is_escaped(new):
            continue
        new_lines = added_lines(new) - added_lines(old)
        if final and new_lines and not new_lines <= final:
            return None                      # первую правку потом переделали: она не та, что решила задачу
        return messages[k], path, old, new
    return None


def short_thought(text, limit=MAX_THOUGHT_CHARS):
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    return text[:cut if cut > limit // 3 else limit].rstrip()


def chosen_thought(rejected_msg, win_msg, path, rng):
    """Размышление решённого эпизода. В логах без размышлений (traj_v3) у rejected его нет -- тогда и у chosen нет,
    чтобы пара отличалась только вызовом; если нет только у решённого -- короткий шаблон."""
    if not (rejected_msg.get("thought") or "").strip():
        return ""
    own = (win_msg.get("thought") or win_msg.get("reasoning_content") or "").strip()
    return short_thought(own) if own else rng.choice(THOUGHTS).format(path=path)


def chosen_turn(rejected_msg, win_msg, path, old, new, rng):
    out = copy.deepcopy(rejected_msg)
    out.pop("reasoning_content", None)
    out["thought"] = chosen_thought(rejected_msg, win_msg, path, rng)
    out["tool_calls"] = [{"id": rejected_msg["tool_calls"][0].get("id", "call_fix"), "type": "function", "function": {
        "name": "edit_file",
        "arguments": json.dumps({"filepath": path, "old_string": old, "new_string": new}, ensure_ascii=False)}}]
    return out


def target_turn(messages, w_path):
    """Ход проваленного эпизода, который заменяем: первая правка исходника (любого файла), а если правок не было --
    первый обход (repro-скрипт) после чтения файла верной правки. -> (k, причина)."""
    edits = source_edits(messages)
    if edits:
        return edits[0][0], "wrong_edit"
    status = tool_status(messages)
    seen = False
    for k, m in enumerate(messages):
        call = single_call(m) if m.get("role") == "assistant" else None
        if call is None:
            continue
        name, args = call
        if name == "read_file" and norm(args.get("filepath") or args.get("path")) == w_path \
                and status.get(m["tool_calls"][0].get("id")) != "error":
            seen = True
        elif seen and is_detour(call, {w_path}):
            return k, "detour"
    return None, "no_edit"


def episode_pair(failed, win, system, tools, rng, keep_exchanges=2, max_keep=8, max_tool_chars=1500):
    """failed -- диалог проваленного эпизода, win -- winning_edit() решённого. -> (пара или None, причина)."""
    messages = failed["messages"]
    win_msg, w_path, w_old, w_new = win
    k, case = target_turn(messages, w_path)
    if k is None:
        return None, case
    call = single_call(messages[k])
    if call[0] == "edit_file" and (call[1].get("old_string"), call[1].get("new_string")) == (w_old, w_new):
        return None, "same_edit"
    history, at = window_read(messages[:k], w_old)
    if at is None:
        return None, "old_not_in_history"
    keep = sum(1 for x in history[at - 1:] if x.get("role") == "assistant")
    if keep > max_keep:
        return None, "read_too_far"
    prefix = [{"role": "system", "content": system}] + trim_history(
        history, max(keep_exchanges, keep), max(max_tool_chars, 2 * WINDOW + 40))
    rejected = copy.deepcopy(messages[k])
    rejected.pop("reasoning_content", None)
    return {"chosen": prefix + [chosen_turn(messages[k], win_msg, w_path, w_old, w_new, rng)],
            "rejected": prefix + [rejected], "tools": tools, "thinking": True, "kind": "onpolicy"}, case


def build(runs, system, exclude=(), max_per_task=4, seed=0):
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    by_task = defaultdict(lambda: ([], []))
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            reason = (row.get("result") or {}).get("reason")
            if row["instance_id"] in exclude or row.get("error") or row.get("crashed"):
                continue
            if row.get("passed"):
                by_task[row["instance_id"]][0].append(dialog)
            elif reason == "tests_failed":
                by_task[row["instance_id"]][1].append(dialog)
    rng = random.Random(seed)
    out, stats = [], Counter()
    for iid, (wins, fails) in sorted(by_task.items()):
        if not (wins and fails):
            continue
        stats["tasks_mixed"] += 1
        good = [w for w in map(winning_edit, wins) if w]
        if not good:
            stats["no_winning_edit"] += 1
            continue
        pairs = []
        for failed in fails:
            reasons = []
            for win in good:
                pair, why = episode_pair(failed, win, system, tools, rng)
                if pair:
                    pair["task"] = iid
                    pairs.append(pair)
                    stats["pair_" + why] += 1
                    break                    # одна пара на проваленный эпизод
                reasons.append(why)
            else:
                stats[reasons[0]] += 1
        rng.shuffle(pairs)
        out += pairs[:max_per_task]
    stats["tasks_with_pairs"] = len({p["task"] for p in out})
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True)
    ap.add_argument("--exclude", default=None, help="json с {'ids': [...]}: задачи, которые в пары не берём")
    ap.add_argument("--max-per-task", type=int, default=4)
    ap.add_argument("--out", default="data/pref_onpolicy.jsonl")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    exclude = set()
    if args.exclude:
        with open(args.exclude, encoding="utf-8") as f:
            exclude = set(json.load(f)["ids"])
    pairs, stats = build(args.runs, system, exclude, args.max_per_task)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"пар записано: {len(pairs)} -> {args.out}; {dict(stats)}")


if __name__ == "__main__":
    main()
