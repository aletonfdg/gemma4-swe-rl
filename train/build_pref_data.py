# train/build_pref_data.py -- пары (chosen, rejected) для обучения формату вызовов edit_file
"""
Модель в ~28% вызовов edit_file печатает в old_string/new_string литеральные символы "\\n" вместо
перевода строки; такой old_string не находится в файле (успешны 6 из 247 вызовов), а в run_command и
write_file того же дефекта нет. Награда проверяема: вызов применился или нет. Пары строим из траекторий:

  * real_fix  -- вызов модели с литеральным "\\n" (rejected) и тот же вызов с настоящими переводами
                 строк (chosen); контекст -- ровно тот, в котором модель ошиблась;
  * synthetic -- успешный вызов с переводами строк (chosen) и он же с "\\n" вместо них (rejected).

Записи: {"chosen": messages, "rejected": messages, "tools", "thinking": True, "kind", "task"}; оба списка
различаются только аргументами последнего вызова (рассуждение хода одинаковое, поэтому в разности
логарифмов правдоподобия оно сокращается). Системный промпт добавляем сами, как в build_sft_data.

Запуск:
  python -m train.build_pref_data --runs results/traj_think --system submission_v3/prompts/system.md \
      --out data/pref_edit.jsonl --exclude fastapi_14303 ...
v2 (--v2): все виды лишнего экранирования (\\n, \\", \\t, \\\\); chosen real_fix принимается, только если его
old_string дословно есть в тексте файла, который модель видела раньше в эпизоде (см. make_pair_v2):
  python -m train.build_pref_data --v2 --runs <логи> --system submission_v5/prompts/system.md \
      --exclude-file eval/mimo50.json --out data/pref_edit_v2.jsonl
"""
import argparse
import copy
import json
import os
import random
from collections import Counter

from sim.tool_schemas import ROOT_TOOL_SCHEMAS
from train.build_sft_data import ALLOWED_TOOLS, load_run

BACKSLASH_N = chr(92) + "n"
FIELDS = ("old_string", "new_string")
MAX_THOUGHT_CHARS = 6000


def escape_newlines(text):
    return text.replace("\n", BACKSLASH_N)


def is_escaped(text):
    """Поле, в котором все переводы строк заменены на литеральный \\n. Одиночный \\n встречается и в
    настоящем коде (print("a\\n")), поэтому требуем не меньше двух и ни одного настоящего перевода."""
    return text.count(BACKSLASH_N) >= 2 and "\n" not in text


def unescape_newlines(text):
    return text.replace(BACKSLASH_N, "\n")


def edit_args(message):
    """Аргументы единственного вызова edit_file в сообщении ассистента или None."""
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "edit_file":
        return None
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return None
    return args if isinstance(args, dict) else None


def with_args(message, args):
    out = copy.deepcopy(message)
    out["tool_calls"][0]["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
    return out


def trim_history(messages, keep_exchanges, max_tool_chars):
    """Условие задачи (messages[0]) и последние keep_exchanges обменов (ход ассистента + ответы инструментов)
    перед целью. Потолок обучения 8192 токена, а к первой правке история обычно длиннее; формат вызова от
    дальнего прошлого почти не зависит, нужен лишь код, который модель только что прочитала. Длинные ответы
    инструментов обрезаем, оставляя начало."""
    starts = [i for i, m in enumerate(messages) if i > 0 and m.get("role") == "assistant"]
    first_kept = starts[-keep_exchanges] if len(starts) > keep_exchanges else (starts[0] if starts else len(messages))
    out = [messages[0]] + [dict(m) for m in messages[first_kept:]]
    for m in out:
        if m.get("role") == "tool" and len(m.get("content") or "") > max_tool_chars:
            m["content"] = m["content"][:max_tool_chars] + " ...[truncated]"
    return out


def tool_status(messages):
    out = {}
    for m in messages:
        if m.get("role") == "tool":
            try:
                out[m.get("tool_call_id")] = json.loads(m.get("content") or "{}").get("status")
            except ValueError:
                out[m.get("tool_call_id")] = None
    return out


def make_pair(args, ok):
    """-> (kind, chosen_args, rejected_args) или None. Только литеральный \\n (данные pref_edit_v1)."""
    texts = {f: args.get(f) or "" for f in FIELDS}
    if any(is_escaped(t) for t in texts.values()):
        if ok:                                   # удачный вызов с литеральным \n -- не учим на нём
            return None
        fixed = dict(args, **{f: unescape_newlines(t) if is_escaped(t) else t for f, t in texts.items()})
        return ("real_fix", fixed, args) if fixed != args else None
    if ok and any("\n" in t for t in texts.values()):
        broken = dict(args, **{f: escape_newlines(t) for f, t in texts.items()})
        return ("synthetic", args, broken) if broken != args else None
    return None


# --- v2: все виды лишнего экранирования, исправление сверяется с текстом файла -------------------------
# В pref_edit_v1 исправлялся только \n, а \" оставался: 23 из 87 chosen сами не применялись (detail=\"x\").
# Модель экранирует old_string как строку JSON: \n, \", \t, \\. Обычный код тоже бывает с \n или \" внутри
# строк, поэтому исправление принимаем, только если исправленный old_string дословно есть в тексте файла,
# который модель видела раньше в этом эпизоде (read_file или вывод run_command), а исходный -- нет.
BS = chr(92)
UNESCAPES = {"newline": (BS + "n", "\n"), "tab": (BS + "t", "\t"), "quote": (BS + '"', '"'),
             "squote": (BS + "'", "'")}


def unescape(text, kinds):
    """Снять экранирование выбранных видов; "backslash" (\\\\ -> \\) обрабатывается так, чтобы \\\\n не стал \\+перевод."""
    out = text.replace(BS + BS, "\0") if "backslash" in kinds else text
    for k in kinds:
        if k in UNESCAPES:
            out = out.replace(*UNESCAPES[k])
    return out.replace("\0", BS) if "backslash" in kinds else out


def fix_variants(text):
    """Кандидаты исправления: по одному виду, newline+quote, затем все сразу -> [(kinds, fixed)]."""
    present = [k for k, (esc, _) in UNESCAPES.items() if esc in text]
    if BS + BS in text:
        present.append("backslash")
    variants = [(k,) for k in present]
    if "quote" in present and "newline" in present:
        variants.append(("newline", "quote"))
    variants.append(tuple(present))
    seen, out = set(), []
    for kinds in variants:
        fixed = unescape(text, kinds)
        if fixed != text and fixed not in seen:
            seen.add(fixed)
            out.append((kinds, fixed))
    return out


def seen_texts(messages):
    """Тексты файлов, которые модель видела: content у read_file, stdout у run_command."""
    out = []
    for m in messages:
        if m.get("role") != "tool":
            continue
        try:
            body = json.loads(m.get("content") or "{}")
        except ValueError:
            continue
        if isinstance(body, dict):
            out += [body[k] for k in ("content", "stdout") if isinstance(body.get(k), str)]
    return out


def escape_quotes(text):
    return text.replace('"', BS + '"')


def make_pair_v2(args, ok, seen):
    """-> (kind, defect, chosen_args, rejected_args) или None."""
    old, new = args.get("old_string") or "", args.get("new_string") or ""
    if not ok:
        if not old.strip() or any(old in t for t in seen):
            return None                          # провал не из-за экранирования
        for kinds, fixed_old in fix_variants(old):
            if any(fixed_old in t for t in seen):
                fixed = dict(args, old_string=fixed_old, new_string=unescape(new, kinds))
                return "real_fix", "+".join(kinds), fixed, args
        return None
    if '"' in old and BS + '"' not in old + new:
        broken = dict(args, old_string=escape_quotes(old), new_string=escape_quotes(new))
        return "synthetic", "quote", args, broken
    if "\n" in old + new and not any(is_escaped(t) for t in (old, new)):
        broken = dict(args, old_string=escape_newlines(old), new_string=escape_newlines(new))
        return "synthetic", "newline", args, broken
    return None


# Ловушка харнесса: write_file убирает ведущий "/" и пишет "/tmp/x.py" в /workspace/tmp/x.py, а run_command
# работает с настоящим /tmp. Модель запускает старый файл, видит ложный результат и теряет время (база: 70%
# эпизодов); файл к тому же попадает в патч. Правильно -- записать файл через run_command и heredoc.
def tmp_write_args(message):
    """Аргументы единственного write_file в /tmp или None."""
    calls = message.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "write_file":
        return None
    try:
        args = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError):
        return None
    if not isinstance(args, dict) or not str(args.get("filepath", "")).startswith("/tmp/"):
        return None
    return args if isinstance(args.get("content"), str) and args["content"].strip() else None


def heredoc_command(path, content):
    """run_command, создающий файл с точно таким содержимым; маркер в кавычках -- без подстановок shell."""
    tag = "EOF"
    while any(line.strip() == tag for line in content.splitlines()):
        tag += "_"
    body = content if content.endswith("\n") else content + "\n"
    return f"cat > {path} <<'{tag}'\n{body}{tag}"


def as_run_command(message, command):
    out = copy.deepcopy(message)
    fn = out["tool_calls"][0]["function"]
    fn["name"], fn["arguments"] = "run_command", json.dumps({"command": command}, ensure_ascii=False)
    return out


def episode_pairs(dialog, system, tools, max_per_episode, max_thought_chars=MAX_THOUGHT_CHARS,
                  keep_exchanges=3, max_tool_chars=3000, v2=False):
    messages = dialog["messages"]
    status = tool_status(messages)
    pairs = []
    for k, m in enumerate(messages):
        if m.get("role") != "assistant" or len(pairs) >= max_per_episode:
            continue
        thought = (m.get("thought") or "").strip()
        if not thought or len(thought) > max_thought_chars:
            continue
        tmp = tmp_write_args(m) if v2 else None
        if tmp is not None:
            prefix = [{"role": "system", "content": system}] + trim_history(messages[:k], keep_exchanges,
                                                                             max_tool_chars)
            good = as_run_command(m, heredoc_command(tmp["filepath"], tmp["content"]))
            pairs.append({"chosen": prefix + [good], "rejected": prefix + [copy.deepcopy(m)],
                          "tools": tools, "thinking": True, "kind": "real_fix", "defect": "tmp_write"})
            continue
        args = edit_args(m)
        if args is None:
            continue
        ok = status.get(m["tool_calls"][0]["id"]) == "ok"
        if v2:
            made = make_pair_v2(args, ok, seen_texts(messages[:k]))
            if made is None:
                continue
            kind, defect, good, bad = made
        else:
            made = make_pair(args, ok)
            if made is None:
                continue
            (kind, good, bad), defect = made, "newline"
        prefix = [{"role": "system", "content": system}] + trim_history(messages[:k], keep_exchanges, max_tool_chars)
        pairs.append({"chosen": prefix + [with_args(m, good)], "rejected": prefix + [with_args(m, bad)],
                      "tools": tools, "thinking": True, "kind": kind, "defect": defect})
    return pairs


def balance_synthetic(real, synthetic, per_real):
    """v2: синтетических пар каждого вида (quote / newline) -- per_real на каждую реальную с этим дефектом."""
    used = []
    for d in ("quote", "newline"):
        need = per_real * sum(d in p["defect"].split("+") for p in real)
        used += [p for p in synthetic if p["defect"] == d][:need]
    return used


def build(runs, system, exclude=(), max_per_episode=20, synthetic_per_real=1, max_thought_chars=MAX_THOUGHT_CHARS,
          keep_exchanges=3, max_tool_chars=3000, v2=False, max_tmp=150):
    """synthetic_per_real=0 -- брать все доступные синтетические пары; max_tmp -- потолок пар tmp_write (v2)."""
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    real, synthetic, tmp = [], [], []
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            if row["instance_id"] in exclude or row.get("error"):
                continue
            for pair in episode_pairs(dialog, system, tools, max_per_episode, max_thought_chars,
                                       keep_exchanges, max_tool_chars, v2):
                pair["task"] = row["instance_id"]
                if pair["defect"] == "tmp_write":
                    tmp.append(pair)
                else:
                    (real if pair["kind"] == "real_fix" else synthetic).append(pair)
    random.Random(0).shuffle(tmp)
    tmp_used = tmp[:max_tmp]
    if not synthetic_per_real:
        used = synthetic
    elif v2:
        used = balance_synthetic(real, synthetic, synthetic_per_real)
    else:
        used = synthetic[:len(real) * synthetic_per_real]
    stats = Counter({"real_fix": len(real), "synthetic_available": len(synthetic), "synthetic_used": len(used),
                     "tmp_write_available": len(tmp)})
    stats.update(f"{p['kind']}:{p['defect']}" for p in real + used + tmp_used)
    return real + used + tmp_used, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/")
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", default="data/pref_edit.jsonl")
    ap.add_argument("--exclude", nargs="*", default=[], help="задачи, которые не берём (оценочный набор)")
    ap.add_argument("--max-per-episode", type=int, default=20)
    ap.add_argument("--max-thought-chars", type=int, default=MAX_THOUGHT_CHARS)
    ap.add_argument("--keep-exchanges", type=int, default=3, help="сколько последних обменов оставлять в истории")
    ap.add_argument("--max-tool-chars", type=int, default=3000, help="длиннее ответы инструментов обрезаются")
    ap.add_argument("--synthetic-per-real", type=int, default=1,
                    help="синтетических пар на одну реальную (0 = все доступные)")
    ap.add_argument("--v2", action="store_true",
                    help="все виды лишнего экранирования (\\n, \\\", \\t, \\\\), исправление сверяется с текстом файла")
    ap.add_argument("--exclude-file", default=None, help="json с {'ids': [...]}, например eval/mimo50.json")
    ap.add_argument("--max-tmp", type=int, default=150, help="--v2: потолок пар write_file(/tmp) -> heredoc")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    exclude = set(args.exclude)
    if args.exclude_file:
        with open(args.exclude_file, encoding="utf-8") as f:
            exclude |= set(json.load(f)["ids"])
    pairs, stats = build(args.runs, system, exclude, args.max_per_episode, args.synthetic_per_real,
                         args.max_thought_chars, args.keep_exchanges, args.max_tool_chars, args.v2, args.max_tmp)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"пар записано: {len(pairs)} -> {args.out}; {dict(stats)}")
    print("задач в данных:", len({p["task"] for p in pairs}))


if __name__ == "__main__":
    main()
