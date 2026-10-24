# train/pg_lora.py -- обучение с наградой по группам повторов одной задачи (GRPO-подобное, офлайн)
"""
Учим не «повторяй удачное», а «какие рассуждения и действия ведут к успеху»: у каждой задачи несколько
повторов (sim.run_mimo_baseline --repeats N), у каждого эпизода награда, преимущество хода = нормированная
разность награды его эпизода и средней награды группы. Ходы удачных попыток (рассуждение + вызов)
становятся вероятнее, ходы неудачных -- менее вероятными. Группы с одинаковой наградой сигнала не дают
и отбрасываются.

Награда эпизода (числа -- параметры, а не проверенные константы):
    1, если скрытые тесты прошли, иначе 0;
    минус --no-edit-penalty, если не было ни одного edit_file/write_file (против эпизодов no_patch);
    минус --length-penalty * доля ходов с рассуждением длиннее --long-thought-chars (учим думать коротко).

Потери на ходе (один проход по данным, поэтому без отношения вероятностей и обрезки, как в PPO):
    loss = -A * mean_t logp(токен цели) + kl_coef * mean_t KL_k3(policy || base)
База для KL -- та же модель с выключенной LoRA (lora_disabled), второй копии весов нет.
Цель хода -- рассуждение и вызов (как в build_sft_data --turns); история без рассуждений, как в бою.

Запуск (Colab, A100 80 ГБ, vLLM выключен):
  python -m train.pg_lora --runs <папки> --system prompts/system.md --exclude-from eval_ids.json \
      --name pg_v1 --decompress
  python -m train.pg_lora ... --dry-run    # только собрать группы и показать статистику (без GPU)
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from train.build_sft_data import ALLOWED_TOOLS, MAX_ASSISTANT_TEXT, load_run, to_turn_records

WRITE_TOOLS = ("edit_file", "write_file")


def broken_reason(dialog):
    """Структурный брак эпизода (обучать на нём нельзя ни в плюс, ни в минус) -> причина или None."""
    for m in dialog.get("messages") or []:
        if m.get("role") != "assistant":
            continue
        if len(m.get("content") or "") > MAX_ASSISTANT_TEXT:
            return "runaway_text"
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") not in ALLOWED_TOOLS:
                return "bad_tool_name"
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except (TypeError, ValueError):
                return "bad_json_args"
            if not isinstance(args, dict):
                return "bad_json_args"
    return None


def episode_reward(passed, dialog, no_edit_penalty=0.2, long_thought_chars=2400, length_penalty=0.2):
    turns = [m for m in dialog.get("messages") or [] if m.get("role") == "assistant"]
    reward = 1.0 if passed else 0.0
    edited = any(tc["function"]["name"] in WRITE_TOOLS for m in turns for tc in m.get("tool_calls") or [])
    if not edited:
        reward -= no_edit_penalty
    if turns:
        long_turns = sum(len(m.get("thought") or "") > long_thought_chars for m in turns)
        reward -= length_penalty * long_turns / len(turns)
    return reward


def group_advantages(rewards, eps=1e-6):
    """(r - mean) / std по группе; группа без разброса -> нули (сигнала нет)."""
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    mean = sum(rewards) / n
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / n)
    if std < eps:
        return [0.0] * n
    return [(r - mean) / std for r in rewards]


def build_records(runs, system, exclude=(), turns=6, max_thought_chars=6000, min_group=2, **reward_kw):
    """Папки прогонов -> (записи ходов с полем adv, статистика). Повторы одной задачи из разных папок
    объединяются в одну группу."""
    from sim.tool_schemas import ROOT_TOOL_SCHEMAS
    tools = [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in ALLOWED_TOOLS]
    exclude = set(exclude or ())
    stats = Counter()
    groups = defaultdict(list)
    for run_dir in runs:
        for row, dialog in load_run(run_dir):
            iid = row["instance_id"]
            if iid in exclude:
                stats["excluded"] += 1
                continue
            if row.get("error") or row.get("crashed"):
                stats["infra_error"] += 1
                continue
            reason = broken_reason(dialog)
            if reason:
                stats[reason] += 1
                continue
            groups[iid].append((bool(row.get("passed")), dialog))
    records = []
    for iid in sorted(groups):
        episodes = groups[iid]
        if len(episodes) < min_group:
            stats["group_too_small"] += 1
            continue
        rewards = [episode_reward(p, d, **reward_kw) for p, d in episodes]
        advs = group_advantages(rewards)
        if not any(advs):
            stats["group_no_spread"] += 1
            continue
        stats["groups_used"] += 1
        stats["groups_with_success"] += any(p for p, _ in episodes)
        for (passed, dialog), adv in zip(episodes, advs):
            for rec in to_turn_records(dialog, system, tools, turns, max_thought_chars):
                rec.update(adv=adv, task=iid, passed=passed)
                records.append(rec)
                stats["turns_pos" if adv > 0 else "turns_neg"] += 1
    return records, stats


def drop_oldest_turns(messages, n):
    """Убрать n самых старых ходов (ассистент + ответы инструментов) между условием задачи и целью.
    Условие (system/user в начале) и последний ход-цель остаются: длинные поздние ходы иначе не влезают
    в --max-len, а именно там видно, где модель застревает. None, если столько ходов убрать нельзя."""
    start = next((k + 1 for k, m in enumerate(messages) if m.get("role") == "user"), None)
    if start is None:
        return None
    body, target = messages[start:-1], messages[-1]
    blocks, cur = [], []
    for m in body:
        if m.get("role") == "assistant" and cur:
            blocks.append(cur)
            cur = []
        cur.append(m)
    if cur:
        blocks.append(cur)
    if n > len(blocks):
        return None
    return messages[:start] + [m for b in blocks[n:] for m in b] + [target]


def split_by_task(records, eval_frac):
    from train.pref_lora import split_by_task as split
    return split(records, eval_frac)


def train(args, records):
    import torch
    from rl.logprobs import token_logprobs
    from train.manual_lora_qat import build_example, load_model, lora_disabled, print_vram, save_lora

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.r, args.alpha_lora, args.dropout, args.decompress)
    rows, too_long, truncated = [], 0, 0
    for rec in records:
        ex, drop = None, 0
        while ex is None:
            msgs = drop_oldest_turns(rec["messages"], drop)
            if msgs is None:
                break
            ex = build_example(msgs, tokenizer, rec["tools"], args.max_len, thinking=True, target="last")
            drop += 1
        if ex is None:
            too_long += 1
            continue
        truncated += drop > 1
        rows.append({"ex": ex, "adv": rec["adv"], "task": rec["task"]})
    train_rows, eval_rows = split_by_task(rows, args.eval_frac)
    print(f"   ходов: обучение {len(train_rows)}, оценка {len(eval_rows)}, с обрезанной историей {truncated}, "
          f"отброшено длинных {too_long}", flush=True)
    # та же метрика на части обучающих ходов: отличает «не учится» от «учится, но не обобщает»
    train_probe = random.Random(args.seed).sample(train_rows, min(len(train_rows), max(len(eval_rows), 64)))
    if len(train_rows) < 4:
        raise RuntimeError("слишком мало ходов для обучения")
    params = [p for p in model.parameters() if p.requires_grad]
    try:
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    except ImportError:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = max(1, args.epochs * len(train_rows) // args.grad_accum)
    warm = max(1, int(total * 0.05))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1.0, s / total))))
    dev = next(model.parameters()).device

    def mean_logp(row, grad=True):
        ids = torch.tensor([row["ex"]["input_ids"]], device=dev)
        mask = torch.tensor([row["ex"]["mask"]], device=dev)
        m = mask[:, 1:].float()
        n = m.sum().clamp(min=1)
        with torch.set_grad_enabled(grad):
            lp = token_logprobs(model, ids, mask)
        return lp, m, n, ids, mask

    def evaluate(sample):
        # средний logp хода у удачных минус у неудачных попыток: должен расти
        model.eval()
        pos, neg = [], []
        with torch.no_grad():
            for row in sample:
                lp, m, n, _, _ = mean_logp(row, grad=False)
                (pos if row["adv"] > 0 else neg).append(((lp * m).sum() / n).item())
        model.train()
        if not pos or not neg:
            return float("nan")
        return sum(pos) / len(pos) - sum(neg) / len(neg)

    def report(label):
        print(f"   {label}: разность logp удачных и неудачных -- отложенные задачи {evaluate(eval_rows):.4f}, "
              f"обучающие {evaluate(train_probe):.4f}", flush=True)

    print(f"   шагов оптимизатора {total}", flush=True)
    report("eval до обучения")
    model.train()
    step, micro, run_pg, run_kl = 0, 0, 0.0, 0.0
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        for row in train_rows:
            lp, m, n, ids, mask = mean_logp(row)
            pg = -row["adv"] * (lp * m).sum() / n
            loss = pg
            if args.kl_coef > 0:
                with torch.no_grad(), lora_disabled(model):
                    ref = token_logprobs(model, ids, mask)
                d = ref - lp
                kl = ((torch.exp(d) - d - 1) * m).sum() / n
                loss = loss + args.kl_coef * kl
                run_kl += kl.item() / args.grad_accum
            (loss / args.grad_accum).backward()
            run_pg += pg.item() / args.grad_accum
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total} pg {run_pg / args.log_every:.4f} "
                      f"kl {run_kl / args.log_every:.5f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
                run_pg = run_kl = 0.0
                print_vram(f"шаг {step}")
        report(f"eval после эпохи {epoch}")
    save_lora(model, Path(args.out) / args.name, args.r, args.alpha_lora, args.dropout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="папки с results.jsonl и messages/ (с повторами)")
    ap.add_argument("--system", required=True, help="prompts/system.md той отправки, под которую учим")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--exclude-from", nargs="*", default=[], help="JSON: список id или словарь списков")
    ap.add_argument("--name", default="pg_v1")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--turns", type=int, default=6, help="ходов на эпизод")
    ap.add_argument("--max-thought-chars", type=int, default=6000,
                    help="ходы с более длинным рассуждением пропускаются (длинные неудачные полезны как минус)")
    ap.add_argument("--no-edit-penalty", type=float, default=0.2)
    ap.add_argument("--long-thought-chars", type=int, default=2400, help="~600 токенов")
    ap.add_argument("--length-penalty", type=float, default=0.2)
    ap.add_argument("--kl-coef", type=float, default=0.05)
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha-lora", type=float, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-5, help="2e-5 в первом раунде почти не сдвинуло модель")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decompress", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="только собрать группы и напечатать статистику")
    args = ap.parse_args()
    with open(args.system, encoding="utf-8") as f:
        system = f.read()
    exclude = set(args.exclude)
    for path in args.exclude_from:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        exclude.update(data if isinstance(data, list) else [i for v in data.values() for i in v])
    records, stats = build_records(args.runs, system, exclude, args.turns, args.max_thought_chars,
                                   no_edit_penalty=args.no_edit_penalty, long_thought_chars=args.long_thought_chars,
                                   length_penalty=args.length_penalty)
    print(f"групп в обучении: {stats['groups_used']} (с хотя бы одним успехом {stats['groups_with_success']}); "
          f"ходов: + {stats['turns_pos']}, - {stats['turns_neg']}", flush=True)
    print(f"отброшено: {dict((k, v) for k, v in stats.items() if not k.startswith(('groups_', 'turns_')))}", flush=True)
    if args.dry_run:
        return
    if not records:
        raise SystemExit("нет групп со смешанным исходом: нужны повторы задач (--repeats) с разной наградой")
    train(args, records)


if __name__ == "__main__":
    main()
