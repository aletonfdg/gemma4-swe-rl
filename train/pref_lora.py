# train/pref_lora.py -- обучение на парах (chosen, rejected): формат вызовов edit_file
"""
Оптимизация предпочтений без reference-модели с якорем на SFT:
    loss = -log sigmoid(beta * (logp(chosen) - logp(rejected))) + alpha * NLL(chosen)
В парах из build_pref_data оба варианта различаются только аргументами последнего вызова, поэтому
рассуждение хода и вся история в разности логарифмов правдоподобия сокращаются.

Память: градиент потерь по паре = beta*sigmoid(-z) * (grad logp(chosen) - grad logp(rejected)),
z = beta*(logp(c) - logp(r)). Поэтому вес считаем без градиента (два прохода вперёд), а затем делаем два
отдельных прохода вперёд-назад: цепочки вычислений не держатся одновременно, и потолок длины тот же
(8192 токена), что у SFT. Dropout по умолчанию 0: проход без градиента должен совпадать с обучающим.

Запуск (Colab, A100 80 ГБ, vLLM выключен):
  python -m train.pref_lora --data data/pref_edit.jsonl --name pref_edit_v1 --decompress --epochs 2
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path


def pref_weight(logp_chosen, logp_rejected, beta):
    """-> (вес beta*sigmoid(-z), z) для градиента -log sigmoid(z); z = beta*(logp_c - logp_r)."""
    z = beta * (logp_chosen - logp_rejected)
    return beta / (1.0 + math.exp(min(z, 50.0))), z


def split_by_task(rows, eval_frac):
    """Детерминированно: задачи по sha1 идентификатора; доля eval_frac уходит в оценку."""
    tasks = sorted({r["task"] for r in rows}, key=lambda t: hashlib.sha1(t.encode()).hexdigest())
    n_eval = max(1, int(round(len(tasks) * eval_frac))) if len(tasks) > 1 else 0
    held = set(tasks[:n_eval])
    return [r for r in rows if r["task"] not in held], [r for r in rows if r["task"] in held]


def load_pairs(path, tokenizer, max_len):
    from train.manual_lora_qat import build_example
    rows, stats = [], Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            c = build_example(rec["chosen"], tokenizer, rec["tools"], max_len, thinking=True, target="last")
            r = build_example(rec["rejected"], tokenizer, rec["tools"], max_len, thinking=True, target="last")
            if c is None or r is None:
                stats["long"] += 1
                continue
            rows.append({"c": c, "r": r, "kind": rec.get("kind", "?"), "task": rec.get("task", "")})
            stats[rec.get("kind", "?")] += 1
    kinds = {k: v for k, v in stats.items() if k != "long"}
    print(f"   пар: {len(rows)}, отброшено слишком длинных: {stats['long']}, по видам: {kinds}", flush=True)
    return rows


def train(args):
    import torch
    from rl.logprobs import token_logprobs
    from train.manual_lora_qat import load_model, print_vram, save_lora

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.r, args.alpha_lora, args.dropout, args.decompress)
    rows = load_pairs(args.data, tokenizer, args.max_len)
    train_rows, eval_rows = split_by_task(rows, args.eval_frac)
    if len(train_rows) < 4:
        raise RuntimeError("слишком мало пар для обучения")
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

    def tensors(item):
        return (torch.tensor([item["input_ids"]], device=dev), torch.tensor([item["mask"]], device=dev))

    def logp_sum(item):
        ids, mask = tensors(item)
        lp = token_logprobs(model, ids, mask)
        return (lp * mask[:, 1:].float()).sum()

    def evaluate():
        model.eval()
        margins = []
        with torch.no_grad():
            for row in eval_rows:
                margins.append((logp_sum(row["c"]) - logp_sum(row["r"])).item())
        model.train()
        if not margins:
            return float("nan"), float("nan")
        return sum(m > 0 for m in margins) / len(margins), sum(margins) / len(margins)

    print(f"   train {len(train_rows)} пар, eval {len(eval_rows)}, шагов оптимизатора {total}", flush=True)
    acc, margin = evaluate()
    print(f"   eval до обучения: доля пар chosen>rejected {acc:.2f}, средняя разность logp {margin:.2f}", flush=True)
    model.train()
    step, micro, run_loss, run_z = 0, 0, 0.0, 0.0
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        for row in train_rows:
            with torch.no_grad():
                w, z = pref_weight(logp_sum(row["c"]).item(), logp_sum(row["r"]).item(), args.beta)
            n_tok = max(1, sum(row["c"]["mask"]))
            lp_c = logp_sum(row["c"])
            (-(w + args.sft_weight / n_tok) * lp_c / args.grad_accum).backward()
            lp_r = logp_sum(row["r"])
            (w * lp_r / args.grad_accum).backward()
            run_loss += -math.log(1.0 / (1.0 + math.exp(-max(min(z, 50.0), -50.0)))) / args.grad_accum
            run_z += z / args.grad_accum
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total} pref-loss {run_loss / args.log_every:.4f} "
                      f"z {run_z / args.log_every:.3f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
                run_loss = run_z = 0.0
                print_vram(f"шаг {step}")
    acc, margin = evaluate()
    print(f"   eval в конце: доля пар chosen>rejected {acc:.2f}, средняя разность logp {margin:.2f}", flush=True)
    save_lora(model, Path(args.out) / args.name, args.r, args.alpha_lora, args.dropout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--name", default="pref_edit_v1")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha-lora", type=float, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--sft-weight", type=float, default=0.2, help="вес NLL по chosen (якорь против деградации)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--log-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decompress", action="store_true")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
