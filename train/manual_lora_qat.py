# train/manual_lora_qat.py -- LoRA вручную поверх gemma-4-31b-it-qat-w4a16-ct (compressed-tensors)
"""
Почему вручную: peft.get_peft_model() падает на CompressedLinear (нет base_layer.weight).

Память. CompressedLinear распаковывает int4 -> bf16 на каждом forward. Без checkpoint autograd
хранит распакованный вес КАЖДОГО слоя для градиента по входу: 410 x вес ~ 62 ГБ -> OOM.
Решение: checkpoint на уровне слоёв декодера (gradient_checkpointing_enable, non-reentrant):
между forward и backward хранятся только входы слоёв, веса распаковываются заново по одному слою.

Данные. Тексты строятся родным шаблоном Gemma 4 (tokenizer.apply_chat_template с tools), потому что
боевой vLLM разбирает ответы парсерами gemma4: другой формат вызовов модель выучит, а сервер не поймёт.
Маска ассистента -- через разницу префиксов шаблона и offsets токенизатора.

Запуск (Colab, из корня проекта):
  python -m train.manual_lora_qat diagnose
  python -m train.manual_lora_qat train --data data/sft_trajectories.jsonl --name main_lora
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # до первого обращения к CUDA
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import contextlib
import gc
import json
import math
import random
import re
from pathlib import Path

import torch
import torch.nn as nn

MODEL_NAME = "google/gemma-4-31B-it-qat-w4a16-ct"
VISION_SUBTREES = ("vision_tower", "embed_vision", "patch_embedder", "audio_tower", "embed_audio")
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
WRITE_TOOLS = {"edit_file", "write_file", "submit_patch"}


# ───────────────────────────── память ─────────────────────────────

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_vram(label):
    if not torch.cuda.is_available():
        return
    gb = 1024 ** 3
    total = torch.cuda.get_device_properties(0).total_memory / gb
    print(f"   VRAM {label}: allocated={torch.cuda.memory_allocated() / gb:.1f} "
          f"reserved={torch.cuda.memory_reserved() / gb:.1f} / {total:.1f} GB "
          f"(пик {torch.cuda.max_memory_allocated() / gb:.1f})", flush=True)


# ───────────────────────────── LoRA ─────────────────────────────

class ManualLoRALinear(nn.Module):
    """Замороженный базовый слой (в т.ч. CompressedLinear без .weight) + ветка B(A(x)) * scaling."""

    def __init__(self, base_layer, r, alpha, dropout):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad_(False)
        in_f, out_f = getattr(base_layer, "in_features", None), getattr(base_layer, "out_features", None)
        if in_f is None or out_f is None:
            raise ValueError(f"нет in_features/out_features у {type(base_layer).__name__}")
        device = next(base_layer.parameters()).device
        self.lora_A = nn.Linear(in_f, r, bias=False, dtype=torch.bfloat16, device=device)
        self.lora_B = nn.Linear(r, out_f, bias=False, dtype=torch.bfloat16, device=device)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)       # старт = базовая модель; поэтому на 1-м шаге grad(A) == 0
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.enabled = True                      # False = чистая базовая модель (reference для KL в RL)

    def forward(self, x):
        out = self.base_layer(x)
        if not self.enabled:
            return out
        lora = self.lora_B(self.lora_A(self.dropout(x).to(self.lora_A.weight.dtype)))
        return out + lora.to(out.dtype) * self.scaling


def lora_modules(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, ManualLoRALinear)]


def inject_lora(model, r, alpha, dropout, targets=PROJ_NAMES):
    skip = "|".join(re.escape(s) for s in VISION_SUBTREES)
    pattern = re.compile(rf"^(?!.*(?:{skip})).*\.(?:{'|'.join(targets)})$")
    modules = dict(model.named_modules())
    paths = [n for n in modules if pattern.fullmatch(n)]
    if not paths:
        raise RuntimeError("не найдено ни одного слоя под LoRA")
    for path in paths:
        parent_path, _, leaf = path.rpartition(".")
        setattr(modules[parent_path] if parent_path else model, leaf,
                ManualLoRALinear(modules[path], r, alpha, dropout))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   LoRA: {len(paths)} слоёв, обучаемых параметров {trainable:,} "
          f"(~{trainable * 2 / 1024 ** 2:.0f} МБ в bf16)", flush=True)
    return model


@contextlib.contextmanager
def lora_disabled(model):
    """Временно выключить все LoRA-ветки (reference-модель для KL без второй копии весов)."""
    mods = [m for _, m in lora_modules(model)]
    for m in mods:
        m.enabled = False
    try:
        yield model
    finally:
        for m in mods:
            m.enabled = True


def save_lora(model, out_dir, r, alpha, dropout):
    """Только веса LoRA в формате PEFT (ключи base_model.model.<путь>.lora_A.weight).
    Токенизатор сюда НЕ сохраняем: tokenizer.model не входит в разрешённые расширения отправки."""
    from safetensors.torch import save_file
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state, leaves = {}, set()
    for name, m in lora_modules(model):
        leaves.add(name.rsplit(".", 1)[-1])
        state[f"base_model.model.{name}.lora_A.weight"] = m.lora_A.weight.detach().to("cpu").contiguous()
        state[f"base_model.model.{name}.lora_B.weight"] = m.lora_B.weight.detach().to("cpu").contiguous()
    save_file(state, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({
        "base_model_name_or_path": MODEL_NAME, "peft_type": "LORA", "task_type": "CAUSAL_LM",
        "r": r, "lora_alpha": alpha, "lora_dropout": dropout, "target_modules": sorted(leaves),
        "bias": "none", "fan_in_fan_out": False, "modules_to_save": None,
    }, indent=2))
    size = (out / "adapter_model.safetensors").stat().st_size / 1024 ** 2
    print(f"   сохранено {out} ({len(state) // 2} слоёв, {size:.0f} МБ)", flush=True)
    return str(out)


# ───────────────────────────── модель ─────────────────────────────

def describe_first_projection(model):
    """Что за слой стоит на месте q_proj: нужно, чтобы понять формат весов, когда CompressedLinear нет."""
    for n, m in model.named_modules():
        if n.endswith("layers.0.self_attn.q_proj") and not any(s in n for s in VISION_SUBTREES):
            print(f"   {n}: {type(m).__name__}, forward={getattr(m.forward, '__qualname__', m.forward)}", flush=True)
            for pn, p in list(m.named_parameters()) + list(m.named_buffers()):
                print(f"      {pn}: {tuple(p.shape)} {p.dtype}", flush=True)
            print(f"      quantization_status={getattr(m, 'quantization_status', None)}", flush=True)
            return


def load_model(r=32, alpha=64, dropout=0.05, decompress=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Загрузка {MODEL_NAME} (decompress={decompress}) ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if decompress:   # веса сразу в bf16 (~62 ГБ): нужен GPU на 80 ГБ; зато обычный Linear с .weight
        from transformers import CompressedTensorsConfig
        kwargs["quantization_config"] = CompressedTensorsConfig(run_compressed=False)
    # run_compressed читается только при загрузке; менять quantization_config потом бесполезно.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map={"": 0}, torch_dtype="auto", **kwargs)
    model.config.use_cache = False
    n_compressed = sum(1 for _, m in model.named_modules() if type(m).__name__ == "CompressedLinear")
    print(f"   CompressedLinear: {n_compressed}", flush=True)
    describe_first_projection(model)
    if n_compressed == 0 and not decompress:
        print("   ВНИМАНИЕ: CompressedLinear нет; слои остались упакованными, но без .weight -- "
              "forward упадёт. Пробуйте --decompress (80 ГБ) или собственную распаковку.", flush=True)
    model.requires_grad_(False)
    print_vram("после загрузки")
    model = inject_lora(model, r, alpha, dropout)
    # checkpoint по слоям декодера: и активации, и распакованные веса не копятся по всем 60 слоям
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    clear_memory()
    print_vram("после LoRA")
    return model, tokenizer


# ───────────────────────────── данные ─────────────────────────────

class TemplateMismatch(ValueError):
    pass


MODEL_TURN = "<|turn>model\n"   # заголовок хода модели в шаблоне Gemma 4


def _render(tokenizer, msgs, tools, gen_prompt=False, enable_thinking=False):
    return tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=gen_prompt,
                                         enable_thinking=enable_thinking)


def clean_messages(messages):
    """Без рассуждений: шаблон на инференсе вырезает старые рассуждения из истории, и при обучении
    модель видела бы другой контекст, чем в бою. Аргументы вызовов -- словарём: шаблон Gemma 4 не берёт
    строку JSON (в файлах траекторий они лежат строкой, как отдаёт vLLM)."""
    def as_mapping(arguments):
        if isinstance(arguments, dict):
            return arguments
        try:
            parsed = json.loads(arguments or "{}")
        except (TypeError, ValueError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if k not in ("reasoning_content", "reasoning", "thinking")}
        if m.get("tool_calls"):
            m["tool_calls"] = [dict(tc, function=dict(tc["function"],
                                                      arguments=as_mapping(tc["function"].get("arguments"))))
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


def truncate_before_first_write(messages):
    """Для адаптера аналитика: начало успешной траектории (чтение/поиск) до первой правки.
    Фильтр «траектории без правок» брал бы в основном провалы: успешные почти всегда что-то правят."""
    for k, m in enumerate(messages):
        if m.get("role") == "assistant" and any(tc["function"]["name"] in WRITE_TOOLS
                                                for tc in m.get("tool_calls") or []):
            return messages[:k]
    return messages


def _span_start(tokenizer, msgs, k, tools, full, upto, thinking):
    """Где в полном тексте начинается ответ ассистента k. Подсказка генерации годится, если она префикс
    полного текста. Шаблон Gemma 4 этого не обеспечивает на первом ходе: подсказка добавляет пустой
    канал мысли, которого нет в рендере диалога; тогда берём конец рендера без подсказки и
    пропускаем заголовок хода модели."""
    before_g = _render(tokenizer, msgs[:k], tools, gen_prompt=True, enable_thinking=thinking)
    if upto.startswith(before_g):
        return len(before_g)
    before = _render(tokenizer, msgs[:k], tools, enable_thinking=thinking)
    if not upto.startswith(before):
        raise TemplateMismatch(f"шаблон не префиксно-устойчив на сообщении {k}: маску так не построить")
    return len(before) + (len(MODEL_TURN) if upto.startswith(MODEL_TURN, len(before)) else 0)


def build_example(messages, tokenizer, tools, max_length, thinking=False):
    """-> {"input_ids", "mask"}; mask=1 на токенах ответов ассистента (включая конец хода), иначе 0.
    None, если пример длиннее max_length (обрезать слева нельзя: потеряется условие задачи).
    В шаблоне Gemma 4 ответы инструментов вставлены внутрь хода модели сразу после её вызова, поэтому
    ответ ассистента k заканчивается там, где шаблон ставит маркер начала ответа инструмента."""
    msgs = clean_messages(messages)
    full = _render(tokenizer, msgs, tools, enable_thinking=thinking)
    spans = []
    for k, m in enumerate(msgs):
        if m.get("role") != "assistant" or k == 0:
            continue
        upto = _render(tokenizer, msgs[:k + 1], tools, enable_thinking=thinking)
        if not full.startswith(upto):
            raise TemplateMismatch(f"шаблон не префиксно-устойчив на сообщении {k}: маску так не построить")
        spans.append((_span_start(tokenizer, msgs, k, tools, full, upto, thinking), len(upto)))
    if not spans:
        return None
    enc = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    if len(ids) > max_length:
        return None
    mask, j = [], 0
    for start, _ in enc["offset_mapping"]:
        while j < len(spans) and start >= spans[j][1]:
            j += 1
        mask.append(1 if j < len(spans) and spans[j][0] <= start else 0)
    return {"input_ids": ids, "mask": mask}


def load_examples(path, tokenizer, tools, max_length, role="main"):
    from sim.tool_schemas import TOOL_SCHEMAS
    rows, stats = [], {"long": 0, "empty": 0}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            msgs = ex["messages"] if role == "main" else truncate_before_first_write(ex["messages"])
            item = build_example(msgs, tokenizer, ex.get("tools") or tools or TOOL_SCHEMAS, max_length)
            if item is None:
                stats["long" if msgs else "empty"] += 1
                continue
            rows.append(item)
    print(f"   данные: {len(rows)} примеров, отброшено {stats}", flush=True)
    return rows


# ───────────────────────────── обучение ─────────────────────────────

def masked_nll(model, input_ids, mask):
    """Средний NLL по токенам ассистента. Логиты только в позициях маски и по чанкам:
    полный тензор [T, 262k] в fp32 на длинных траекториях не помещается в память."""
    from rl.logprobs import token_logprobs
    lp = token_logprobs(model, input_ids, mask)
    m = mask[:, 1:].float()
    return -(lp * m).sum() / m.sum().clamp(min=1)


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer = load_model(args.r, args.alpha, args.dropout, args.decompress)
    rows = load_examples(args.data, tokenizer, None, args.max_len, role=args.role)
    if len(rows) < 2:
        raise RuntimeError("слишком мало примеров для обучения")
    random.shuffle(rows)
    n_eval = max(1, int(len(rows) * 0.05))
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]

    params = [p for p in model.parameters() if p.requires_grad]
    try:
        from bitsandbytes.optim import AdamW8bit
        opt = AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    except ImportError:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total = max(1, args.epochs * len(train_rows) // args.grad_accum)
    warm = max(1, int(total * 0.03))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 *
                                              (1 + math.cos(math.pi * min(1.0, s / total))))
    dev = next(model.parameters()).device

    def tensors(row):
        return (torch.tensor([row["input_ids"]], device=dev), torch.tensor([row["mask"]], device=dev))

    def evaluate():
        model.eval()
        with torch.no_grad():
            vals = [masked_nll(model, *tensors(r)).item() for r in eval_rows]
        model.train()
        return sum(vals) / len(vals)

    print(f"   train {len(train_rows)}, eval {len(eval_rows)}, шагов оптимизатора {total}", flush=True)
    print(f"   eval NLL до обучения: {evaluate():.4f}", flush=True)
    model.train()
    step, micro, run_loss, best = 0, 0, 0.0, float("inf")
    for epoch in range(args.epochs):
        random.shuffle(train_rows)
        for row in train_rows:
            loss = masked_nll(model, *tensors(row)) / args.grad_accum
            loss.backward()
            run_loss += loss.item()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch} step {step}/{total} loss {run_loss / args.log_every:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
                run_loss = 0.0
                print_vram(f"шаг {step}")
            if step % args.eval_every == 0:
                ev = evaluate()
                print(f"   eval NLL {ev:.4f}", flush=True)
                if ev < best:
                    best = ev
                    save_lora(model, Path(args.out) / args.name, args.r, args.alpha, args.dropout)
    ev = evaluate()
    print(f"   eval NLL в конце: {ev:.4f} (лучший {min(best, ev):.4f})", flush=True)
    if ev <= best:
        save_lora(model, Path(args.out) / args.name, args.r, args.alpha, args.dropout)


def diagnose(args):
    """Загрузка + один forward/backward на коротком примере: проверка памяти и того, что градиент доходит до LoRA."""
    model, tokenizer = load_model(args.r, args.alpha, args.dropout, args.decompress)
    model.train()
    text = "def add(a, b):\n    return a + b\n" * max(1, args.seq_len // 12)
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids
    ids = ids.to(next(model.parameters()).device)
    torch.cuda.reset_peak_memory_stats()
    loss = masked_nll(model, ids, torch.ones_like(ids))
    print_vram("после forward")
    loss.backward()
    print_vram("после backward")
    with_grad = sum(1 for _, m in lora_modules(model) if m.lora_B.weight.grad is not None
                    and m.lora_B.weight.grad.abs().sum() > 0)
    print(f"   слоёв с ненулевым градиентом lora_B: {with_grad} из {len(lora_modules(model))} "
          f"(grad lora_A на первом шаге равен нулю, так и должно быть)", flush=True)
    if with_grad == 0:
        raise RuntimeError("градиент не доходит до LoRA")
    save_lora(model, Path(args.out) / "diagnose_lora", args.r, args.alpha, args.dropout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("diagnose", "train"))
    ap.add_argument("--data")
    ap.add_argument("--name", default="main_lora")
    ap.add_argument("--role", choices=("main", "tool"), default="main",
                    help="tool = только начало траектории до первой правки (адаптер аналитика)")
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=64)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-grad-norm", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--seq-len", type=int, default=2048, help="длина примера в diagnose")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decompress", action="store_true",
                    help="загрузить веса сразу в bf16 (run_compressed=False): ~62 ГБ, нужен GPU на 80 ГБ")
    args = ap.parse_args()
    if args.mode == "train" and not args.data:
        ap.error("--data обязателен для train")
    (train if args.mode == "train" else diagnose)(args)


if __name__ == "__main__":
    main()
