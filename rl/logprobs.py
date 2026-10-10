# rl/logprobs.py -- экономные по памяти logprobs токенов (только torch, без transformers/peft)
"""
Проблема: у Gemma словарь ~262k токенов. Полные логиты на 32k контексте в fp32 это ~34 ГБ
на ОДНУ копию (а их две: policy и reference), плюс градиент. Поэтому:
  1. логиты считаются только в позициях, где loss_mask=1 (токены ассистента);
  2. по чанкам, с checkpoint на чанк: в памяти одновременно живёт один чанк [chunk, vocab].
Основной путь берёт hidden states из backbone и применяет lm_head по частям. Если модель не
отдаёт get_decoder()/get_output_embeddings(), запасной путь использует model(...).logits
(полный тензор логитов всё равно создаётся, но fp32-копия -- только по чанкам).
"""
import contextlib

import torch
from torch.utils.checkpoint import checkpoint

DEFAULT_CHUNK = 1024


def _chunk_logp(head, h, t, softcap):
    logits = head(h).float()
    if softcap:
        logits = torch.tanh(logits / softcap) * softcap
    return logits.gather(-1, t.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)


def chunked_token_logprobs(head, hidden, targets, chunk=DEFAULT_CHUNK, softcap=None):
    """hidden [N, H], targets [N] -> logprob выбранных токенов [N] (float32).
    head: callable [n, H] -> [n, V] (lm_head или identity, если hidden уже логиты)."""
    out = []
    use_ckpt = torch.is_grad_enabled() and hidden.requires_grad
    for i in range(0, hidden.size(0), chunk):
        h, t = hidden[i:i + chunk], targets[i:i + chunk]
        if use_ckpt:
            out.append(checkpoint(_chunk_logp, head, h, t, softcap, use_reentrant=False))
        else:
            out.append(_chunk_logp(head, h, t, softcap))
    if not out:
        return hidden.new_zeros(0, dtype=torch.float32)
    return torch.cat(out)


def _softcap(lm):
    cfg = getattr(lm, "config", None)
    for c in (cfg, getattr(cfg, "text_config", None)):
        val = getattr(c, "final_logit_softcapping", None) if c is not None else None
        if val:
            return val
    return None


def _decoder_and_head(model):
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    get_dec = getattr(lm, "get_decoder", None)
    get_head = getattr(lm, "get_output_embeddings", None)
    if get_dec is None or get_head is None:
        return lm, None, None
    backbone, head = get_dec(), get_head()
    if backbone is None or head is None:
        return lm, None, None
    return lm, backbone, head


def token_logprobs(model, input_ids, mask=None, chunk=DEFAULT_CHUNK):
    """logprob токена input_ids[:, t+1] при контексте до t. Результат [B, T-1] (float32).
    mask [B, T] (как loss_mask): вне mask[:, 1:]==1 возвращаются нули и логиты не считаются."""
    targets = input_ids[:, 1:]
    sel = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask[:, 1:].bool()

    lm, backbone, head = _decoder_and_head(model)
    if backbone is not None:
        hidden = backbone(input_ids=input_ids).last_hidden_state[:, :-1]
        head_fn, softcap = head, _softcap(lm)
    else:
        hidden = model(input_ids=input_ids).logits[:, :-1]
        head_fn, softcap = (lambda x: x), None   # logits модели уже с softcap

    lp = chunked_token_logprobs(head_fn, hidden[sel], targets[sel], chunk=chunk, softcap=softcap)
    out = torch.zeros(targets.shape, dtype=torch.float32, device=targets.device)
    out[sel] = lp
    return out


def ref_token_logprobs(model, input_ids, mask=None, chunk=DEFAULT_CHUNK):
    """Reference policy = та же модель с отключённым LoRA -- без второй копии весов."""
    with torch.no_grad(), model.disable_adapter():
        logp = token_logprobs(model, input_ids, mask, chunk).detach().clone()
    with contextlib.suppress(Exception):   # без CUDA (тесты на CPU) cleanup не нужен
        import memory_utils
        memory_utils.hard_cleanup()
    return logp
