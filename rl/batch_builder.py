# rl/batch_builder.py
"""
TODO/FIXME (известное ограничение): при работе через OpenAI-совместимый /v1/chat/completions
API мы получаем ТЕКСТ ответа + per-token logprobs по СТРОКОВЫМ токенам ("token": str,
"logprob": float), но НЕ прямой доступ к token_ids модели в adapter-friendly форме. Чтобы
построить input_ids для HF backward (model_setup.compute_logprobs), сообщения здесь
ре-токенизируются целиком через tokenizer.apply_chat_template(..., tools=...), а
old_logprobs/advantage выравниваются приближённо на уровне ассистентских сегментов
(пропорционально равномерно, если точное количество токенов не совпадает 1:1).
ПЕРЕД боевым прогоном train_gigpo.py -- обязательно явно сверить на 5-10 примерах, что
(a) число реконструированных токенов ассистента совпадает с len(token_logprobs) из vLLM
ответа, и (b) если нет -- заменить на прямой доступ к vLLM `prompt_logprobs`/custom endpoint
вместо OpenAI-совместимого API, где это гарантируется точно.
"""
import torch


def _render_full_conversation(tokenizer, messages, tools):
    return tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=False
    )


def build_training_batch(tokenizer, all_messages, all_steps_with_adv, tools, device,
                          max_len_cap=32768):
    sequences = []
    for messages, steps in zip(all_messages, all_steps_with_adv):
        text = _render_full_conversation(tokenizer, messages, tools)
        ids = tokenizer(text, add_special_tokens=False).input_ids

        # Приближённое построение loss_mask/old_logprobs/advantages:
        # размечаем токены по сегментам ассистента через повторную посегментную токенизацию
        # (как раньше), помечая известные проблемы в module docstring выше.
        mask, old_lp, adv = [], [], []
        step_i = 0
        running_ids = []
        for m in messages:
            seg_text = tokenizer.apply_chat_template([m], tools=tools, tokenize=False)
            seg_ids = tokenizer(seg_text, add_special_tokens=False).input_ids
            running_ids.extend(seg_ids)
            if m.get("role") == "assistant" and step_i < len(steps):
                step = steps[step_i]
                lp_src = step.get("old_logprobs", [])
                if lp_src:
                    k = len(lp_src)
                    n = len(seg_ids)
                    lp_resampled = [lp_src[min(int(j * k / max(n, 1)), k - 1)] for j in range(n)]
                else:
                    lp_resampled = [0.0] * len(seg_ids)
                mask.extend([1] * len(seg_ids))
                old_lp.extend(lp_resampled)
                adv.extend([step.get("advantage", 0.0)] * len(seg_ids))
                step_i += 1
            else:
                mask.extend([0] * len(seg_ids))
                old_lp.extend([0.0] * len(seg_ids))
                adv.extend([0.0] * len(seg_ids))

        ids_final = running_ids[-max_len_cap:]
        mask, old_lp, adv = mask[-max_len_cap:], old_lp[-max_len_cap:], adv[-max_len_cap:]
        sequences.append((ids_final, mask, old_lp, adv))

    max_len = max(len(s[0]) for s in sequences)
    pad_id = tokenizer.pad_token_id or 0
    B = len(sequences)

    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((B, max_len), dtype=torch.float32)
    old_logprobs = torch.zeros((B, max_len), dtype=torch.float32)
    advantages = torch.zeros((B, max_len), dtype=torch.float32)

    for i, (ids, mask, lp, adv) in enumerate(sequences):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids)
        loss_mask[i, :L] = torch.tensor(mask[:L])
        old_logprobs[i, :L] = torch.tensor(lp[:L])
        advantages[i, :L] = torch.tensor(adv[:L])

    return (input_ids.to(device), loss_mask.to(device),
            old_logprobs.to(device), advantages.to(device))
