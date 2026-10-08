# rl/gigpo_loss.py
import torch
from model_setup import compute_logprobs, compute_ref_logprobs


def gigpo_loss(model, input_ids, loss_mask, old_logprobs, advantages,
               eps_low=0.2, eps_high=0.28, beta=0.02):
    new_logp = compute_logprobs(model, input_ids)
    ref_logp = compute_ref_logprobs(model, input_ids)

    mask = loss_mask[:, 1:]
    old_lp = old_logprobs[:, 1:]
    adv = advantages[:, 1:]

    ratio = torch.exp(new_logp - old_lp)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv
    policy_loss_tok = -torch.min(surr1, surr2)

    log_ratio_ref = ref_logp - new_logp
    kl_tok = torch.exp(log_ratio_ref) - log_ratio_ref - 1

    loss_tok = policy_loss_tok + beta * kl_tok
    denom = mask.sum().clamp(min=1)
    loss = (loss_tok * mask).sum() / denom
    mean_kl = (kl_tok * mask).sum() / denom
    return loss, mean_kl.item()
