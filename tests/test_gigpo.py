import math

import numpy as np
import torch

from rl.gigpo_advantage import compute_gigpo_advantages, compute_step_returns
from rl.gigpo_loss import gigpo_loss
from rl.logprobs import token_logprobs
from tests.fakes import FakeLM


def S(reward, anchor):
    return {"reward": reward, "anchor": anchor}


# ---------- advantage ----------

def test_step_returns_discounting():
    r = compute_step_returns([S(1, "a"), S(0, "b"), S(10, "c")], gamma=0.5)
    assert r == [1 + 0.5 * (0 + 0.5 * 10), 0 + 0.5 * 10, 10]


def test_macro_advantage_is_zero_mean_and_ordered():
    groups = [[S(0, f"x{i}")] for i in range(4)]
    total, macro = compute_gigpo_advantages(groups, [1.0, 0.0, 0.0, 0.0], micro_weight=0.0)
    assert abs(sum(macro)) < 1e-4
    assert macro[0] > 0 > macro[1]
    assert math.isclose(total[0][0], macro[0], rel_tol=1e-6)


def test_equal_rewards_give_zero_macro_advantage():
    groups = [[S(0, "a")], [S(0, "a")]]
    _, macro = compute_gigpo_advantages(groups, [1.0, 1.0])
    assert np.allclose(macro, 0.0, atol=1e-6)


def test_micro_advantage_splits_trajectories_sharing_an_anchor():
    # одинаковый якорь "A", но у траектории 0 возврат выше -> положительный micro, у 1 -- отрицательный
    groups = [[S(5.0, "A")], [S(-5.0, "A")]]
    total, _ = compute_gigpo_advantages(groups, [1.0, 1.0], micro_weight=1.0)
    assert total[0][0] > 0 > total[1][0]


def test_unique_anchor_gets_no_micro_advantage():
    groups = [[S(5.0, "only-here")], [S(-5.0, "elsewhere")]]
    total, macro = compute_gigpo_advantages(groups, [1.0, 0.0], micro_weight=1.0)
    assert math.isclose(total[0][0], macro[0], rel_tol=1e-6)
    assert math.isclose(total[1][0], macro[1], rel_tol=1e-6)


# ---------- loss ----------

def _batch(adv_value, mask_from=4, T=10):
    torch.manual_seed(0)
    ids = torch.randint(0, 50, (1, T))
    mask = torch.zeros(1, T)
    mask[0, mask_from:] = 1
    adv = torch.full((1, T), adv_value)
    return ids, mask, adv


def _old_logprobs(model, ids, mask, shift=0.0):
    old = torch.zeros(ids.shape)
    with torch.no_grad():
        old[:, 1:] = token_logprobs(model, ids, mask) + shift
    return old


def test_loss_at_ratio_one_equals_negative_advantage():
    model = FakeLM()
    ids, mask, adv = _batch(2.0)
    old = _old_logprobs(model, ids, mask)
    loss, kl = gigpo_loss(model, ids, mask, old, adv, beta=0.0)   # beta=0 -> чистый policy loss
    assert math.isclose(loss.item(), -2.0, rel_tol=1e-5)
    assert kl >= 0.0


def test_positive_advantage_is_clipped_above_and_gives_zero_gradient():
    model = FakeLM()
    ids, mask, adv = _batch(1.0)
    old = _old_logprobs(model, ids, mask, shift=-1.0)             # ratio = e ~ 2.7 > 1 + eps_high
    loss, _ = gigpo_loss(model, ids, mask, old, adv, eps_low=0.2, eps_high=0.28, beta=0.0)
    assert math.isclose(loss.item(), -1.28, rel_tol=1e-5)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.count_nonzero(g) == 0 for g in grads)


def test_negative_advantage_is_not_clipped_when_ratio_is_high():
    # для A<0 и ratio>1+eps min(r*A, clip(r)*A) = r*A (более пессимистично) -> градиент есть
    model = FakeLM()
    ids, mask, adv = _batch(-1.0)
    old = _old_logprobs(model, ids, mask, shift=-1.0)
    loss, _ = gigpo_loss(model, ids, mask, old, adv, beta=0.0)
    assert loss.item() > 1.28
    loss.backward()
    assert any(torch.count_nonzero(p.grad) > 0 for p in model.parameters() if p.grad is not None)


def test_kl_penalty_positive_when_policy_differs_from_reference():
    model = FakeLM()
    ids, mask, adv = _batch(0.0)
    old = _old_logprobs(model, ids, mask)
    loss, kl = gigpo_loss(model, ids, mask, old, adv, beta=0.5)
    assert kl > 0.0
    assert loss.item() > 0.0


def test_unmasked_tokens_do_not_affect_loss():
    model = FakeLM()
    ids, mask, adv = _batch(1.0, mask_from=6)
    old = _old_logprobs(model, ids, mask)
    l1, _ = gigpo_loss(model, ids, mask, old, adv, beta=0.0)
    adv2 = adv.clone()
    adv2[:, :5] = 100.0                                           # мусор вне маски
    l2, _ = gigpo_loss(model, ids, mask, old, adv2, beta=0.0)
    assert math.isclose(l1.item(), l2.item(), rel_tol=1e-6)
