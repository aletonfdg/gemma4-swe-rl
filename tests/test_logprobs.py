import torch
import torch.nn.functional as F

from rl.logprobs import chunked_token_logprobs, token_logprobs, ref_token_logprobs
from tests.fakes import FakeLM, NoDecoderLM


def reference_logprobs(logits, targets):
    return F.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def test_chunked_matches_full_softmax_and_gradients():
    torch.manual_seed(0)
    head = torch.nn.Linear(8, 30, bias=False)
    h = torch.randn(23, 8, requires_grad=True)
    t = torch.randint(0, 30, (23,))

    got = chunked_token_logprobs(head, h, t, chunk=5)      # 5 не делит 23 -> неполный хвост
    want = reference_logprobs(head(h), t)
    assert torch.allclose(got, want, atol=1e-5)

    g1, = torch.autograd.grad(got.sum(), h)
    h2 = h.detach().clone().requires_grad_(True)
    g2, = torch.autograd.grad(reference_logprobs(head(h2), t).sum(), h2)
    assert torch.allclose(g1, g2, atol=1e-5)


def test_chunked_softcap():
    head = torch.nn.Linear(8, 30, bias=False)
    h, t = torch.randn(10, 8), torch.randint(0, 30, (10,))
    got = chunked_token_logprobs(head, h, t, chunk=4, softcap=5.0)
    capped = torch.tanh(head(h) / 5.0) * 5.0
    assert torch.allclose(got, reference_logprobs(capped, t), atol=1e-5)


def test_empty_selection_returns_empty():
    head = torch.nn.Linear(4, 6)
    out = chunked_token_logprobs(head, torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))
    assert out.shape == (0,)


def test_token_logprobs_matches_full_logits_and_masks_positions():
    torch.manual_seed(1)
    model = FakeLM()
    ids = torch.randint(0, 50, (2, 12))
    mask = torch.zeros(2, 12)
    mask[0, 5:9] = 1
    mask[1, 3:12] = 1

    got = token_logprobs(model, ids, mask, chunk=4)
    full = reference_logprobs(model(ids).logits[:, :-1], ids[:, 1:])

    sel = mask[:, 1:].bool()
    assert got.shape == (2, 11)
    assert torch.allclose(got[sel], full[sel], atol=1e-5)
    assert torch.all(got[~sel] == 0)


def test_token_logprobs_without_mask_covers_everything():
    model = FakeLM(softcap=3.0)
    ids = torch.randint(0, 50, (1, 9))
    got = token_logprobs(model, ids)
    assert torch.allclose(got, reference_logprobs(model(ids).logits[:, :-1], ids[:, 1:]), atol=1e-5)


def test_fallback_path_without_decoder_matches():
    torch.manual_seed(2)
    model = NoDecoderLM()
    ids = torch.randint(0, 50, (2, 10))
    mask = torch.ones(2, 10)
    got = token_logprobs(model, ids, mask, chunk=3)
    assert torch.allclose(got, reference_logprobs(model(ids).logits[:, :-1], ids[:, 1:]), atol=1e-5)


def test_gradients_flow_through_selected_tokens():
    model = FakeLM()
    ids = torch.randint(0, 50, (1, 10))
    mask = torch.zeros(1, 10)
    mask[0, 6:] = 1
    token_logprobs(model, ids, mask).sum().backward()
    assert model.lm_head.weight.grad is not None
    assert model.lm_head.weight.grad.abs().sum() > 0


def test_ref_logprobs_use_disabled_adapter_and_restore_weights():
    torch.manual_seed(3)
    model = FakeLM()
    ids = torch.randint(0, 50, (1, 8))
    before = model.backbone.mix.weight.detach().clone()
    policy = token_logprobs(model, ids).detach()
    ref = ref_token_logprobs(model, ids)
    assert not ref.requires_grad
    assert not torch.allclose(policy, ref)                     # адаптер реально отключался
    assert torch.equal(model.backbone.mix.weight, before)      # и веса вернулись на место
