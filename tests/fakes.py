"""Крошечная фейковая модель, имитирующая интерфейс HF CausalLM (+ PEFT-обёртку)."""
import contextlib
import types

import torch
import torch.nn as nn


class FakeBackbone(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        self.emb = nn.Embedding(vocab, hidden)
        self.mix = nn.Linear(hidden, hidden)

    def forward(self, input_ids):
        # префиксная сумма: hidden в позиции t зависит от токенов <= t (как у каузальной модели)
        h = torch.cumsum(self.emb(input_ids), dim=1)
        return types.SimpleNamespace(last_hidden_state=torch.tanh(self.mix(h)))


class FakeLM(nn.Module):
    """Есть get_decoder()/get_output_embeddings() -> основной (chunked) путь."""

    def __init__(self, vocab=50, hidden=16, softcap=None):
        super().__init__()
        self.backbone = FakeBackbone(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = types.SimpleNamespace(final_logit_softcapping=softcap)

    def get_decoder(self):
        return self.backbone

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        logits = self.lm_head(self.backbone(input_ids).last_hidden_state)
        if self.config.final_logit_softcapping:
            c = self.config.final_logit_softcapping
            logits = torch.tanh(logits / c) * c
        return types.SimpleNamespace(logits=logits)

    @contextlib.contextmanager
    def disable_adapter(self):
        # "адаптер" = backbone.mix; при отключении ref-модель отличается от policy
        saved = self.backbone.mix.weight.data.clone()
        self.backbone.mix.weight.data.zero_()
        try:
            yield
        finally:
            self.backbone.mix.weight.data.copy_(saved)


class NoDecoderLM(FakeLM):
    """Без get_decoder() -> запасной путь через model(...).logits."""
    get_decoder = None
