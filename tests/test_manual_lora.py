import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file  # noqa: E402

from train.manual_lora_qat import (ManualLoRALinear, TemplateMismatch, build_example,  # noqa: E402
                                   inject_lora, lora_disabled, lora_modules, save_lora,
                                   truncate_before_first_write)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.ModuleDict({
            "q_proj": torch.nn.Linear(8, 8, bias=False), "down_proj": torch.nn.Linear(8, 8, bias=False)})])
        self.vision_tower = torch.nn.ModuleDict({"q_proj": torch.nn.Linear(8, 8, bias=False)})

    def forward(self, x):
        layer = self.layers[0]
        return layer["down_proj"](layer["q_proj"](x))


def tiny_model():
    m = Tiny().to(torch.bfloat16)
    m.requires_grad_(False)
    return inject_lora(m, r=4, alpha=8, dropout=0.0)


def test_lora_wraps_text_layers_only_and_starts_as_identity():
    torch.manual_seed(0)
    base = Tiny().to(torch.bfloat16)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    ref = base(x)
    model = inject_lora(base, r=4, alpha=8, dropout=0.0)
    assert sorted(n for n, _ in lora_modules(model)) == ["layers.0.down_proj", "layers.0.q_proj"]
    assert not isinstance(model.vision_tower["q_proj"], ManualLoRALinear)
    assert torch.equal(model(x), ref)                      # lora_B = 0 -> выход базовой модели


def test_gradient_reaches_lora_and_base_stays_frozen():
    model = tiny_model()
    model(torch.randn(3, 8, dtype=torch.bfloat16)).float().pow(2).sum().backward()
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(m.lora_B.weight.grad.abs().sum() > 0 for _, m in lora_modules(model))


def test_lora_disabled_restores_base_output():
    model = tiny_model()
    for _, m in lora_modules(model):
        torch.nn.init.normal_(m.lora_B.weight)
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    with lora_disabled(model):
        off = model(x)
    assert not torch.equal(model(x), off)
    base = Tiny().to(torch.bfloat16)
    base.load_state_dict({k.replace(".base_layer", ""): v for k, v in model.state_dict().items()
                          if "lora_" not in k})
    assert torch.equal(off, base(x))


def test_saved_adapter_has_peft_keys_and_no_tokenizer(tmp_path):
    model = tiny_model()
    save_lora(model, tmp_path / "main_lora", r=4, alpha=8, dropout=0.0)
    assert sorted(p.name for p in (tmp_path / "main_lora").iterdir()) == \
        ["adapter_config.json", "adapter_model.safetensors"]
    keys = set(load_file(str(tmp_path / "main_lora" / "adapter_model.safetensors")))
    assert "base_model.model.layers.0.q_proj.lora_A.weight" in keys and len(keys) == 4
    cfg = json.loads((tmp_path / "main_lora" / "adapter_config.json").read_text())
    assert cfg["r"] == 4 and cfg["target_modules"] == ["down_proj", "q_proj"]


class FakeTokenizer:
    """Посимвольный «токенизатор» с шаблоном в духе Gemma: позволяет проверить маску без модели."""

    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        out = "<bos>"
        for m in msgs:
            body = m.get("content") or ""
            for tc in m.get("tool_calls") or []:
                args = tc["function"]["arguments"]
                if not isinstance(args, dict):     # как настоящий шаблон Gemma 4
                    raise ValueError("arguments must be a JSON object (mapping), not a string")
                body += f"[call {tc['function']['name']} {json.dumps(args)}]"
            out += f"<{m['role']}>{body}<end>"
        return out + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


def conversation():
    return [
        {"role": "user", "content": "fix it"},
        {"role": "assistant", "content": "", "reasoning_content": "SECRET",
         "tool_calls": [{"id": "1", "type": "function",
                         "function": {"name": "read_file", "arguments": {"filepath": "a.py"}}}]},
        {"role": "tool", "tool_call_id": "1", "content": "code"},
        {"role": "assistant", "content": "done"},
    ]


def test_mask_covers_exactly_assistant_turns_without_reasoning():
    ex = build_example(conversation(), FakeTokenizer(), tools=None, max_length=10_000)
    text = "".join(chr(i) for i in ex["input_ids"])
    marked = "".join(ch for ch, m in zip(text, ex["mask"]) if m)
    assert "SECRET" not in text                                # рассуждения вырезаны
    assert marked == '[call read_file {"filepath": "a.py"}]<end>done<end>'
    assert "fix it" not in marked and "code" not in marked


def test_string_arguments_from_trajectory_files_become_mappings():
    msgs = conversation()
    msgs[1]["tool_calls"][0]["function"]["arguments"] = '{"filepath": "a.py"}'   # так лежит в jsonl (vLLM)
    ex = build_example(msgs, FakeTokenizer(), tools=None, max_length=10_000)
    text = "".join(chr(i) for i in ex["input_ids"])
    assert '[call read_file {"filepath": "a.py"}]' in text


def test_invalid_json_arguments_do_not_break_rendering():
    msgs = conversation()
    msgs[1]["tool_calls"][0]["function"]["arguments"] = "{oops"
    ex = build_example(msgs, FakeTokenizer(), tools=None, max_length=10_000)
    assert "[call read_file {}]" in "".join(chr(i) for i in ex["input_ids"])


class GemmaLikeTokenizer(FakeTokenizer):
    """Шаблон в духе Gemma 4: весь диалог агента -- один ход модели, ответы инструментов стоят внутри хода
    сразу после вызова, у вызова без ответа в конце стоит маркер начала ответа, а подсказка генерации
    после сообщения пользователя добавляет пустой канал мысли (его нет в рендере диалога)."""

    def apply_chat_template(self, msgs, tools=None, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False, **kw):
        out, in_model_turn = "<bos>" + ("<|think|>" if enable_thinking else ""), False
        for k, m in enumerate(msgs):
            if m["role"] == "tool":
                continue
            if m["role"] == "user":
                out += f"<|turn>user\n{m['content']}<turn|>\n"
                in_model_turn = False
                continue
            if not in_model_turn:
                out += "<|turn>model\n"
                in_model_turn = True
            if m.get("reasoning_content"):
                out += f"<|channel>thought\n{m['reasoning_content']}\n<channel|>"
            calls = m.get("tool_calls") or []
            for tc in calls:
                args = tc["function"]["arguments"]
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object (mapping), not a string")
                out += f"<|tool_call>call:{tc['function']['name']}{json.dumps(args)}<tool_call|>"
            if calls:
                followed = False
                for nxt in msgs[k + 1:]:
                    if nxt["role"] != "tool":
                        break
                    out += f"<|tool_response>response:{nxt['content']}<tool_response|>"
                    followed = True
                if not followed:
                    out += "<|tool_response>"
            else:
                out += (m.get("content") or "") + "<turn|>\n"
                in_model_turn = False
        if add_generation_prompt and msgs and msgs[-1]["role"] == "user":
            out += "<|turn>model\n" + ("" if enable_thinking else "<|channel>thought\n<channel|>")
        elif add_generation_prompt and msgs and msgs[-1]["role"] == "tool" and enable_thinking:
            out += "<|channel>thought\n"
        return out


def test_gemma_like_template_marks_calls_and_the_marker_that_starts_the_tool_response():
    ex = build_example(conversation(), GemmaLikeTokenizer(), tools=None, max_length=10_000)
    text = "".join(chr(i) for i in ex["input_ids"])
    marked = "".join(ch for ch, m in zip(text, ex["mask"]) if m)
    assert marked == '<|tool_call>call:read_file{"filepath": "a.py"}<tool_call|><|tool_response>done<turn|>\n'
    assert "response:code" in text and "fix it" not in marked           # ответ инструмента не обучается
    assert "<|turn>model\n" not in marked                                # заголовок хода пропущен


def test_last_turn_target_trains_thought_and_call_with_history_without_thoughts():
    msgs = conversation()
    msgs[1]["thought"], msgs[3]["thought"] = "old plan", "final check"
    ex = build_example(msgs, GemmaLikeTokenizer(), tools=None, max_length=10_000, thinking=True, target="last")
    text = "".join(chr(i) for i in ex["input_ids"])
    marked = "".join(ch for ch, m in zip(text, ex["mask"]) if m)
    assert marked == "final check\n<channel|>done<turn|>\n"      # после подсказки генерации `<|channel>thought\n`
    assert "old plan" not in text and text.startswith("<bos><|think|>")   # история без рассуждений


def test_last_turn_target_on_the_first_assistant_turn_includes_the_thought_channel():
    msgs = conversation()[:2]
    msgs[1]["thought"] = "read the file first"
    ex = build_example(msgs, GemmaLikeTokenizer(), tools=None, max_length=10_000, thinking=True, target="last")
    text = "".join(chr(i) for i in ex["input_ids"])
    marked = "".join(ch for ch, m in zip(text, ex["mask"]) if m)
    assert marked == ('<|channel>thought\nread the file first\n<channel|>'
                      '<|tool_call>call:read_file{"filepath": "a.py"}<tool_call|><|tool_response>')


def test_last_turn_target_rejects_examples_that_do_not_end_with_the_assistant():
    assert build_example(conversation()[:3], GemmaLikeTokenizer(), tools=None, max_length=10_000,
                         thinking=True, target="last") is None


def test_too_long_example_is_dropped_not_truncated():
    assert build_example(conversation(), FakeTokenizer(), tools=None, max_length=20) is None


def test_template_that_is_not_prefix_stable_is_reported():
    class Unstable(FakeTokenizer):
        def apply_chat_template(self, msgs, **kw):
            return super().apply_chat_template(msgs, **kw) + f"#{len(msgs)}"
    with pytest.raises(TemplateMismatch):
        build_example(conversation(), Unstable(), tools=None, max_length=10_000)


def test_tool_role_keeps_prefix_before_first_write():
    msgs = conversation()[:3] + [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2", "type": "function",
         "function": {"name": "edit_file", "arguments": "{}"}}]}]
    assert truncate_before_first_write(msgs) == msgs[:3]
