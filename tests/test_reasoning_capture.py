from types import SimpleNamespace

import pytest

pytest.importorskip("openai")
from llm.llm_client import LLMClient  # noqa: E402


def fake_openai(captured, **message_fields):
    message = SimpleNamespace(content="", tool_calls=None, **message_fields)
    choice = SimpleNamespace(message=message, finish_reason="stop", logprobs=None)
    response = SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3))

    def create(**kwargs):
        captured.update(kwargs)
        return response
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_thought_is_never_sent_back_to_the_model_and_reasoning_is_returned():
    sent = {}
    client = LLMClient(base_url="http://127.0.0.1:1/v1")
    client.client = fake_openai(sent, reasoning_content="because the test names the function")
    history = [{"role": "user", "content": "fix"},
               {"role": "assistant", "content": "", "thought": "older reasoning", "tool_calls": []}]
    out = client.chat(history, [])
    assert out["reasoning"] == "because the test names the function"
    assert all("thought" not in m for m in sent["messages"])
    assert history[1]["thought"] == "older reasoning"            # исходный список не испорчен


def test_reasoning_field_name_differs_between_vllm_versions():
    for field in ("reasoning_content", "reasoning"):
        client = LLMClient(base_url="http://127.0.0.1:1/v1")
        client.client = fake_openai({}, **{field: "r"})
        assert client.chat([{"role": "user", "content": "x"}], [])["reasoning"] == "r"


def test_no_reasoning_gives_empty_string():
    client = LLMClient(base_url="http://127.0.0.1:1/v1")
    client.client = fake_openai({})
    assert client.chat([{"role": "user", "content": "x"}], [])["reasoning"] == ""
