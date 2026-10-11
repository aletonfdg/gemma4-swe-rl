# llm/llm_client.py -- chat.completions с tools= и логпробами (реальный протокол харнесса)
from openai import OpenAI


class LLMClient:
    def __init__(self, base_url="http://127.0.0.1:8000/v1", api_key="EMPTY",
                 model="gemma-4-31b-it-qat-w4a16-ct"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(self, messages, tools, temperature=0.9, max_tokens=4096, lora_name: str | None = None):
        resp = self.client.chat.completions.create(
            model=lora_name or self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        choice = resp.choices[0]
        msg = choice.message
        tool_calls = msg.tool_calls or []
        token_logprobs = []
        if choice.logprobs and choice.logprobs.content:
            token_logprobs = [(tl.token, tl.logprob) for tl in choice.logprobs.content]
        usage = getattr(resp, "usage", None)
        return {
            "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                      "completion_tokens": getattr(usage, "completion_tokens", None)},
            "content": msg.content or "",
            "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments,
                             "id": tc.id} for tc in tool_calls],
            "finish_reason": choice.finish_reason,
            "token_logprobs": token_logprobs,
        }
