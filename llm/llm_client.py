# llm/llm_client.py -- chat.completions с tools= и логпробами (реальный протокол харнесса)
from openai import OpenAI

# Без этого vLLM не знает, где у Gemma 4 рассуждение, и thinking_token_budget не действует
THINKING_BUDGET_SERVER_ARGS = ["--reasoning-config",
                               '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}']


class LLMClient:
    def __init__(self, base_url="http://127.0.0.1:8000/v1", api_key="EMPTY",
                 model="gemma-4-31b-it-qat-w4a16-ct"):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(self, messages, tools, temperature=0.9, max_tokens=4096, lora_name: str | None = None,
             thinking: bool = True, thinking_budget: int | None = None):
        # `thought` -- рассуждение прошлых ходов, которое эпизод хранит для обучения; модели оно не уходит
        messages = [{k: v for k, v in m.items() if k != "thought"} for m in messages]
        extra = {"chat_template_kwargs": {"enable_thinking": bool(thinking)}}
        if thinking and thinking_budget:
            # как thinking_budget из sampling.yaml на Kaggle: vLLM закрывает рассуждение после N токенов
            # (сервер нужен с --reasoning-config, см. THINKING_BUDGET_SERVER_ARGS)
            extra["thinking_token_budget"] = int(thinking_budget)
        kwargs = dict(
            model=lora_name or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            extra_body=extra,
        )
        if tools:   # пустой список (например, запрос на сводку) -- без tools и tool_choice, иначе сервер отвечает 400
            kwargs.update(tools=tools, tool_choice="auto")
        resp = self.client.chat.completions.create(**kwargs)
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
            # рассуждение, которое парсер vLLM отделил от ответа (поле называется по-разному в версиях)
            "reasoning": getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or "",
            "tool_calls": [{"name": tc.function.name, "arguments": tc.function.arguments,
                             "id": tc.id} for tc in tool_calls],
            "finish_reason": choice.finish_reason,
            "token_logprobs": token_logprobs,
        }
