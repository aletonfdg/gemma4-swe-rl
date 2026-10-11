# sim/compaction.py -- эмуляция сжатия контекста боевого харнесса.
"""
Боевой swegemma включает у Google ADK сжатие истории событий (HARNESS_README 7.2:
token_threshold=14336, event_retention_size=5): когда контекст вырастает, старые события заменяются
краткой сводкой, а последние остаются как есть. Без этого в нашем симуляторе промпт доходит до
лимита 32768 токенов (в MiMo-прогонах 3 из 15 эпизодов кончались `context_full`).

Точный алгоритм ADK нам неизвестен, это приближение:
  * порог оцениваем по реальному числу токенов промпта из последнего ответа сервера (плюс прирост);
  * к модели уходит [первое сообщение + сводка] + последние `keep` сообщений;
  * сводку пишет сама модель отдельным запросом (его время учитывается в трассе как `compact`);
  * сообщения role=tool никогда не остаются без своего вызова ассистента.
Работает на копии: список сообщений эпизода, который ведёт run_episode, не меняется.
"""
THRESHOLD_TOKENS = 14336
KEEP_EVENTS = 5
CHARS_PER_TOKEN = 3.0        # код токенизируется плотнее английского текста, берём с запасом
OVERHEAD_TOKENS = 3000       # системный промпт и описания инструментов, которых нет в списке сообщений
SUMMARY_CHARS = 1800

SUMMARY_PROMPT = """You are compacting the work log of an autonomous software engineering agent so it can keep working with a smaller context.

Write a concise progress summary (at most 200 words) covering:
- files and symbols already examined, and the key findings;
- edits already made (file and what changed) -- these are already applied on disk;
- tests or checks already run and their results;
- what still remains to be done.
Do not call tools. Reply with the summary text only.

## Previous summary
{previous}

## New events since then
{events}
"""


def _trim(text, cap):
    text = (text or "").replace("\r", "")
    return text if len(text) <= cap else text[:cap] + f"...[+{len(text) - cap} chars]"


def msg_chars(m):
    n = len(m.get("content") or "")
    for tc in m.get("tool_calls") or []:
        n += len(tc["function"]["arguments"]) + len(tc["function"]["name"])
    return n


def render_events(msgs, call_cap=300, out_cap=500):
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            if m.get("content"):
                out.append("ASSISTANT: " + _trim(m["content"], call_cap))
            for tc in m.get("tool_calls") or []:
                out.append(f"CALL {tc['function']['name']}({_trim(tc['function']['arguments'], call_cap)})")
        elif m["role"] == "tool":
            out.append("RESULT: " + _trim(m["content"], out_cap))
        else:
            out.append("USER: " + _trim(m["content"], call_cap))
    return "\n".join(out)


class CompactingClient:
    """Обёртка над клиентом модели эпизода. `inner.chat(messages, tools, **kw)` должен принимать kw
    `kind` (метка запроса для трассы) и возвращать словарь с ключом usage (как sim.run_mimo_baseline)."""

    def __init__(self, inner, threshold=THRESHOLD_TOKENS, keep=KEEP_EVENTS, summary_tokens=400):
        self.inner, self.threshold, self.keep = inner, threshold, keep
        self.summary_tokens = summary_tokens
        self.summary = ""
        self.cut = 1                  # messages[1:cut] свёрнуты в сводку
        self.compactions = 0
        self._last_ptok = None        # токенов в последнем реальном запросе
        self._last_chars = 0          # и сколько символов было в представлении тогда

    def _view(self, messages):
        first = dict(messages[0])
        if self.summary:
            first["content"] = (first["content"] + "\n\n## Progress so far "
                                "(older steps were compacted)\n" + self.summary)
        return [first, *messages[self.cut:]]

    def _estimate(self, view):
        chars = sum(msg_chars(m) for m in view)
        if self._last_ptok is not None:
            return self._last_ptok + max(0, chars - self._last_chars) / CHARS_PER_TOKEN
        return chars / CHARS_PER_TOKEN + OVERHEAD_TOKENS

    def _compact(self, messages, **kw):
        new_cut = len(messages) - self.keep
        while new_cut > self.cut and messages[new_cut]["role"] == "tool":
            new_cut -= 1              # не оставляем ответ инструмента без вызова ассистента
        if new_cut <= self.cut:
            return False
        dropped = messages[self.cut:new_cut]
        prompt = SUMMARY_PROMPT.format(previous=self.summary or "(none)", events=render_events(dropped))
        text = ""
        try:
            out = self.inner.chat([{"role": "user", "content": prompt}], [], kind="compact",
                                  max_tokens=self.summary_tokens, **kw)
            text = (out.get("content") or "").strip()
        except Exception:
            pass                       # сводку не получили -- ниже запасной вариант без модели
        if not text:
            text = render_events(dropped, call_cap=120, out_cap=120)
        self.summary = _trim(text, SUMMARY_CHARS)
        self.cut = new_cut
        self.compactions += 1
        self._last_ptok = None
        return True

    def chat(self, messages, tools, **kw):
        view = self._view(messages)
        if self._estimate(view) > self.threshold and len(messages) - self.cut > self.keep + 1:
            if self._compact(messages, **{k: v for k, v in kw.items() if k in ("temperature",)}):
                view = self._view(messages)
        out = self.inner.chat(view, tools, **kw)
        used = (out.get("usage") or {}).get("prompt_tokens")
        if used:
            self._last_ptok, self._last_chars = used, sum(msg_chars(m) for m in view)
        return out
