from sim.compaction import CompactingClient, KEEP_EVENTS


class Inner:
    """Поддельный клиент модели: запоминает запросы, на сводку отвечает SUMMARY-<n>."""

    def __init__(self, ptok=1000, fail_compact=False):
        self.calls, self.ptok, self.fail_compact, self.n = [], ptok, fail_compact, 0

    def chat(self, messages, tools, kind="llm", **kw):
        self.calls.append((kind, messages, tools))
        if kind == "compact":
            if self.fail_compact:
                raise RuntimeError("модель недоступна")
            self.n += 1
            return {"content": f"SUMMARY-{self.n}", "usage": {}}
        return {"content": "", "tool_calls": [], "usage": {"prompt_tokens": self.ptok}}


def history(pairs, size=8000, start=0):
    """Первое сообщение + пары (вызов ассистента, ответ инструмента)."""
    msgs = [] if start else [{"role": "user", "content": "FIX THE BUG"}]
    for i in range(start, start + pairs):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": "read_file", "arguments": '{"filepath": "a%d.py"}' % i}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * size})
    return msgs


def assert_no_orphan_tool_messages(view):
    seen = set()
    for m in view:
        for tc in m.get("tool_calls") or []:
            seen.add(tc["id"])
        if m["role"] == "tool":
            assert m["tool_call_id"] in seen, "ответ инструмента без вызова ассистента"


def test_short_history_is_sent_untouched():
    inner = Inner()
    client = CompactingClient(inner)
    msgs = history(3, size=200)
    client.chat(msgs, ["tool"])
    assert [c[0] for c in inner.calls] == ["llm"] and inner.calls[0][1] == msgs
    assert client.compactions == 0


def test_long_history_is_compacted_into_summary_plus_recent_events():
    inner = Inner()
    client = CompactingClient(inner)
    msgs = history(10)                               # ~80 тыс. символов -> заведомо за порогом
    before = [dict(m) for m in msgs]
    client.chat(msgs, ["tool"])
    kinds = [c[0] for c in inner.calls]
    assert kinds == ["compact", "llm"] and inner.calls[0][2] == []   # сводка без инструментов
    view = inner.calls[1][1]
    assert "FIX THE BUG" in view[0]["content"] and "SUMMARY-1" in view[0]["content"]
    assert len(view) <= 1 + KEEP_EVENTS + 1 and view[-1] is msgs[-1]
    assert_no_orphan_tool_messages(view)
    assert msgs == before                            # исходный список эпизода не тронут
    assert client.compactions == 1


def test_boundary_on_a_tool_message_moves_back_to_its_assistant_call():
    inner = Inner()
    client = CompactingClient(inner, keep=5)         # 21 сообщение: граница (индекс 16) попадает на role=tool
    msgs = history(10)
    assert msgs[len(msgs) - 5]["role"] == "tool"     # условие теста: без отката ответ остался бы сиротой
    client.chat(msgs, ["tool"])
    view = inner.calls[1][1]
    assert view[1]["role"] == "assistant" and view[1]["tool_calls"][0]["id"] == msgs[-6]["tool_calls"][0]["id"]
    assert_no_orphan_tool_messages(view)


def test_second_compaction_carries_the_previous_summary_forward():
    inner = Inner(ptok=2000)
    client = CompactingClient(inner)
    msgs = history(10)
    client.chat(msgs, ["tool"])                      # первая сводка
    msgs += history(6, start=10)                     # агент продолжил работать
    inner.ptok = 30000                               # сервер сообщает: контекст снова большой
    client.chat(msgs, ["tool"])
    second_prompt = [c for c in inner.calls if c[0] == "compact"][1][1][0]["content"]
    assert "SUMMARY-1" in second_prompt              # старая сводка вошла в новую
    assert "SUMMARY-2" in inner.calls[-1][1][0]["content"]
    assert client.compactions == 2


def test_compaction_is_not_repeated_while_context_fits():
    inner = Inner(ptok=2000)
    client = CompactingClient(inner)
    msgs = history(10)
    client.chat(msgs, ["tool"])
    client.chat(msgs + history(1, size=100, start=10), ["tool"])   # чуть подросло, но влезает
    assert [c[0] for c in inner.calls].count("compact") == 1


def test_failed_summary_falls_back_to_a_digest_and_work_continues():
    inner = Inner(fail_compact=True)
    client = CompactingClient(inner)
    client.chat(history(10), ["tool"])
    assert client.compactions == 1 and "CALL read_file" in client.summary
    assert inner.calls[-1][0] == "llm"
