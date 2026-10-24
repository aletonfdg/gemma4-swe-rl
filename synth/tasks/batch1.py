# synth/tasks/batch1.py -- задачи 1-3: счета (деньги), cron-расписание, шина событий

INVOICING = {
    "name": "invoicing",
    "topic": "финансы: счета, налоги, округление денег",
    "kind": "bug",
    "issue": r"""
Invoice totals are off by a cent for half-cent prices

When a line item amount ends in exactly half a cent, the invoice rounds it down instead of up, so our invoices disagree with the accounting system by one cent.

```python
from invoicing import Invoice, LineItem

LineItem("Widget", 1, "2.675").subtotal                     # 2.67, expected Decimal('2.68')
LineItem("Cable", 3, "4.35", discount_pct=10).subtotal      # 11.74, expected Decimal('11.75')

inv = Invoice([LineItem("Widget", 1, "2.675"),
               LineItem("Cable", 3, "4.35", discount_pct=10)], region="CA")
inv.total                                                   # Decimal('15.45'), expected Decimal('15.48')
```

Amounts must be rounded half-up to the cent (that is what `round_money` already does). Line item subtotals should use exact decimal arithmetic and return a `Decimal`.
""",
    "files": {
        "README.md": r"""# invoicing

Small helpers to build customer invoices: line items with discounts, regional sales tax and
money formatting. All amounts are `decimal.Decimal` rounded half-up to the cent.
""",
        "pyproject.toml": r"""[project]
name = "invoicing"
version = "0.4.1"
requires-python = ">=3.9"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
""",
        "invoicing/__init__.py": r"""from .invoice import Invoice, LineItem
from .money import format_money, round_money
from .tax import UnknownRegion, tax_for

__all__ = ["Invoice", "LineItem", "UnknownRegion", "format_money", "round_money", "tax_for"]
""",
        "invoicing/money.py": r"""from decimal import ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")


def to_decimal(value):
    # floats go through repr() so that 0.1 becomes Decimal('0.1'), not its binary expansion
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(repr(value))
    return Decimal(str(value))


def round_money(value):
    return to_decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def format_money(value, currency="USD"):
    amount = round_money(value)
    sign = "-" if amount < 0 else ""
    whole, _, frac = f"{abs(amount):.2f}".partition(".")
    groups = []
    while len(whole) > 3:
        groups.insert(0, whole[-3:])
        whole = whole[:-3]
    groups.insert(0, whole)
    return f"{sign}{','.join(groups)}.{frac} {currency}"
""",
        "invoicing/tax.py": r"""from decimal import Decimal

from .money import round_money, to_decimal

RATES = {
    "CA": Decimal("0.0725"),
    "NY": Decimal("0.08875"),
    "OR": Decimal("0"),
    "TX": Decimal("0.0625"),
}


class UnknownRegion(KeyError):
    pass


def rate_for(region):
    try:
        return RATES[region.upper()]
    except KeyError:
        raise UnknownRegion(f"no tax rate for region {region!r}") from None


def tax_for(amount, region):
    return round_money(to_decimal(amount) * rate_for(region))
""",
        "invoicing/invoice.py": r"""from .money import format_money, round_money, to_decimal
from .tax import tax_for


class LineItem:
    def __init__(self, description, quantity, unit_price, discount_pct=0):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if not 0 <= discount_pct <= 100:
            raise ValueError("discount_pct must be between 0 and 100")
        self.description = description
        self.quantity = quantity
        self.unit_price = unit_price
        self.discount_pct = discount_pct

    @property
    def subtotal(self):
        gross = self.quantity * float(self.unit_price)
        return round(gross * (1 - self.discount_pct / 100), 2)

    def __repr__(self):
        return f"LineItem({self.description!r}, {self.quantity}, {self.unit_price!r})"


class Invoice:
    def __init__(self, items, region, shipping=0):
        self.items = list(items)
        self.region = region
        self.shipping = to_decimal(shipping)

    @property
    def subtotal(self):
        return round_money(sum(item.subtotal for item in self.items))

    @property
    def tax(self):
        return tax_for(self.subtotal, self.region)

    @property
    def total(self):
        return round_money(self.subtotal + self.tax + self.shipping)

    def render(self, currency="USD"):
        lines = [f"{item.description:<20} {format_money(item.subtotal, currency):>16}" for item in self.items]
        lines.append(f"{'Subtotal':<20} {format_money(self.subtotal, currency):>16}")
        lines.append(f"{'Tax':<20} {format_money(self.tax, currency):>16}")
        if self.shipping:
            lines.append(f"{'Shipping':<20} {format_money(self.shipping, currency):>16}")
        lines.append(f"{'Total':<20} {format_money(self.total, currency):>16}")
        return "\n".join(lines)
""",
        "tests/test_money.py": r"""from decimal import Decimal

from invoicing.money import format_money, round_money


def test_round_half_up():
    assert round_money("1.005") == Decimal("1.01")
    assert round_money(Decimal("2.344")) == Decimal("2.34")


def test_format_groups_thousands():
    assert format_money(1234567.5) == "1,234,567.50 USD"
    assert format_money(-3, "EUR") == "-3.00 EUR"
""",
        "tests/test_invoice.py": r"""from decimal import Decimal

import pytest

from invoicing import Invoice, LineItem, UnknownRegion


def test_simple_total():
    inv = Invoice([LineItem("Book", 2, "10.00")], region="OR", shipping="5")
    assert inv.subtotal == Decimal("20.00")
    assert inv.total == Decimal("25.00")


def test_tax_ny():
    inv = Invoice([LineItem("Pen", 4, "2.50")], region="ny")
    assert inv.tax == Decimal("0.89")


def test_quantity_must_be_positive():
    with pytest.raises(ValueError, match="quantity must be positive"):
        LineItem("Bad", 0, "1.00")


def test_unknown_region():
    with pytest.raises(UnknownRegion):
        Invoice([LineItem("Pen", 1, "1.00")], region="ZZ").tax
""",
    },
    "fix": [
        ("invoicing/invoice.py",
         "        gross = self.quantity * float(self.unit_price)\n"
         "        return round(gross * (1 - self.discount_pct / 100), 2)\n",
         "        gross = to_decimal(self.unit_price) * self.quantity\n"
         "        return round_money(gross * (100 - to_decimal(self.discount_pct)) / 100)\n"),
    ],
    "tests": {
        "tests/test_invoice.py": r"""from decimal import Decimal

import pytest

from invoicing import Invoice, LineItem, UnknownRegion


def test_simple_total():
    inv = Invoice([LineItem("Book", 2, "10.00")], region="OR", shipping="5")
    assert inv.subtotal == Decimal("20.00")
    assert inv.total == Decimal("25.00")


def test_tax_ny():
    inv = Invoice([LineItem("Pen", 4, "2.50")], region="ny")
    assert inv.tax == Decimal("0.89")


def test_quantity_must_be_positive():
    with pytest.raises(ValueError, match="quantity must be positive"):
        LineItem("Bad", 0, "1.00")


def test_unknown_region():
    with pytest.raises(UnknownRegion):
        Invoice([LineItem("Pen", 1, "1.00")], region="ZZ").tax


def test_half_cent_rounds_up():
    subtotal = LineItem("Widget", 1, "2.675").subtotal
    assert isinstance(subtotal, Decimal)
    assert subtotal == Decimal("2.68")


def test_discount_rounding():
    assert LineItem("Cable", 3, "4.35", discount_pct=10).subtotal == Decimal("11.75")


def test_mixed_invoice_total():
    inv = Invoice([LineItem("Widget", 1, "2.675"),
                   LineItem("Cable", 3, "4.35", discount_pct=10)], region="CA")
    assert inv.subtotal == Decimal("14.43")
    assert inv.tax == Decimal("1.05")
    assert inv.total == Decimal("15.48")
""",
    },
    "solution": {
        "input": "Сумма позиции считается во float и округляется встроенным round(), поэтому половина цента уходит вниз; нужно точное Decimal-вычисление с округлением half-up.",
        "locate": "`git grep -n \"def subtotal\" -- '*.py'` -> invoicing/invoice.py, свойство LineItem.subtotal (2 строки); рядом уже импортированы to_decimal и round_money.",
        "change": "В LineItem.subtotal заменить float-вычисление на `to_decimal(self.unit_price) * self.quantity` и вернуть `round_money(gross * (100 - to_decimal(self.discount_pct)) / 100)`.",
        "verify": "python -m pytest tests/test_invoice.py -q",
        "output": "Патч на 2 строки в invoicing/invoice.py; LineItem('Widget', 1, '2.675').subtotal == Decimal('2.68').",
    },
}

CRONLITE = {
    "name": "cronlite",
    "topic": "планировщик: разбор cron-выражений",
    "kind": "feature",
    "issue": r"""
Support step values in cron fields (`*/15`, `1-10/3`)

`CronSchedule` rejects the standard step syntax:

```python
>>> from cronlite import CronSchedule
>>> CronSchedule("*/15 * * * *")
cronlite.errors.CronError: invalid value '*/15' for minute
```

Please support steps the way crontab does:

* `*/15` in the minute field means 0, 15, 30, 45;
* `1-10/3` in the day field means 1, 4, 7, 10;
* `5/20` (a single start value with a step) runs from 5 to the end of the field range: 5, 25, 45 for minutes;
* a zero step is an error: `CronError` whose message contains `step must be positive`.
""",
    "files": {
        "README.md": r"""# cronlite

A tiny cron expression parser: `CronSchedule("30 9 * * 1-5").next_after(datetime.now())`.
Fields: minute, hour, day of month, month, weekday (0 = Sunday).
""",
        "pyproject.toml": r"""[project]
name = "cronlite"
version = "0.2.0"
requires-python = ">=3.9"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
""",
        "cronlite/__init__.py": r"""from .errors import CronError
from .fields import FIELD_NAMES, parse_field
from .schedule import CronSchedule

__all__ = ["CronError", "CronSchedule", "FIELD_NAMES", "parse_field"]
""",
        "cronlite/errors.py": r"""class CronError(ValueError):
    pass
""",
        "cronlite/fields.py": r"""from .errors import CronError

FIELD_NAMES = ("minute", "hour", "day", "month", "weekday")
FIELD_RANGES = {
    "minute": (0, 59),
    "hour": (0, 23),
    "day": (1, 31),
    "month": (1, 12),
    "weekday": (0, 6),
}


def _to_int(text, part, name):
    if not text.isdigit():
        raise CronError(f"invalid value {part!r} for {name}")
    return int(text)


def _check_range(value, name, lo, hi):
    if not lo <= value <= hi:
        raise CronError(f"{name} value {value} out of range {lo}-{hi}")


def parse_field(expr, name):
    # one cron field -> frozenset of allowed integer values
    lo, hi = FIELD_RANGES[name]
    values = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            raise CronError(f"empty item in {name} field {expr!r}")
        if part == "*":
            values.update(range(lo, hi + 1))
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = _to_int(start_text, part, name)
            end = _to_int(end_text, part, name)
        else:
            start = end = _to_int(part, part, name)
        _check_range(start, name, lo, hi)
        _check_range(end, name, lo, hi)
        if start > end:
            raise CronError(f"descending range {part!r} in {name} field")
        values.update(range(start, end + 1))
    return frozenset(values)
""",
        "cronlite/schedule.py": r"""from datetime import timedelta

from .errors import CronError
from .fields import FIELD_NAMES, parse_field

MAX_MINUTES = 366 * 24 * 60


class CronSchedule:
    def __init__(self, expr):
        parts = expr.split()
        if len(parts) != len(FIELD_NAMES):
            raise CronError(f"expected {len(FIELD_NAMES)} fields, got {len(parts)}")
        self.expr = expr
        self.fields = {name: parse_field(part, name) for name, part in zip(FIELD_NAMES, parts)}

    def matches(self, dt):
        f = self.fields
        return (dt.minute in f["minute"] and dt.hour in f["hour"] and dt.day in f["day"]
                and dt.month in f["month"] and dt.isoweekday() % 7 in f["weekday"])

    def next_after(self, dt):
        t = dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
        for _ in range(MAX_MINUTES):
            if self.matches(t):
                return t
            t += timedelta(minutes=1)
        raise CronError(f"no run time within a year for {self.expr!r}")

    def __repr__(self):
        return f"CronSchedule({self.expr!r})"
""",
        "tests/test_fields.py": r"""import pytest

from cronlite import CronError, parse_field


def test_star_and_lists():
    assert parse_field("*", "hour") == frozenset(range(24))
    assert parse_field("1,5-7", "day") == {1, 5, 6, 7}


def test_out_of_range():
    with pytest.raises(CronError, match="minute value 60 out of range 0-59"):
        parse_field("60", "minute")


def test_garbage():
    with pytest.raises(CronError, match="invalid value 'x' for hour"):
        parse_field("x", "hour")
""",
        "tests/test_schedule.py": r"""from datetime import datetime

import pytest

from cronlite import CronError, CronSchedule


def test_next_weekday_morning():
    s = CronSchedule("30 9 * * 1-5")
    assert s.next_after(datetime(2026, 10, 23, 10, 0)) == datetime(2026, 10, 26, 9, 30)


def test_field_count():
    with pytest.raises(CronError, match="expected 5 fields, got 4"):
        CronSchedule("* * * *")
""",
    },
    "fix": [
        ("cronlite/fields.py",
         "        if part == \"*\":\n"
         "            values.update(range(lo, hi + 1))\n"
         "            continue\n",
         "        step = 1\n"
         "        stepped = \"/\" in part\n"
         "        if stepped:\n"
         "            base, _, step_text = part.partition(\"/\")\n"
         "            step = _to_int(step_text, part, name)\n"
         "            if step == 0:\n"
         "                raise CronError(f\"step must be positive in {part!r}\")\n"
         "            part = base\n"
         "        if part == \"*\":\n"
         "            values.update(range(lo, hi + 1, step))\n"
         "            continue\n"),
        ("cronlite/fields.py",
         "            start = end = _to_int(part, part, name)\n",
         "            start = _to_int(part, part, name)\n"
         "            end = hi if stepped else start\n"),
        ("cronlite/fields.py",
         "        values.update(range(start, end + 1))\n",
         "        values.update(range(start, end + 1, step))\n"),
    ],
    "tests": {
        "tests/test_fields.py": r"""import pytest

from cronlite import CronError, parse_field


def test_star_and_lists():
    assert parse_field("*", "hour") == frozenset(range(24))
    assert parse_field("1,5-7", "day") == {1, 5, 6, 7}


def test_out_of_range():
    with pytest.raises(CronError, match="minute value 60 out of range 0-59"):
        parse_field("60", "minute")


def test_garbage():
    with pytest.raises(CronError, match="invalid value 'x' for hour"):
        parse_field("x", "hour")


def test_star_step():
    assert parse_field("*/15", "minute") == {0, 15, 30, 45}


def test_range_step():
    assert parse_field("1-10/3", "day") == {1, 4, 7, 10}


def test_single_value_step_runs_to_end_of_range():
    assert parse_field("5/20", "minute") == {5, 25, 45}


def test_step_in_list():
    assert parse_field("0,30-40/5", "minute") == {0, 30, 35, 40}


def test_zero_step():
    with pytest.raises(CronError, match="step must be positive"):
        parse_field("*/0", "minute")
""",
        "tests/test_schedule.py": r"""from datetime import datetime

import pytest

from cronlite import CronError, CronSchedule


def test_next_weekday_morning():
    s = CronSchedule("30 9 * * 1-5")
    assert s.next_after(datetime(2026, 10, 23, 10, 0)) == datetime(2026, 10, 26, 9, 30)


def test_field_count():
    with pytest.raises(CronError, match="expected 5 fields, got 4"):
        CronSchedule("* * * *")


def test_every_half_hour():
    s = CronSchedule("*/30 * * * *")
    assert s.next_after(datetime(2026, 10, 23, 10, 7)) == datetime(2026, 10, 23, 10, 30)
""",
    },
    "solution": {
        "input": "Нужна поддержка шага `/n` в полях cron: `*/15`, `a-b/n`, `a/n` (до конца диапазона), шаг 0 -> CronError со словами `step must be positive`.",
        "locate": "`git grep -n \"invalid value\" -- '*.py'` -> cronlite/fields.py, функция parse_field.",
        "change": "В parse_field отделить `/шаг` от части поля, проверить шаг > 0, передать шаг в range(...) для `*` и диапазонов; для одиночного значения со шагом конец = верх диапазона.",
        "verify": "python -m pytest tests/test_fields.py tests/test_schedule.py -q",
        "output": "Патч ~10 строк в cronlite/fields.py; parse_field('*/15', 'minute') == {0, 15, 30, 45}.",
    },
}

EVENTBUS = {
    "name": "eventbus",
    "topic": "архитектура: шина событий с подписками",
    "kind": "bug",
    "issue": r"""
`once=True` handlers are skipped when several are subscribed to the same event

If two one-shot handlers listen to the same event, only the first one runs. The same happens when a handler cancels its own subscription while the event is being emitted: the next handler in line is silently skipped.

```python
from eventbus import EventBus

bus = EventBus()
calls = []
bus.subscribe("saved", lambda: calls.append("a"), once=True)
bus.subscribe("saved", lambda: calls.append("b"), once=True)
bus.subscribe("saved", lambda: calls.append("c"))
bus.emit("saved")
print(calls)   # ['a', 'c'] -- expected ['a', 'b', 'c']
```

Every handler that was subscribed when `emit` started must be called exactly once for that emit, even if subscriptions change during the emit.
""",
    "files": {
        "README.md": r"""# eventbus

In-process publish/subscribe. Handlers are called in priority order (higher first, then in
subscription order). `subscribe(..., once=True)` removes the handler after its first call.
""",
        "pyproject.toml": r"""[project]
name = "eventbus"
version = "1.1.0"
requires-python = ">=3.9"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
""",
        "eventbus/__init__.py": r"""from .bus import EventBus
from .subscription import Subscription

__all__ = ["EventBus", "Subscription"]
""",
        "eventbus/subscription.py": r"""class Subscription:
    def __init__(self, bus, event, handler, once=False, priority=0):
        self.bus = bus
        self.event = event
        self.handler = handler
        self.once = once
        self.priority = priority

    @property
    def active(self):
        return self in self.bus.handlers(self.event)

    def cancel(self):
        self.bus.unsubscribe(self)

    def __repr__(self):
        name = getattr(self.handler, "__name__", repr(self.handler))
        return f"Subscription({self.event!r}, {name}, once={self.once}, priority={self.priority})"
""",
        "eventbus/bus.py": r"""from collections import defaultdict

from .subscription import Subscription


class EventBus:
    def __init__(self):
        self._handlers = defaultdict(list)

    def subscribe(self, event, handler, *, once=False, priority=0):
        if not callable(handler):
            raise TypeError("handler must be callable")
        sub = Subscription(self, event, handler, once=once, priority=priority)
        handlers = self._handlers[event]
        handlers.append(sub)
        handlers.sort(key=lambda s: -s.priority)   # stable: equal priorities keep subscription order
        return sub

    def unsubscribe(self, sub):
        handlers = self._handlers.get(sub.event, [])
        if sub in handlers:
            handlers.remove(sub)

    def handlers(self, event):
        return tuple(self._handlers.get(event, ()))

    def emit(self, event, *args, **kwargs):
        results = []
        for sub in self._handlers.get(event, []):
            if sub.once:
                self.unsubscribe(sub)
            results.append(sub.handler(*args, **kwargs))
        return results

    def clear(self, event=None):
        if event is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event, None)
""",
        "tests/test_bus.py": r"""import pytest

from eventbus import EventBus


def test_priority_order():
    bus = EventBus()
    calls = []
    bus.subscribe("e", lambda: calls.append("low"))
    bus.subscribe("e", lambda: calls.append("high"), priority=10)
    bus.emit("e")
    assert calls == ["high", "low"]


def test_once_single_handler():
    bus = EventBus()
    calls = []
    bus.subscribe("e", lambda x: calls.append(x), once=True)
    bus.emit("e", 1)
    bus.emit("e", 2)
    assert calls == [1]


def test_results_and_args():
    bus = EventBus()
    bus.subscribe("sum", lambda a, b: a + b)
    assert bus.emit("sum", 2, b=3) == [5]


def test_handler_must_be_callable():
    with pytest.raises(TypeError):
        EventBus().subscribe("e", "not callable")
""",
    },
    "fix": [
        ("eventbus/bus.py",
         "        for sub in self._handlers.get(event, []):\n",
         "        for sub in list(self._handlers.get(event, ())):\n"),
    ],
    "tests": {
        "tests/test_bus.py": r"""import pytest

from eventbus import EventBus


def test_priority_order():
    bus = EventBus()
    calls = []
    bus.subscribe("e", lambda: calls.append("low"))
    bus.subscribe("e", lambda: calls.append("high"), priority=10)
    bus.emit("e")
    assert calls == ["high", "low"]


def test_once_single_handler():
    bus = EventBus()
    calls = []
    bus.subscribe("e", lambda x: calls.append(x), once=True)
    bus.emit("e", 1)
    bus.emit("e", 2)
    assert calls == [1]


def test_results_and_args():
    bus = EventBus()
    bus.subscribe("sum", lambda a, b: a + b)
    assert bus.emit("sum", 2, b=3) == [5]


def test_handler_must_be_callable():
    with pytest.raises(TypeError):
        EventBus().subscribe("e", "not callable")


def test_all_once_handlers_run():
    bus = EventBus()
    calls = []
    bus.subscribe("saved", lambda: calls.append("a"), once=True)
    bus.subscribe("saved", lambda: calls.append("b"), once=True)
    bus.subscribe("saved", lambda: calls.append("c"))
    bus.emit("saved")
    bus.emit("saved")
    assert calls == ["a", "b", "c", "c"]


def test_cancel_inside_handler_does_not_skip_next():
    bus = EventBus()
    calls = []

    def first():
        calls.append(1)
        sub.cancel()

    sub = bus.subscribe("e", first)
    bus.subscribe("e", lambda: calls.append(2))
    bus.emit("e")
    bus.emit("e")
    assert calls == [1, 2, 2]
""",
    },
    "solution": {
        "input": "emit перебирает список подписок и одновременно удаляет из него one-shot подписки, поэтому следующий обработчик пропускается.",
        "locate": "`git grep -n \"def emit\" -- '*.py'` -> eventbus/bus.py, цикл в EventBus.emit.",
        "change": "Перебирать копию списка: `for sub in list(self._handlers.get(event, ())):`.",
        "verify": "python -m pytest tests/test_bus.py -q",
        "output": "Патч в одну строку в eventbus/bus.py; все one-shot обработчики вызываются при первом emit.",
    },
}

TASKS = [INVOICING, CRONLITE, EVENTBUS]
