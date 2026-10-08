# reward/anchor.py
import hashlib
import json
import re


def canonicalize_observation(tool_name, tool_result):
    try:
        parsed = json.loads(tool_result)
    except Exception:
        parsed = {"status": "raw"}
    status = parsed.get("status", "unknown")

    if tool_name == "run_command":
        out = (parsed.get("stdout", "") or "") + (parsed.get("stderr", "") or "")
        m_fail = re.search(r"(\d+) failed", out)
        m_pass = re.search(r"(\d+) passed", out)
        m_err = re.search(r"(Error|Exception): ?(\w+)?", out)
        return (f"pytest|failed={m_fail.group(1) if m_fail else '?'}"
                f"|passed={m_pass.group(1) if m_pass else '?'}"
                f"|err={m_err.group(2) if m_err else 'none'}")
    if tool_name in ("edit_file", "write_file"):
        return f"{tool_name}|{status}|{parsed.get('filepath', '')}"
    if tool_name == "read_file":
        return f"read|{parsed.get('filepath', '')}"
    return f"{tool_name}|{status}"


def anchor_key(task_id, tool_name, tool_result):
    """ВАЖНО: ключ НЕ включает turn_index -- иначе GiGPO micro-advantage группировка
    вырождается почти в пустые бакеты (разные роллауты редко попадают в одинаковое
    состояние на одном и том же номере хода)."""
    sig = canonicalize_observation(tool_name, tool_result)
    raw = f"{task_id}|{sig}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]
