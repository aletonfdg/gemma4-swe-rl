# sim/tools.py
import os
import re
import json
import difflib
import functools
import subprocess


def _err(error_type, message, **details):
    return {"status": "error", "error_type": error_type, "error_message": message, "details": details}


def _ok(**payload):
    payload["status"] = "ok"
    return payload


def _truncate(text, limit):
    if text is None:
        return "", False
    return (text[:limit], True) if len(text) > limit else (text, False)


def _maybe_add_budget_warning(ctx, payload):
    used = ctx.tool_calls_used
    remaining = ctx.remaining_tool_calls()
    if used >= 20 and remaining <= 10:
        payload["budget_warning"] = (
            f"Only {remaining} tool call(s) remaining ({used}/{ctx.budget.tool_calls} used). "
            "Finalize your edits and call submit_patch soon."
        )
    return payload


def budget_gated(count_tool_call=True):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(ctx, **kwargs):
            if count_tool_call:
                if ctx.remaining_tool_calls() <= 0 or ctx.remaining_time_seconds() <= 0:
                    return json.dumps(_err("BudgetExceeded", "Budget exhausted."), ensure_ascii=False)
                ctx.tool_calls_used += 1
            result = fn(ctx, **kwargs)
            result = _maybe_add_budget_warning(ctx, result)
            return json.dumps(result, ensure_ascii=False)
        return wrapper
    return deco


@budget_gated()
def run_command(ctx, command: str):
    timeout = min(ctx.harness.command_timeout_seconds, max(1, int(ctx.remaining_time_seconds())))
    try:
        result = subprocess.run(command, shell=True, executable="/bin/bash", cwd=ctx.workspace,
                                 capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _err("TimeoutExceeded", f"Timed out after {timeout}s")
    except Exception as e:
        return _err("ExecutionError", str(e))
    stdout, _ = _truncate(result.stdout, ctx.harness.max_stdout_chars)
    stderr, _ = _truncate(result.stderr, ctx.harness.max_stdout_chars)
    if result.returncode == 0:
        return _ok(stdout=stdout, stderr=stderr, exit_code=0)
    return _err("CommandError", f"exit {result.returncode}", stdout=stdout, stderr=stderr,
                exit_code=result.returncode)


@budget_gated(count_tool_call=False)
def submit_patch(ctx):
    subprocess.run(["git", "add", "-N", "."], cwd=ctx.workspace, capture_output=True)
    diff = subprocess.run(
        "git diff --binary _swegemma_baseline 2>/dev/null || git diff --binary HEAD",
        shell=True, executable="/bin/bash", cwd=ctx.workspace, capture_output=True, text=True)
    ctx.submitted_patch = diff.stdout or ""
    ctx.patch_submitted = True
    stat = subprocess.run("git diff --name-only HEAD 2>/dev/null | wc -l", shell=True,
                           executable="/bin/bash", cwd=ctx.workspace, capture_output=True, text=True)
    files_changed = int((stat.stdout or "0").strip() or 0)
    return _ok(patch_size=len(ctx.submitted_patch), files_changed=files_changed)


@budget_gated(count_tool_call=False)
def get_status(ctx):
    return _ok(
        tool_calls_used=ctx.tool_calls_used,
        patch_submitted=ctx.patch_submitted,
        patch_size=len(ctx.submitted_patch),
        tool_calls_remaining=ctx.remaining_tool_calls(),
        max_tool_calls=ctx.budget.tool_calls,
        time_seconds_remaining=ctx.remaining_time_seconds(),
        max_time_minutes=ctx.budget.time_minutes,
        agent_elapsed_seconds=ctx.agent_elapsed_seconds(),
        max_turns=ctx.budget.turns,
        command_timeout_seconds=ctx.harness.command_timeout_seconds,
    )


@budget_gated()
def read_file(ctx, filepath, start_line=None, end_line=None):
    try:
        full = ctx.resolve(filepath)
    except ValueError as e:
        return _err("ValidationError", str(e))
    if not os.path.isfile(full):
        return _err("FileNotFoundError", f"'{filepath}' does not exist.")
    lines = open(full, "r", encoding="utf-8", errors="replace").readlines()
    total = len(lines)
    start = max(0, (start_line - 1)) if start_line else 0
    stop = min(end_line, total) if end_line else total
    sel = lines[start:stop]
    truncated = False
    if len(sel) > ctx.harness.max_file_lines:
        sel = sel[:ctx.harness.max_file_lines]
        truncated = True
    content, char_trunc = _truncate("".join(sel), ctx.harness.max_file_chars)
    end_returned = start + len(sel)
    return _ok(filepath=filepath, content=content or "File is empty.",
               start_line=start + 1, end_line=end_returned, total_lines=total,
               is_truncated=truncated or char_trunc)


def _apply_replacement(content, old_string, new_string, allow_multiple):
    count = content.count(old_string)
    if count >= 1:
        if count > 1 and not allow_multiple:
            return None, count, "exact"
        return (content.replace(old_string, new_string) if allow_multiple
                else content.replace(old_string, new_string, 1)), count, "exact"

    c_lines = content.splitlines(keepends=True)
    stripped = [l.strip() for l in c_lines]
    o_lines = [l.strip() for l in old_string.splitlines()]
    n = len(o_lines)
    matches = [i for i in range(len(c_lines) - n + 1) if stripped[i:i + n] == o_lines] if n else []
    if matches:
        if len(matches) > 1 and not allow_multiple:
            return None, len(matches), "flexible"
        targets = matches if allow_multiple else matches[:1]
        for start in sorted(targets, reverse=True):
            indent = re.match(r"[ \t]*", c_lines[start]).group(0)
            block = "\n".join((indent + l) if l.strip() else l for l in new_string.splitlines())
            if not block.endswith("\n"):
                block += "\n"
            c_lines[start:start + n] = [block]
        return "".join(c_lines), len(matches), "flexible"

    delims = r"()\[\]{}:><=,"
    tokens = [re.escape(t) for t in re.split(f"([{delims}])", old_string) if t != ""]
    pattern = re.compile(r"\s*".join(tokens), re.MULTILINE)
    rmatches = list(pattern.finditer(content))
    if rmatches:
        if len(rmatches) > 1 and not allow_multiple:
            return None, len(rmatches), "regex"
        new_content = content
        for m in reversed(rmatches if allow_multiple else rmatches[:1]):
            new_content = new_content[:m.start()] + new_string + new_content[m.end():]
        return new_content, len(rmatches), "regex"
    return None, 0, None


@budget_gated()
def edit_file(ctx, filepath, old_string, new_string, allow_multiple=False):
    try:
        full = ctx.resolve(filepath)
    except ValueError as e:
        return _err("ValidationError", str(e))
    if not os.path.isfile(full) or os.path.getsize(full) == 0:
        return _err("FileEditError", "File missing or empty.")
    if not old_string:
        return _err("FileEditError", "old_string must not be empty.")
    original = open(full, "r", encoding="utf-8", errors="replace").read()
    new_content, occ, strategy = _apply_replacement(original, old_string, new_string, allow_multiple)
    if new_content is None:
        return _err("FileEditError", f"old_string not matched (tried exact/flexible/regex, best={occ}).")
    open(full, "w", encoding="utf-8").write(new_content)
    diff = "".join(difflib.unified_diff(original.splitlines(keepends=True),
                                         new_content.splitlines(keepends=True),
                                         fromfile=f"a/{filepath}", tofile=f"b/{filepath}"))
    diff, trunc = _truncate(diff, ctx.harness.max_stdout_chars)
    return _ok(filepath=filepath, occurrences=occ, strategy=strategy, diff=diff, is_truncated=trunc)


@budget_gated()
def write_file(ctx, filepath, content):
    try:
        full = ctx.resolve(filepath)
    except ValueError as e:
        return _err("ValidationError", str(e))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    open(full, "w", encoding="utf-8").write(content)
    return _ok(filepath=filepath, size=len(content.encode("utf-8")))


@budget_gated()
def get_code_neighbors(ctx, node, edge_type=None, max_neighbors=50):
    if ctx.graph is None:
        return _err("GraphUnavailable", "No graph for this repo.")
    resolved = ctx.resolve_node(node)
    if resolved is None:
        return _err("NodeNotFound", f"'{node}' not found.")
    neighbors = set()
    for _, t, d in ctx.graph.out_edges(resolved, data=True):
        if edge_type is None or d.get("type") == edge_type:
            neighbors.add(t)
    for s, _, d in ctx.graph.in_edges(resolved, data=True):
        if edge_type is None or d.get("type") == edge_type:
            neighbors.add(s)
    result = list(neighbors)[:max_neighbors]
    return _ok(node=resolved, neighbors=result, count=len(result))


def _cosine(a, b):
    import numpy as np
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


@budget_gated()
def search_similar_code(ctx, query, k=10):
    if not ctx.embeddings:
        return _err("EmbeddingsUnavailable", "No embeddings for this repo.")
    resolved = ctx.resolve_node(query) if ctx.graph is not None else None
    if resolved is None or resolved not in ctx.embeddings:
        cands = [key for key in ctx.embeddings if query.lower() in key.lower()]
        resolved = cands[0] if cands else None
    if resolved is None:
        return _err("NodeNotFound", f"Could not resolve '{query}'.")
    qv = ctx.embeddings[resolved]
    scored = sorted(((n, _cosine(qv, v)) for n, v in ctx.embeddings.items() if n != resolved),
                     key=lambda x: x[1], reverse=True)[:k]
    results = [{"node_name": n, "code": ctx.node_text.get(n, ""), "similarity": round(s, 4)}
               for n, s in scored]
    return _ok(query=query, results=results, count=len(results))


@budget_gated()
def get_code_subgraph(ctx, nodes):
    if ctx.graph is None:
        return _err("GraphUnavailable", "No graph for this repo.")
    resolved = [r for n in nodes if (r := ctx.resolve_node(n))]
    if not resolved:
        return _err("NodeNotFound", "No requested symbols found.")
    sub = ctx.graph.subgraph(resolved)
    edges = [{"from": u, "to": v, "type": d.get("type", "")} for u, v, d in sub.edges(data=True)]
    return _ok(nodes=resolved, edges=edges, node_count=sub.number_of_nodes(), edge_count=sub.number_of_edges())


TOOL_REGISTRY = {
    "run_command": run_command, "read_file": read_file, "edit_file": edit_file,
    "write_file": write_file, "submit_patch": submit_patch, "get_status": get_status,
    "get_code_neighbors": get_code_neighbors, "search_similar_code": search_similar_code,
    "get_code_subgraph": get_code_subgraph,
}
