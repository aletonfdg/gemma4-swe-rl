# sim/prompts.py -- дословная реконструкция build_agent_prompt() из HARNESS_README.md
import subprocess

STANDARD_INSTRUCTIONS = """## Standard Instructions
0. Work strictly under /workspace. Do not touch files outside this directory.
1. Begin by inspecting the repository structure and relevant files to understand existing conventions before writing code.
2. Make minimal, targeted changes that directly address the problem statement.
3. Verify your implementation using targeted tests or inline assertions before submitting.
4. Do not modify test files, pytest.ini, conftest.py, or other test-runner configuration -- such changes will be discarded before grading.
5. Call submit_patch() once you are confident the fix is complete and verified, then provide a brief final summary."""

STANDARD_INSTRUCTIONS_NO_SANDBOX_TESTING = STANDARD_INSTRUCTIONS.replace(
    "3. Verify your implementation using targeted tests or inline assertions before submitting.",
    "3. Verify your implementation using inline assertions (e.g. `python3 -c \"...\"`) before "
    "submitting; pytest/unittest execution is disabled in this sandbox.",
)

CODE_INTEL_BLOCK = """## Code Intelligence Tools
This repository has pre-built code graph and embedding data. Use these tools for fast, targeted navigation:
- `search_similar_code(query)`: Find semantically similar functions/classes by keyword.
- `get_code_neighbors(node)`: Find callers, callees, and definitions related to a symbol.
- `get_code_subgraph(nodes)`: Get the induced subgraph for a set of symbols."""

NUDGE_UNCLOSED_TOOL_CALL = (
    "Your previous response reached the token limit before the tool call finished closing "
    "(<|tool_call|> was cut off). Do NOT repeat your prior reasoning in thought\u2014emit your "
    "next tool call immediately, and if calling edit_file or write_file, split the change into "
    "smaller incremental edits."
)
NUDGE_MAX_TOKENS = (
    "Your previous response reached the token limit while thinking before a tool call was "
    "completed. Do NOT repeat your analysis in thought\u2014keep reasoning under a few sentences "
    "and emit your next tool call immediately, or call submit_patch when you have completed and "
    "verified your changes."
)
NUDGE_NORMAL_STOP = (
    "Please continue your work using the available tools, or call submit_patch when you have "
    "completed and verified your changes."
)


def workspace_layout(workdir: str, max_entries: int = 150) -> str:
    try:
        r = subprocess.run(
            ["find", ".", "-maxdepth", "3"],
            cwd=workdir, capture_output=True, text=True, timeout=10,
        )
        lines = [l for l in r.stdout.splitlines()
                 if l.strip() and ".git" not in l and "__pycache__" not in l and not l.endswith(".pyc")]
        return "\n".join(lines[:max_entries])
    except Exception:
        return "(unavailable)"


def build_agent_prompt(task: dict, workdir: str, budget, harness, has_graph: bool,
                        enable_sandbox_testing: bool = True, layout: str | None = None) -> str:
    """layout -- готовый текст дерева файлов (для удалённой песочницы, где workdir не существует
    на этой машине); None = посчитать локально через workspace_layout(workdir)."""
    sections = [
        f"You are evaluating a software engineering task for repository {task['repo']}.\n\n"
        f"Problem Statement:\n{task['problem_statement']}"
    ]
    if task.get("hints_text"):
        sections.append(f"## Hints:\n{task['hints_text'].strip()}")

    budget_lines = [
        "## Task Budget (Session terminates when any budget is exhausted)",
        f"- Time allowance: {budget.time_minutes:.1f} minutes",
        f"- Tool calls allowance: {budget.tool_calls} calls",
        f"- Max loop iterations: {budget.turns} turns",
    ]
    if budget.cost_usd is not None:
        budget_lines.append(f"- Cost budget: ${budget.cost_usd:.2f} USD")
    sections.append("\n".join(budget_lines))

    sections.append(
        "## Execution Environment Rules\n"
        f"- Single command timeout: {harness.command_timeout_seconds} seconds "
        "(commands exceeding this fail without ending the session)\n"
        f"- Command output limit: {harness.max_stdout_chars} characters\n"
        f"- File view limit: {harness.max_file_lines} lines per read_file call\n"
        f"- File character limit: {harness.max_file_chars} characters per read_file call\n"
        "- Environment is offline (no network/PyPI access). All repository and test "
        "dependencies are ALREADY pre-installed. Do NOT attempt to run pip install or "
        "download packages."
    )

    sections.append(STANDARD_INSTRUCTIONS if enable_sandbox_testing
                     else STANDARD_INSTRUCTIONS_NO_SANDBOX_TESTING)

    if has_graph:
        sections.append(CODE_INTEL_BLOCK)

    sections.append(f"## Workspace Layout\n{layout if layout is not None else workspace_layout(workdir)}")
    return "\n\n".join(sections)
