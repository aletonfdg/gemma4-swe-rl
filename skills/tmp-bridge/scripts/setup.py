"""tmp-bridge setup: make write_file("/tmp/x") and run_command("... /tmp/x") refer to the same file.

The harness strips the leading "/" in write_file, so write_file("/tmp/repro.py") creates <repo>/tmp/repro.py,
while run_command sees the real /tmp (the sandbox's tmp directory). The agent then runs a stale script, sees
a wrong result and spends minutes debugging its tools (70% of our baseline episodes wrote to /tmp this way).
This script links <repo>/tmp to the real tmp directory and keeps it, and root-level scratch scripts, out of
the patch via .git/info/exclude. Run once, with no arguments. Safe to run more than once.

find_workspace() follows robust-edit (magicsword001/robust-edit, Apache 2.0): in the subprocess sandbox that
Kaggle uses the repository is a temporary directory and ADK runs skill scripts from a temporary copy of the
skill, so the repository is searched for instead of assuming /workspace.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

EXCLUDE_MARKER = "# tmp-bridge"
EXCLUDE_PATTERNS = ["/tmp", "/repro*", "/reproduce*", "/debug_*", "/scratch*"]


def find_workspace() -> Path | None:
    """First git repository among: /workspace, the cwd before ADK's chdir, the parent process's cwd, $HOME, cwd."""
    candidates = [Path("/workspace")]
    frame = sys._getframe()
    while frame is not None:  # ADK's skill wrapper keeps the original cwd in a local named _orig_cwd
        if "_orig_cwd" in frame.f_locals:
            candidates.append(Path(frame.f_locals["_orig_cwd"]))
            break
        frame = frame.f_back
    try:
        candidates.append(Path(os.readlink(f"/proc/{os.getppid()}/cwd")))
    except OSError:
        pass
    if os.environ.get("HOME"):
        candidates.append(Path(os.environ["HOME"]))
    candidates.append(Path.cwd())
    return next((c for c in candidates if (c / ".git").is_dir()), None)


def real_tmp(repo: Path) -> Path:
    """Docker sandbox: /workspace -> /tmp. Subprocess sandbox: swegemma_sandbox_<id>/{workspace,tmp}."""
    sibling = repo.parent / "tmp"
    return sibling if sibling.is_dir() else Path("/tmp")


def bridge(repo: Path) -> str:
    link = repo / "tmp"
    target = real_tmp(repo)
    if link.is_symlink():
        status = f"already linked to {os.readlink(link)}"
    elif link.exists():
        return "the repository has its own tmp/; nothing linked"
    else:
        os.symlink(target, link, target_is_directory=True)
        status = f"linked to {target}"
    exclude = repo / ".git" / "info" / "exclude"
    exclude.parent.mkdir(parents=True, exist_ok=True)
    current = exclude.read_text() if exclude.exists() else ""
    if EXCLUDE_MARKER not in current:
        exclude.write_text(current.rstrip("\n") + f"\n{EXCLUDE_MARKER}\n" + "\n".join(EXCLUDE_PATTERNS) + "\n")
    return status


def main() -> None:
    repo = find_workspace()
    if repo is None:
        print("tmp-bridge: repository not found; nothing changed.")
        return
    try:
        status = bridge(repo)
    except OSError as e:
        print(f"tmp-bridge: could not link tmp ({e}); nothing changed.")
        return
    print(f"tmp-bridge: {status}. /tmp is now the same for write_file and run_command; "
          "files there and root-level repro*/debug_* scripts stay out of the patch.")


if __name__ == "__main__":
    main()
