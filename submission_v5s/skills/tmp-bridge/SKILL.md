---
name: tmp-bridge
description: One-time setup that makes /tmp the same directory for write_file and run_command, and keeps scratch files out of the patch.
---

# tmp-bridge

Setup (once, first): `run_skill_script` with skill_name `tmp-bridge` and file_path `scripts/setup.py`, with no other arguments.

Afterwards `write_file("/tmp/repro.py", ...)` and `run_command("python3 /tmp/repro.py")` use the same file,
and files in /tmp or root-level `repro*` / `debug_*` scripts never appear in the submitted patch.
