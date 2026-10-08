# sim/container_setup.py
"""
Реконструкция container_setup.py харнесса по HARNESS_README.md.
ВНИМАНИЕ: содержимое pytest.ini/conftest.py -- наша best-effort реконструкция ПО ОПИСАНИЮ,
НЕ побайтовая копия. Если inspect_harness.py покажет, что swegemma.harness.container_setup
импортируется напрямую -- заменить bootstrap_task() на вызов оригинальной функции.
"""
import tarfile
import tempfile
import shutil

PYTEST_INI = """[pytest]
addopts = -p no:anyio
norecursedirs = hardware device examples .git __pycache__
"""

CONFTEST_HOOK = """
# --- swegemma hermetic test collection hook (sim reconstruction) ---
import os
collect_ignore_glob = ["hardware/*", "device/*", "examples/*"]
# --- end hook ---
"""


def bootstrap_task(container, snapshot_tgz: str, repo: str, base_commit: str,
                    commit_message: str = "baseline"):
    tmp = tempfile.mkdtemp(prefix="snap_")
    try:
        with tarfile.open(snapshot_tgz) as tf:
            tf.extractall(tmp)
        container.cp_in(tmp + "/.", "/workspace")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    container.exec(f"git checkout {base_commit} -q || true")

    exclude_patterns = ["__pycache__/", "*.pyc", ".pytest_cache/", "*.egg-info/", "build/", "dist/", ".coverage"]
    container.exec("mkdir -p .git/info && " +
                    " && ".join(f"echo '{p}' >> .git/info/exclude" for p in exclude_patterns))

    container.exec("pip install --no-index --find-links=/wheels --no-build-isolation "
                    "--no-deps -e /workspace 2>/dev/null || true")
    container.exec(f"python3 /sandbox/setup.py --fast-path {repo}")

    container.exec(f"cat > pytest.ini << 'SWEGEMMA_EOF'\n{PYTEST_INI}SWEGEMMA_EOF")
    container.exec(f"cat >> conftest.py << 'SWEGEMMA_EOF'\n{CONFTEST_HOOK}SWEGEMMA_EOF")

    container.exec(f'git add -A && git commit -m "{commit_message}" --allow-empty -q')
