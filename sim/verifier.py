# sim/verifier.py -- Phase 2: anti-tampering + JUnit XML валидация (как в харнессе)
import os
import re
import subprocess
import tempfile
import shutil
import xml.etree.ElementTree as ET
from sim.container import DockerSandbox
from sim.container_setup import bootstrap_task

PROTECTED_PATTERNS = [
    r"test_.*\.py$", r".*_test\.py$", r"^tests?/", r"^testing/",
    r"conftest\.py$", r"pytest\.ini$", r"pyproject\.toml$", r"tox\.ini$",
    r"setup\.cfg$", r"\.pytest\.ini$", r"sitecustomize\.py$", r"usercustomize\.py$",
    r"_swegemma_stubs\.py$", r".*\.pth$",
]


def _is_protected(path: str) -> bool:
    return any(re.search(p, path) for p in PROTECTED_PATTERNS)


def _extract_touched_files(patch: str) -> set:
    return {m.group(1) for m in re.finditer(r"^\+\+\+ b/(.+)$", patch, re.MULTILINE)}


def _apply_patch_4pass(container, patch_path: str) -> bool:
    passes = [
        f"git apply --unsafe-paths -p1 {patch_path}",
        f"git apply --unsafe-paths -p1 -3 {patch_path}",
        f"git apply --unsafe-paths -p1 --ignore-space-change --ignore-whitespace {patch_path}",
        f"git apply --unsafe-paths -p1 --recount {patch_path}",
        f"git apply --unsafe-paths -p0 {patch_path}",
        f"patch -p1 --batch --forward < {patch_path}",
        f"patch -p0 --batch --forward < {patch_path}",
    ]
    for cmd in passes:
        code, _, _ = container.exec(cmd)
        if code == 0:
            return True
    return False


def _validate_junit_xml(xml_path: str, required_nodes: set | None = None) -> bool:
    if not os.path.exists(xml_path):
        return False
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return False
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        return False
    total = int(suite.get("tests", 0))
    failures = int(suite.get("failures", 0))
    errors = int(suite.get("errors", 0))
    skipped = int(suite.get("skipped", 0))
    passed = total - failures - errors - skipped
    if passed <= 0 or failures > 0 or errors > 0:
        return False
    if required_nodes:
        passed_names = set()
        for tc in suite.findall("testcase"):
            if tc.find("failure") is None and tc.find("error") is None and tc.find("skipped") is None:
                passed_names.add(f"{tc.get('classname', '')}::{tc.get('name', '')}")
        for req in required_nodes:
            if not any(req in p for p in passed_names):
                return False
    return True


def verify_task(task: dict, patch: str, snapshots_dir: str, docker_cfg: dict, timeout=600) -> bool:
    if not patch or not patch.strip() or patch.strip() == "NO_PATCH":
        return False

    container = DockerSandbox(docker_cfg["image"], docker_cfg["wheels_dir"],
                               docker_cfg["sandbox_scripts_dir"],
                               docker_cfg["mem_limit"], docker_cfg["cpus"])
    container.start()
    tmp = tempfile.mkdtemp(prefix="verify_")
    try:
        tgz = os.path.join(snapshots_dir, f"{task['instance_id']}.tgz")
        bootstrap_task(container, tgz, task["repo"], task["base_commit"], commit_message="eval_baseline")

        patch_file_local = os.path.join(tmp, "_agent.patch")
        open(patch_file_local, "w").write(patch)
        container.cp_in(patch_file_local, "/tmp/_agent.patch")
        if not _apply_patch_4pass(container, "/tmp/_agent.patch"):
            return False

        touched = _extract_touched_files(patch)
        protected = [f for f in touched if _is_protected(f)]
        if protected:
            files_arg = " ".join(protected)
            container.exec(f"git checkout HEAD -- {files_arg} 2>/dev/null || true")
            container.exec(f"git clean -f -- {files_arg} 2>/dev/null || true")

        test_patch_local = os.path.join(tmp, "_test.patch")
        open(test_patch_local, "w").write(task["test_patch"])
        container.cp_in(test_patch_local, "/tmp/_test.patch")
        if not _apply_patch_4pass(container, "/tmp/_test.patch"):
            return False

        junit_path_container = "/tmp/_junit.xml"
        cmd = (
            "PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -m pytest "
            f"--junitxml={junit_path_container} -p no:anyio -o timeout=0 "
            '-o python_classes="Test* *Test" -q'
        )
        code, stdout, stderr = container.exec(cmd, timeout=timeout)

        junit_local = os.path.join(tmp, "junit.xml")
        try:
            container.cp_out(junit_path_container, junit_local)
        except subprocess.CalledProcessError:
            return False

        required = {m.group(1) for m in re.finditer(r"def (test_\w+)", task.get("test_patch", ""))}
        return code == 0 and _validate_junit_xml(junit_local, required)
    except Exception:
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        container.stop()
