"""Фейковый docker для тестов песочницы MiMo: «контейнер» = временная папка, команды исполняются
настоящим bash/git на этой машине. Пути /testbed, /workspace, /tmp/ в argv переписываются на
папку контейнера, конфиг git изолирован (HOME и GIT_CONFIG_GLOBAL внутри папки контейнера)."""
import os
import re
import shutil
import subprocess
from pathlib import Path

TEMPLATE = {
    "pkg/__init__.py": b"",
    "pkg/mod.py": b"def add(a, b):\n    return a - b\n",
    "tests/test_helper.py": b"X = 1\n",
    "README.md": b"hello\n",
}

_MAP = re.compile(r"/testbed|/workspace|/tmp/")


def find_bash():
    cand = os.environ.get("SWE_TEST_BASH") or shutil.which("bash")
    if cand and "system32" in cand.replace("\\", "/").lower():   # WSL-обёртка, не Git Bash
        cand = None
    if cand is None:
        for p in ("C:/Program Files/Git/bin/bash.exe", "/bin/bash", "/usr/bin/bash"):
            if os.path.exists(p):
                return p
    return cand


class FakeDocker:
    def __init__(self, base, template=None):
        self.base = Path(base)
        self.template = template or TEMPLATE
        self.bash = find_bash()
        self.containers = {}
        self.images = []
        self.removed_images = []

    def _map(self, s, root):
        def repl(m):
            return {"/testbed": (root / "testbed").as_posix(),
                    "/workspace": (root / "workspace").as_posix(),
                    "/tmp/": (root / "tmp").as_posix() + "/"}[m.group(0)]
        return _MAP.sub(repl, s)

    def run(self, args, input=None, timeout=None):
        op = args[0]
        if op == "run":
            name = args[args.index("--name") + 1]
            self.images.append(args[-3])
            root = self.base / name
            for rel, data in self.template.items():
                p = root / "testbed" / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            (root / "tmp").mkdir(parents=True, exist_ok=True)
            (root / "home").mkdir(parents=True, exist_ok=True)
            self.containers[name] = root
            return 0, b"cid\n", b""
        if op == "rm":
            root = self.containers.pop(args[-1], None)
            if root:
                shutil.rmtree(root, ignore_errors=True)
            return 0, b"", b""
        if op == "rmi":
            self.removed_images.append(args[-1])
            return 0, b"", b""
        if op == "exec":
            name = args[4]
            root = self.containers[name]
            argv = [self._map(a, root) for a in args[5:]]
            env = dict(os.environ, HOME=(root / "home").as_posix(),
                       GIT_CONFIG_GLOBAL=(root / "home" / ".gitconfig").as_posix(),
                       GIT_CONFIG_NOSYSTEM="1", CHERE_INVOKING="1")
            try:
                p = subprocess.run([self.bash, "-c", 'exec "$@"', "_", *argv],
                                   input=input if input is not None else b"",
                                   cwd=str(root / "testbed"), env=env, capture_output=True,
                                   timeout=timeout)
                return p.returncode, p.stdout, p.stderr
            except subprocess.TimeoutExpired:
                return -1, b"", b"TimeoutExpired"
        raise AssertionError(f"неожиданная docker-команда: {args}")

    def container_files(self, name):
        return self.containers[name] / "testbed"
