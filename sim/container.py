# sim/container.py
import subprocess
import uuid
import os


class DockerSandbox:
    """Docker-песочница с лимитами, соответствующими харнессу (4GiB RAM, 2 vCPU, no network).

    FIXME/TODO (архитектурная развилка -- решить до первого полномасштабного прогона):
    Текущая реализация копирует snapshot ВНУТРЬ контейнера через `docker cp` и исполняет
    run_command через `docker exec`, но read_file/edit_file/write_file (sim/tools.py)
    работают с os.path на ХОСТЕ. Для единого адресного пространства рекомендуется bind-mount:
        -v {host_workdir}:/workspace
    и использовать host_workdir как ctx.workspace, тогда docker нужен только для run_command
    (ограничение CPU/RAM/network) и верификации, а read_file/edit_file/write_file могут
    работать напрямую с host_workdir без докер-слоя (быстрее на тысячи RL-эпизодов).
    """

    def __init__(self, image, wheels_dir, sandbox_scripts_dir, mem_limit="4g", cpus=2.0,
                 host_workdir: str | None = None):
        self.image = image
        self.wheels_dir = os.path.abspath(wheels_dir)
        self.sandbox_scripts_dir = os.path.abspath(sandbox_scripts_dir)
        self.mem_limit = mem_limit
        self.cpus = cpus
        self.host_workdir = os.path.abspath(host_workdir) if host_workdir else None
        self.name = f"swegemma_sim_{uuid.uuid4().hex[:10]}"
        self._started = False

    def start(self):
        cmd = [
            "docker", "run", "-d", "--name", self.name,
            "--network", "none",
            "--memory", self.mem_limit,
            "--cpus", str(self.cpus),
            "-v", f"{self.wheels_dir}:/wheels:ro",
            "-v", f"{self.sandbox_scripts_dir}:/sandbox:ro",
        ]
        if self.host_workdir:
            os.makedirs(self.host_workdir, exist_ok=True)
            cmd += ["-v", f"{self.host_workdir}:/workspace"]
        cmd += [self.image, "sleep", "infinity"]
        subprocess.run(cmd, check=True, capture_output=True)
        self._started = True

    def cp_in(self, local_path, container_path):
        subprocess.run(["docker", "cp", local_path, f"{self.name}:{container_path}"], check=True)

    def cp_out(self, container_path, local_path):
        subprocess.run(["docker", "cp", f"{self.name}:{container_path}", local_path], check=True)

    def exec(self, command, cwd="/workspace", timeout=300):
        try:
            r = subprocess.run(
                ["docker", "exec", "-w", cwd, self.name, "/bin/bash", "-c", command],
                capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "TimeoutExpired"

    def stop(self):
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self._started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()
