# llm/vllm_server.py -- настоящий OpenAI-совместимый сервер (НЕ LLM.generate() напрямую)
import memory_utils  # noqa
import subprocess
import time
import os
import signal
import requests
import yaml

CFG = yaml.safe_load(open("config.yaml"))


class VLLMServerProcess:
    def __init__(self):
        self.proc = None
        self.port = CFG["vllm"]["port"]
        self._lora_modules: dict[str, str] = {}

    def start(self, lora_paths: dict[str, str] | None = None):
        self._lora_modules = dict(lora_paths or {})
        hf_id = CFG["model"]["id"]
        cmd = [
            "vllm", "serve", hf_id,
            "--quantization", "compressed-tensors",
            "--max-model-len", str(CFG["vllm"]["max_model_len"]),
            "--gpu-memory-utilization", str(CFG["vllm"]["gpu_memory_utilization"]),
            "--enable-auto-tool-choice",
            "--tool-call-parser", CFG["vllm"]["tool_call_parser"],
            "--reasoning-parser", CFG["vllm"]["reasoning_parser"],
            "--port", str(self.port),
            "--dtype", "bfloat16",
        ]
        if self._lora_modules:
            cmd += ["--enable-lora", "--max-lora-rank", str(CFG["vllm"]["max_lora_rank"]),
                    "--lora-modules"] + [f"{name}={path}" for name, path in self._lora_modules.items()]
        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
        self._wait_healthy()

    def _wait_healthy(self, timeout=900):
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"http://127.0.0.1:{self.port}/health", timeout=3)
                if r.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(2)
        raise RuntimeError("vLLM server did not become healthy in time")

    def reload_lora(self, name: str, path: str):
        """TODO/FIXME: горячая подгрузка LoRA без рестарта процесса.
        vLLM >=0.6 экспериментально поддерживает POST /v1/load_lora_adapter на части версий --
        ПРОВЕРИТЬ на закреплённой версии перед использованием в train_gigpo.py. Если endpoint
        недоступен -- единственный надёжный путь: self.stop() + self.start(lora_paths={...})
        (дороже по времени, но гарантированно корректно)."""
        try:
            r = requests.post(
                f"http://127.0.0.1:{self.port}/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": path}, timeout=60,
            )
            if r.status_code == 200:
                self._lora_modules[name] = path
                return True
        except requests.RequestException:
            pass
        return False

    def restart_with_lora(self, name: str, path: str):
        self.stop()
        self._lora_modules[name] = path
        self.start(lora_paths=self._lora_modules)

    def stop(self):
        if self.proc:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            self.proc = None
