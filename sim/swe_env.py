# sim/swe_env.py
import os
import json
import tempfile
import yaml
from sim.container import DockerSandbox
from sim.container_setup import bootstrap_task
from sim.context import SwegemmaContext
from sim.tools import TOOL_REGISTRY, submit_patch as _submit_patch_fn
from sim.prompts import (build_agent_prompt, NUDGE_UNCLOSED_TOOL_CALL,
                          NUDGE_MAX_TOKENS, NUDGE_NORMAL_STOP)
from sim.config import default_budget_harness, tight_budget_harness

_CFG = yaml.safe_load(open("config.yaml"))
MAX_NUDGES = 3


class SWEEnv:
    def __init__(self, tasks_path, snapshots_dir, graphs_dir=None, embeddings_dir=None,
                 use_tight_budget=False, eval_config_path="eval_config.yaml"):
        self.tasks = [json.loads(l) for l in open(tasks_path)]
        self.snapshots_dir = snapshots_dir
        self.graphs_dir = graphs_dir
        self.embeddings_dir = embeddings_dir
        self.docker_cfg = _CFG["docker"]
        self.use_tight_budget = use_tight_budget
        self.eval_config_path = eval_config_path
        self.enable_sandbox_testing = _CFG["env_budget"]["enable_sandbox_testing"]

    def _current_budget_harness(self):
        if self.use_tight_budget:
            return tight_budget_harness(self.eval_config_path)
        return default_budget_harness()

    def reset(self, task_idx: int, force_tight_budget: bool | None = None):
        self.task = self.tasks[task_idx]
        use_tight = self.use_tight_budget if force_tight_budget is None else force_tight_budget
        self.budget, self.harness = (tight_budget_harness(self.eval_config_path) if use_tight
                                      else default_budget_harness())

        self.host_workdir = tempfile.mkdtemp(prefix=f"swe_{self.task['instance_id']}_")
        self.container = DockerSandbox(
            image=self.docker_cfg["image"],
            wheels_dir=self.docker_cfg["wheels_dir"],
            sandbox_scripts_dir=self.docker_cfg["sandbox_scripts_dir"],
            mem_limit=self.docker_cfg["mem_limit"], cpus=self.docker_cfg["cpus"],
            host_workdir=self.host_workdir,   # bind-mount -- см. FIXME в sim/container.py
        )
        self.container.start()
        tgz = os.path.join(self.snapshots_dir, f"{self.task['instance_id']}.tgz")
        bootstrap_task(self.container, tgz, self.task["repo"], self.task["base_commit"])

        self.ctx = SwegemmaContext(
            workspace=self.host_workdir,  # благодаря bind-mount это = /workspace внутри контейнера
            budget=self.budget, harness=self.harness,
            graph_path=self._maybe(self.graphs_dir, ".json"),
            embeddings_path=self._maybe(self.embeddings_dir, ".npz"),
        )
        self.ctx.start_agent_session()
        self.consecutive_nudges = 0
        self.done = False
        prompt = build_agent_prompt(self.task, "/workspace", self.budget, self.harness,
                                     has_graph=self.ctx.has_code_intel(),
                                     enable_sandbox_testing=self.enable_sandbox_testing)
        return prompt

    def _maybe(self, d, ext):
        if not d:
            return None
        p = os.path.join(d, self.task["instance_id"] + ext)
        return p if os.path.exists(p) else None

    def call_tool(self, tool_name, tool_args):
        fn = TOOL_REGISTRY.get(tool_name)
        if fn is None:
            return json.dumps({"status": "error", "error_message": f"Unknown tool {tool_name}"})
        obs = fn(self.ctx, **tool_args)
        if tool_name == "submit_patch":
            self.ctx.patch_submitted = True
        return obs

    def nudge_message(self, last_assistant_text, finish_reason, had_tool_call):
        if had_tool_call:
            self.consecutive_nudges = 0
            return None
        self.consecutive_nudges += 1
        if self.consecutive_nudges > MAX_NUDGES:
            self.done = True
            return None
        if last_assistant_text and "<|tool_call|>" in last_assistant_text \
                and "<|/tool_call|>" not in last_assistant_text:
            return NUDGE_UNCLOSED_TOOL_CALL
        if finish_reason in ("length", "MAX_TOKENS"):
            return NUDGE_MAX_TOKENS
        return NUDGE_NORMAL_STOP

    def finalize(self):
        if not self.ctx.patch_submitted:
            _submit_patch_fn(self.ctx)
        return self.ctx.submitted_patch

    def cleanup(self):
        self.container.stop()
        import shutil
        shutil.rmtree(self.host_workdir, ignore_errors=True)
