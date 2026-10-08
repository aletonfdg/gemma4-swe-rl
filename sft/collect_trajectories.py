# sft/collect_trajectories.py -- rejection sampling для warm-start (через настоящий vLLM-сервер)
import memory_utils  # noqa
import json
import yaml
from llm.vllm_server import VLLMServerProcess
from llm.llm_client import LLMClient
from sim.swe_env import SWEEnv
from rl.rollout import run_episode

CFG = yaml.safe_load(open("config.yaml"))


def main():
    server = VLLMServerProcess()
    server.start(lora_paths=None)   # teacher = базовая it-модель, без LoRA
    client = LLMClient(model=CFG["model"]["id"])

    env = SWEEnv(CFG["paths"]["tasks_jsonl"], CFG["paths"]["snapshots_dir"],
                 CFG["paths"]["graphs_dir"], CFG["paths"]["embeddings_dir"])

    out_f = open(CFG["paths"]["sft_data_out"], "w")
    kept, total = 0, 0

    try:
        for task_idx in range(len(env.tasks)):
            for _ in range(CFG["sft"]["rollouts_per_task"]):
                steps, messages, passed = run_episode(
                    env, task_idx, client, CFG["paths"]["snapshots_dir"], CFG["docker"],
                    temperature=CFG["sft"]["teacher_temperature"],
                )
                total += 1
                reward = 1.0 if passed else 0.0
                if reward >= CFG["sft"]["min_reward_to_keep"]:
                    out_f.write(json.dumps({
                        "instance_id": env.tasks[task_idx]["instance_id"],
                        "messages": messages,
                    }) + "\n")
                    kept += 1
            print(f"task {task_idx+1}/{len(env.tasks)} | kept={kept}/{total}")
    finally:
        out_f.close()
        server.stop()

    print(f"Done. Kept {kept}/{total} successful trajectories -> {CFG['paths']['sft_data_out']}")


if __name__ == "__main__":
    main()
