# train_gigpo.py
import memory_utils  # noqa -- обязательно первым
import os
import yaml
import numpy as np
import torch

from model_setup import (load_base_model_compressed, load_tokenizer,
                          apply_lora_with_safe_checkpointing, build_optimizer)
from sim.swe_env import SWEEnv
from sim.tool_schemas import TOOL_SCHEMAS
from llm.vllm_server import VLLMServerProcess
from llm.llm_client import LLMClient
from rl.curriculum import CurriculumSampler
from rl.rollout import run_episode
from rl.gigpo_advantage import compute_gigpo_advantages
from rl.batch_builder import build_training_batch
from rl.gigpo_loss import gigpo_loss

CFG = yaml.safe_load(open("config.yaml"))
G = CFG["gigpo"]
CURRENT_LORA_NAME = "policy"


def main():
    model = load_base_model_compressed()
    model = apply_lora_with_safe_checkpointing(model, lora_path=CFG["paths"]["sft_adapter_out"])
    tokenizer = load_tokenizer()
    optimizer = build_optimizer(model, G["lr"])

    env = SWEEnv(CFG["paths"]["tasks_jsonl"], CFG["paths"]["snapshots_dir"],
                 CFG["paths"]["graphs_dir"], CFG["paths"]["embeddings_dir"])

    n_tasks = len(env.tasks)
    n_eval = max(1, int(n_tasks * G["held_out_fraction"]))
    import random
    rng = random.Random(42)
    shuffled = list(range(n_tasks))
    rng.shuffle(shuffled)
    eval_indices = set(shuffled[:n_eval])
    train_indices = [i for i in shuffled if i not in eval_indices]

    sampler = CurriculumSampler(len(train_indices))
    os.makedirs(CFG["paths"]["checkpoints_dir"], exist_ok=True)

    server = VLLMServerProcess()
    server.start(lora_paths={CURRENT_LORA_NAME: CFG["paths"]["sft_adapter_out"]})
    client = LLMClient(model=CFG["model"]["id"])

    for step in range(G["num_steps"]):
        local_idx = sampler.sample()
        task_idx = train_indices[local_idx]
        use_tight = (np.random.rand() < G["tight_budget_fraction"])

        # === ROLLOUT PHASE (vLLM-сервер уже поднят и обслуживает запросы) ===
        group_steps, group_messages, group_rewards = [], [], []
        for _ in range(G["group_size"]):
            steps, messages, passed = run_episode(
                env, task_idx, client, CFG["paths"]["snapshots_dir"], CFG["docker"],
                temperature=G["rollout_temperature"], lora_name=CURRENT_LORA_NAME,
                force_tight_budget=use_tight,
            )
            group_steps.append(steps)
            group_messages.append(messages)
            group_rewards.append(1.0 if passed else 0.0)
        memory_utils.hard_cleanup()

        sampler.update(local_idx, group_rewards)

        if np.std(group_rewards) == 0.0:
            print(f"step={step} task={env.tasks[task_idx]['instance_id']} "
                  f"zero-variance group ({group_rewards[0]}), skip update")
            continue

        advantages, macro_adv = compute_gigpo_advantages(
            group_steps, group_rewards, micro_weight=G["micro_weight"], gamma=G["gamma"])
        for steps, adv_list in zip(group_steps, advantages):
            for s, a in zip(steps, adv_list):
                s["advantage"] = a

        input_ids, loss_mask, old_logprobs, adv_tensor = build_training_batch(
            tokenizer, group_messages, group_steps, TOOL_SCHEMAS, model.device,
            max_len_cap=CFG["model"]["max_seq_length"])

        # === TRAIN PHASE (HF+PEFT backward; vLLM-сервер в это время простаивает) ===
        model.train()
        optimizer.zero_grad()
        kls = []
        mb = G["micro_batch_size"]
        n_micro = max(1, input_ids.size(0) // mb)
        for i in range(0, input_ids.size(0), mb):
            sl = slice(i, i + mb)
            loss, kl = gigpo_loss(
                model, input_ids[sl], loss_mask[sl], old_logprobs[sl], adv_tensor[sl],
                eps_low=G["clip_eps_low"], eps_high=G["clip_eps_high"], beta=G["kl_coef"],
            )
            (loss / n_micro).backward()
            kls.append(kl)
            del loss

        memory_utils.hard_cleanup()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()

        print(f"step={step} task={env.tasks[task_idx]['instance_id']} tight={use_tight} "
              f"reward_mean={np.mean(group_rewards):.3f} kl={np.mean(kls):.4f} "
              f"curriculum_pass_rate={sampler.mean_pass_rate():.3f} "
              f"peak_mem={torch.cuda.max_memory_allocated()/1e9:.1f}GB")

        if step % G["save_every"] == 0 and step > 0:
            ckpt_path = f"{CFG['paths']['checkpoints_dir']}/main_lora_step{step}"
            model.save_pretrained(ckpt_path)
            if not server.reload_lora(CURRENT_LORA_NAME, ckpt_path):
                server.restart_with_lora(CURRENT_LORA_NAME, ckpt_path)

        if step % G["eval_every"] == 0 and step > 0:
            evaluate(env, eval_indices, client, G)

    server.stop()


def evaluate(env, eval_indices, client, G):
    passed_count = 0
    for idx in eval_indices:
        _, _, passed = run_episode(env, idx, client, env.snapshots_dir, CFG["docker"],
                                    temperature=0.0, lora_name=CURRENT_LORA_NAME,
                                    force_tight_budget=True)
        passed_count += int(passed)
    rate = passed_count / max(1, len(eval_indices))
    print(f"[EVAL] held-out resolution rate (tight budget): {rate:.3f} "
          f"({passed_count}/{len(eval_indices)})")


if __name__ == "__main__":
    main()
