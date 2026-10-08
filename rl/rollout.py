# rl/rollout.py (заменить целиком)
import json
from sim.swe_env import SWEEnv
from sim.tool_schemas import ROOT_TOOL_SCHEMAS, ANALYZER_TOOL_SCHEMAS
from sim.verifier import verify_task
from reward.reward_shaping import RewardTracker
from reward.anchor import anchor_key

ANALYZER_SYSTEM_PROMPT = (
    "You are a specialized code analysis sub-agent. Your role is to inspect repository "
    "source files, trace symbol relationships, and identify the root cause of the reported "
    "issue.\n\nInstructions\nUse search_similar_code, get_code_neighbors, get_code_subgraph, "
    "and read_file to locate the exact functions, classes, and line ranges involved in the "
    "bug.\nProvide a concise, structured report containing:\nExact file paths and line "
    "numbers to modify.\nRoot cause explanation.\nRecommended minimal code change."
)


def _run_analyzer_subcall(env, llm_client, request_text, analyzer_lora_name,
                           temperature, max_sub_turns=10):
    """Прогоняет code_analyzer_agent как вложенный мини-эпизод. Возвращает (report_text, steps).
    ВАЖНО: вложенные tool-вызовы считаются как против ОБЩЕГО бюджета root-агента
    (env.ctx), т.к. харнесс выставляет единый budget на всю сессию, а не отдельно
    на под-агента -- см. HARNESS_README.md раздел 2.3 (sub_agents используют общий
    SwegemmaContext инструментов)."""
    sub_messages = [
        {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
        {"role": "user", "content": request_text},
    ]
    sub_steps = []
    for _ in range(max_sub_turns):
        if env.ctx.budget_exhausted():
            break
        out = llm_client.chat(sub_messages, ANALYZER_TOOL_SCHEMAS, temperature=temperature,
                               lora_name=analyzer_lora_name)
        sub_messages.append({"role": "assistant", "content": out["content"]})
        if not out["tool_calls"]:
            return out["content"], sub_steps
        for tc in out["tool_calls"]:
            try:
                args = json.loads(tc["arguments"])
            except Exception:
                continue
            obs = env.call_tool(tc["name"], args)
            sub_messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
            sub_steps.append(dict(
                old_logprobs=[lp for _, lp in out["token_logprobs"]],
                reward=0.0,  # reward для tool_lora считается отдельным прогоном (см. ниже)
                anchor=anchor_key(env.task["instance_id"], tc["name"], obs),
            ))
    return "No conclusive report produced before budget exhausted.", sub_steps


def run_episode(env: SWEEnv, task_idx: int, llm_client, snapshots_dir: str, docker_cfg,
                 temperature=0.9, main_lora_name="main_lora", analyzer_lora_name="tool_lora",
                 force_tight_budget: bool | None = None, collect_analyzer_steps=False):
    prompt = env.reset(task_idx, force_tight_budget=force_tight_budget)
    task_id = env.task["instance_id"]
    tracker = RewardTracker(env.task)

    messages = [{"role": "user", "content": prompt}]
    steps = []
    analyzer_steps_all = []
    turn = 0

    while not env.done and turn < env.budget.turns:
        out = llm_client.chat(messages, ROOT_TOOL_SCHEMAS, temperature=temperature,
                               lora_name=main_lora_name)
        env.ctx.turns_used += 1
        turn += 1

        assistant_msg = {"role": "assistant", "content": out["content"]}
        if out["tool_calls"]:
            assistant_msg["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in out["tool_calls"]
            ]
        messages.append(assistant_msg)

        had_tool_call = bool(out["tool_calls"])
        token_logprobs_flat = [lp for _, lp in out["token_logprobs"]]

        if not had_tool_call:
            nudge = env.nudge_message(out["content"], out["finish_reason"], had_tool_call=False)
            if nudge:
                messages.append({"role": "user", "content": nudge})
            steps.append(dict(old_logprobs=token_logprobs_flat, reward=0.0,
                               anchor=f"{task_id}|no_tool_call"))
            continue

        env.nudge_message(out["content"], out["finish_reason"], had_tool_call=True)

        for tc in out["tool_calls"]:
            try:
                args = json.loads(tc["arguments"])
            except Exception:
                r = tracker.on_malformed_tool_call()
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                  "content": "Invalid JSON arguments."})
                steps.append(dict(old_logprobs=token_logprobs_flat, reward=r,
                                   anchor=f"{task_id}|bad_json"))
                continue

            if tc["name"] == "code_analyzer_agent":
                report, sub_steps = _run_analyzer_subcall(
                    env, llm_client, args.get("request", ""), analyzer_lora_name,
                    temperature=min(temperature, 0.3),  # аналитик обучается на низкой T (см. sampling.yaml=0.2)
                )
                obs = json.dumps({"status": "ok", "report": report}, ensure_ascii=False)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
                steps.append(dict(old_logprobs=token_logprobs_flat, reward=0.0,
                                   anchor=f"{task_id}|code_analyzer_call"))
                if collect_analyzer_steps:
                    analyzer_steps_all.append(sub_steps)
                continue

            obs = env.call_tool(tc["name"], args)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})

            r = tracker.on_step(tc["name"], args, obs)
            a_key = anchor_key(task_id, tc["name"], obs)
            steps.append(dict(old_logprobs=token_logprobs_flat, reward=r, anchor=a_key))

            if tc["name"] == "submit_patch":
                env.done = True

        if env.ctx.budget_exhausted():
            env.done = True

    hit_budget = not env.ctx.patch_submitted and env.ctx.budget_exhausted()
    patch = env.finalize()

    verifier_passed = verify_task(env.task, patch, snapshots_dir, docker_cfg)

    final_r = tracker.terminal_reward(verifier_passed, hit_budget_limit=hit_budget, sandbox_crashed=False)
    if steps:
        steps[-1]["reward"] += final_r
    else:
        steps.append(dict(old_logprobs=[], reward=final_r, anchor=f"{task_id}|terminal"))

    env.cleanup()
    if collect_analyzer_steps:
        return steps, messages, verifier_passed, analyzer_steps_all
    return steps, messages, verifier_passed