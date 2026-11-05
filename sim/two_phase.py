# sim/two_phase.py -- двухфазный эпизод: следователь (только чтение) -> исправитель (с диагнозом в промпте)
"""
Зачем: в 47% проваленных эпизодов база открывает нужный файл (в среднем на 32-50 с) и правильно называет
причину, а потом ~17 вызовов читает и проверяет дальше и не правит. Двухфазный агент делает переход к
правке частью устройства, а не просьбой в промпте:

  фаза 1 (следователь): инструменты только для чтения; закончить можно одним способом -- ответить текстом
           (диагноз: файл, строки, причина, правка). Нет edit/write/submit.
  фаза 2 (исправитель): все инструменты; системный промпт = боевой промпт + диагноз; история -- только
           условие задачи (как include_contents: none в ADK), без 20 ходов чтения следователя.

Бюджет (вызовы, время, ходы) у фаз общий -- как у SequentialAgent на Kaggle (один SwegemmaContext).
На Kaggle это agent.yaml с agent_class: SequentialAgent, output_key: diagnosis и {diagnosis} в промпте
исправителя. Здесь -- его эмуляция в нашем раннере, чтобы измерить эффект до отправки.
"""
import json

from reward.anchor import anchor_key
from reward.reward_shaping import RewardTracker
from sim.tool_schemas import ROOT_TOOL_SCHEMAS

INVESTIGATOR_TOOLS = ("read_file", "run_command", "get_status")
FIXER_TOOLS = ("read_file", "run_command", "edit_file", "write_file", "get_status", "submit_patch")


class SystemClient:
    """Подставляет свой системный промпт перед сообщениями (базовый клиент -- без системного)."""

    def __init__(self, base, system):
        self.base, self.system = base, system

    def chat(self, messages, tools, **kw):
        return self.base.chat([{"role": "system", "content": self.system}] + list(messages), tools, **kw)


def _tools(names):
    return [t for t in ROOT_TOOL_SCHEMAS if t["function"]["name"] in names]


def _assistant(out):
    msg = {"role": "assistant", "content": out["content"]}
    if out.get("reasoning"):
        msg["thought"] = out["reasoning"]
    if out["tool_calls"]:
        msg["tool_calls"] = [{"id": tc["id"], "type": "function",
                              "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                             for tc in out["tool_calls"]]
    return msg


def run_investigator(env, client, prompt, temperature, tracker, steps, stats):
    """-> (messages, diagnosis). Заканчивается на первом ответе без вызова инструмента или по бюджету."""
    task_id = env.task["instance_id"]
    allowed = set(INVESTIGATOR_TOOLS)
    tools = _tools(allowed)
    messages = [{"role": "user", "content": prompt}]
    while not env.done and env.ctx.turns_used < env.budget.turns:
        out = client.chat(messages, tools, temperature=temperature)
        env.ctx.turns_used += 1
        stats["inv_turns"] += 1
        messages.append(_assistant(out))
        if not out["tool_calls"]:
            stats["inv_stopped"] = True
            return messages, (out["content"] or "").strip()
        for tc in out["tool_calls"]:
            if tc["name"] not in allowed:   # на Kaggle вызов чужого инструмента -- фатальная ошибка ADK
                stats["inv_forbidden"] += 1
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(
                    {"status": "error", "error_message": f"{tc['name']} is not available in the investigation "
                     "step. Reply with your diagnosis as plain text to hand over to the fix step."})})
                continue
            try:
                args = json.loads(tc["arguments"])
            except Exception:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Invalid JSON arguments."})
                continue
            obs = env.call_tool(tc["name"], args)
            stats["inv_calls"] += 1
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
            steps.append(dict(old_logprobs=[], reward=tracker.on_step(tc["name"], args, obs),
                              anchor=anchor_key(task_id, tc["name"], obs)))
        if env.ctx.budget_exhausted():
            env.done = True
    return messages, ""


def run_fixer(env, client, prompt, temperature, tracker, steps, stats):
    task_id = env.task["instance_id"]
    tools = _tools(FIXER_TOOLS)
    messages = [{"role": "user", "content": prompt}]
    while not env.done and env.ctx.turns_used < env.budget.turns:
        out = client.chat(messages, tools, temperature=temperature)
        env.ctx.turns_used += 1
        messages.append(_assistant(out))
        if not out["tool_calls"]:
            nudge = env.nudge_message(out["content"], out["finish_reason"], had_tool_call=False)
            if nudge:
                messages.append({"role": "user", "content": nudge})
            continue
        env.nudge_message(out["content"], out["finish_reason"], had_tool_call=True)
        for tc in out["tool_calls"]:
            try:
                args = json.loads(tc["arguments"])
            except Exception:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": "Invalid JSON arguments."})
                continue
            obs = env.call_tool(tc["name"], args)
            if tc["name"] in ("edit_file", "write_file") and stats["fix_first_edit_call"] is None:
                stats["fix_first_edit_call"] = stats["fix_calls"] + 1
            stats["fix_calls"] += 1
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": obs})
            steps.append(dict(old_logprobs=[], reward=tracker.on_step(tc["name"], args, obs),
                              anchor=anchor_key(task_id, tc["name"], obs)))
            if tc["name"] == "submit_patch":
                env.done = True
        if env.ctx.budget_exhausted():
            env.done = True
    return messages


def fixer_system(base_system, diagnosis):
    note = diagnosis or "(the investigation step ended without a diagnosis; investigate briefly yourself)"
    return (f"{base_system}\n\n## Diagnosis from the investigation step\n"
            "A previous step already read the code and wrote this diagnosis. Start from it: open only the lines "
            "you are about to change, apply the fix with edit_file, check it once, then call submit_patch.\n\n"
            f"{note}\n")


def run_episode_two_phase(env, make_client, investigator_system, fixer_base_system, temperature=0.2):
    """-> (steps, messages, passed, stats). make_client(system) -- свежий клиент модели с этим системным
    промптом (у каждой фазы свой: сжатие контекста хранит состояние своей истории)."""
    prompt = env.reset(0)
    tracker = RewardTracker(env.task)
    steps = []
    stats = {"inv_turns": 0, "inv_calls": 0, "inv_forbidden": 0, "inv_stopped": False, "diagnosis_chars": 0,
             "fix_calls": 0, "fix_first_edit_call": None, "inv_budget_left": None}
    inv_msgs, diagnosis = run_investigator(env, make_client(investigator_system), prompt,
                                           temperature, tracker, steps, stats)
    stats["diagnosis_chars"] = len(diagnosis)
    stats["inv_budget_left"] = not env.ctx.budget_exhausted()
    fix_msgs = []
    if not env.done:
        fix_msgs = run_fixer(env, make_client(fixer_system(fixer_base_system, diagnosis)), prompt,
                             temperature, tracker, steps, stats)
    hit_budget = not env.ctx.patch_submitted and env.ctx.budget_exhausted()
    patch = env.finalize()
    passed = env.verify(patch)
    final_r = tracker.terminal_reward(passed, hit_budget_limit=hit_budget, sandbox_crashed=getattr(env, "crashed", False))
    steps.append(dict(old_logprobs=[], reward=final_r, anchor=f"{env.task['instance_id']}|terminal"))
    env.cleanup()
    messages = inv_msgs + [{"role": "user", "content": f"[phase 2: fixer]\n{diagnosis}"}] + fix_msgs
    return steps, messages, passed, stats
