# export_submission.py (ЗАМЕНИТЬ ЦЕЛИКОМ)
import os
import shutil
import zipfile
import argparse
import json

MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"

AGENT_YAML = f"""\
name: swe_baseline_agent
model: {MODEL_ALIAS}
adapter: main_lora
instruction: !include prompts/system.md
tools:
  - run_command
  - read_file
  - edit_file
  - write_file
  - get_status
  - submit_patch
  - get_code_neighbors
  - search_similar_code
  - get_code_subgraph
  - agent_tool:
      config_path: sub_agents/code_analyzer.yaml
      skip_summarization: true
generate_content_config: !include configs/sampling.yaml
"""

# Подтверждено дословно пользователем (sample_submission/sub_agents/code_analyzer.yaml)
CODE_ANALYZER_YAML = f"""\
name: code_analyzer_agent
description: Analyzes repository source files and symbol graphs to locate root causes.
model: {MODEL_ALIAS}
adapter: tool_lora
instruction: !include ../prompts/analyzer.md
tools:
  - read_file
  - search_similar_code
  - get_code_neighbors
  - get_code_subgraph
generate_content_config: !include ../configs/sampling.yaml
"""

# Подтверждено дословно пользователем (sample_submission/configs/sampling.yaml)
# ОДИН файл, используется и root-агентом, и code_analyzer -- НЕ два разных.
SAMPLING_YAML = """\
temperature: 0.2
top_p: 0.95
max_output_tokens: 16384
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
"""

# Подтверждено дословно пользователем (sample_submission/prompts/analyzer.md)
ANALYZER_PROMPT = """\
You are a specialized code analysis sub-agent. Your role is to inspect repository source \
files, trace symbol relationships, and identify the root cause of the reported issue.

Instructions
Use search_similar_code, get_code_neighbors, get_code_subgraph, and read_file to locate the \
exact functions, classes, and line ranges involved in the bug.
Provide a concise, structured report containing:
Exact file paths and line numbers to modify.
Root cause explanation.
Recommended minimal code change.
"""

SYSTEM_PROMPT = """You are an autonomous software engineering agent.
Navigate the repository under /workspace using the available tools, understand the
problem statement, implement a fix, verify it with targeted tests, and call
submit_patch when finished. Think briefly, then act -- prefer small incremental edits.
For broad code exploration or root-cause localization, delegate to the code_analyzer_agent
tool instead of reading many files directly yourself -- it will return exact file paths,
line numbers, and a recommended minimal change, keeping your own context window focused
on applying and verifying the fix.
Do not modify test files or pytest.ini/conftest.py/pyproject.toml -- such changes are
discarded before grading. Create scratch/reproduction scripts only under /tmp, and remove
them before calling submit_patch (any stray file left under /workspace is included in your
diff)."""

# NOTE: sample_submission's own eval_config.yaml uses an extremely tight demo budget
# (timeout_seconds=60, max_tool_calls=10, max_time_minutes=1, max_turns=50) -- почти
# наверняка это smoke-test значение, а не рекомендация. Эмпирически подобрать через
# `swegemma eval --max-time-minutes ... --max-tool-calls ...` на train split, замеряя
# resolution_rate vs бюджет, и держать total time в пределах 12h / ~120 задач с запасом
# на container setup. Текущие значения -- осторожная стартовая гипотеза, НЕ финал.
EVAL_CONFIG_YAML = """\
evaluation:
  timeout_seconds: 120
  max_tool_calls: 25
  max_time_minutes: 4
  max_turns: 60
"""


def validate_adapter_config(adapter_dir: str, name: str):
    """Отсеивает placeholder-адаптеры до упаковки: ловит пустые/заглушечные checkpoint'ы."""
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    weights_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"[{name}] adapter_config.json не найден в {adapter_dir}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[{name}] adapter_model.safetensors не найден в {adapter_dir}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    if cfg.get("peft_type") != "LORA":
        raise ValueError(f"[{name}] peft_type={cfg.get('peft_type')!r}, ожидался 'LORA'")
    if cfg.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"[{name}] task_type={cfg.get('task_type')!r}, ожидался 'CAUSAL_LM'")

    layers_restricted = cfg.get("layers_to_transform")
    if layers_restricted is not None:
        # PEFT допускает и список слоёв, и одно число -- len() на int упал бы TypeError
        n_layers = len(layers_restricted) if isinstance(layers_restricted, (list, tuple, set)) else 1
        if n_layers <= 1:
            print(f"ВНИМАНИЕ [{name}]: layers_to_transform={layers_restricted} -- похоже на "
                  f"placeholder-адаптер (как в sample_submission), а не на реально обученный. "
                  f"Проверьте, что вы упаковываете финальный checkpoint, а не SFT-заглушку.")

    target_modules = cfg.get("target_modules") or []
    n_modules = len(target_modules)
    if n_modules < 4:
        print(f"ВНИМАНИЕ [{name}]: target_modules содержит только {n_modules} модулей "
              f"({target_modules}) -- значительно меньше, чем ожидаемый набор "
              f"из 7 проекций. Убедитесь, что это не placeholder-адаптер.")

    size_mb = os.path.getsize(weights_path) / 1e6
    print(f"[{name}] r={cfg.get('r')} alpha={cfg.get('lora_alpha')} "
          f"modules={cfg.get('target_modules')} size={size_mb:.1f}MB")
    if size_mb < 1.0:
        print(f"ВНИМАНИЕ [{name}]: вес адаптера подозрительно мал ({size_mb:.2f} MB) -- "
              f"похоже на placeholder, а не на r=16/32 адаптер на все 7 проекций.")


def build_submission(main_adapter_dir: str, tool_adapter_dir: str, out_path: str = "submission.zip"):
    # Валидация ДО создания staging-каталога: иначе битый адаптер затирает предыдущую сборку
    validate_adapter_config(main_adapter_dir, "main_lora")
    validate_adapter_config(tool_adapter_dir, "tool_lora")

    staging = "_submission_staging"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(f"{staging}/prompts")
    os.makedirs(f"{staging}/configs")
    os.makedirs(f"{staging}/sub_agents")
    os.makedirs(f"{staging}/adapters/main_lora")
    os.makedirs(f"{staging}/adapters/tool_lora")

    open(f"{staging}/agent.yaml", "w").write(AGENT_YAML)
    open(f"{staging}/eval_config.yaml", "w").write(EVAL_CONFIG_YAML)
    open(f"{staging}/configs/sampling.yaml", "w").write(SAMPLING_YAML)
    open(f"{staging}/sub_agents/code_analyzer.yaml", "w").write(CODE_ANALYZER_YAML)
    open(f"{staging}/prompts/system.md", "w").write(SYSTEM_PROMPT)
    open(f"{staging}/prompts/analyzer.md", "w").write(ANALYZER_PROMPT)

    for fname in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(os.path.join(main_adapter_dir, fname), f"{staging}/adapters/main_lora/{fname}")
        shutil.copy(os.path.join(tool_adapter_dir, fname), f"{staging}/adapters/tool_lora/{fname}")

    if os.path.exists(out_path):
        os.remove(out_path)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(staging):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, staging))

    shutil.rmtree(staging)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"submission.zip создан: {out_path} ({size_mb:.1f} MB)")
    if size_mb > 3072:
        print("ВНИМАНИЕ: превышен лимит 3 GiB на unpacked submission!")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_adapter_dir", required=True)
    ap.add_argument("--tool_adapter_dir", required=True)
    ap.add_argument("--out", default="submission.zip")
    args = ap.parse_args()
    build_submission(args.main_adapter_dir, args.tool_adapter_dir, args.out)