#!/usr/bin/env python3
# inspect_sample_submission.py
"""
Запустить ПЕРВЫМ после скачивания датасета — распечатывает содержимое ВСЕХ файлов
sample_submission/, чтобы зафиксировать точную схему agent.yaml / eval_config.yaml /
sub_agents/*.yaml / configs/sampling.yaml / prompts/*.md / adapters/.

Без этого export_submission.py в create_repo.py остаётся реконструкцией "по памяти"
из README, а не точной копией baseline-схемы.
"""
import os
import sys
import json

DATASET_ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SAMPLE_DIR = os.path.join(DATASET_ROOT, "sample_submission")


def dump_tree(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fname in sorted(filenames):
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root)
            print("=" * 100)
            print(f"### {rel}")
            print("=" * 100)
            try:
                if fname.endswith((".yaml", ".yml", ".md", ".txt", ".json")):
                    with open(full, "r", encoding="utf-8", errors="replace") as f:
                        print(f.read())
                elif fname.endswith(".safetensors"):
                    size = os.path.getsize(full)
                    print(f"<binary safetensors, {size/1e6:.1f} MB>")
                else:
                    print(f"<binary or unrecognized file, {os.path.getsize(full)} bytes>")
            except Exception as e:
                print(f"<error reading file: {e}>")
            print()


def dump_top_level_layout(root):
    print("TOP-LEVEL DATASET LAYOUT (relative to", root, "):")
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        kind = "dir" if os.path.isdir(full) else "file"
        print(f"  [{kind}] {entry}")
    print()


if __name__ == "__main__":
    if not os.path.isdir(SAMPLE_DIR):
        print(f"НЕ НАЙДЕНО: {SAMPLE_DIR}. Проверьте DATASET_ROOT (передайте первым аргументом).")
        dump_top_level_layout(DATASET_ROOT)
        sys.exit(1)

    dump_top_level_layout(DATASET_ROOT)
    print(f"Содержимое {SAMPLE_DIR}:\n")
    dump_tree(SAMPLE_DIR)

    # Отдельно -- зафиксировать список adapters (если есть) и их реальный adapter_config.json
    adapters_dir = os.path.join(SAMPLE_DIR, "adapters")
    if os.path.isdir(adapters_dir):
        print("=" * 100)
        print("ADAPTERS SUMMARY")
        print("=" * 100)
        for name in sorted(os.listdir(adapters_dir)):
            cfg_path = os.path.join(adapters_dir, name, "adapter_config.json")
            if os.path.exists(cfg_path):
                print(f"- {name}:")
                print(json.dumps(json.load(open(cfg_path)), indent=2))