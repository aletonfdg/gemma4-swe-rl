# Gemma 4 SWE-RL: Fix the Failure, Not the Trajectory

> **Minimal-difference preference adapters for a 31B on-device coding agent: a gain where imitation fine-tuning gave none**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model: Gemma 4 31B QAT](https://img.shields.io/badge/Model-Gemma%204%2031B%20QAT-orange.svg)]()

Official implementation and training scripts for our paper submitted to the **Google - The Gemma 4 Developer Agent Paper Track**.

---

## Abstract

A common view in the Gemma 4 Developer Agent competition is that LoRA adapters do not help: SFT and distillation adapters failed to beat prompt-only agents, and adapters slow down decoding. In this work, we reproduce the negative result for standard SFT and show that it does not extend to every adapter. 

By auditing 312 agent episodes, we discovered a major mechanical defect: **28% of `edit_file` calls contain literal `\n` string escapes instead of real newlines**, causing 98% of those calls to fail. To fix this without diluting the gradient over thousands of well-formed reasoning tokens, we introduce **minimal-difference preference adapters** trained on 188 pairs that share the entire context and reasoning, differing *only* in the problematic tool call arguments. 

Using an NLL-anchored pairwise loss, our preference adapter completely eliminates the format defect (28% $\to$ 0%) and improves the public leaderboard score from **0.15 to 0.20**.

---

## Repository Structure

```text
.
├── train/
│   ├── build_pref_data.py      # Builds real_fix and synthetic minimal-difference pairs
│   ├── build_pref_onpolicy.py  # Constructs on-policy preference pairs from sometimes-solved tasks
│   ├── pref_lora.py            # Reference-free and reference-based pairwise LoRA training script
│   └── manual_lora_qat.py      # QAT w4a16 LoRA workaround for transformers 5.x on 1x A100 80GB
├── sim/
│   └── mimo_server.py          # Docker sandbox server for local environment rollouts
└── notebooks/
    ├── colab_pref.ipynb        # Interactive pipeline for preference data curation
    └── colab_pref_onpolicy.ipynb # On-policy rollout collection notebook
```
---

## ⚙️ Key Technical Highlights
 * Failure Census (§3): Systematic analysis of 312 agent episodes revealing that formatting defects (\n escapes) and infinite test-script creation loops (/tmp/repro*.py) dominate 31B agent failures.
 * Minimal-Difference Pairs (§4.1): Pairs sharing the system prompt, trimmed history (\le 8,192 tokens), and reasoning block (reasoning_content), isolating the gradient updates strictly to the defect.
 * QAT w4a16 LoRA Workaround (§4.3): A single-GPU memory recipe for transformers 5.x and compressed-tensors that decompresses packed weights via run_compressed=False and computes gradients in a memory-efficient three-pass scheme, fitting within 72 GB VRAM on an A100 80GB.

---

## Quick Start & Reproduction
Prerequisites
 * Python 3.10+
 * PyTorch with CUDA support
 * 1x NVIDIA A100 (80GB VRAM recommended for QAT LoRA training)
1. Installatio
```bash
git clone [https://github.com/aletonfdg/gemma4-swe-rl.git](https://github.com/aletonfdg/gemma4-swe-rl.git)
cd gemma4-swe-rl
pip install -r requirements.txt # or uv sync
```
2. Build Preference Datasets
```bash
python train/build_pref_data.py --logs_dir ./logs --output_dir ./data/pairs
```
3. Train Preference Adapter
```bash
python train/pref_lora.py \
    --model_name_or_path "google/gemma-4-31b-it-qat-w4a16-ct" \
    --data_path "./data/pairs/edit_pairs.json" \
    --output_dir "./adapters/pref_edit_v1" \
    --lora_r 32 \
    --lora_alpha 64 \
    --learning_rate 3e-5
```
---

## 📊 Summary of Results
| Model / Configuration | Edit Defect Rate (\text{literal } \textbackslash n) | Public Leaderboard Score (58 tasks) |
|---|---|---|
| Base Gemma 4 31B-it (Prompt P1) | 28% | 0.15 |
| Standard SFT Adapter | 48% | — |
| Our Preference Adapter (pref_edit_v1) | 0% | 0.20 |

---

## 📖 Citation

If you build upon this work, use our minimal-difference preference pairs, or find our findings useful in your research, please cite it as follows:

```bibtex
@misc{aleton2026fixfailure,
  author = {Aleksey Vishnevskiy},
  title = {Fix the Failure, Not the Trajectory: Minimal-difference preference adapters for a 31B on-device coding agent},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/aletonfdg/gemma4-swe-rl](https://github.com/aletonfdg/gemma4-swe-rl)}}
}