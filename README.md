# Gemma 4 SWE-RL: Fix the Failure, Not the Trajectory

> Minimal-difference preference adapters for a 31B on-device coding agent.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model: Gemma 4 31B QAT](https://img.shields.io/badge/Model-Gemma%204%2031B%20QAT-orange.svg)](#)

Code for our paper submitted to the **Google - The Gemma 4 Developer Agent Paper Track**.
Paper: https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/fix-the-failure-not-the-trajectory

## Summary

Auditing 312 episodes of a Gemma 4 31B agent, we found that **28% of `edit_file` calls
contain a literal `\n` instead of a real newline, and 98% of those calls fail** (6 of 247 applied).
We train a LoRA adapter on 188 *minimal-difference* preference pairs: chosen and rejected share
the whole context and reasoning and differ only in one tool call's arguments. The loss is a
pairwise preference term plus an NLL anchor on the chosen call.

- Defect rate on locally generated calls: **28% → 0%** (SFT adapter: 48%).
- Public leaderboard (58 tasks, 1 task ≈ 0.017): **0.15 → 0.20** under prompt P1 and
  **0.15 → 0.17** under prompt P2. These are one-to-three-task differences, within
  leaderboard noise, so we read them as *consistent with* a gain, not as proof.
- Cost: decoding is about 20% slower (27.5 vs 34 tok/s).

## Repository structure

```text
.
├── train/
│   ├── build_pref_data.py      # builds real_fix and synthetic minimal-difference pairs
│   ├── build_pref_onpolicy.py  # builds on-policy pairs from sometimes-solved tasks
│   ├── pref_lora.py            # pairwise LoRA training (reference-free and reference variants)
│   └── manual_lora_qat.py      # LoRA workaround for a QAT w4a16 checkpoint on one A100 80GB
├── sim/mimo_server.py          # Docker sandbox server for local rollouts
├── evaluation/                 # TODO: describe exactly what is stored here (see "Data" below)
├── data/                       # TODO: describe
├── submission_*/               # submission packages (v1 no-adapter ... v5)
├── colab_pref.ipynb            # preference data curation + training
├── colab_collect_sometimes.ipynb
├── colab_pref_onpolicy.ipynb
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch with CUDA
- 1x NVIDIA A100 80 GB for training (peak about 72 GB at 8,192 tokens)
- Docker for sandbox rollouts

## Quick start

```bash
git clone https://github.com/aletonfdg/gemma4-swe-rl.git
cd gemma4-swe-rl
pip install -r requirements.txt
```

**1. Build preference pairs** from your own episode logs:

```bash
python train/build_pref_data.py --logs_dir <PATH_TO_YOUR_EPISODE_LOGS> --output_dir ./data/pairs
```

**2. Train the adapter** (about 4 GPU-hours on an A100 80 GB):

```bash
python train/pref_lora.py \
  --model_name_or_path "google/gemma-4-31b-it-qat-w4a16-ct" \
  --data_path "./data/pairs/edit_pairs.json" \
  --output_dir "./adapters/pref_edit_v1" \
  --lora_r 32 --lora_alpha 64 --learning_rate 3e-5
```

Other settings from the paper: beta = 0.1, NLL weight = 0.2, grad-accum 8, one epoch with each pair repeated twice.

## Data

Pairs and logs derived from competition tasks are **not redistributed** (competition rules),
so the repository contains the builders, not the data. Adapter weights are not released;
the pipeline regenerates them from your own episode logs.
TODO: make sure `evaluation/` does not contain raw competition logs or task text.

## Results

| Configuration | Prompt | Literal `\n` in `edit_file` | Leaderboard (58 tasks) |
|---|---|---|---|
| Base Gemma 4 31B-it | P1 | 28% | 0.15 |
| SFT adapter (own solved runs) | – | 48% | – |
| `pref_edit_v1` | P1 | 0% | 0.20 |
| Base | P2 | – | 0.15 |
| `pref_edit` retrained under P2 | P2 | – | 0.17 |

The 0% rate comes from a small local check (22 held-out pairs for pair accuracy).
Repeated submission of the same package varies by about 3–5 points, so use the defect rate
and held-out pair accuracy as the main evidence, not the leaderboard.

## Limitations

- Two paired leaderboard comparisons on 58 tasks, differing by one to three tasks.
- The census and `real_fix` pairs come from logs on the public tasks; overlap with the
  leaderboard tasks is not verified.
- Pair type and the reference term vary together in the on-policy ablation.
- One model, one harness, one defect.

## Citation

```bibtex
@misc{aleton2026fixfailure,
  author       = {Aleksey Vishnevskiy},
  title        = {Fix the Failure, Not the Trajectory: Minimal-difference preference adapters for a 31B on-device coding agent},
  year         = {2026},
  howpublished = {\url{https://github.com/aletonfdg/gemma4-swe-rl}}
}
```

## License

Apache 2.0
