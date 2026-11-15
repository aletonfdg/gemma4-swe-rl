# Gemma 4 SWE-RL: Fix the Failure, Not the Trajectory

> Minimal-difference preference adapters for a 31B on-device coding agent.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model: Gemma 4 31B QAT](https://img.shields.io/badge/Model-Gemma%204%2031B%20QAT-orange.svg)](https://huggingface.co/google)

Code and training logs for our paper in the **Google - The Gemma 4 Developer Agent Paper Track**:
https://www.kaggle.com/competitions/gemma-4-developer-agent-paper/writeups/fix-the-failure-not-the-trajectory

---

## Summary

Auditing 312 episodes of a Gemma 4 31B agent, we found that **28% of `edit_file` calls contain
a literal `\n` instead of a real newline, and 98% of those calls fail** (only 6 of 247 applied).
SFT on our own solved trajectories did not help (the defect rose to 48%).

We train a LoRA adapter on **188 minimal-difference preference pairs**: chosen and rejected share
the system prompt, history and reasoning, and differ only in one tool call's arguments. The loss is
a pairwise preference term plus an NLL anchor on the chosen call. On locally generated calls the
defect drops from 28% to 0%. The public leaderboard score moves from 0.15 to 0.20 (and from 0.15
to 0.17 under a second prompt). These are one-to-three-task differences on 58 tasks, within
leaderboard noise, so we read them as **consistent with a gain, not as proof**.

Cost: decoding is about 20% slower (27.5 vs 34 tok/s).

## Results

| Configuration | Prompt | Literal `\n` in `edit_file` | Held-out pair acc. | Leaderboard (58 tasks) |
|---|---|:---:|:---:|:---:|
| Base Gemma 4 31B-it (`v3`) | P1 | 28% | 0.57 | 0.15 |
| SFT adapter (171 per-turn examples) | - | 48% | - | - |
| Prompt rule "use real newlines" (`v4`) | P1 | - | - | 0.12 |
| `pref_edit_v1` (`v5`) | P1 | 0% | 1.00 | 0.20 |
| Base (`v16`) | P2 | - | - | 0.15 |
| `pref_edit` retrained under P2 (`pe16`) | P2 | - | — | 0.17 |

Defect rate and pair accuracy are small local checks (22 held-out pairs). Resubmitting the same
package varies by roughly 3-5 points on the leaderboard, so the defect rate is the main evidence,
not the leaderboard score. Prompts P1 and P2 are two different system prompts (see `submission_*`).

## Repository structure

```text
.
├── train/
│   ├── build_pref_data.py        # real_fix (87) and synthetic (101) pairs (paper §4.1)
│   ├── build_pref_onpolicy.py    # on-policy pairs from sometimes-solved tasks (§5.3)
│   ├── pref_lora.py              # pairwise LoRA training, reference-free and reference variants (§4.2)
│   └── manual_lora_qat.py        # LoRA on a QAT w4a16 checkpoint, one A100 80GB (§4.3)
├── evaluation/                   # training logs and analysis script (see evaluation/README.md)
├── data/                         # task IDs and our agent's boolean results only (see data/README.md)
├── sim/                          # Docker sandbox server (mimo_server.py) and episode analysis
├── sft/                          # per-turn SFT baseline (§5.1, §6)
├── rl/, reward/, synth/          # reward-weighted / policy-gradient experiments (§6), train_gigpo.py
├── eval/, prompts/two_phase/     # two-phase investigator agent (NOT reported in the paper)
├── skills/tmp-bridge/            # ADK setup skill linking <repo>/tmp
├── submission_v1_noadapter/ ... submission_v5s/   # submission packages (prompts, configs, adapter hooks)
├── tests/
├── colab_*.ipynb                 # Colab pipelines
└── requirements.txt
```

Submissions: `v3` prompt-only P1 (LB 0.15), `v4` prompt rules (0.12), `v5` with `pref_edit_v1`
(0.20), `v16` P2 prompt-only (0.15).

## Requirements

- Python 3.10+, PyTorch with CUDA
- 1x NVIDIA A100 80 GB for training (peak about 72 GB at 8,192 tokens)
- Docker for sandbox rollouts

## Quick start

```bash
git clone https://github.com/aletonfdg/gemma4-swe-rl.git
cd gemma4-swe-rl
pip install -r requirements.txt
```

**1. Build preference pairs** from your own episode logs (task texts and derived episode logs are
not redistributed, see "Data"):

```bash
python train/build_pref_data.py --logs_dir <PATH_TO_YOUR_EPISODE_LOGS> --output_dir ./data/pairs
```

**2. Train the adapter** (about 4 GPU-hours):

```bash
python train/pref_lora.py \
  --model_name_or_path "google/gemma-4-31B-it-qat-w4a16-ct" \
  --data_path "./data/pairs/edit_pairs.json" \
  --output_dir "./adapters/pref_edit_v1" \
  --lora_r 32 --lora_alpha 64 --learning_rate 3e-5
```

Other settings: beta = 0.1, NLL weight = 0.2, grad-accum 8, one epoch, each pair repeated twice.

**QAT loading.** The paper used `run_compressed=False`. Newer transformers versions deprecate it
in favour of `dequantize=True` (the training logs show this warning). The model decompresses to
bf16 (about 60 GB).

## Data

Competition task texts, hidden tests, gold patches, raw agent episodes and adapter weights are
**not** in this repository (competition rules). `data/` holds only task IDs and boolean results
of our own agent. The pipeline regenerates the adapters from your own episode logs.

## Not covered in the paper

Experimental code whose results are not reported: the two-phase investigator
(`eval/`, `prompts/two_phase/`, `inv_v1/inv_v2` logs), `pref_fix` pairs, and `submission_v5s`.
Policy-gradient and GiGPO code (`rl/`, `train_gigpo.py`) is only summarized in the paper's
negative results. Treat all of it as exploratory.

## Limitations

- Two paired leaderboard comparisons on 58 tasks, differing by one to three tasks.
- The census and `real_fix` pairs come from logs on the 104 public tasks; overlap with the
  leaderboard tasks is not verified, so part of the gain could be task familiarity.
- Pair type and the reference term vary together in the on-policy ablation (§5.3).
- The `mimo50` local set was selected from a run of `pref_edit_v1`, so comparisons on it are
  biased against that adapter (regression to the mean).
- The 0% defect rate and 1.00 pair accuracy come from a small local check (22 held-out pairs).
- One model, one harness, one defect with a verified fix.

## Citation

```bibtex
@misc{vishnevskiy2026fixfailure,
  author       = {Vishnevskiy, Aleksey},
  title        = {Fix the Failure, Not the Trajectory: Minimal-difference preference adapters for a 31B on-device coding agent},
  year         = {2026},
  howpublished = {\url{https://github.com/aletonfdg/gemma4-swe-rl}}
}
```

## License

Apache 2.0. Gemma 4 weights are used under their own license.
