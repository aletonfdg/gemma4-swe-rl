#!/bin/bash
set -e
python inspect_harness.py
python sft/collect_trajectories.py
python sft/train_sft.py
