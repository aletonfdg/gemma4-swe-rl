# memory_utils.py -- импортировать ПЕРВЫМ в любом entrypoint, до torch/vllm/transformers
import os
import yaml

_CFG = yaml.safe_load(open("config.yaml"))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _CFG["memory"]["alloc_conf"]

import gc
import torch


def report_memory(tag: str):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"[mem:{tag}] allocated={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB")


def hard_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def reset_peak():
    torch.cuda.reset_peak_memory_stats()
