# model_setup.py
import memory_utils  # noqa: F401 -- должен импортироваться первым
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

CFG = yaml.safe_load(open("config.yaml"))
MODEL_ID = CFG["model"]["id"]


def load_base_model_compressed():
    """Загружает модель строго в compressed (packed int4) режиме.
    NB: локально для тренировки используем полный HF id с префиксом, если репозиторий
    требует его для скачивания; в agent.yaml / ModelRegistry харнесса используется
    короткий alias без префикса -- см. export_submission.py."""
    hf_id = MODEL_ID if "/" in MODEL_ID else f"google/{MODEL_ID}"
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation=CFG["model"]["attn_implementation"],
    )

    quant_cfg = getattr(model.config, "quantization_config", None)
    if quant_cfg is None:
        raise RuntimeError(
            "quantization_config не найден -- проверьте пакет compressed-tensors "
            "и config.json модели."
        )
    quant_cfg.run_compressed = True
    model.config.quantization_config = quant_cfg

    n_compressed = sum(
        1 for _, m in model.named_modules() if m.__class__.__name__ == "CompressedLinear"
    )
    print(f"[model_setup] CompressedLinear modules: {n_compressed}")
    if n_compressed == 0:
        raise RuntimeError(
            "Модель уже декомпрессирована при загрузке -- "
            "проверьте версию transformers/compressed-tensors."
        )

    model.requires_grad_(False)
    return model


def load_tokenizer():
    hf_id = MODEL_ID if "/" in MODEL_ID else f"google/{MODEL_ID}"
    return AutoTokenizer.from_pretrained(hf_id)


def apply_lora_with_safe_checkpointing(model, lora_path: str | None = None):
    if lora_path is None:
        lora_cfg = LoraConfig(
            r=CFG["lora"]["r"],
            lora_alpha=CFG["lora"]["alpha"],
            lora_dropout=CFG["lora"]["dropout"],
            target_modules=CFG["lora"]["target_modules"],
            task_type="CAUSAL_LM",
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
    else:
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def build_optimizer(model, lr: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if CFG["memory"]["use_8bit_optimizer"]:
        from bitsandbytes.optim import AdamW8bit
        return AdamW8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


from rl.logprobs import token_logprobs as compute_logprobs  # noqa: E402 -- chunked, только токены ассистента
from rl.logprobs import ref_token_logprobs as compute_ref_logprobs  # noqa: E402
