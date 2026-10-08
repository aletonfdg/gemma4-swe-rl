# sft/train_sft.py
import memory_utils  # noqa
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from sim.tool_schemas import TOOL_SCHEMAS
from model_setup import (load_base_model_compressed, load_tokenizer,
                          apply_lora_with_safe_checkpointing, build_optimizer)

CFG = yaml.safe_load(open("config.yaml"))


class SFTDataset(Dataset):
    def __init__(self, path, tokenizer, max_len):
        self.examples = [json.loads(l) for l in open(path)]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]["messages"]
        text = self.tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=False)
        ids = self.tokenizer(text, add_special_tokens=False).input_ids

        mask = []
        for m in messages:
            seg = self.tokenizer.apply_chat_template([m], tools=TOOL_SCHEMAS, tokenize=False)
            seg_ids = self.tokenizer(seg, add_special_tokens=False).input_ids
            mask.extend([1 if m.get("role") == "assistant" else 0] * len(seg_ids))

        ids = ids[-self.max_len:]
        mask = mask[-self.max_len:]
        if len(mask) < len(ids):
            mask = mask + [0] * (len(ids) - len(mask))
        return {"input_ids": ids, "loss_mask": mask[:len(ids)]}


def collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((len(batch), max_len), dtype=torch.float32)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"])
        loss_mask[i, :L] = torch.tensor(b["loss_mask"])
    return input_ids, loss_mask


def sft_loss(model, input_ids, loss_mask):
    logits = model(input_ids=input_ids).logits[:, :-1]
    targets = input_ids[:, 1:]
    mask = loss_mask[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = -token_logp
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def main():
    model = load_base_model_compressed()
    model = apply_lora_with_safe_checkpointing(model)
    tokenizer = load_tokenizer()
    optimizer = build_optimizer(model, CFG["sft"]["lr"])

    dataset = SFTDataset(CFG["paths"]["sft_data_out"], tokenizer, CFG["model"]["max_seq_length"])
    loader = DataLoader(dataset, batch_size=CFG["sft"]["batch_size"], shuffle=True,
                         collate_fn=lambda b: collate(b, tokenizer.pad_token_id or 0))

    model.train()
    step = 0
    for epoch in range(CFG["sft"]["epochs"]):
        optimizer.zero_grad()
        for i, (input_ids, loss_mask) in enumerate(loader):
            input_ids, loss_mask = input_ids.to(model.device), loss_mask.to(model.device)
            loss = sft_loss(model, input_ids, loss_mask)
            (loss / CFG["sft"]["grad_accum_steps"]).backward()

            if (i + 1) % CFG["sft"]["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()
                memory_utils.hard_cleanup()
                step += 1
                if step % 10 == 0:
                    print(f"epoch={epoch} step={step} loss={loss.item():.4f}")
                    memory_utils.report_memory("sft")

        model.save_pretrained(f"{CFG['paths']['sft_adapter_out']}_epoch{epoch}")

    model.save_pretrained(CFG["paths"]["sft_adapter_out"])
    print(f"SFT done -> {CFG['paths']['sft_adapter_out']}")


if __name__ == "__main__":
    main()
