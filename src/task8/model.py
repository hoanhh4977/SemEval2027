"""Multi-task model: shared encoder (or LoRA-tuned decoder LLM) + task heads A / B / C.

Heads
  A  : stereotype yes/no                         (softmax, 2)
  B  : hate no / implicit / explicit             (softmax, 3)
  C-T: target type group / individual            (softmax, 2)  trained on hateful rows only
  C-I: 9 targeted identities                     (sigmoid, 9)  trained on hateful rows only

`hier=True` conditions the A and C heads on the (detached) softmax of head B, i.e. the hierarchy
"is it hateful?" -> "who is targeted?" is exposed to the model instead of being left implicit.
"""
from __future__ import annotations

import re
from typing import Dict, Set

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _load_encoder(cfg: dict):
    name = cfg["backbone"]
    if cfg["arch"] != "decoder":
        # .float(): some checkpoints (e.g. mDeBERTa) are stored in fp16 and newer transformers keep that dtype,
        # but mixed-precision training needs float32 master weights (autocast does the fp16 compute).
        return AutoModel.from_pretrained(name).float(), False
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    kw = {}
    quant = bool(cfg.get("load_4bit"))
    if quant:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": torch.cuda.current_device()}
    enc = AutoModel.from_pretrained(name, torch_dtype=torch.float16, **kw)
    if quant:
        enc = prepare_model_for_kbit_training(enc, use_gradient_checkpointing=True)
    elif cfg.get("grad_ckpt"):
        enc.gradient_checkpointing_enable()
        enc.enable_input_require_grads()
    enc = get_peft_model(enc, LoraConfig(r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32),
                                         lora_dropout=0.05, target_modules=LORA_TARGETS, bias="none"))
    for p in enc.parameters():  # GradScaler cannot unscale fp16 params
        if p.requires_grad:
            p.data = p.data.float()
    return enc, quant


class MTModel(nn.Module):
    def __init__(self, cfg: dict, tasks: Set[str]):
        super().__init__()
        self.cfg, self.tasks = cfg, set(tasks)
        self.encoder, self.quantized = _load_encoder(cfg)
        H = AutoConfig.from_pretrained(cfg["backbone"]).hidden_size
        self.pooling = "last" if cfg["arch"] == "decoder" else cfg["pooling"]
        self.drop = nn.Dropout(cfg["dropout"])
        self.hier = bool(cfg["hier"]) and "B" in self.tasks
        extra = 3 if self.hier else 0

        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, H), nn.GELU(), nn.Dropout(cfg["dropout"]), nn.Linear(H, o))

        if "B" in self.tasks:
            self.head_b = mlp(H, 3)
        if "A" in self.tasks:
            self.head_a = mlp(H + extra, 2)
        if "C" in self.tasks:
            self.head_t = mlp(H + extra, 2)
            self.head_i = mlp(H + extra, 9)
        # Kendall-style homoscedastic uncertainty weights, order: A, B, T, I
        self.log_vars = nn.Parameter(torch.zeros(4)) if cfg["weighting"] == "uncertainty" else None

    def trunk_params(self):
        return [(n, p) for n, p in self.encoder.named_parameters() if p.requires_grad]

    def head_params(self):
        enc_ids = {id(p) for p in self.encoder.parameters()}
        return [(n, p) for n, p in self.named_parameters() if id(p) not in enc_ids]

    def pool(self, hs, mask):
        if self.pooling == "cls":
            return hs[:, 0]
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).to(hs.dtype)
            return (hs * m).sum(1) / m.sum(1).clamp(min=1)
        idx = mask.sum(1) - 1  # last real token (right padding)
        return hs[torch.arange(hs.size(0), device=hs.device), idx]

    def forward(self, input_ids, attention_mask) -> Dict[str, torch.Tensor]:
        hs = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.drop(self.pool(hs, attention_mask).float())
        out = {"h": h}  # pooled embedding (used by the supervised-contrastive loss; ignored at inference)
        cond = None
        if "B" in self.tasks:
            out["b"] = self.head_b(h)
            if self.hier:
                cond = torch.softmax(out["b"], -1).detach()
        hc = torch.cat([h, cond], -1) if cond is not None else h
        if "A" in self.tasks:
            out["a"] = self.head_a(hc)
        if "C" in self.tasks:
            out["t"] = self.head_t(hc)
            out["i"] = self.head_i(hc)
        return out


def layer_index(name: str, n_layers: int) -> int:
    """Depth of a parameter for layer-wise LR decay: 0=embeddings ... n_layers+1=top."""
    m = re.search(r"layer\.(\d+)\.", name)
    if m:
        return int(m.group(1)) + 1
    if "embeddings" in name:
        return 0
    return n_layers + 1
