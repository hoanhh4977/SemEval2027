"""Domain-adaptive pre-training (continued masked-LM) on the unlabeled task text.

Top EDOS (SemEval-2023 Task 10) systems gained several F1 points by continuing MLM pre-training on
in-domain text before fine-tuning. Only *unlabeled text* is used (comments, titles, descriptions of
the training files + the FA development comments), never labels, so there is no label leakage.
The adapted checkpoint is cached once per backbone and shared by all seeds.

    python -m t8.dapt --backbone xlm-roberta-base --data data/task8 --out dapt_cache/xlm-roberta-base
"""
from __future__ import annotations

import argparse
import math
import os
import random

import numpy as np
import torch
from transformers import (AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling)

from .data import load_task8


def collect_texts(data_dir: str):
    train, fa = load_task8(data_dir)
    texts = set()
    for df in (train, fa):
        texts.update(df["yt_comment"].astype(str).str.strip().tolist())
        texts.update((df["yt_title"].astype(str).str.strip() + ". " + df["yt_description"].astype(str).str.strip()
                      ).drop_duplicates().tolist())
    return [t for t in texts if len(t) > 3]


def run_dapt(backbone: str, data_dir: str, out_dir: str, epochs: int = 3, bs: int = 8, lr: float = 4e-5,
             max_len: int = 128, seed: int = 0, log=print) -> str:
    if os.path.exists(os.path.join(out_dir, "DONE")):
        return out_dir
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(backbone)
    model = AutoModelForMaskedLM.from_pretrained(backbone).to(dev)
    max_len = min(max_len, getattr(model.config, "max_position_embeddings", max_len) - 4)
    texts = collect_texts(data_dir)
    enc = tok(texts, truncation=True, max_length=max_len)["input_ids"]
    coll = DataCollatorForLanguageModeling(tok, mlm=True, mlm_probability=0.15)
    steps = epochs * math.ceil(len(enc) / bs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / (0.06 * steps), max(0.0, (steps - s) / (0.94 * steps))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    log(f"DAPT {backbone}: {len(enc)} texts, {steps} steps")
    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(len(enc))
        tot, n = 0.0, 0
        for s in range(0, len(perm), bs):
            batch = coll([{"input_ids": enc[j]} for j in perm[s: s + bs]])
            batch = {k: v.to(dev) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            tot += float(loss)
            n += 1
        log(f"  DAPT ep{ep} mlm_loss={tot / max(n, 1):.4f}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    open(os.path.join(out_dir, "DONE"), "w").write("ok")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    a = ap.parse_args()
    run_dapt(a.backbone, a.data, a.out, epochs=a.epochs)
