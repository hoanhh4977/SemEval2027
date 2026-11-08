"""Loss functions of the multi-task model.

Total loss = A + B + (target-type + identity)  [+ 0.1 * supervised-contrastive]
  * C terms are computed on hateful rows only (C is only defined / evaluated there).
  * `cw` (class weights) make rare classes count more; see class_weights().
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def class_weights(train_df: pd.DataFrame, mode: str, device) -> Dict[str, torch.Tensor]:
    """Inverse-frequency class weights. mode: 'none' | 'sqrt' (softened) | 'balanced' (full)."""
    if mode == "none":
        return {}
    power = 0.5 if mode == "sqrt" else 1.0

    def per_class(labels, n_classes):
        counts = np.bincount(labels, minlength=n_classes).astype(float) + 1.0  # +1 avoids division by zero
        w = (counts.sum() / (n_classes * counts)) ** power
        return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    hateful = train_df[train_df["y_b"] > 0]
    weights = {"a": per_class(train_df["y_a"].values, 2), "b": per_class(train_df["y_b"].values, 3)}
    if len(hateful):
        weights["t"] = per_class(hateful["y_t"].values, 2)
        y_identity = np.stack(hateful["y_i"].values)
        pos = y_identity.sum(0) + 1.0
        neg = len(y_identity) - y_identity.sum(0) + 1.0
        # pos_weight for BCE: rare identities (e.g. 'a', 'q', 'i') get a larger weight, capped at 20
        weights["i"] = torch.tensor(np.clip((neg / pos) ** power, 1.0, 20.0), dtype=torch.float32, device=device)
    return weights


def cross_entropy(logits, target, weight, focal_gamma: float):
    """Cross-entropy, optionally focal (down-weights easy examples when focal_gamma > 0)."""
    if focal_gamma > 0:
        p_correct = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        per_example = F.cross_entropy(logits, target, weight=weight, reduction="none")
        return (((1 - p_correct) ** focal_gamma) * per_example).mean()
    return F.cross_entropy(logits, target, weight=weight)


def supervised_contrastive(h, labels, temperature: float):
    """Pull together pooled embeddings with the same label, push apart the others (Khosla et al. 2020)."""
    z = F.normalize(h.float(), dim=-1)
    sim = (z @ z.t()) / temperature
    n = labels.size(0)
    self_mask = torch.eye(n, dtype=torch.bool, device=h.device)
    sim = sim.masked_fill(self_mask, -1e9)                       # an example is not its own positive
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    positives = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = positives.sum(1)
    has_pos = n_pos > 0
    if not has_pos.any():
        return h.sum() * 0.0
    return -((log_prob * positives).sum(1)[has_pos] / n_pos[has_pos]).mean()


def compute_loss(outputs: dict, batch: dict, class_w: dict, cfg: dict, model) -> torch.Tensor:
    gamma = cfg["focal_gamma"] if cfg["loss"] == "focal" else 0.0
    parts = {}
    if "a" in outputs:
        parts["a"] = cross_entropy(outputs["a"], batch["y_a"], class_w.get("a"), gamma)
    if "b" in outputs:
        parts["b"] = cross_entropy(outputs["b"], batch["y_b"], class_w.get("b"), gamma)
    if "t" in outputs:
        hateful = batch["y_b"] > 0
        if hateful.any():
            parts["t"] = cross_entropy(outputs["t"][hateful], batch["y_t"][hateful], class_w.get("t"), gamma)
            parts["i"] = F.binary_cross_entropy_with_logits(outputs["i"][hateful], batch["y_i"][hateful],
                                                            pos_weight=class_w.get("i"))
        else:  # no hateful row in this batch: keep the graph connected with a zero loss
            parts["t"] = outputs["t"].sum() * 0.0
            parts["i"] = outputs["i"].sum() * 0.0

    if model.log_vars is not None:  # learned task weights (Kendall et al. 2018)
        index = {"a": 0, "b": 1, "t": 2, "i": 3}
        total = sum(torch.exp(-model.log_vars[index[k]]) * v + model.log_vars[index[k]] for k, v in parts.items())
    else:
        total = (cfg["w_a"] * parts.get("a", 0.0) + cfg["w_b"] * parts.get("b", 0.0)
                 + cfg["w_c"] * (parts.get("t", 0.0) + parts.get("i", 0.0)))

    if cfg["supcon_w"] > 0 and "h" in outputs:
        labels = batch["y_b"] if "b" in outputs else batch["y_a"]
        total = total + cfg["supcon_w"] * supervised_contrastive(outputs["h"], labels, cfg["supcon_temp"])
    return total
