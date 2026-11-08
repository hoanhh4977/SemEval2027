"""Train one neural model (one seed, one set of active tasks) and return its probabilities.

Reading order:  fit_predict()  ->  _build_optimizer()  ->  _train_one_epoch()  ->  predict()
"""
from __future__ import annotations

import gc
import math
import time
from typing import Dict, Set

import numpy as np
import torch
from transformers import AutoConfig

from .encoding import build_encodings, collate, load_tokenizer, make_items
from .losses import class_weights, compute_loss
from .metrics import score_split
from .model import MTModel, layer_index


# ------------------------------------------------------------------------------------------ inference
@torch.no_grad()
def predict(model, items, pad_id, device, batch_size=64) -> Dict[str, np.ndarray]:
    """Probabilities for every example. Sorting by length makes batches short -> fast."""
    model.eval()
    order = np.argsort([len(it["ids"]) for it in items])
    chunks: Dict[str, list] = {}
    for start in range(0, len(order), batch_size):
        idx = order[start: start + batch_size]
        batch = collate([items[j] for j in idx], pad_id)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        for name, logits in outputs.items():
            if name == "h":  # pooled embedding, only used for the contrastive loss
                continue
            probs = torch.sigmoid(logits.float()) if name == "i" else torch.softmax(logits.float(), -1)
            chunks.setdefault(name, []).append(probs.cpu().numpy())
    back_to_original_order = np.argsort(order)
    return {name: np.concatenate(c)[back_to_original_order] for name, c in chunks.items()}


def _selection_score(val_df, probs, tasks: Set[str]) -> float:
    """Epoch-selection metric: mean macro-F1 over the tasks this model was trained on."""
    scores = score_split(val_df, probs)
    chosen = [scores[k] for k in ("A", "B", "C") if k in tasks and k in scores]
    return float(np.mean(chosen)) if chosen else 0.0


# ------------------------------------------------------------------------------------------ optimiser
def _build_optimizer(model, cfg: dict):
    """AdamW with (a) no weight decay on biases/norms, (b) optional layer-wise LR decay, (c) larger head LR."""
    n_layers = getattr(AutoConfig.from_pretrained(cfg["backbone"]), "num_hidden_layers", 12)
    groups = []
    for name, p in model.trunk_params():
        lr = cfg["lr"]
        if cfg["arch"] != "decoder" and cfg["llrd"] < 1.0:
            # layer k gets lr * llrd^(distance from the top): lower layers move more slowly
            lr = cfg["lr"] * cfg["llrd"] ** (n_layers + 1 - layer_index(name, n_layers))
        no_decay = p.ndim == 1 or "bias" in name or "LayerNorm" in name
        groups.append({"params": [p], "lr": lr, "weight_decay": 0.0 if no_decay else cfg["wd"]})
    for name, p in model.head_params():
        groups.append({"params": [p], "lr": cfg["head_lr"], "weight_decay": 0.0 if p.ndim == 1 else cfg["wd"]})
    return torch.optim.AdamW(groups), groups


def _build_scheduler(opt, n_train: int, cfg: dict):
    """Linear warm-up for 10 % of the steps, then linear decay to 0."""
    steps_per_epoch = math.ceil(n_train / cfg["bs"] / cfg["accum"])
    total = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(cfg["warmup"] * total))
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))


# ------------------------------------------------------------------------------------------ one epoch
def _fgm_extra_backward(model, batch, class_w, cfg, scaler, device):
    """FGM adversarial training: nudge the word embeddings along their gradient, add the loss on the
    perturbed input, then restore the embeddings."""
    emb = model.encoder.get_input_embeddings().weight
    if emb.grad is None:
        return
    backup = emb.data.clone()
    grad_norm = torch.norm(emb.grad)
    if torch.isfinite(grad_norm) and grad_norm > 0:
        emb.data.add_(cfg["fgm_eps"] * emb.grad / grad_norm)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(batch["input_ids"], batch["attention_mask"])
        adv_loss = compute_loss(outputs, batch, class_w, cfg, model)
        if torch.isfinite(adv_loss):
            scaler.scale(adv_loss / cfg["accum"]).backward()
    emb.data.copy_(backup)


def _train_one_epoch(model, items, pad_id, class_w, opt, sched, scaler, groups, cfg, device, rng, log):
    model.train()
    order = rng.permutation(len(items))
    total_loss, n_batches = 0.0, 0
    for b, start in enumerate(range(0, len(order), cfg["bs"])):
        batch = collate([items[j] for j in order[start: start + cfg["bs"]]], pad_id)
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(batch["input_ids"], batch["attention_mask"])
        loss = compute_loss(outputs, batch, class_w, cfg, model)
        if not torch.isfinite(loss):
            log(f"  non-finite loss in batch {b}; skipped")
            opt.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss / cfg["accum"]).backward()
        if cfg["fgm_eps"] > 0:
            _fgm_extra_backward(model, batch, class_w, cfg, scaler, device)
        total_loss += float(loss)
        n_batches += 1

        is_update_step = (b + 1) % cfg["accum"] == 0 or start + cfg["bs"] >= len(order)
        if is_update_step:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
    return total_loss / max(n_batches, 1)


# ------------------------------------------------------------------------------------------ main entry
def fit_predict(cfg: dict, tasks: Set[str], train_df, val_df, extra_dfs: Dict[str, "pd.DataFrame"],
                seed: int, device, log) -> dict:
    """Train on train_df, pick the best epoch on val_df, return probabilities for val and for each
    dataframe in `extra_dfs` (e.g. the held-out test split and the FA development set)."""
    start_time = time.time()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    # ---- data
    tok = load_tokenizer(cfg["backbone"])
    pad_id = tok.pad_token_id
    if tasks == {"C"}:  # a C-only model never needs non-hateful rows
        train_df = train_df[train_df["y_b"] > 0].reset_index(drop=True)
    train_items = make_items(train_df, build_encodings(train_df, tok, cfg), with_labels=True)
    val_items = make_items(val_df, build_encodings(val_df, tok, cfg), with_labels=False)
    extra_items = {k: make_items(d, build_encodings(d, tok, cfg), with_labels=False) for k, d in extra_dfs.items()}
    class_w = class_weights(train_df, cfg["cw"], device)

    # ---- model, optimiser
    model = MTModel(cfg, tasks)
    if not model.quantized:  # 4-bit models are already on the GPU
        model.encoder.to(device)
    for child in model.children():
        if child is not model.encoder:
            child.to(device)
    if model.log_vars is not None:
        model.log_vars.data = model.log_vars.data.to(device)
    opt, groups = _build_optimizer(model, cfg)
    sched = _build_scheduler(opt, len(train_items), cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # ---- training with early stopping on the validation score
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    best_state, best_score, best_epoch, epochs_without_gain, history = None, -1.0, -1, 0, []
    for epoch in range(cfg["epochs"]):
        train_loss = _train_one_epoch(model, train_items, pad_id, class_w, opt, sched, scaler, groups, cfg,
                                      device, rng, log)
        val_score = _selection_score(val_df, predict(model, val_items, pad_id, device), tasks)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_score": val_score})
        log(f"  epoch {epoch}: train_loss={train_loss:.4f} val_score={val_score:.4f}")
        if val_score > best_score + 1e-6:
            best_score, best_epoch, epochs_without_gain = val_score, epoch, 0
            params = dict(model.named_parameters())
            best_state = {n: params[n].detach().cpu().clone() for n in trainable_names}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg["patience"]:
                break

    # ---- restore the best epoch and predict
    if best_state is not None:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in best_state.items():
                params[name].copy_(value.to(params[name].device))
    probs = {"val": predict(model, val_items, pad_id, device)}
    for name, items in extra_items.items():
        probs[name] = predict(model, items, pad_id, device)

    info = {"best_epoch": best_epoch, "best_val": best_score, "history": history,
            "train_time_s": time.time() - start_time, "n_train": len(train_items)}
    del model, opt, best_state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"probs": probs, "info": info}
