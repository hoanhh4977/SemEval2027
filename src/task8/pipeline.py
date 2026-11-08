"""Run one (experiment, seed): split -> train -> predict -> score (raw + tuned thresholds) -> save."""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from typing import Dict

import numpy as np
import pandas as pd

from . import classical
from .data import split_by_seed
from .metrics import score_by_lang, tune_thresholds


def _select_train(split: Dict[str, pd.DataFrame], cfg: dict, seed: int):
    langs = cfg["train_langs"]
    tr = split["train"][split["train"]["lang"].isin(langs)]
    va = split["val"][split["val"]["lang"].isin(langs)]
    if cfg["train_frac"] < 1.0:
        tr = tr.sample(frac=cfg["train_frac"], random_state=seed)
    return tr.reset_index(drop=True), va.reset_index(drop=True)


def _merge(a: dict, b: dict) -> dict:
    return {s: {**a.get(s, {}), **b.get(s, {})} for s in set(a) | set(b)}


def run_seed(name: str, cfg: dict, seed: int, train_df, fa_df, out_dir: str, device, data_dir: str,
             dapt_root: str, log) -> dict:
    t0 = time.time()
    split = split_by_seed(train_df, seed)
    tr, va = _select_train(split, cfg, seed)
    extra = {"test": split["test"].reset_index(drop=True), "fa": fa_df.reset_index(drop=True)}
    infos = []

    if cfg["backbone"] in ("majority", "random_stratified", "tfidf"):
        probs = classical.fit_predict(cfg, tr, {"val": va, **extra}, seed)
    else:
        from .dapt import run_dapt
        from .trainer import fit_predict
        run_cfg = dict(cfg)
        if cfg["dapt"]:
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cfg["backbone"])
            run_cfg["backbone"] = run_dapt(cfg["backbone"], data_dir, os.path.join(dapt_root, safe), log=log)
        task_sets = [{"A"}, {"B"}, {"C"}] if cfg["mode"] == "single" else [{"A", "B", "C"}]
        probs = {}
        for ts in task_sets:
            log(f"  training tasks={''.join(sorted(ts))}")
            r = fit_predict(run_cfg, ts, tr, va, extra, seed, device, log)
            probs = _merge(probs, r["probs"])
            infos.append({"tasks": "".join(sorted(ts)), **r["info"]})

    frames = {"val": va, **extra}
    thr = tune_thresholds(va, probs["val"])
    metrics = {"raw": {}, "tuned": {}}
    for s, df in frames.items():
        metrics["raw"][s] = score_by_lang(df, probs[s], None)
        metrics["tuned"][s] = score_by_lang(df, probs[s], thr)

    d = os.path.join(out_dir, name, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    arrs = {}
    for s, df in frames.items():
        arrs[f"{s}_ids"] = df["StereoQueerEval_id"].astype(str).values
        for k, v in probs[s].items():
            arrs[f"{s}_{k}"] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(os.path.join(d, "preds.npz"), **arrs)
    res = {"exp": name, "seed": seed, "cfg": cfg, "thresholds": thr, "metrics": metrics, "train_info": infos,
           "elapsed_s": time.time() - t0, "n_train": int(len(tr)), "n_val": int(len(va)),
           "n_test": int(len(extra["test"])), "n_fa": int(len(extra["fa"]))}
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def safe_run_seed(log, errors_path: str, **kw):
    """Never let one failing run kill the session; record the traceback instead."""
    try:
        return run_seed(log=log, **kw)
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc()
        log(f"!! FAILED {kw['name']} seed{kw['seed']}\n{tb}")
        with open(errors_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"exp": kw["name"], "seed": kw["seed"], "traceback": tb}) + "\n")
        try:
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        return None
