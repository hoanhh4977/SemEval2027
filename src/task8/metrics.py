"""Official-equivalent metrics (A, B, C) + threshold tuning."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data import IDENTS


def decode(P: Dict[str, np.ndarray], thr: Optional[dict] = None):
    """probs -> hard predictions. thr = {'a': float, 'i': list(9)} or None (=0.5 / argmax)."""
    ta = 0.5 if thr is None else thr["a"]
    ti = np.full(len(IDENTS), 0.5) if thr is None else np.asarray(thr["i"])
    out = {}
    if "a" in P:
        out["a"] = (P["a"][:, 1] >= ta).astype(int)
    if "b" in P:
        out["b"] = P["b"].argmax(1)
    if "t" in P:
        out["t"] = P["t"].argmax(1)
    if "i" in P:
        pi = (P["i"] >= ti[None, :]).astype(int)
        empty = pi.sum(1) == 0
        pi[empty, P["i"][empty].argmax(1)] = 1  # every hateful comment targets >= 1 identity
        out["i"] = pi
    return out


def score_split(df: pd.DataFrame, P: Dict[str, np.ndarray], thr: Optional[dict] = None) -> Dict[str, float]:
    """Scores on one dataframe. C is computed on gold-hateful rows only (as in the official task)."""
    pr = decode(P, thr)
    r: Dict[str, float] = {}
    if "a" in pr:
        r["A"] = float(f1_score(df["y_a"].values, pr["a"], labels=[0, 1], average="macro", zero_division=0))
    if "b" in pr:
        r["B"] = float(f1_score(df["y_b"].values, pr["b"], labels=[0, 1, 2], average="macro", zero_division=0))
    if "t" in pr and "i" in pr:
        m = df["y_b"].values > 0
        if m.sum() > 0:
            yt = df["y_t"].values[m]
            yi = np.stack(df["y_i"].values)[m]
            r["C_target"] = float(f1_score(yt, pr["t"][m], labels=[0, 1], average="macro", zero_division=0))
            # official: macro over all 9 identity columns, zero_division=0
            r["C_identity"] = float(f1_score(yi, pr["i"][m], average="macro", zero_division=0))
            r["C"] = 0.5 * r["C_target"] + 0.5 * r["C_identity"]
    present = [r[k] for k in ("A", "B", "C") if k in r]
    if present:
        r["mean"] = float(np.mean(present))
    r["n"] = int(len(df))
    return r


def score_by_lang(df: pd.DataFrame, P: Dict[str, np.ndarray], thr: Optional[dict] = None) -> Dict[str, dict]:
    res = {"ALL": score_split(df, P, thr)}
    langs = df["lang"].values
    for lang in sorted(set(langs)):
        m = langs == lang
        res[lang] = score_split(df[m], {k: v[m] for k, v in P.items()}, thr)
    lang_list = [k for k in res if k != "ALL"]
    if len(lang_list) > 1:  # every language weighted equally
        keys = [k for k in res["ALL"] if k != "n"]
        res["LANGMEAN"] = {k: float(np.mean([res[l][k] for l in lang_list if k in res[l]])) for k in keys}
    return res


def tune_thresholds(df_val: pd.DataFrame, P_val: Dict[str, np.ndarray]) -> dict:
    """Threshold for stereotype (macro-F1) and per-identity thresholds (binary F1 on hateful rows)."""
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    thr = {"a": 0.5, "i": np.full(len(IDENTS), 0.5)}

    def pick(scores):
        best, bt = -1.0, 0.5
        for t, f in scores:
            if f > best + 1e-9 or (abs(f - best) <= 1e-9 and abs(t - 0.5) < abs(bt - 0.5)):
                best, bt = f, t
        return float(bt)

    if "a" in P_val:
        thr["a"] = pick([(t, f1_score(df_val["y_a"].values, (P_val["a"][:, 1] >= t).astype(int), labels=[0, 1],
                                      average="macro", zero_division=0)) for t in grid])
    if "i" in P_val:
        m = df_val["y_b"].values > 0
        if m.sum() > 0:
            yi = np.stack(df_val["y_i"].values)[m]
            pi = P_val["i"][m]
            for j in range(len(IDENTS)):
                if yi[:, j].sum() == 0:  # no positives in val: keep default
                    continue
                thr["i"][j] = pick([(t, f1_score(yi[:, j], (pi[:, j] >= t).astype(int), zero_division=0))
                                    for t in grid])
    thr["i"] = [float(x) for x in thr["i"]]
    return thr
