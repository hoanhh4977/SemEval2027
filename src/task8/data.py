"""Data loading, label encoding and leakage-safe splitting for Task 8."""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

IDENTS = ["l", "g", "b", "t", "q", "i", "a", "nb", "lgbtqia+"]
HATE = ["no", "yes_implicit", "yes_explicit"]
TTYPE = ["group", "individual"]
TRAIN_LANGS = ["EN", "IT", "NL"]


def _read_tsv(path: str) -> pd.DataFrame:
    # keep_default_na=False: the string "none" / "NA"-like values must stay strings.
    return pd.read_csv(path, sep="\t", keep_default_na=False, dtype=str)


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_a"] = (df["stereotype"].str.strip().str.lower() == "yes").astype(int)
    df["y_b"] = df["hate_speech"].str.strip().str.lower().map({h: i for i, h in enumerate(HATE)}).astype(int)
    y_t, y_i = [], np.zeros((len(df), len(IDENTS)), dtype=np.float32)
    for k, t in enumerate(df["target"].tolist()):
        t = str(t).strip().lower()
        if t in ("none", ""):
            y_t.append(-1)
            continue
        ty, rest = t.split("_", 1)
        y_t.append(TTYPE.index(ty))
        for x in rest.split(","):
            y_i[k, IDENTS.index(x.strip())] = 1.0
    df["y_t"] = y_t
    df["y_i"] = list(y_i)
    return df


def load_task8(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df over EN/IT/NL with a `lang` column, FA dev df)."""
    parts = []
    for lang in TRAIN_LANGS:
        p = os.path.join(data_dir, "train", f"StereoQueerEval_{lang}_training.tsv")
        d = _read_tsv(p)
        d["lang"] = lang
        parts.append(d)
    train = encode_labels(pd.concat(parts, ignore_index=True))
    fa = _read_tsv(os.path.join(data_dir, "dev", "StereoQueerEval_FA_development.tsv"))
    fa["lang"] = "FA"
    return train, encode_labels(fa)


def split_by_seed(train: pd.DataFrame, seed: int, test_size: float = 0.20, val_size: float = 0.10
                  ) -> Dict[str, pd.DataFrame]:
    """Per-language group split by video title, deterministic in `seed`.

    Several comments share one video (title/description); splitting at comment level would leak
    the video context between train and test and inflate scores. The same seed always gives the same
    split, so different methods are compared on identical data (needed for paired tests).
    """
    out = {"train": [], "val": [], "test": []}
    for _, d in train.groupby("lang", sort=True):
        d = d.reset_index(drop=True)
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(d, groups=d["yt_title"]))
        rest = d.iloc[tr].reset_index(drop=True)
        tr2, va = next(GroupShuffleSplit(1, test_size=val_size / (1 - test_size), random_state=seed + 1000)
                       .split(rest, groups=rest["yt_title"]))
        out["train"].append(rest.iloc[tr2])
        out["val"].append(rest.iloc[va])
        out["test"].append(d.iloc[te])
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


def context_text(row, mode: str) -> str:
    if mode == "none":
        return ""
    if mode == "title":
        return str(row["yt_title"]).strip()
    return (str(row["yt_title"]).strip() + " | " + str(row["yt_description"]).strip()).strip()
