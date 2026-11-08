"""Non-neural baselines producing the same probability dict as the neural models.

backbone == "majority"           : always predicts the majority class of the training data
backbone == "random_stratified"  : samples labels from the training label distribution
backbone == "tfidf"              : word(1-2) + char(2-5) TF-IDF + balanced logistic regression
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import IDENTS, context_text


def _onehot(idx, k):
    o = np.zeros((len(idx), k), dtype=np.float32)
    o[np.arange(len(idx)), idx] = 1.0
    return o


def _texts(df: pd.DataFrame, mode: str):
    c = df["yt_comment"].astype(str).str.strip()
    if mode == "none":
        return c.tolist()
    return [f"{a} ||| {context_text(r, mode)}" for a, (_, r) in zip(c, df.iterrows())]


def fit_predict(cfg: dict, train: pd.DataFrame, dfs: Dict[str, pd.DataFrame], seed: int) -> Dict[str, dict]:
    kind = cfg["backbone"]
    hate = train[train["y_b"] > 0]
    yi_h = np.stack(hate["y_i"].values)
    rng = np.random.RandomState(seed)
    out = {}

    if kind in ("majority", "random_stratified"):
        pa = np.bincount(train["y_a"], minlength=2) / len(train)
        pb = np.bincount(train["y_b"], minlength=3) / len(train)
        pt = np.bincount(hate["y_t"], minlength=2) / len(hate)
        pi = yi_h.mean(0)
        for name, df in dfs.items():
            n = len(df)
            if kind == "majority":
                a, b, t = (np.full(n, pa.argmax()), np.full(n, pb.argmax()), np.full(n, pt.argmax()))
                i = np.tile((pi >= 0.5).astype(np.float32), (n, 1))
            else:
                a, b, t = (rng.choice(2, n, p=pa), rng.choice(3, n, p=pb), rng.choice(2, n, p=pt))
                i = (rng.rand(n, len(IDENTS)) < pi[None, :]).astype(np.float32)
            out[name] = {"a": _onehot(a, 2), "b": _onehot(b, 3), "t": _onehot(t, 2), "i": i}
        return out

    # ---- TF-IDF + logistic regression
    vw = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, max_features=200000)
    vc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300000)
    tr_txt = _texts(train, cfg["context"])
    vw.fit(tr_txt)
    vc.fit(tr_txt)

    def feats(df):
        t = _texts(df, cfg["context"])
        return hstack([vw.transform(t), vc.transform(t)]).tocsr()

    X, Xh = feats(train), feats(hate)

    def lr():
        return LogisticRegression(C=cfg.get("C", 4.0), class_weight="balanced", max_iter=3000)

    ma = lr().fit(X, train["y_a"].values)
    mb = lr().fit(X, train["y_b"].values)
    mt = lr().fit(Xh, hate["y_t"].values)
    mi = []
    for j in range(len(IDENTS)):
        if yi_h[:, j].sum() < 1 or yi_h[:, j].sum() == len(yi_h):
            mi.append(float(yi_h[:, j].mean()))
        else:
            mi.append(lr().fit(Xh, yi_h[:, j]))
    for name, df in dfs.items():
        Xd = feats(df)
        pi_ = np.stack([np.full(len(df), m, dtype=np.float32) if isinstance(m, float)
                        else m.predict_proba(Xd)[:, 1].astype(np.float32) for m in mi], 1)
        out[name] = {"a": ma.predict_proba(Xd).astype(np.float32), "b": mb.predict_proba(Xd).astype(np.float32),
                     "t": mt.predict_proba(Xd).astype(np.float32), "i": pi_}
    return out
