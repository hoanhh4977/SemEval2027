"""Experiment registry. Every experiment = DEFAULT config + overrides, run for several seeds.

Naming
  B_*   baselines                 (non-neural / single-task / no-context neural)
  P     PROPOSED method           context-aware multi-task encoder with hierarchical conditioning,
                                  class-balanced losses, layer-wise LR decay and supervised contrastive loss
  A_*   ablations of P (leave one component out) and design-choice variants
  X_*   extensions from the literature of similar shared tasks (FGM, domain-adaptive pretraining)
  BB_*  other backbones with the P recipe
  XL_*  cross-lingual training-language ablations (all runs are evaluated on EN/IT/NL/FA)
  DS_*  learning-curve (training data fraction)
  LLM_* decoder LLM + LoRA with the P recipe
`cost` is only used to balance work between parallel shards (1.0 ~ one xlm-roberta-base multi-task run).
"""
from __future__ import annotations

from typing import Dict, List

XLMRB = "xlm-roberta-base"
XLMRL = "xlm-roberta-large"
MDEB = "microsoft/mdeberta-v3-base"
TWXLMR = "cardiffnlp/twitter-xlm-roberta-base"
MBERT = "bert-base-multilingual-cased"
LABSE = "sentence-transformers/LaBSE"
QWEN15 = "Qwen/Qwen2.5-1.5B"
QWEN7 = "Qwen/Qwen2.5-7B"

DEFAULT: dict = dict(
    backbone=XLMRB, arch="encoder",
    mode="multi",                    # multi: one model, heads A/B/C | single: one model per task
    context="title_desc",            # none | title | title_desc   (video context for the comment)
    hier=False,                      # condition A and C heads on the softmax of head B
    loss="ce", focal_gamma=2.0,      # ce | focal
    cw="none",                       # class weights: none | sqrt | balanced (also identity pos_weight)
    weighting="fixed",               # fixed | uncertainty (Kendall et al.)
    w_a=1.0, w_b=1.0, w_c=1.0,
    pooling="cls",                   # cls | mean   (decoder: last token)
    dropout=0.1, lr=2e-5, head_lr=5e-4, llrd=1.0, wd=0.01,
    epochs=6, bs=16, accum=1, warmup=0.1, patience=2,
    max_len=256, max_comment_len=128,
    train_langs=["EN", "IT", "NL"], train_frac=1.0,
    supcon_w=0.0, supcon_temp=0.1,
    fgm_eps=0.0,
    dapt=False,
    load_4bit=False, grad_ckpt=False, lora_r=16, lora_alpha=32,
    C=4.0,                           # tf-idf logistic regression
    cost=1.0,
)

PROPOSED = dict(hier=True, cw="sqrt", llrd=0.9, supcon_w=0.1)

EXPERIMENTS: Dict[str, dict] = {}
GROUPS: Dict[str, List[str]] = {}


def add(name: str, group: str, cost: float = 1.0, **over):
    cfg = {**DEFAULT, **over, "cost": cost}
    EXPERIMENTS[name] = cfg
    GROUPS.setdefault(group, []).append(name)


# ------------------------------------------------------------------ smoke test (a few minutes)
add("SMOKE", "smoke", cost=0.1, epochs=1, train_frac=0.1, **PROPOSED)

# ------------------------------------------------------------------ baselines + main comparison
add("B_majority", "baselines", cost=0.01, backbone="majority", arch="none", context="none")
add("B_random_stratified", "baselines", cost=0.01, backbone="random_stratified", arch="none", context="none")
add("B_tfidf_comment", "baselines", cost=0.05, backbone="tfidf", arch="none", context="none")
add("B_tfidf_context", "baselines", cost=0.05, backbone="tfidf", arch="none", context="title_desc")
add("B_xlmrb_single_comment", "main", cost=3.0, mode="single", context="none")
add("B_xlmrb_single_context", "main", cost=3.0, mode="single")
add("B_xlmrb_mt_comment", "main", cost=1.0, context="none")
add("B_xlmrb_mt_context", "main", cost=1.0)
add("P", "main", cost=1.0, **PROPOSED)

# ------------------------------------------------------------------ ablations of P
add("A_P_no_context", "ablation", context="none", **PROPOSED)
add("A_P_title_only", "ablation", context="title", **PROPOSED)
add("A_P_no_hier", "ablation", **{**PROPOSED, "hier": False})
add("A_P_no_cw", "ablation", **{**PROPOSED, "cw": "none"})
add("A_P_no_llrd", "ablation", **{**PROPOSED, "llrd": 1.0})
add("A_P_no_supcon", "ablation", **{**PROPOSED, "supcon_w": 0.0})
add("A_P_single_task", "ablation", cost=3.0, mode="single", **PROPOSED)
add("A_P_balanced_cw", "ablation", **{**PROPOSED, "cw": "balanced"})
add("A_P_focal", "ablation", **{**PROPOSED, "loss": "focal"})
add("A_P_uncertainty_w", "ablation", **{**PROPOSED, "weighting": "uncertainty"})
add("A_P_mean_pool", "ablation", **{**PROPOSED, "pooling": "mean"})
add("A_P_supcon_0.3", "ablation", **{**PROPOSED, "supcon_w": 0.3})
add("A_P_maxlen128", "ablation", cost=0.6, **{**PROPOSED, "max_len": 128})
add("A_P_maxlen384", "ablation", cost=1.6, **{**PROPOSED, "max_len": 384})

# ------------------------------------------------------------------ literature extensions
add("X_P_fgm", "extensions", cost=1.8, fgm_eps=0.5, **PROPOSED)
add("X_P_dapt", "extensions", cost=1.3, dapt=True, **PROPOSED)
add("X_P_dapt_fgm", "extensions", cost=2.1, dapt=True, fgm_eps=0.5, **PROPOSED)

# ------------------------------------------------------------------ other backbones (P recipe)
add("BB_mdeberta", "backbone", cost=1.6, backbone=MDEB, **PROPOSED)
add("BB_twitter_xlmr", "backbone", backbone=TWXLMR, **PROPOSED)
add("BB_mbert", "backbone", backbone=MBERT, **PROPOSED)
add("BB_labse", "backbone", backbone=LABSE, **PROPOSED)
add("BB_xlmr_large", "backbone", cost=3.5, backbone=XLMRL, lr=1e-5, bs=4, accum=4, epochs=5,
    **{k: v for k, v in PROPOSED.items()})

# ------------------------------------------------------------------ cross-lingual transfer
for _l in (["EN"], ["IT"], ["NL"], ["EN", "IT"], ["EN", "NL"], ["IT", "NL"]):
    add("XL_train_" + "+".join(_l), "crosslingual", cost=0.4 * len(_l), train_langs=_l, **PROPOSED)

# ------------------------------------------------------------------ learning curve
for _f in (0.25, 0.5, 0.75):
    add(f"DS_frac_{int(_f * 100)}", "datasize", cost=_f, train_frac=_f, **PROPOSED)

# ------------------------------------------------------------------ decoder LLM + LoRA (heaviest)
add("LLM_qwen1.5b_lora", "llm", cost=6.0, backbone=QWEN15, arch="decoder", lr=2e-4, head_lr=5e-4, llrd=1.0,
    bs=8, accum=2, epochs=3, grad_ckpt=True, **{k: v for k, v in PROPOSED.items() if k != "llrd"})
add("LLM_qwen7b_qlora", "llm", cost=19.0, backbone=QWEN7, arch="decoder", lr=1e-4, head_lr=5e-4, llrd=1.0,
    bs=4, accum=4, epochs=2, load_4bit=True, max_len=224, **{k: v for k, v in PROPOSED.items() if k != "llrd"})

GROUP_ORDER = ["smoke", "baselines", "main", "ablation", "extensions", "backbone", "crosslingual", "datasize", "llm"]


def resolve(names_or_groups: List[str]) -> List[str]:
    out: List[str] = []
    for n in names_or_groups:
        if n == "all":
            items = [e for g in GROUP_ORDER if g != "smoke" for e in GROUPS.get(g, [])]
        elif n in GROUPS:
            items = GROUPS[n]
        elif n in EXPERIMENTS:
            items = [n]
        else:
            raise KeyError(f"unknown experiment/group: {n}")
        out += [i for i in items if i not in out]
    return out
