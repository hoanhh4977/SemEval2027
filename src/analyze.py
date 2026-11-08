"""Turn the result zips from Kaggle into tables and significance tests (run this on your own computer).

    python src/analyze.py --inputs r1s1.zip --ref P --out report_r1
    python src/analyze.py --inputs r1s1.zip r2s1.zip --ref P --out report_all \
        --ensemble P BB_mdeberta BB_twitter_xlmr --data data/task8

Output folder: report.md (all tables), table_*.csv, paired_tests.csv.

Statistics: for every experiment vs --ref we use the seeds both have (the split is identical for the same
seed, so the comparison is paired) and report: mean difference, bootstrap 95 % CI, paired t-test, Wilcoxon,
Cohen's d_z, and Holm-corrected p-values across experiments.
With 5 seeds the smallest possible Wilcoxon p is 0.0625, so it can never be < 0.05: trust the t-test / CI.
"""
import argparse
import glob
import json
import os
import sys
import tempfile
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task8.metrics import score_by_lang, tune_thresholds  # noqa: E402

KEYS = ["A", "B", "C", "mean"]


# ------------------------------------------------------------------------------------------ loading
def load_results(inputs: List[str]) -> Dict[str, Dict[int, dict]]:
    """{experiment: {seed: result.json content}} from any mix of zip files and folders."""
    tmp = tempfile.mkdtemp(prefix="t8_analysis_")
    roots = []
    for i, path in enumerate(inputs):
        if path.endswith(".zip"):
            target = os.path.join(tmp, f"zip{i}")
            zipfile.ZipFile(path).extractall(target)
            roots.append(target)
        else:
            roots.append(path)
    results: Dict[str, Dict[int, dict]] = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "result.json"), recursive=True):
            r = json.load(open(f, encoding="utf-8"))
            r["_dir"] = os.path.dirname(f)
            results.setdefault(r["exp"], {})[r["seed"]] = r
    return results


def metric(result: dict, thr: str, split: str, group: str, key: str):
    return result["metrics"][thr][split].get(group, {}).get(key)


# ------------------------------------------------------------------------------------------ tables
def mean_std_table(results, thr, split, group) -> pd.DataFrame:
    rows = []
    for exp, seeds in results.items():
        row = {"exp": exp, "n_seeds": len(seeds)}
        for key in KEYS:
            vals = [metric(r, thr, split, group, key) for r in seeds.values()]
            vals = [v for v in vals if v is not None]
            row[key] = f"{np.mean(vals):.4f} +- {np.std(vals, ddof=1) if len(vals) > 1 else 0:.4f}" if vals else "-"
            row[key + "_mean"] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_mean", ascending=False)


# ------------------------------------------------------------------------------------------ tests
def holm_correction(p_values: List[float]) -> List[float]:
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def paired_tests(results, ref, thr, split, group, n_boot=10000) -> pd.DataFrame:
    if ref not in results:
        return pd.DataFrame()
    rng = np.random.RandomState(0)
    rows = []
    for key in KEYS:
        block = []
        for exp, seeds in results.items():
            if exp == ref:
                continue
            common = sorted(set(seeds) & set(results[ref]))
            a = np.array([metric(seeds[s], thr, split, group, key) for s in common], float)
            b = np.array([metric(results[ref][s], thr, split, group, key) for s in common], float)
            ok = ~(np.isnan(a) | np.isnan(b))
            a, b = a[ok], b[ok]
            if len(a) < 2:
                continue
            diff = a - b  # experiment minus reference
            t_p = stats.ttest_rel(a, b).pvalue if np.std(diff) > 0 else 1.0
            try:
                w_p = stats.wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
            except ValueError:
                w_p = 1.0
            boot = rng.choice(diff, (n_boot, len(diff)), replace=True).mean(1)
            sd = diff.std(ddof=1)
            block.append({"metric": key, "exp": exp, "n_seeds": len(diff), "ref_mean": b.mean(), "exp_mean": a.mean(),
                          "diff": diff.mean(), "ci95_low": np.percentile(boot, 2.5),
                          "ci95_high": np.percentile(boot, 97.5), "cohen_dz": diff.mean() / sd if sd > 0 else np.nan,
                          "p_ttest": t_p, "p_wilcoxon": w_p})
        for row, p_adj in zip(block, holm_correction([r["p_ttest"] for r in block])):
            row["p_ttest_holm"] = p_adj
        rows += block
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------ ensemble
def add_ensemble(results, names: List[str], data_dir: str):
    """Average the probabilities of several experiments seed by seed (same seed = same split) and re-score."""
    from task8.data import load_task8
    train, fa = load_task8(data_dir)
    labels = pd.concat([train, fa]).set_index("StereoQueerEval_id")
    ens_name = "ENSEMBLE(" + "+".join(names) + ")"
    results[ens_name] = {}
    for seed in sorted(set.intersection(*[set(results[n]) for n in names])):
        files = [np.load(os.path.join(results[n][seed]["_dir"], "preds.npz"), allow_pickle=True) for n in names]
        probs, frames = {}, {}
        for split in ("val", "test", "fa"):
            ids = files[0][f"{split}_ids"]
            assert all((f[f"{split}_ids"] == ids).all() for f in files), "different splits: cannot ensemble"
            df = labels.loc[ids].copy()
            df["lang"] = "FA" if split == "fa" else [i.split("_")[1] for i in ids]
            frames[split] = df
            heads = [h for h in "abti" if f"{split}_{h}" in files[0]]
            probs[split] = {h: np.mean([f[f"{split}_{h}"] for f in files], 0) for h in heads}
        thr = tune_thresholds(frames["val"], probs["val"])
        metrics = {"raw": {}, "tuned": {}}
        for split, df in frames.items():
            metrics["raw"][split] = score_by_lang(df, probs[split], None)
            metrics["tuned"][split] = score_by_lang(df, probs[split], thr)
        results[ens_name][seed] = {"exp": ens_name, "seed": seed, "metrics": metrics}


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="zip files and/or folders")
    ap.add_argument("--ref", default="P", help="reference experiment for the paired tests")
    ap.add_argument("--out", default="report")
    ap.add_argument("--thr", default="raw", choices=["raw", "tuned"], help="decision rule used for the paired tests")
    ap.add_argument("--split", default="test", choices=["test", "fa", "val"])
    ap.add_argument("--group", default="LANGMEAN", help="ALL | LANGMEAN | EN | IT | NL | FA")
    ap.add_argument("--ensemble", nargs="*", default=[], help="experiment names to average")
    ap.add_argument("--data", default="data/task8")
    a = ap.parse_args()

    results = load_results(a.inputs)
    print(f"{len(results)} experiments loaded: " + ", ".join(f"{e}({len(s)})" for e, s in results.items()))
    if a.ensemble:
        add_ensemble(results, a.ensemble, a.data)
    os.makedirs(a.out, exist_ok=True)

    md = []
    for thr in ("raw", "tuned"):
        for split, group in (("test", "LANGMEAN"), ("test", "ALL"), ("test", "EN"), ("test", "IT"), ("test", "NL"),
                             ("fa", "ALL")):
            table = mean_std_table(results, thr, split, group)
            if table["mean_mean"].isna().all():
                continue
            md.append(f"\n## {thr} decisions | split={split} | group={group}\n")
            md.append(table[["exp", "n_seeds"] + KEYS].to_markdown(index=False))
            table.to_csv(os.path.join(a.out, f"table_{thr}_{split}_{group}.csv"), index=False)
    tests = paired_tests(results, a.ref, a.thr, a.split, a.group)
    if not tests.empty:
        tests.to_csv(os.path.join(a.out, "paired_tests.csv"), index=False)
        md.append(f"\n## Paired tests vs `{a.ref}` ({a.thr}, {a.split}, {a.group})\n")
        md.append(tests.round(4).to_markdown(index=False))
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"report written to {a.out}/report.md")


if __name__ == "__main__":
    main()
