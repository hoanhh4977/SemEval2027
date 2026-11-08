"""The experiment loop: for every experiment, for every seed -> run, save, summarise, zip.

Safety rules (so that a crash or the 12 h Kaggle limit never loses finished work):
  1. every finished (experiment, seed) is already on disk (result.json + preds.npz);
  2. the output folder is zipped after every experiment and once more when the process exits;
  3. an exception inside one run is written to errors.jsonl and the loop moves on;
  4. a run is NOT started if its estimated duration would pass the session deadline;
  5. re-running the same command skips (experiment, seed) pairs that are already done.
"""
from __future__ import annotations

import atexit
import glob
import json
import os
import time
import zipfile
from typing import List, Optional

import numpy as np

from .data import load_task8
from .experiments import EXPERIMENTS
from .pipeline import safe_run_seed

METRICS = ["A", "B", "C", "C_target", "C_identity", "mean"]


def zip_dir(folder: str, zip_path: str):
    """Zip `folder` atomically: write a temp file first, then rename, so a good zip is never half-overwritten."""
    tmp = zip_path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root, _, files in os.walk(folder):
            for name in files:
                path = os.path.join(root, name)
                z.write(path, os.path.relpath(path, os.path.dirname(folder)))
    for attempt in range(5):  # rename can fail briefly if a sync tool locks the target (seen with OneDrive)
        try:
            os.replace(tmp, zip_path)
            return
        except OSError:
            time.sleep(1.0 + attempt)
    os.replace(tmp, zip_path[:-4] + time.strftime("_%H%M%S") + ".zip")  # last resort: keep a timestamped copy


def summarize(out_dir: str, exp_name: str):
    """mean/std over seeds of the main metrics -> <exp>/summary.json (handy for a quick look)."""
    results = [json.load(open(p, encoding="utf-8"))
               for p in sorted(glob.glob(os.path.join(out_dir, exp_name, "seed*", "result.json")))]
    if not results:
        return
    summary = {"exp": exp_name, "seeds": [r["seed"] for r in results], "metrics": {}}
    for thr in ("raw", "tuned"):
        for split in ("test", "fa"):
            for group in ("ALL", "LANGMEAN", "EN", "IT", "NL", "FA"):
                entry = {}
                for metric in METRICS:
                    vals = [r["metrics"][thr][split].get(group, {}).get(metric) for r in results]
                    vals = [v for v in vals if v is not None]
                    if vals:
                        entry[metric] = {"mean": float(np.mean(vals)),
                                         "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
                if entry:
                    summary["metrics"][f"{thr}/{split}/{group}"] = entry
    with open(os.path.join(out_dir, exp_name, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)


def find_data_dir(given: Optional[str]) -> str:
    """The Task 8 folder is the one that contains train/ and dev/ (search Kaggle inputs if not given)."""
    candidates = [given] if given else []
    candidates += ["data/task8", "../data/task8"]
    candidates += [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/train", recursive=True)]
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "train")) and os.path.isdir(os.path.join(c, "dev")):
            return c
    raise FileNotFoundError("Task 8 data not found: expected a folder with train/ and dev/. Pass --data <folder>.")


def run_experiments(names: List[str], seeds: List[int], out_dir: str, zip_path: str, data_dir: Optional[str],
                    lane: str = "", deadline: Optional[float] = None, minutes_per_unit: float = 10.0):
    import torch  # imported here so that `--help` and the plan tools work without torch

    os.makedirs(out_dir, exist_ok=True)
    tag = lane or "main"
    log_file = open(os.path.join(out_dir, f"log_{tag}.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}][{tag}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")
        log_file.flush()

    def save_zip(reason):
        try:
            zip_dir(out_dir, zip_path)
            log(f"zip updated ({os.path.getsize(zip_path) / 1e6:.1f} MB) {reason}")
        except Exception as e:  # noqa: BLE001
            log(f"ZIP FAILED: {e}")

    atexit.register(save_zip, "(on exit)")

    data_dir = find_data_dir(data_dir)
    train_df, fa_df = load_task8(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    log(f"device={gpu_name} data={data_dir}")
    log(f"experiments={names} seeds={seeds}")
    dapt_root = os.path.join(os.path.dirname(os.path.abspath(out_dir)), f"dapt_cache_{tag}")  # not zipped
    errors_path = os.path.join(out_dir, f"errors_{tag}.jsonl")
    skipped = []

    for name in names:
        cfg = EXPERIMENTS[name]
        log(f"=== {name} ===")
        for seed in seeds:
            if os.path.exists(os.path.join(out_dir, name, f"seed{seed}", "result.json")):
                log(f"  seed {seed}: already done, skipping")
                continue
            estimate_s = cfg["cost"] * minutes_per_unit * 60 * 1.25  # 25 % safety margin
            if deadline is not None and time.time() + estimate_s > deadline:
                log(f"  seed {seed}: NOT STARTED (would pass the session deadline)")
                skipped.append({"exp": name, "seed": seed})
                continue
            log(f"  seed {seed}: start")
            t0 = time.time()
            result = safe_run_seed(log=log, errors_path=errors_path, name=name, cfg=cfg, seed=seed,
                                   train_df=train_df, fa_df=fa_df, out_dir=out_dir, device=device,
                                   data_dir=data_dir, dapt_root=dapt_root)
            if result is not None:
                test = result["metrics"]["raw"]["test"]["LANGMEAN"]
                fa = result["metrics"]["raw"]["fa"]["ALL"]
                log(f"  seed {seed}: done in {(time.time() - t0) / 60:.1f} min | test A={test.get('A', 0):.3f} "
                    f"B={test.get('B', 0):.3f} C={test.get('C', 0):.3f} | FA mean={fa.get('mean', 0):.3f}")
            if cfg["cost"] >= 3.0:  # heavy runs: save after every seed
                summarize(out_dir, name)
                save_zip(f"(after {name} seed {seed})")
        summarize(out_dir, name)
        save_zip(f"(after experiment {name})")

    if skipped:
        with open(os.path.join(out_dir, f"skipped_{tag}.json"), "w") as f:
            json.dump(skipped, f, indent=1)
        log(f"{len(skipped)} runs were skipped for lack of time (listed in skipped_{tag}.json)")
    log("LANE FINISHED")
