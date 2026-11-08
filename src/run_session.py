"""Run ONE Kaggle session of the plan. This is the only command you normally need:

    python src/run_session.py --session r1s1

What it does
  1. reads src/plan.json and finds the session;
  2. starts one worker per GPU lane (gpu0 and gpu1 run in parallel on a T4 x2 machine);
  3. each worker saves + zips its results after every experiment;
  4. at the end it also writes one combined zip:  /kaggle/working/<session>.zip   <- download this one.
If a lane crashes or the time runs out, the zips written earlier are still complete.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from task8.runner import zip_dir  # noqa: E402


def load_session(plan_path, session_id):
    plan = json.load(open(plan_path, encoding="utf-8"))
    for rnd in plan["rounds"]:
        for s in rnd["sessions"]:
            if s["id"] == session_id:
                return plan["meta"], rnd, s
    ids = [s["id"] for r in plan["rounds"] for s in r["sessions"]]
    sys.exit(f"Unknown session '{session_id}'. Available: {ids}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="session id from plan.json, e.g. r1s1")
    ap.add_argument("--plan", default=os.path.join(HERE, "plan.json"))
    ap.add_argument("--out_root", default="/kaggle/working" if os.path.isdir("/kaggle/working") else "session_out")
    ap.add_argument("--data", default=None)
    ap.add_argument("--one_gpu", action="store_true", help="only 1 GPU available: run the lanes one after another")
    ap.add_argument("--dry_run", action="store_true", help="print what would run, run nothing")
    a = ap.parse_args()

    meta, rnd, session = load_session(a.plan, a.session)
    default_seeds = session.get("seeds", [0, 1, 2, 3, 4])
    deadline = time.time() + meta["hard_stop_hours"] * 3600
    session_dir = os.path.join(a.out_root, a.session)
    os.makedirs(session_dir, exist_ok=True)

    print(f"=== session {a.session}  (round {rnd['id']}: {rnd['goal']})")
    jobs = []
    for lane, spec in session["lanes"].items():
        if not spec["experiments"]:
            continue
        seeds = spec.get("seeds", default_seeds)
        out = os.path.join(session_dir, lane)
        cmd = [sys.executable, os.path.join(HERE, "run_experiments.py"),
               "--exps", *spec["experiments"], "--seeds", *map(str, seeds),
               "--out", out, "--zip", os.path.join(a.out_root, f"{a.session}_{lane}.zip"),
               "--lane", lane, "--deadline", str(deadline), "--minutes_per_unit", str(meta["minutes_per_cost_unit"])]
        if a.data:
            cmd += ["--data", a.data]
        if not a.one_gpu:
            cmd += ["--gpu", str(int(lane.replace("gpu", "")))]
        print(f"  {lane}: seeds={seeds} experiments={spec['experiments']}")
        jobs.append((lane, cmd))
    if a.dry_run:
        return

    if a.one_gpu:  # sequential
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        for lane, cmd in jobs:
            subprocess.call(cmd, env=env)
    else:          # parallel, one process per GPU
        env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}  # less memory fragmentation
        procs = [(lane, subprocess.Popen(cmd, env=env)) for lane, cmd in jobs]
        for lane, p in procs:
            p.wait()
            print(f"lane {lane} exited with code {p.returncode}")

    zip_dir(session_dir, os.path.join(a.out_root, f"{a.session}.zip"))
    print(f"\nDONE. Download: {os.path.join(a.out_root, a.session + '.zip')}")


if __name__ == "__main__":
    main()
