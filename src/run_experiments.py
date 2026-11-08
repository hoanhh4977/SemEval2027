"""Worker: run some experiments on ONE GPU. Normally started by run_session.py, but you can also use it by hand:

    python src/run_experiments.py --exps P B_majority --seeds 0 1 2 --out out_test
    python src/run_experiments.py --list                      # show every experiment name and its cost
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from task8.experiments import EXPERIMENTS, GROUPS, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=[], help="experiment names, group names, or 'all'")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="out")
    ap.add_argument("--zip", default=None, help="zip file updated after every experiment (default: <out>.zip)")
    ap.add_argument("--data", default=None, help="folder containing train/ and dev/ (auto-detected on Kaggle)")
    ap.add_argument("--gpu", type=int, default=None, help="physical GPU index to use")
    ap.add_argument("--lane", default="", help="label used in log lines, e.g. gpu0")
    ap.add_argument("--deadline", type=float, default=None, help="unix time after which no new run is started")
    ap.add_argument("--minutes_per_unit", type=float, default=10.0)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    if a.list:
        for group, names in GROUPS.items():
            print(f"[{group}]")
            for n in names:
                print(f"    {n:26s} cost={EXPERIMENTS[n]['cost']}")
        return
    if a.gpu is not None:  # must be set before torch is imported
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    from task8.runner import run_experiments
    out = os.path.abspath(a.out)
    run_experiments(resolve(a.exps), a.seeds, out, a.zip or out.rstrip("/\\") + ".zip", a.data,
                    lane=a.lane, deadline=a.deadline, minutes_per_unit=a.minutes_per_unit)


if __name__ == "__main__":
    main()
