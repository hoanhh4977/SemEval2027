"""Print (and optionally store) the estimated duration of every session of plan.json.

    python src/plan_summary.py            # just print
    python src/plan_summary.py --write    # also write "estimated_hours" into plan.json

estimated hours of a lane = sum over its experiments of  cost x n_seeds x minutes_per_cost_unit
                            + n_runs x overhead_minutes_per_run
Change `minutes_per_cost_unit` in plan.json after round R0 (when you know the real speed) and run --write again.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from task8.experiments import EXPERIMENTS  # noqa: E402


def lane_hours(meta, spec, default_seeds):
    seeds = spec.get("seeds", default_seeds)
    minutes = 0.0
    for name in spec["experiments"]:
        cost = EXPERIMENTS[name]["cost"]
        minutes += len(seeds) * (cost * meta["minutes_per_cost_unit"] + meta["overhead_minutes_per_run"])
    return minutes / 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=os.path.join(HERE, "plan.json"))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    plan = json.load(open(a.plan, encoding="utf-8"))
    meta = plan["meta"]
    print(f"minutes per cost unit = {meta['minutes_per_cost_unit']}, hard stop = {meta['hard_stop_hours']} h\n")
    for rnd in plan["rounds"]:
        print(f"{rnd['id']}: {rnd['goal'][:100]}")
        for s in rnd["sessions"]:
            default_seeds = s.get("seeds", [0, 1, 2, 3, 4])
            hours = {lane: round(lane_hours(meta, spec, default_seeds), 1) for lane, spec in s["lanes"].items()}
            s["estimated_hours"] = hours
            n_exp = sum(len(spec["experiments"]) for spec in s["lanes"].values())
            flag = "  <-- longer than the hard stop!" if max(hours.values()) > meta["hard_stop_hours"] else ""
            print(f"   {s['id']:6s} {n_exp:2d} experiments | hours per lane: {hours}{flag}")
        if not rnd["sessions"]:
            print("   (no session yet)")
    if a.write:
        json.dump(plan, open(a.plan, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print("\nplan.json updated with estimated_hours")


if __name__ == "__main__":
    main()
