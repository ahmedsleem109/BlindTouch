"""Robustness check: the jobs, code unchanged, on worlds wider than the ones they were developed on.

    python -m blindtouch.eval.robustness --trials 30 --workers 8

One variation at a time (a job's ``RANGES`` widened past what it was tuned on), plus all five realism effects on
at once (developed with each on at p = 0.5). New seeds 850000+, the same seeds for every variant, including the
tuned ranges as the reference. No re-tuning: a drop is reported as it is. Every finished run is appended to
results/robustness.jsonl (a rerun resumes); table in results/robustness.md.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp

from blindtouch.paths import RESULTS
from blindtouch.eval.jobs import job_class, summary

SEED0 = 850_000
# job -> variant -> (range overrides, realism p, plain-language description)
VARIANTS = {
    "socket": {
        "tuned": ({}, 0.5, "socket +-2 cm, turned +-15 deg, pin friction 0.2-0.4"),
        "position x2": ({"pos": 0.04}, 0.5, "socket anywhere within +-4 cm"),
        "angle x2": ({"yaw": math.radians(30)}, 0.5, "socket turned up to +-30 deg (the pin search covers +-25 deg)"),
        "friction": ({"pin_mu": (0.1, 0.6)}, 0.5, "pin friction 0.1-0.6"),
        "all effects": ({}, 1.0, "all five motor/model imperfections at once"),
    },
    "whiteboard": {
        "tuned": ({}, 0.5, "slope +-4 deg, height +-1 cm, turned +-10 deg, +-2 cm, board friction 0.30-0.45"),
        "slope x2": ({"tilt_deg": 8.0}, 0.5, "slope +-8 deg"),
        "height x2": ({"dz": 0.02}, 0.5, "board height +-2 cm"),
        "placement x2": ({"yaw_deg": 20.0, "xy": 0.04}, 0.5, "board turned +-20 deg, moved +-4 cm"),
        "friction": ({"board_mu": (0.2, 0.6), "felt_mu": (0.3, 0.7)}, 0.5, "board 0.2-0.6, felt 0.3-0.7"),
        "all effects": ({}, 1.0, "all five motor/model imperfections at once"),
    },
    "peg": {
        "tuned": ({}, 0.5, "hole within +-8 mm of where the robot expects it"),
        "position x2": ({"pos": 0.016}, 0.5, "hole within +-16 mm (the spiral reaches 16 mm)"),
        "all effects": ({}, 1.0, "all five motor/model imperfections at once"),
    },
    "cans": {
        "tuned": ({}, 0.5, "cans placed +-3 mm, friction 0.4-0.7"),
        "placement x3": ({"jitter": 0.010}, 0.5, "cans placed +-10 mm off their spots"),
        "friction": ({"can_mu": (0.25, 0.9)}, 0.5, "can friction 0.25-0.9 (greasy to rubbery)"),
        "all effects": ({}, 1.0, "all five motor/model imperfections at once"),
    },
}


def _work(task):
    job, variant, seed = task
    over, p, _ = VARIANTS[job][variant]
    row = job_class(job)(**over).run(seed, realism_p=p).row(seed)
    return {"job": job, "variant": variant, **row}


def report(rows: list[dict]) -> str:
    L = ["# Robustness: the jobs on worlds they were not developed on", "",
         "`python -m blindtouch.eval.robustness` -- code unchanged, one variation at a time, new seeds 850000+ (the same seeds in "
         "every row). Success needs the job done *and* the peak true contact force within the job's budget.", "",
         "| job | variant | what changed | success (95% CI) | n | change vs tuned | peak force (N, median) |",
         "|---|---|---|---|---|---|---|"]
    for job, vs in VARIANTS.items():
        base = None
        for v, (_, _, desc) in vs.items():
            rr = [r for r in rows if r["job"] == job and r["variant"] == v]
            if not rr:
                continue
            s = summary(rr)
            base = s["success"] if v == "tuned" else base
            d = "" if v == "tuned" or base is None else f"{100 * (s['success'] - base):+.0f} pts"
            L.append(f"| {job} | {v} | {desc} | {s['success']:.0%} ({s['ci95'][0]:.0%}-{s['ci95'][1]:.0%}) | {s['n']} | {d} | "
                     f"{s['peak_force_N_median']:.1f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--jobs", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()
    log = RESULTS / "robustness.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    done = {(r["job"], r["variant"], r["seed"]) for r in rows}
    todo = [(j, v, SEED0 + i) for j in args.jobs for v in VARIANTS[j] for i in range(args.trials)
            if (j, v, SEED0 + i) not in done]
    print(f"{len(rows)} runs already done, {len(todo)} to go", flush=True)
    # multiprocessing.Pool, not ProcessPoolExecutor: the latter's max_tasks_per_child hangs on Windows (Python 3.12)
    with mp.Pool(args.workers, maxtasksperchild=4) as pool, log.open("a") as f:
        for r in pool.imap_unordered(_work, todo):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(r["job"], r["variant"], r["seed"], "ok" if r["success"] else "FAIL", f"{r['time']:.0f}s", flush=True)
    (RESULTS / "robustness.md").write_text(report(rows), encoding="utf-8")
    print(report(rows))


if __name__ == "__main__":
    main()
