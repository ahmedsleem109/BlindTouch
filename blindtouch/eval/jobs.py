"""Evaluate the contact jobs on the shared touch layer: one table, same layer, same settings.

    python -m blindtouch.eval.jobs --jobs peg socket whiteboard cans --trials 50 --workers 3

Every job: seeds 800000+, realism effects each on with p=0.5, success judged from simulator truth, peak true
contact force (20 ms average), time, number of touches. Results: results/jobs/<job>.json, results/jobs.md.
"""

from __future__ import annotations

import argparse
import json
import time
import multiprocessing as mp

import numpy as np

from blindtouch.paths import RESULTS

RES = RESULTS / "jobs"
SEED0 = 800_000


JOBS = {"socket": ("blindtouch.jobs.socket", "SocketJob"), "whiteboard": ("blindtouch.jobs.whiteboard", "WhiteboardJob"),
        "cans": ("blindtouch.jobs.cans", "CansJob"), "peg": ("blindtouch.jobs.peg", "PegJob")}


def job_class(name: str):
    """The job's class, imported on demand (each job is its own module)."""
    import importlib

    mod, cls = JOBS[name]
    return getattr(importlib.import_module(mod), cls)


def _work(args):
    name, seeds, p = args
    job = job_class(name)()
    rows = []
    for s in seeds:
        rows.append(job.run(s, realism_p=p).row(s))
    return rows


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def summary(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["success"]]
    lo, hi = wilson(len(ok), len(rows))
    return {"n": len(rows), "success": len(ok) / len(rows), "ci95": [lo, hi],
            "time_s": float(np.mean([r["time"] for r in ok])) if ok else float("nan"),
            "peak_force_N_median": float(np.median([r["peak_force"] for r in rows])),
            "touches_mean": float(np.mean([r["touches"] for r in rows]))}


def write_table() -> None:
    lines = ["# Everyday jobs on one touch layer", "",
             "Simulation (OpenArm v2, MuJoCo). Felt force only from motor signals; no force sensor, no camera. "
             "Seeds 800000+, realism effects each on with p=0.5.", "",
             "| job | success (95% CI) | n | time (s, successes) | peak true force (N, median) | touches |", "|---|---|---|---|---|---|"]
    for f in sorted(RES.glob("*.json")):
        d = json.load(open(f))
        s = d["summary"]
        lines.append(f"| {d['title']} | {s['success']:.0%} ({s['ci95'][0]:.0%}-{s['ci95'][1]:.0%}) | {s['n']} | {s['time_s']:.1f} | {s['peak_force_N_median']:.1f} | {s['touches_mean']:.1f} |")
    (RESULTS / "jobs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--realism-p", type=float, default=0.5)
    args = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    for name in args.jobs:
        t0 = time.time()
        seeds = list(range(SEED0, SEED0 + args.trials))
        chunks = [(name, seeds[k::args.workers], args.realism_p) for k in range(args.workers)]
        with mp.Pool(args.workers) as pool:
            rows = sorted((r for part in pool.map(_work, chunks) for r in part), key=lambda r: r["seed"])
        s = summary(rows)
        json.dump({"job": name, "title": job_class(name).title, "summary": s, "raw": rows}, open(RES / f"{name}.json", "w"), indent=1, default=float)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, f"{time.time() - t0:.0f}s", flush=True)
    write_table()


if __name__ == "__main__":
    main()
