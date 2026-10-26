"""Sensing report: felt vs true contact force, per imperfection and per fix.

    python -m blindtouch.eval.sensing --n 20 --workers 3

The same scripted touch protocol runs on the flat hole plate (4-5 cm beside the hole, so the only contact is the
peg tip on the plate), with each realism effect alone, all together, and the clean plant. Every condition runs
at four cumulative fix levels on the same seeds:

    raw       momentum-observer residual mapped to the tip (the classic sensorless estimate); no feed-forward
    +dob      contact-gated disturbance observer: free-space estimate fed forward and subtracted, frozen at the
              hover before a deliberate contact (blindtouch.skills.touch_robot.TouchRobot.hold_contact, as every skill does)
    +cog      + anticogging map from the free-space sweeps (blindtouch.sensing.anticogging), as every job does
    +int      + lateral and free-space vertical integral (the job settings)

Protocol (positions from the known plate height; the felt force decides nothing, it is only recorded):
hover 3 cm above the plate -> hold free 1 s -> free move 3 cm and back -> descend -> press 3 N, 8 N (1 s each)
-> slide 3 cm at 2 cm/s pressing 3 N -> lift 3 cm -> hold free 1 s.

Per phase: error of the felt force (rob.force) against the true contact force (20 ms average), the achieved vs
commanded press force, and in free space the phantom force and the share of ticks above the 2.5 N contact
threshold. Results: results/sensing.json, results/sensing.md.
"""

from __future__ import annotations

import argparse
import json
import time
import multiprocessing as mp

import numpy as np

from blindtouch.paths import RESULTS
from blindtouch.scenes.build import GEOM
from blindtouch.sim.env import Env
from blindtouch.sim.realism import RealismConfig
from blindtouch.skills.touch_robot import KP, TouchRobot

SEED0 = 960_000
CONDS = ("clean", "friction", "noise", "ripple", "latency", "model_error", "all")
LEVELS = ("raw", "+dob", "+cog", "+int")
F_CONTACT = 2.5     # N: the touch layer's contact threshold (touch_down)
PRESS = (3.0, 8.0)


class _Cal:
    """Just the calibration routine of blindtouch.jobs.base.Job (lift 4 cm, sweep, return)."""

    CALIB_LIFT = 0.04

    def __init__(self):
        from blindtouch.jobs.base import Job

        self.calibrate_motors = Job.calibrate_motors.__get__(self)


def run_one(env: Env, cond: str, level: str, seed: int, yaw_deg: float | None = None, limit: bool = True) -> dict:
    """``yaw_deg``: hold the tool at this yaw (as the jobs do) instead of leaving it free; ``limit``: null-space
    joint-limit avoidance on (the job setting) or off."""
    rng = np.random.default_rng(seed)
    hole = np.asarray(GEOM.hole_nominal)
    side = 1 if rng.random() < 0.5 else -1
    xy0 = hole + [side * rng.uniform(0.035, 0.055), rng.uniform(-0.035, -0.025)]
    top = GEOM.plate_top
    env.realism.cfg = RealismConfig.only(cond)
    yaw = None if yaw_deg is None else np.radians(yaw_deg)
    env.reset(seed=seed, hole_xy=hole, tip_start=np.r_[xy0, top + 0.03], tip_yaw=yaw)
    g = env.ctrl.g
    keep = (g.dob, g.ki_pos.copy(), g.ki_z_free, g.limit_margin, g.f_cap)
    g.dob = True
    g.ki_pos = np.array([1000.0, 1000.0, 0.0]) if level == "+int" else np.zeros(3)
    g.ki_z_free = 1000.0 if level == "+int" else 0.0
    g.limit_margin, g.f_cap = (0.25 if limit else None), 15.0
    env.ctrl.cogging = None
    env.ctrl.i_force[:] = 0.0
    if level in ("+cog", "+int"):
        _Cal().calibrate_motors(env)
    rob = TouchRobot(env)
    rob.yaw_cmd = yaw
    if level == "raw":   # the residual is computed, but no estimate is ever learned or subtracted
        env.ctrl.dob.reset()
        env.ctrl.dob.hold = True
    rec: dict[str, list] = {}
    lo, hi = env.dyn.q_lo, env.dyn.q_hi
    margin = [np.inf]

    def run(phase: str | None, seconds: float, cmd=None, cmd_end=None):
        n = int(round(seconds / 0.01))
        c0 = rob.cmd.copy() if cmd is None else np.asarray(cmd, float)
        for k in range(n):
            rob.cmd = c0 if cmd_end is None else c0 + (np.asarray(cmd_end) - c0) * (k + 1) / n
            rob.tick()
            margin[0] = min(margin[0], float(np.min(np.minimum(env.last.q - lo, hi - env.last.q))))
            if phase is not None:
                rec.setdefault(phase, []).append(np.r_[rob.force(), np.mean(rob._fwin, axis=0), env.tip_pos(), rob.cmd])

    hover = np.r_[xy0, top + 0.03]
    rob.cmd = hover.copy()
    run(None, 1.0, hover)
    run("free hold", 1.0, hover)
    far = hover + [0, 0.03, 0]
    run("free move", 1.0, hover, far)
    run("free move", 1.0, far, hover)
    run(None, 0.5, hover)
    if level != "raw":
        rob.hold_contact(snapshot=False)
    p3 = np.r_[xy0, top - PRESS[0] / KP]
    run(None, 0.03 / 0.015, hover, p3)         # descend at 1.5 cm/s
    run(None, 0.5, p3)
    run("press 3 N", 1.0, p3)
    p8 = np.r_[xy0, top - PRESS[1] / KP]
    run(None, 0.5, p8)
    run("press 8 N", 1.0, p8)
    run(None, 0.5, p3)
    run("slide 3 N", 1.5, p3, p3 + [0, 0.03, 0])
    end = p3 + [0, 0.03, 0]
    up = np.r_[end[:2], top + 0.03]
    run(None, 1.0, end, up)
    if level != "raw":
        rob.release_contact()
    run(None, 0.5, up)
    run("free after", 1.0, up)
    g.dob, g.ki_pos, g.ki_z_free, g.limit_margin, g.f_cap = keep
    env.ctrl.cogging = None

    out = {"cond": cond, "level": level, "seed": seed, "yaw_deg": yaw_deg, "limit": limit,
           "margin_deg": float(np.degrees(margin[0]))}
    for ph, rows in rec.items():
        a = np.array(rows)
        felt, true, tip, cmd = a[:, :3], a[:, 3:6], a[:, 6:9], a[:, 9:12]
        err = np.linalg.norm(felt - true, axis=1)
        m = {"err_rms": float(np.sqrt(np.mean(err ** 2))), "err_max": float(err.max()),
             "felt_z": float(np.mean(felt[:, 2])), "true_z": float(np.mean(true[:, 2])),
             "true_max": float(np.linalg.norm(true, axis=1).max())}
        if ph.startswith("free"):
            fn = np.linalg.norm(felt, axis=1)
            m["false_contact"] = float(np.mean(fn > F_CONTACT))
            m["pos_err_mm"] = float(np.median(np.linalg.norm(tip - cmd, axis=1)) * 1e3)
        if ph.startswith("press"):
            m["cmd_z"] = float(ph.split()[1])
        if ph.startswith("slide"):
            m["true_lat"] = float(np.mean(np.linalg.norm(true[:, :2], axis=1)))
            m["felt_lat"] = float(np.mean(np.linalg.norm(felt[:, :2], axis=1)))
        out[ph] = m
    return out


def _work(jobs):
    env = Env("peg")
    rows = []
    for cond, level, seed, *kw in jobs:
        t0 = time.time()
        rows.append(run_one(env, cond, level, seed, **(kw[0] if kw else {})))
        print(f"{cond:12s} {level:5s} {seed} {time.time() - t0:5.1f}s  press3 err {rows[-1]['press 3 N']['err_rms']:.2f} N", flush=True)
    return rows


PHASES = ("free hold", "free move", "press 3 N", "press 8 N", "slide 3 N", "free after")


def report(rows: list[dict]) -> str:
    def med(cond, level, ph, key):
        v = [r[ph][key] for r in rows if r["cond"] == cond and r["level"] == level]
        return float(np.median(v)) if v else float("nan")

    n = len({r["seed"] for r in rows})
    L = ["# Sensing report: felt vs true contact force without a force sensor", "",
         f"`python -m blindtouch.eval.sensing` -- {n} seeds per cell ({SEED0}+), same seeds for every fix level.",
         "Scripted protocol on the flat plate (see the script). Medians over seeds. Felt force = momentum-observer",
         "residual mapped to the tip, after each level's fixes; true force = simulator contact force, 20 ms average.", ""]
    L += ["## 1. Felt force error in contact (RMS of |felt - true|, N; press 3 N / press 8 N / slide at 3 N)", "",
          "| imperfection | " + " | ".join(LEVELS) + " |", "|---" * (len(LEVELS) + 1) + "|"]
    for c in [c for c in CONDS if any(r["cond"] == c for r in rows)]:
        cells = [" / ".join(f"{med(c, lv, ph, 'err_rms'):.2f}" for ph in ("press 3 N", "press 8 N", "slide 3 N")) for lv in LEVELS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 2. Phantom force in free space (RMS |felt|, N; hold / moving 3 cm/s / after a contact) and false-contact share", "",
          "| imperfection | " + " | ".join(LEVELS) + " |", "|---" * (len(LEVELS) + 1) + "|"]
    for c in [c for c in CONDS if any(r["cond"] == c for r in rows)]:
        cells = []
        for lv in LEVELS:
            e = " / ".join(f"{med(c, lv, ph, 'err_rms'):.2f}" for ph in ("free hold", "free move", "free after"))
            fc = np.mean([med(c, lv, ph, "false_contact") for ph in ("free hold", "free move", "free after")])
            cells.append(f"{e} ({fc:.0%})")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Achieved press force (true F_z, N) for a commanded 3 N / 8 N", "",
          "| imperfection | " + " | ".join(LEVELS) + " |", "|---" * (len(LEVELS) + 1) + "|"]
    for c in [c for c in CONDS if any(r["cond"] == c for r in rows)]:
        cells = [" / ".join(f"{med(c, lv, ph, 'true_z'):.2f}" for ph in ("press 3 N", "press 8 N")) for lv in LEVELS]
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L += ["", "## 4. Free-space position error (true tip vs command, hold, mm)", "",
          "| imperfection | " + " | ".join(LEVELS) + " |", "|---" * (len(LEVELS) + 1) + "|"]
    for c in [c for c in CONDS if any(r["cond"] == c for r in rows)]:
        L.append(f"| {c} | " + " | ".join(f"{med(c, lv, 'free hold', 'pos_err_mm'):.2f}" for lv in LEVELS) + " |")
    return "\n".join(L) + "\n"


LIMIT_YAWS = (85, 155, 235, 260, 280)   # deg: 85 inside the usable window; the others 1.5-3.5 deg from a wrist stop


def report_limits(rows: list[dict]) -> str:
    L = ["# Joint end stops read as contact force: null-space limit avoidance", "",
         "`python -m blindtouch.eval.sensing --limits` -- all imperfections on, all other fixes on (+int level), tool yaw",
         "held as the jobs do. Medians over seeds; margin = smallest distance of any joint to its stop during the run.", "",
         "| yaw | avoidance | min margin (deg) | phantom free hold / after (N) | false contacts | error press 3 N / 8 N / slide (N) |",
         "|---|---|---|---|---|---|"]
    for y in LIMIT_YAWS:
        for lim in (False, True):
            rr = [r for r in rows if r["yaw_deg"] == y and r["limit"] == lim]
            if not rr:
                continue
            md = lambda ph, k: float(np.median([r[ph][k] for r in rr]))
            fc = np.mean([md(ph, "false_contact") for ph in ("free hold", "free move", "free after")])
            L.append(f"| {y} | {'on' if lim else 'off'} | {np.median([r['margin_deg'] for r in rr]):.1f} | "
                     f"{md('free hold', 'err_rms'):.2f} / {md('free after', 'err_rms'):.2f} | {fc:.0%} | "
                     + " / ".join(f"{md(ph, 'err_rms'):.2f}" for ph in ("press 3 N", "press 8 N", "slide 3 N")) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--conds", nargs="*", default=list(CONDS))
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limits", action="store_true", help="the end-stop test instead of the main table")
    args = ap.parse_args()
    if args.limits:
        jobs = [("all", "+int", SEED0 + i, {"yaw_deg": y, "limit": lim}) for y in LIMIT_YAWS for lim in (False, True)
                for i in range(args.n)]
    else:
        jobs = [(c, lv, SEED0 + i) for c in args.conds for lv in args.levels for i in range(args.n)]
    out = args.out or ("sensing_limits" if args.limits else "sensing")
    rep = report_limits if args.limits else report
    res = RESULTS
    # every finished run is appended at once (a crash loses nothing; a rerun resumes); workers are recycled
    # after a few jobs so memory cannot creep up over a long run
    log = res / f"{out}.jsonl"
    rows = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    key = lambda c, lv, sd, kw=None: (c, lv, sd, (kw or {}).get("yaw_deg"), (kw or {}).get("limit", True))
    done = {key(r["cond"], r["level"], r["seed"], r) for r in rows}
    todo = [j for j in jobs if key(*j) not in done]
    print(f"{len(rows)} runs already done, {len(todo)} to go", flush=True)
    # multiprocessing.Pool, not ProcessPoolExecutor: the latter's max_tasks_per_child hangs on Windows (Python 3.12)
    with mp.Pool(args.workers, maxtasksperchild=4) as pool, log.open("a") as f:
        for part in pool.imap_unordered(_work, [[j] for j in todo]):
            for r in part:
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
    rows = [r for r in rows if key(r["cond"], r["level"], r["seed"], r) in {key(*j) for j in jobs}]
    (res / f"{out}.json").write_text(json.dumps(rows, indent=1))
    (res / f"{out}.md").write_text(rep(rows))
    print(rep(rows))


if __name__ == "__main__":
    main()
