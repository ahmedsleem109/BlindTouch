"""Short clip of one everyday job: real episodes (some successes, some failures), sped up, with the procedure
step as a title, a live force plot (true force vs what the robot felt from its motors) and the verdict.

    python -m blindtouch.video.job_clip whiteboard --seeds 800003 800011 800017 800020 --out out/job_whiteboard.mp4

Each episode is re-run with a recorder (the same seed gives the same episode as in the evaluation), the motor
calibration at the start is skipped in the clip, and every rendered frame is checked for penetration between
collision geoms (worst value printed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mujoco
import numpy as np

from blindtouch.eval.jobs import job_class
from blindtouch.paths import OUT
from blindtouch.video.studio import FONT, GREEN, RED, TRUE_C, ACCENT, Recorder, Studio, write_mp4

W, H, FPS = 1280, 720, 30
SPEED = 4.0          # playback speed (at least)
MAX_EP_S = 14.0      # s of video per trial at most
FAIL_WHY = {
    "socket": lambda i: {"turn": "pins never found their holes", "seat": "pins stuck part-way in" if float(i.get("pin_depth_mm", 0)) > 2 else "felt a drop, but the pins missed their holes",
                         "off the strip": "slid off the power strip", "centre": "could not feel the socket wall"}.get(
                             i.get("stage"), "not pushed in far enough" if i.get("stage") == "done" else f"stopped at '{i.get('stage')}'"),
    "cans": lambda i: ("weighed wrong" if not all(i.get("classified", [0])) else "") +
                      (" / not set down on its spot" if not all(i.get("placed", [0])) else ""),
    "whiteboard": lambda i: ("force not steady" if (i.get("force_rmse") or 99) > 2.0 else "") +
                            (" / ink left" if i.get("wiped", 0) < 0.9 else ""),
}
SUMMARY = {
    "socket": lambda i: f"pins {max(0.0, float(i.get('pin_depth_mm', 0))):.0f} mm into the socket (seated = 19 mm)",
    "cans": lambda i: "weighing error " + ", ".join(f"{e:.0f} g" for e in i.get("mass_err_g", [])) if "mass_err_g" in i
                      else i.get("aborted", ""),
    "whiteboard": lambda i: f"force error {i['force_rmse']:.1f} N RMS (limit 2.0) - {i['wiped']:.0%} of the ink wiped"
                  if i.get("force_rmse") is not None else f"{i.get('aborted', 'stopped')}",
}
CAMS = {   # lookat offset from the fixture (cans: from the table centre), distance, azimuth, elevation
    "socket": ((0.0, 0.0, 0.05), 0.42, 35.0, -42.0),
    "cans": ((0.28, -0.22, 0.51), 0.85, 20.0, -28.0),
    "whiteboard": ((0.0, 0.0, 0.08), 0.80, 30.0, -32.0),
}


class EpisodeView:
    """One recorded episode: frames, felt force (from the touch layer's trace), steps, verdict."""

    def __init__(self, kind: str, seed: int, **ranges) -> None:
        job = job_class(kind)(**ranges)
        rec = Recorder()
        self.ink = None
        hooks = [rec]
        if kind == "whiteboard":
            hooks.append(lambda env, s: rec.note(erased=np.flatnonzero(job.erased).tolist() if hasattr(job, "erased") else []))
        res = job.run(seed, hooks=hooks)
        self.kind, self.seed, self.res, self.job = kind, seed, res, job
        ev = job.rob.events
        t_first = next((t for t, n, _ in ev if n == "step"), 0.0)
        self.frames = [f for f in rec.frames if f.t >= t_first - 0.3]
        self.steps = [(t, d["title"]) for t, n, d in ev if n == "step"]
        tr = job.rob.trace
        self.tt = np.array([x[0] for x in tr])
        self.felt = np.array([x[2] for x in tr])
        env = job.env
        self.model_pose = (env.model.body_pos.copy(), env.model.body_quat.copy(),
                           env.model.geom_friction.copy())
        self.n = getattr(job, "n_true", np.array([0, 0, 1.0]))
        self.fixture = env.data.mocap_pos[env.fixture_mocap].copy() if env.fixture_mocap >= 0 else np.zeros(3)
        self.mocap_quat = env.data.mocap_quat.copy()     # the fixture's turn (the recorder keeps positions only)

    def felt_at(self, t: float) -> np.ndarray:
        if not len(self.tt):
            return np.zeros(3)
        k = int(np.clip(np.searchsorted(self.tt, t), 0, len(self.tt) - 1))
        return self.felt[k]

    def step_at(self, t: float) -> tuple[int, str]:
        k, title = 0, ""
        for i, (ts, ti) in enumerate(self.steps):
            if ts <= t:
                k, title = i + 1, ti
        return k, title


def plot(img, ep: EpisodeView, i: int, window: float = 8.0) -> None:
    """Normal force along the board (whiteboard) or |F|: true (white), felt (accent), target band (green)."""
    x0, y0, w, h = 40, H - 175, W - 80, 140
    over = img.copy()
    cv2.rectangle(over, (x0 - 10, y0 - 28), (x0 + w + 10, y0 + h + 12), (18, 18, 20), -1)
    cv2.addWeighted(over, 0.7, img, 0.3, 0, img)
    t_now = ep.frames[i].t
    y_max = 16.0

    def xy(t, f):
        return int(x0 + (t - (t_now - window)) / window * w), int(y0 + h - np.clip(f, 0, y_max) / y_max * h)

    if ep.kind == "whiteboard":
        lo, hi = xy(t_now, 6.0)[1], xy(t_now, 10.0)[1]
        band = img.copy()
        cv2.rectangle(band, (x0, hi), (x0 + w, lo), GREEN, -1)
        cv2.addWeighted(band, 0.18, img, 0.82, 0, img)
        cv2.putText(img, "target 8 N +-25%", (x0 + w - 200, hi - 6), FONT, 0.5, GREEN, 1, cv2.LINE_AA)
        proj = lambda f: float(f @ ep.n)
    else:
        proj = lambda f: float(np.linalg.norm(f))
    j0 = max(0, i - int(window * 500))
    sel = [f for f in ep.frames[j0: i + 1: 5] if f.t >= t_now - window]
    # true force as a sensor reports it: 20 ms average (raw 2 ms contact samples flicker)
    true20 = lambda k: proj(np.mean([g.f_true for g in ep.frames[max(0, k - 9): k + 1]], axis=0))
    idx = list(range(j0, i + 1, 5))[-len(sel):] if sel else []
    if len(sel) > 1:
        cv2.polylines(img, [np.array([xy(f.t, true20(k)) for f, k in zip(sel, idx)], np.int32)], False, TRUE_C, 2, cv2.LINE_AA)
        cv2.polylines(img, [np.array([xy(f.t, proj(ep.felt_at(f.t))) for f in sel], np.int32)], False, ACCENT, 2, cv2.LINE_AA)
    lab = "pressing force" if ep.kind == "whiteboard" else "contact force"
    cv2.putText(img, f"{lab}:", (x0, y0 - 8), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "true (simulator)", (x0 + 150, y0 - 8), FONT, 0.55, TRUE_C, 1, cv2.LINE_AA)
    cv2.putText(img, "felt from the motors (no force sensor)", (x0 + 330, y0 - 8), FONT, 0.55, ACCENT, 1, cv2.LINE_AA)
    cv2.putText(img, f"{true20(i):4.1f} N", (x0 + w - 90, y0 - 8), FONT, 0.6, TRUE_C, 1, cv2.LINE_AA)


def banner(img, text: str, sub: str = "", color=(245, 245, 245)) -> None:
    over = img.copy()
    cv2.rectangle(over, (0, 0), (W, 84 if sub else 58), (14, 14, 16), -1)
    cv2.addWeighted(over, 0.72, img, 0.28, 0, img)
    scale = 1.0   # shrink a long step title to fit the frame
    while scale > 0.55 and cv2.getTextSize(text, FONT, scale, 2)[0][0] > W - 56:
        scale -= 0.05
    cv2.putText(img, text, (28, 40), FONT, scale, color, 2, cv2.LINE_AA)
    if sub:
        cv2.putText(img, sub, (28, 72), FONT, 0.6, (190, 190, 190), 1, cv2.LINE_AA)


def card(lines: list[tuple[str, float, tuple]], seconds: float = 2.5) -> list[np.ndarray]:
    img = np.full((H, W, 3), 22, np.uint8)
    y = H // 2 - 30 * len(lines)
    for text, scale, color in lines:
        (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
        cv2.putText(img, text, ((W - tw) // 2, y), FONT, scale, color, 2, cv2.LINE_AA)
        y += int(60 * scale) + 18
    return [img] * int(seconds * FPS)


def render_episode(studio: Studio, ep: EpisodeView, n_ep: int, k_ep: int) -> tuple[list[np.ndarray], float]:
    m = studio.model
    pos, quat, _ = ep.model_pose
    m.body_pos[:] = pos
    m.body_quat[:] = quat
    ink = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("ink")]
    ink_mat, ink_rgba = studio.base_mat[ink].copy(), studio.base_rgba[ink].copy()
    look, dist, az, el = CAMS.get(ep.kind, ((0, 0, 0.08), 0.9, 200.0, -35.0))
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (ep.fixture if ep.kind != "cans" else 0.0) + np.asarray(look)
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    out, worst = [], 0.0
    times = np.array([f.t for f in ep.frames])
    speed = max(SPEED, (times[-1] - times[0]) / MAX_EP_S)     # long trials play faster: each lasts <= MAX_EP_S
    ts = np.arange(times[0], times[-1], speed / FPS)
    ok = ep.res.success
    why = FAIL_WHY.get(ep.kind, lambda i: "")(ep.res.info).strip(" /")
    for t in ts:
        i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
        fr = ep.frames[i]
        if ink:
            m.geom_matid[ink] = ink_mat
            m.geom_rgba[ink] = ink_rgba
            for k in fr.tags.get("erased", []):
                m.geom_matid[ink[k]] = -1
                m.geom_rgba[ink[k]] = [0, 0, 0, 0]
        studio.data.mocap_quat[:] = ep.mocap_quat
        studio.set_state(fr)
        for c in range(studio.data.ncon):
            worst = min(worst, studio.data.contact[c].dist)
        studio.r.update_scene(studio.data, camera=cam)
        img = cv2.cvtColor(studio.r.render(), cv2.COLOR_RGB2BGR)
        k, title = ep.step_at(t)
        banner(img, f"{k}. {title}" if title else "", f"trial {k_ep}/{n_ep}  (seed {ep.seed}, imperfections: {ep.res.info.get('realism', '')})  x{speed:.0f} speed")
        plot(img, ep, i)
        out.append(img)
    # verdict, held on the last frame
    last = out[-1].copy()
    over = last.copy()
    cv2.rectangle(over, (W // 2 - 360, H // 2 - 80), (W // 2 + 360, H // 2 + 70), (14, 14, 16), -1)
    cv2.addWeighted(over, 0.8, last, 0.2, 0, last)
    txt = "SUCCESS" if ok else "FAILED" + (f": {why}" if why else "")
    (tw, _), _ = cv2.getTextSize(txt, FONT, 1.3, 3)
    cv2.putText(last, txt, ((W - tw) // 2, H // 2 - 15), FONT, 1.3, GREEN if ok else RED, 3, cv2.LINE_AA)
    sm = SUMMARY.get(ep.kind, lambda i: "")(ep.res.info)
    (tw, _), _ = cv2.getTextSize(sm, FONT, 0.65, 1)
    cv2.putText(last, sm, ((W - tw) // 2, H // 2 + 35), FONT, 0.65, (220, 220, 220), 1, cv2.LINE_AA)
    out += [last] * int(2.2 * FPS)
    return out, worst


def montage(plan: list[tuple[str, list[int], str]], out: Path) -> None:
    """One video of several jobs: per job a title card (with its status), then its trials."""
    frames = card([("Blind Touch: everyday jobs by touch", 1.3, (245, 245, 245)),
                   ("a low-cost arm feels contact through its motors only - no force sensor, no camera at contact", 0.62, (200, 200, 200)),
                   ("simulation (MuJoCo, OpenArm v2), real trials: successes and failures", 0.62, (200, 200, 200))], 3.0)
    worst = 0.0
    for kind, seeds, status in plan:
        title = job_class(kind).title
        frames += card([(title, 1.3, (245, 245, 245)), (status, 0.75, (200, 200, 200))], 2.5)
        studio = Studio(kind, W, H)
        for k, s_ in enumerate(seeds, 1):
            ep = EpisodeView(kind, s_)
            print(f"{kind} seed {s_}: success={ep.res.success} {SUMMARY.get(kind, lambda i: '')(ep.res.info)}", flush=True)
            f, w = render_episode(studio, ep, len(seeds), k)
            frames += f
            worst = min(worst, w)
        del studio
    write_mp4(str(out), frames, fps=FPS, crf=24)
    print(f"wrote {out} ({len(frames) / FPS:.0f} s); worst penetration {-worst * 1e3:.2f} mm")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rate", default="", help="evaluation result to show on the title card, e.g. '46/50 (92 %)'")
    args = ap.parse_args()
    out = Path(args.out or OUT / f"job_{args.kind}.mp4")
    title = job_class(args.kind).title
    eps = []
    for s in args.seeds:
        ep = EpisodeView(args.kind, s)
        print(f"seed {s}: success={ep.res.success} {SUMMARY.get(args.kind, lambda i: '')(ep.res.info)}", flush=True)
        eps.append(ep)
    studio = Studio(args.kind, W, H)
    frames = card([(title, 1.4, (245, 245, 245)),
                   ("felt only through the motors - no force sensor, no camera at contact", 0.7, (200, 200, 200)),
                   (f"{len(eps)} real trials: successes and failures" + (f" - evaluation {args.rate}" if args.rate else ""), 0.7, (200, 200, 200))])
    worst = 0.0
    for k, ep in enumerate(eps, 1):
        f, w = render_episode(studio, ep, len(eps), k)
        frames += f
        worst = min(worst, w)
    write_mp4(str(out), frames, fps=FPS, crf=23)
    print(f"wrote {out} ({len(frames) / FPS:.0f} s); worst penetration {-worst * 1e3:.2f} mm")


if __name__ == "__main__":
    main()
