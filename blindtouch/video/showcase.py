"""Showcase cut (~80 s) for LinkedIn / Enact: real simulated episodes, successes and failures, little text.

    python -m blindtouch.video.showcase                 # -> out/blind_touch_showcase.mp4
    python -m blindtouch.video.showcase --only socket          # one part, for a quick look

Storyboard (agreed with the user 2026-10-25): title -> the idea (a press on a plate: felt from the motors vs the
true force) -> three jobs, each one success and one real failure, slow camera move, one line of text, small force
plot (cans: the weight felt vs the true weight) -> results card -> end card. Episodes are re-run from their
evaluation seeds (same seed = same episode); playback is faster in free motion and slower in contact. Every
rendered frame is checked for penetration between collision geoms (worst value printed per shot).
Sound: calm generated music, a soft tick on each title, a light tap on first contacts, a bell on success, a soft
two-note fall on failure (video/explainer.py).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.io import wavfile

from blindtouch.paths import OUT, RESULTS
from blindtouch.video.canvas import (ACCENT, FPS, GREEN, GREY, H, RED, SR, W, WHITE, Canvas, Mixer, Writer, bell, check_icon, cross_icon,
                             ease, marimba, soft_fail, tap, text_w, tick)
from blindtouch.video.job_clip import EpisodeView
from blindtouch.video.studio import Recorder, Studio

TRUE_C = (235, 235, 235)
BG = (22, 18, 16)
NAME = "Ahmed Sleem"

# camera per job: lookat offset from the fixture (cans: from the world origin), distance, azimuth from -> to, elevation
# (chosen on stills: the TV behind the socket and the arm's upper links must not hide the work)
SHOW_CAMS = {"socket": ((0.0, 0.0, 0.07), 0.70, (5.0, 30.0), -30.0),
             "whiteboard": ((0.0, 0.0, 0.08), 0.90, (225.0, 255.0), -26.0),
             "cans": ((0.28, -0.22, 0.45), 0.80, (200.0, 230.0), -24.0)}


def make_studio(scene: str) -> Studio:
    st = Studio(scene, W, H)
    st.model.vis.headlight.ambient[:] = 0.30      # a little brighter than the scenes' default (0.18 / 0.35), no grading
    st.model.vis.headlight.diffuse[:] = 0.50
    return st


# why a failure happened, where the numbers alone do not say it (each checked on the rendered frames)
FAIL_NOTE = {("cans", 800047): "can 2 slipped onto its neighbour, "}


JOBS = {   # job -> (title, subtitle, success seed, failure seed, seconds for the success / failure shot)
    "socket": ("Plug in a charger", "find the socket by feel, turn until the pins drop in, push", 800004, 800032, (11.0, 7.0)),
    "whiteboard": ("Erase a sloped whiteboard", "feel the board's angle, then wipe with a steady 8 N", 800004, 800011, (10.0, 6.0)),
    "cans": ("Which cans are still full?", "lift each can and weigh it through the motors", 800004, 800047, (12.0, 8.0)),
}


# ----------------------------------------------------------------------------------------------- helpers
def write_faded(wr: Writer, img: np.ndarray, k: int, n: int, fin: int = 8, fout: int = 8) -> None:
    a = min(1.0, (k + 1) / fin if fin else 1.0, (n - k) / fout if fout else 1.0)
    wr.write(img if a >= 1 else (img.astype(np.float32) * a).astype(np.uint8))


def lower_third(cv: Canvas, title: str, sub: str, a: float) -> None:
    if a <= 0.01:
        return
    w = max(text_w(title, 46, "s"), text_w(sub, 26)) + 60
    cv.panel(36, H - 170, w, 128, 0.62 * a)
    cv.text((66, H - 152), title, 46, "s", WHITE, a)
    cv.text((66, H - 90), sub, 26, "r", (215, 215, 215), a)


def verdict(cv: Canvas, ok: bool, text: str, a: float) -> None:
    if a <= 0.01:
        return
    w = text_w(text, 40, "s") + 124
    x = (W - w) // 2
    cv.panel(x, 70, w, 84, 0.78 * a, r=42)
    if a > 0.5:
        (check_icon if ok else cross_icon)(cv.img, (x + 58, 112), 16, GREEN if ok else RED, 5)
    cv.text((x + 96, 112), text, 40, "s", GREEN if ok else RED, a, "lm")


def speed_tag(cv: Canvas, speed: float, a: float) -> None:
    cv.text((W - 60, 60), f"{speed:.0f}× speed", 24, "r", (200, 200, 200), 0.8 * a, "ra")


def force_plot(cv: Canvas, tt: np.ndarray, true: np.ndarray, felt: np.ndarray, t_now: float, a: float,
               window: float = 6.0, y_max: float = 16.0, label: str = "contact force", big: bool = False) -> None:
    """Felt (accent) vs true (white) force over the last ``window`` s of simulated time."""
    if a <= 0.01:
        return
    w, h = (1100, 300) if big else (560, 170)
    x0, y0 = (W - w) // 2 if big else W - w - 60, H - h - (90 if big else 60)
    cv.panel(x0 - 24, y0 - 64, w + 48, h + 92, 0.62 * a)
    sel = (tt >= t_now - window) & (tt <= t_now)
    xy = lambda t, f: np.stack([x0 + (t - (t_now - window)) / window * w, y0 + h - np.clip(f, 0, y_max) / y_max * h], 1).astype(np.int32)
    img = cv.img
    for f in (0, y_max / 2, y_max):
        y = int(y0 + h - f / y_max * h)
        cv2.line(img, (x0, y), (x0 + w, y), (60, 60, 60), 1, cv2.LINE_AA)
        cv.text((x0 - 10, y), f"{f:.0f} N", 18, "r", GREY, a, "rm")
    if sel.sum() > 1:
        cv2.polylines(img, [xy(tt[sel], true[sel])], False, TRUE_C, 3 if big else 2, cv2.LINE_AA)
        cv2.polylines(img, [xy(tt[sel], felt[sel])], False, ACCENT, 3 if big else 2, cv2.LINE_AA)
    s = 30 if big else 22
    cv.text((x0, y0 - 46), label, s, "s", WHITE, a)
    lx = x0 + w
    cv.text((lx, y0 - 46), "true (simulator)", s, "r", TRUE_C, a, "ra")
    cv.text((lx - text_w("true (simulator)", s) - 30, y0 - 46), "felt from the motors", s, "s", ACCENT, a, "ra")


def time_warp(times: np.ndarray, contact: np.ndarray, seconds: float, slow: float = 3.0) -> np.ndarray:
    """Simulated times to show at FPS so the shot lasts ``seconds``: ``slow`` x slower while in contact."""
    grid = np.arange(times[0], times[-1], 0.01)
    c = np.interp(grid, times, contact.astype(float))
    c = np.convolve(c, np.ones(100) / 100, mode="same")           # 1 s ramps, no jumps in speed
    cost = 0.01 * (1 + (slow - 1) * c)                            # video seconds per sim step, before scaling
    cum = np.cumsum(cost)
    cum *= seconds / cum[-1]
    return np.interp(np.arange(int(seconds * FPS)) / FPS, cum, grid)


# ----------------------------------------------------------------------------------------------- the jobs
class Shot:
    def __init__(self, kind: str, seed: int):
        self.kind, self.seed = kind, seed
        self.ep = ep = EpisodeView(kind, seed)
        self.ok = bool(ep.res.success)
        self.times = np.array([f.t for f in ep.frames])
        ft = np.array([f.f_true for f in ep.frames])
        k = np.ones(10) / 10                                            # 20 ms average, as a sensor reports it
        f20 = np.stack([np.convolve(ft[:, j], k, mode="same") for j in range(3)], 1)
        proj = (lambda f: f @ ep.n) if kind == "whiteboard" else (lambda f: np.linalg.norm(f, axis=-1))
        self.true = proj(f20)
        self.felt = np.array([proj(ep.felt_at(t)) for t in self.times[::5]])
        self.tt5 = self.times[::5]
        self.true5 = self.true[::5]
        self.contact = np.abs(self.true) > 0.8
        info = ep.res.info
        self.mass_true = [float(m) for m in ep.job.env.model.body_mass[ep.job.bid]] if kind == "cans" else []
        self.weighed = [(t, d["can"], d["mass"], d["cls"]) for t, n, d in ep.job.rob.events if n == "weighed"]
        if kind == "socket":
            d = max(0.0, float(info.get("pin_depth_mm", 0)))
            self.text = "Plugged in" if self.ok else f"Failed: pins stuck {d:.0f} mm in (need 19)"
        elif kind == "whiteboard":
            self.text = "Wiped, steady force" if self.ok else f"Failed: pressed unevenly, {1 - info.get('wiped', 1):.0%} of the ink left"
        else:
            bad = [i for i, c in enumerate(info.get("classified", [])) if not c]
            if self.ok:
                self.text = "All three sorted correctly"
            else:
                i = bad[0] if bad else 0
                m = next((w[2] for w in self.weighed if w[1] == i), float("nan"))
                why = FAIL_NOTE.get((kind, seed), "")
                self.text = f"Failed: {why}felt {1e3 * m:.0f} g, really {1e3 * self.mass_true[i]:.0f} g"
        print(f"{kind} {seed}: success={self.ok} '{self.text}' realism={info.get('realism')}", flush=True)

    def window(self, fail: bool) -> tuple[float, float]:
        """Simulated time span to show: the whole job for a success; for a failure, the part where it goes wrong."""
        t0, t1 = self.times[0], self.times[-1]
        if not fail:
            return t0, t1
        if self.kind == "cans":
            i = next((i for i, c in enumerate(self.ep.res.info.get("classified", [])) if not c), 0)
            st = [t for t, n, d in self.ep.job.rob.events if n == "step" and d.get("can") == i]
            return (st[0], st[-1] + 4.0) if st else (t0, t1)
        return max(t0, t1 - 30.0), t1


def play_shot(wr: Writer, mx: Mixer, studio: Studio, sh: Shot, seconds: float, title: str, sub: str, fail: bool) -> float:
    ep, m = sh.ep, studio.model
    pos, quat, _ = ep.model_pose
    m.body_pos[:] = pos
    m.body_quat[:] = quat
    ink = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("ink")]
    ink_mat, ink_rgba = studio.base_mat[ink].copy(), studio.base_rgba[ink].copy()
    look, dist, (az0, az1), el = SHOW_CAMS[sh.kind]
    lookat = (ep.fixture if sh.kind != "cans" else 0.0) + np.asarray(look)
    a, b = sh.window(fail)
    sel = (sh.times >= a) & (sh.times <= b)
    ts = time_warp(sh.times[sel], sh.contact[sel], seconds)
    hold = int(1.7 * FPS)
    n = len(ts) + hold
    cam = mujoco.MjvCamera()
    worst, t_tap, was = 0.0, -9.0, False
    start = wr.now
    mx.put(start + 0.1, tick(), 0.25)
    for k in range(n):
        t = ts[min(k, len(ts) - 1)]
        i = int(np.clip(np.searchsorted(sh.times, t), 0, len(sh.times) - 1))
        fr = ep.frames[i]
        if ink:
            m.geom_matid[ink] = ink_mat
            m.geom_rgba[ink] = ink_rgba
            for g in fr.tags.get("erased", []):
                m.geom_matid[ink[g]] = -1
                m.geom_rgba[ink[g]] = [0, 0, 0, 0]
        studio.data.mocap_quat[:] = ep.mocap_quat
        studio.set_state(fr)
        for c in range(studio.data.ncon):
            worst = min(worst, studio.data.contact[c].dist)
        s = min(k, len(ts)) / max(len(ts), 1)                       # slow orbit and dolly over the shot
        cam.lookat[:] = lookat
        cam.distance = dist * (1.06 - 0.10 * ease(s))
        cam.azimuth = az0 + (az1 - az0) * s
        cam.elevation = el
        studio.r.update_scene(studio.data, camera=cam)
        cv = Canvas(cv2.cvtColor(studio.r.render(), cv2.COLOR_RGB2BGR))
        ta = ease((k / FPS - 0.2) / 0.6) * (1 - ease((k / FPS - 4.5) / 0.6)) if not fail else ease((k / FPS - 0.2) / 0.6) * (1 - ease((k / FPS - 3.0) / 0.6))
        lower_third(cv, title, sub, ta)
        if k < len(ts) and k > 0:
            speed = (ts[k] - ts[k - 1]) * FPS
            speed_tag(cv, max(1.0, speed), 1.0)
        if sh.kind == "cans":
            done = [w for w in sh.weighed if w[0] <= t]
            if done:
                cv.panel(W - 560, H - 90 - 62 * len(done) - 40, 510, 62 * len(done) + 70, 0.62)
                cv.text((W - 530, H - 90 - 62 * len(done) - 30), "weight felt from the motors", 24, "s", ACCENT, 1.0)
                for r, (tw, can, mass, cls) in enumerate(done):
                    y = H - 90 - 62 * (len(done) - r) + 30
                    cv.text((W - 530, y), f"can {can + 1}:  {1e3 * mass:.0f} g", 30, "s", ACCENT, 1.0)
                    cv.text((W - 300, y), f"(true {1e3 * sh.mass_true[can]:.0f} g)", 26, "r", TRUE_C, 1.0)
        else:
            force_plot(cv, sh.tt5, sh.true5, sh.felt, t, 1.0, label="pressing force" if sh.kind == "whiteboard" else "contact force")
        now = bool(sh.contact[i])
        if now and not was and t - t_tap > 1.5 and k < len(ts):
            mx.put(wr.now, tap(5.0), 0.35)
            t_tap = t
        was = now
        if k >= len(ts):
            va = ease((k - len(ts)) / 10)
            verdict(cv, sh.ok, sh.text, va)
            if k == len(ts):
                if sh.ok:
                    mx.put(wr.now, bell(880.0), 0.22)
                    mx.duck.append(wr.now)
                else:
                    mx.put(wr.now, soft_fail(), 0.3)
        write_faded(wr, cv.done(), k, n)
    print(f"  shot {sh.kind} {sh.seed}: {wr.now - start:.1f} s, worst penetration {-worst * 1e3:.2f} mm", flush=True)
    return -worst * 1e3


def job_part(wr: Writer, mx: Mixer, kind: str) -> None:
    title, sub, s_ok, s_bad, (sec_ok, sec_bad) = JOBS[kind]
    ok, bad = Shot(kind, s_ok), Shot(kind, s_bad)
    studio = make_studio(kind)
    play_shot(wr, mx, studio, ok, sec_ok, title, sub, fail=False)
    play_shot(wr, mx, studio, bad, sec_bad, "A real failure", "same job, another trial", fail=True)
    studio.r.close()      # a second open renderer draws black frames


# ----------------------------------------------------------------------------------------------- the idea
def idea_part(wr: Writer, mx: Mixer, seconds: float = 8.5) -> None:
    """A press and a slide on a plate (scripts/sensing_report protocol, all five imperfections, all fixes)."""
    from blindtouch.eval.sensing import run_one
    from blindtouch.sim.env import Env

    env = Env("peg")
    rec = Recorder()
    felt: list = []

    def feel(e, s):
        e.dyn.update(s.q, s.qd)
        J = e.dyn.tip_jac()
        felt.append(np.linalg.pinv(J.T, rcond=1e-3) @ (e.ctrl.dob.rf - e.ctrl.dob.d))

    env.hooks[:] = [rec, feel]
    seed = 960003
    r = run_one(env, "all", "+int", seed)
    print(f"idea: seed {seed}, press 3 N error {r['press 3 N']['err_rms']:.2f} N, slide {r['slide 3 N']['err_rms']:.2f} N", flush=True)
    env.hooks[:] = []
    fr = rec.frames
    times = np.array([f.t for f in fr])
    ft = np.array([f.f_true for f in fr])
    k = np.ones(10) / 10
    true = np.linalg.norm(np.stack([np.convolve(ft[:, j], k, mode="same") for j in range(3)], 1), axis=1)
    feltn = np.array([np.linalg.norm(f[:3]) for f in felt])
    t_end = times[-1]
    a, b = t_end - 10.5, t_end - 2.4                              # from the descent to the end of the slide
    ts = np.linspace(a, b, int(seconds * FPS))
    studio = make_studio("peg")
    tip0 = env.tip_pos()
    cam = mujoco.MjvCamera()
    hold = int(1.0 * FPS)
    n = len(ts) + hold
    mx.put(wr.now + 0.1, tick(), 0.25)
    t_tap = -9.0
    for kk in range(n):
        t = ts[min(kk, len(ts) - 1)]
        i = int(np.clip(np.searchsorted(times, t), 0, len(times) - 1))
        studio.set_state(fr[i])
        s = min(kk, len(ts)) / len(ts)
        cam.lookat[:] = tip0 + [0.0, -0.015, -0.03]
        cam.distance = 0.42 - 0.06 * ease(s)
        cam.azimuth, cam.elevation = 145 + 20 * s, -18
        studio.r.update_scene(studio.data, camera=cam)
        cv = Canvas(cv2.cvtColor(studio.r.render(), cv2.COLOR_RGB2BGR))
        ta = ease((kk / FPS - 0.2) / 0.6)
        cv.panel(36, 40, text_w("No force sensor: the arm feels the plate through its own motors", 40, "s") + 60, 120, 0.62 * ta)
        cv.text((66, 58), "No force sensor: the arm feels the plate through its own motors", 40, "s", WHITE, ta)
        cv.text((66, 112), "press 3 N, press 8 N, slide  ·  friction, cogging, noise, delay and model error all on", 26, "r", (215, 215, 215), ta)
        force_plot(cv, times[::5], true[::5], feltn[::5], t, ta, window=8.0, y_max=12.0, label="contact force", big=True)
        if true[i] > 0.8 and t - t_tap > 3.0:
            mx.put(wr.now, tap(5.0), 0.35)
            t_tap = t
        write_faded(wr, cv.done(), kk, n)
    studio.r.close()


# ----------------------------------------------------------------------------------------------- cards
def jobs_rates() -> dict:
    out = {}
    for j in JOBS:
        d = json.load(open(RESULTS / "jobs" / f"{j}.json"))["summary"]
        out[j] = d["success"]
    return out


def title_card(wr: Writer, mx: Mixer, sec: float = 3.5) -> None:
    n = int(sec * FPS)
    mx.put(wr.now + 0.3, marimba(523.25), 0.25)
    for k in range(n):
        cv = Canvas(np.full((H, W, 3), BG, np.uint8))
        a, b = ease((k / FPS - 0.2) / 0.8), ease((k / FPS - 1.0) / 0.8)
        cv.text((W // 2, H // 2 - 50), "Touch without a force sensor", 96, "b", WHITE, a, "mm")
        cv.text((W // 2, H // 2 + 50), "A low-cost robot arm feels contact through its own motors", 40, "r", (220, 220, 220), b, "mm")
        write_faded(wr, cv.done(), k, n, 1, 10)


def results_card(wr: Writer, mx: Mixer, sec: float = 8.0) -> None:
    rates = jobs_rates()
    rows = [("Plug in a charger", rates["socket"]), ("Erase a sloped whiteboard", rates["whiteboard"]),
            ("Which cans are full?", rates["cans"])]
    n = int(sec * FPS)
    mx.put(wr.now + 0.3, marimba(659.25), 0.2)
    for k in range(n):
        tt = k / FPS
        cv = Canvas(np.full((H, W, 3), BG, np.uint8))
        cv.text((W // 2, 130), "50 trials per job, motor imperfections switched on at random", 34, "r", (210, 210, 210), ease(tt / 0.6), "mm")
        for r, (name, v) in enumerate(rows):
            a = ease((tt - 0.4 - 0.3 * r) / 0.6)
            y = 250 + 105 * r
            cv.text((560, y), name, 40, "s", WHITE, a, "rm")
            x0, w = 610, 640
            cv.panel(x0, y - 16, w, 32, 0.9 * a, r=16, color=(52, 48, 46))
            fill = int(w * v * ease((tt - 0.6 - 0.3 * r) / 1.0))
            if fill > 32:
                cv.panel(x0, y - 16, fill, 32, a, r=16, color=GREEN)
            cv.text((x0 + w + 30, y), f"{v:.0%}", 44, "b", WHITE, a, "lm")
        a = ease((tt - 2.4) / 0.6)
        cv.text((W // 2, 640), "Felt-force error halved by the fixes (3.6 N → 1.8 N)", 36, "s", ACCENT, a, "mm")
        cv.text((W // 2, 700), "False touches in free air: 37 % → 4 %", 36, "s", ACCENT, a, "mm")
        b = ease((tt - 4.0) / 0.6)
        cv.text((W // 2, 810), "Still hard: joint friction. With it on, plugging in drops from 89 % to 50 %.", 32, "r", (215, 215, 215), b, "mm")
        write_faded(wr, cv.done(), k, n, 8, 12)


def end_card(wr: Writer, mx: Mixer, sec: float = 4.5) -> None:
    n = int(sec * FPS)
    for k in range(n):
        cv = Canvas(np.full((H, W, 3), BG, np.uint8))
        a = ease((k / FPS - 0.2) / 0.8)
        cv.text((W // 2, H // 2 - 40), NAME, 64, "s", WHITE, a, "mm")
        cv.text((W // 2, H // 2 + 40), "Simulation only  ·  OpenArm v2 in MuJoCo", 32, "r", (200, 200, 200), a, "mm")
        write_faded(wr, cv.done(), k, n, 1, 20)


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "blind_touch_showcase.mp4"))
    ap.add_argument("--only", default=None, help="title | idea | socket | whiteboard | cans | results | end")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.only:
        out = out.with_name(f"{out.stem}_{args.only}.mp4")
    silent, wav = out.with_name(out.stem + "_silent.mp4"), out.with_name(out.stem + ".wav")
    wr, mx = Writer(silent), Mixer()
    parts = {"title": lambda: title_card(wr, mx), "idea": lambda: idea_part(wr, mx),
             "socket": lambda: job_part(wr, mx, "socket"), "whiteboard": lambda: job_part(wr, mx, "whiteboard"),
             "cans": lambda: job_part(wr, mx, "cans"), "results": lambda: results_card(wr, mx), "end": lambda: end_card(wr, mx)}
    for name, p in parts.items():
        if args.only in (None, name):
            p()
            print(f"{name} done at {wr.now:.1f} s", flush=True)
    wr.close()
    sec = wr.n / FPS
    wavfile.write(wav, SR, (np.repeat(mx.render(sec)[:, None], 2, 1) * 32767).astype(np.int16))
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav), "-t", f"{sec:.3f}", "-c:v", "libx264", "-preset", "slow",
                    "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)
    silent.unlink()
    wav.unlink()
    sn = wr.snaps
    sheet = np.vstack([np.hstack(sn[i:i + 4] + [np.zeros_like(sn[0])] * (4 - len(sn[i:i + 4]))) for i in range(0, len(sn), 4)])
    cv2.imwrite(str(out.with_name(out.stem + "_sheet.jpg")), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out, f"{sec:.1f} s", flush=True)


if __name__ == "__main__":
    main()
