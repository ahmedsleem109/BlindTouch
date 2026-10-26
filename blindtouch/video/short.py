"""LinkedIn cut (~50 s, portrait 4:5): every job as a real failure (top) and a real success (bottom), played in
sync, with a voice-over and big captions. No music.

    python -m blindtouch.video.short                    # -> out/blind_touch_short.mp4
    python -m blindtouch.video.short --only peg         # one part, for a quick look

Parts: hook (three successes, "no force sensor") -> the idea (felt vs true force while wiping) -> peg -> socket ->
whiteboard -> cans -> results. Episodes are re-run from their evaluation seeds (same seed = same episode); every
rendered frame is checked for penetration between collision geoms (worst value printed per episode).

Voice-over: a Microsoft Edge neural voice through the ``edge-tts`` package (needs internet; only the script text
is sent). Set ``TTS_PYTHON`` to an interpreter that has edge-tts if this one does not. Clips are cached in out/vo.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.io import wavfile

from blindtouch.paths import OUT, RESULTS
from blindtouch.video.canvas import (ACCENT, FPS, GREEN, SR, WHITE, Canvas, Writer, check_icon, cross_icon, ease,
                                     text_w)
from blindtouch.video.job_clip import EpisodeView
from blindtouch.video.showcase import time_warp
from blindtouch.video.studio import Studio

W, H = 1080, 1350
HEAD = 120                       # header band
PH = 545                         # height of each episode panel
TOP_Y, BOT_Y = HEAD, HEAD + PH + 6
CAP_Y = (BOT_Y + PH + H) // 2    # captions: in the band below both panels
BG = (22, 18, 16)
TRUE_C = (235, 235, 235)
GREY = (200, 200, 200)
VOICE = "en-US-AndrewNeural"
REPO = "github.com/ahmedsleem109/BlindTouch"
NAME = "Ahmed Sleem"

VO = {
    "hook": "This robot arm has no force sensor. But it can still feel what it touches.",
    "idea": "It feels through its own motors. Blue is what it feels. White is the real force.",
    "peg": "A peg into a hole it can't see. On top, the hole is too far off, and the search finds nothing. "
           "Below, it feels its way in.",
    "socket": "Plugging in a charger, blind. On top, the pins jam half way. Below, they drop in, and it pushes home.",
    "whiteboard": "Erasing a sloped whiteboard. It feels the angle, then keeps a steady push. "
                  "On top, the push wobbles, and ink is left.",
    "cans": "Which cans are full? It lifts each one and weighs it through its motors. On top, a can slips, and the "
            "weight comes out wrong.",
    "results": "Eighty to a hundred percent success, failures included. Simulation for now. The code is on GitHub.",
}

# job -> title, subtitle, (success seed, ranges), (failure seed, ranges); seeds from the evaluations (results/)
JOBS = {
    "peg": ("Insert a peg into a hidden hole", "1.5 mm clearance, hole position unknown", (800012, {}), (850004, {"pos": 0.016})),
    "socket": ("Plug in a charger, blind", "the TV hides the power strip", (800004, {}), (800032, {})),
    "whiteboard": ("Erase a sloped whiteboard", "steady 8 N push on an unknown slope", (800004, {}), (800011, {})),
    "cans": ("Which cans are still full?", "weigh each can through the motors", (800004, {}), (800047, {})),
}
FAIL_NOTE = {("cans", 800047): "can 2 slipped onto its neighbour, "}


# ----------------------------------------------------------------------------------------------- episodes
class Ep:
    """A re-run episode with what the video needs: true and felt force, contact, verdict, cameras."""

    def __init__(self, kind: str, seed: int, ranges: dict):
        self.kind, self.seed = kind, seed
        self.ep = ep = EpisodeView(kind, seed, **ranges)
        self.ok = bool(ep.res.success)
        self.times = np.array([f.t for f in ep.frames])
        ft = np.array([f.f_true for f in ep.frames])
        k = np.ones(10) / 10                                            # 20 ms average, as a sensor reports it
        f20 = np.stack([np.convolve(ft[:, j], k, mode="same") for j in range(3)], 1)
        proj = (lambda f: f @ ep.n) if kind == "whiteboard" else (lambda f: np.linalg.norm(f, axis=-1))
        self.true = proj(f20)
        self.tt5, self.true5 = self.times[::5], self.true[::5]
        self.felt5 = np.array([proj(ep.felt_at(t)) for t in self.tt5])
        self.contact = np.abs(self.true) > 0.8
        info = ep.res.info
        env = ep.job.env
        self.mass_true = [float(m) for m in env.model.body_mass[ep.job.bid]] if kind == "cans" else []
        self.weighed = [(t, d["can"], d["mass"], d["cls"]) for t, n, d in ep.job.rob.events if n == "weighed"]
        self.hole = env.hole_xy() if kind == "peg" else None
        q = ep.mocap_quat[env.fixture_mocap] if env.fixture_mocap >= 0 else np.array([1.0, 0, 0, 0])
        self.yaw = float(np.degrees(2 * np.arctan2(q[3], q[0])))
        self.text = self._verdict(info)
        self.worst = 0.0
        print(f"{kind} {seed}: success={self.ok} '{self.text}' realism={info.get('realism')}", flush=True)

    def _verdict(self, info: dict) -> str:
        kind = self.kind
        if kind == "peg":
            return f"Inserted {info['depth_mm']:.0f} mm" if self.ok else "Hole too far off: search found nothing"
        if kind == "socket":
            d = max(0.0, float(info.get("pin_depth_mm", 0)))
            return "Plugged in" if self.ok else f"Pins stuck {d:.0f} mm in (need 19)"
        if kind == "whiteboard":
            return "Wiped, steady force" if self.ok else f"Uneven push, {1 - info.get('wiped', 1):.0%} of the ink left"
        if self.ok:
            return "All three sorted correctly"
        bad = [i for i, c in enumerate(info.get("classified", [])) if not c]
        i = bad[0] if bad else 0
        m = next((w[2] for w in self.weighed if w[1] == i), float("nan"))
        return f"{FAIL_NOTE.get((kind, self.seed), '').capitalize()}felt {1e3 * m:.0f} g, really {1e3 * self.mass_true[i]:.0f} g"

    def window(self, fail: bool) -> tuple[float, float]:
        """Simulated time span to show: the whole job for a success; for a failure, the part where it goes wrong."""
        t0, t1 = self.times[0], self.times[-1]
        if not fail or self.kind == "peg":
            return t0, t1
        if self.kind == "cans":
            i = next((i for i, c in enumerate(self.ep.res.info.get("classified", [])) if not c), 0)
            st = [t for t, n, d in self.ep.job.rob.events if n == "step" and d.get("can") == i]
            return (st[0], st[-1] + 4.0) if st else (t0, t1)
        return max(t0, t1 - 30.0), t1

    def camera(self, s: float, portrait: bool = False) -> mujoco.MjvCamera:
        """A slow orbit over the shot (s: 0 -> 1). Portrait frames see less sideways: the camera backs off."""
        ep = self.ep
        if self.kind == "peg":
            look, dist, az, el = np.r_[self.hole, 0.50], 0.30, (-50.0, -30.0), -25.0
        elif self.kind == "socket":
            look, dist, az, el = ep.fixture + [0, 0, 0.045], 0.23, (self.yaw - 95, self.yaw - 80), -16.0
        elif self.kind == "whiteboard":
            look, dist, az, el = ep.fixture + [0, 0, 0.08], 0.80, (225.0, 250.0), -26.0
        else:
            look, dist, az, el = np.array([0.28, -0.22, 0.45]), 0.80, (200.0, 225.0), -24.0
        c = mujoco.MjvCamera()
        c.lookat[:] = look
        c.distance = dist * (1.5 if portrait else 1.0) * (1.04 - 0.08 * ease(s))
        c.azimuth = az[0] + (az[1] - az[0]) * s
        c.elevation = el
        return c

    def index(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))


def make_studio(kind: str, w: int, h: int) -> Studio:
    st = Studio(kind, w, h)
    st.model.vis.headlight.ambient[:] = 0.32
    st.model.vis.headlight.diffuse[:] = 0.52
    st.ink = [g for g in range(st.model.ngeom) if (st.model.geom(g).name or "").startswith("ink")]
    return st


def render(st: Studio, e: Ep, t: float, s: float, portrait: bool = False) -> np.ndarray:
    """The episode's state at simulated time ``t`` (its own fixture placement, ink already wiped)."""
    m, ep = st.model, e.ep
    pos, quat, _ = ep.model_pose
    m.body_pos[:] = pos
    m.body_quat[:] = quat
    fr = ep.frames[e.index(t)]
    if st.ink:
        m.geom_matid[st.ink] = st.base_mat[st.ink]
        m.geom_rgba[st.ink] = st.base_rgba[st.ink]
        for g in fr.tags.get("erased", []):
            m.geom_matid[st.ink[g]] = -1
            m.geom_rgba[st.ink[g]] = [0, 0, 0, 0]
    st.data.mocap_quat[:] = ep.mocap_quat
    st.set_state(fr)
    for c in range(st.data.ncon):
        e.worst = min(e.worst, st.data.contact[c].dist)
    st.r.update_scene(st.data, camera=e.camera(s, portrait))
    return cv2.cvtColor(st.r.render(), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------------------------- overlays
def wrap_text(s: str, size: int, wpx: int, weight: str = "s") -> list[str]:
    lines, cur = [], ""
    for word in s.split():
        if cur and text_w(cur + " " + word, size, weight) > wpx:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    return lines + ([cur] if cur else [])


def caption(cv: Canvas, text: str, y_mid: int, a: float = 1.0, size: int = 44) -> None:
    """Big, centred, readable without sound."""
    if not text or a <= 0.01:
        return
    lh = int(size * 1.27)
    lines = wrap_text(text, size, W - 160)
    w = max(text_w(x, size, "s") for x in lines) + 56
    h = lh * len(lines) + 30
    cv.panel((W - w) // 2, y_mid - h // 2, w, h, 0.80 * a, r=26, color=(12, 10, 9))
    for i, x in enumerate(lines):
        cv.text((W // 2, y_mid - h // 2 + 15 + lh * i + lh // 2), x, size, "s", WHITE, a, "mm")


def chip(cv: Canvas, ok: bool, text: str | None = None) -> None:
    """Top-left label: SUCCESS / FAILURE while it plays, the verdict at the end."""
    text = text or ("SUCCESS" if ok else "FAILURE")
    color = (60, 150, 30) if ok else (50, 50, 200)
    w = text_w(text, 30, "b") + 78
    cv.panel(22, 20, w, 54, 0.92, r=27, color=color)
    (check_icon if ok else cross_icon)(cv.img, (52, 47), 11, WHITE, 4)
    cv.text((74, 47), text, 30, "b", WHITE, 1.0, "lm")


def speed_tag(cv: Canvas, speed: float) -> None:
    cv.text((W - 26, 30), f"{max(1.0, speed):.0f}× speed", 24, "s", GREY, 0.9, "ra")


def force_plot(cv: Canvas, e: Ep, t: float, box: tuple, window: float = 6.0, y_max: float = 14.0, big: bool = False) -> None:
    """Felt (accent) vs true (white) force over the last ``window`` s of simulated time."""
    x0, y0, w, h = box
    cv.panel(x0 - 18, y0 - (64 if big else 46), w + 36, h + (88 if big else 62), 0.70, r=18)
    sel = (e.tt5 >= t - window) & (e.tt5 <= t)
    xy = lambda tt, f: np.stack([x0 + (tt - (t - window)) / window * w, y0 + h - np.clip(f, 0, y_max) / y_max * h], 1).astype(np.int32)
    for f in (0, y_max / 2, y_max):
        yy = int(y0 + h - f / y_max * h)
        cv2.line(cv.img, (x0, yy), (x0 + w, yy), (70, 70, 70), 1, cv2.LINE_AA)
        cv.text((x0 + w + 6, yy), f"{f:.0f} N", 18 if not big else 22, "r", GREY, 0.9, "lm")
    if sel.sum() > 1:
        cv2.polylines(cv.img, [xy(e.tt5[sel], e.true5[sel])], False, TRUE_C, 3 if big else 2, cv2.LINE_AA)
        cv2.polylines(cv.img, [xy(e.tt5[sel], e.felt5[sel])], False, ACCENT, 3 if big else 2, cv2.LINE_AA)
    s = 30 if big else 20
    cv.text((x0, y0 - (40 if big else 28)), "felt from the motors", s, "s", ACCENT, 1.0, "lm")
    cv.text((x0 + text_w("felt from the motors", s, "s") + 24, y0 - (40 if big else 28)), "real force", s, "s", TRUE_C, 1.0, "lm")


def weights(cv: Canvas, e: Ep, t: float) -> None:
    done = [w for w in e.weighed if w[0] <= t]
    if not done:
        return
    x0, lh = W - 400, 40
    y0 = PH - 30 - lh * len(done) - 40
    cv.panel(x0 - 20, y0 - 10, 400, lh * len(done) + 60, 0.70, r=18)
    cv.text((x0, y0 + 14), "weighed through the motors", 20, "s", ACCENT, 1.0, "lm")
    for r, (_, can, mass, _) in enumerate(done):
        y = y0 + 54 + lh * r
        cv.text((x0, y), f"can {can + 1}:  {1e3 * mass:.0f} g", 28, "s", ACCENT, 1.0, "lm")
        cv.text((x0 + 200, y), f"(real {1e3 * e.mass_true[can]:.0f} g)", 22, "r", TRUE_C, 1.0, "lm")


# ----------------------------------------------------------------------------------------------- voice-over and sound
def tts(part: str) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """Voice clip (mono float at SR) and its caption cues (start, end, text), cached by text."""
    text = VO[part]
    d = OUT / "vo"
    d.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{VOICE}|{text}".encode()).hexdigest()[:10]
    mp3, srt, wav = d / f"{part}_{key}.mp3", d / f"{part}_{key}.srt", d / f"{part}_{key}.wav"
    if not wav.exists():
        py = os.environ.get("TTS_PYTHON", sys.executable)
        subprocess.run([py, "-m", "edge_tts", "--voice", VOICE, "--rate", "+6%", "--text", text,
                        "--write-media", str(mp3), "--write-subtitles", str(srt)], check=True)
        import imageio_ffmpeg

        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp3), "-ac", "1",
                        "-ar", str(SR), str(wav)], check=True)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32) / 32768.0
    ts = lambda s: sum(float(v) * m for v, m in zip(re.split("[:,]", s), (3600, 60, 1, 1e-3)))
    cues = [(ts(a), ts(b), " ".join(txt.split())) for a, b, txt in
            re.findall(r"(\d\d:\d\d:\d\d,\d+) --> (\d\d:\d\d:\d\d,\d+)\s*\n(.+?)(?:\n\n|\Z)", srt.read_text(encoding="utf-8"), re.S)]
    return x, cues


def _t(d: float) -> np.ndarray:
    return np.arange(int(d * SR)) / SR


def soft_ok() -> np.ndarray:
    """Two quick, soft rising notes (a gentle 'done'), no ringing tail."""
    out = np.zeros(int(0.35 * SR))
    for k, f in enumerate((587.33, 880.0)):
        t = _t(0.18)
        x = np.sin(2 * np.pi * f * t) * (1 - np.exp(-t / 0.006)) * np.exp(-t / 0.05)
        i = int(k * 0.09 * SR)
        out[i:i + len(x)] += x
    return out


def soft_fail() -> np.ndarray:
    """One low, muted thud."""
    t = _t(0.3)
    return (np.sin(2 * np.pi * 150 * t) + 0.3 * np.sin(2 * np.pi * 300 * t)) * (1 - np.exp(-t / 0.008)) * np.exp(-t / 0.07)


class Sound:
    def __init__(self):
        self.ev: list = []

    def put(self, t: float, x: np.ndarray, gain: float) -> None:
        self.ev.append((t, x, gain))

    def render(self, sec: float) -> np.ndarray:
        n = int(sec * SR)
        out = np.zeros(n + 10 * SR)
        for t, x, g in self.ev:
            i = int(t * SR)
            out[i:i + len(x)] += g * x
        out = out[:n]
        return out * (0.9 / max(np.abs(out).max(), 1e-6))


def cue_at(cues: list, t: float) -> str:
    for a, b, text in cues:
        if a - 0.05 <= t < b + 0.25:
            return text
    return ""


# ----------------------------------------------------------------------------------------------- parts
class Part:
    """Shared state for one part: the writer, the sound track, its voice clip and captions."""

    def __init__(self, wr: Writer, snd: Sound, name: str, pad: float = 0.9, min_s: float = 0.0):
        self.wr, self.snd = wr, snd
        self.vo, self.cues = tts(name)
        self.start = wr.now
        snd.put(self.start + 0.25, self.vo, 1.0)
        self.seconds = max(min_s, 0.25 + len(self.vo) / SR + pad)
        self.n = int(self.seconds * FPS)

    def caption_at(self, k: int) -> str:
        return cue_at(self.cues, k / FPS - 0.25)


def write_faded(wr: Writer, img: np.ndarray, k: int, n: int, fin: int = 4, fout: int = 4) -> None:
    a = min(1.0, (k + 1) / fin if fin else 1.0, (n - k) / fout if fout else 1.0)
    wr.write(img if a >= 1 else (img.astype(np.float32) * a).astype(np.uint8))


def hook(wr: Writer, snd: Sound, eps: dict) -> None:
    p = Part(wr, snd, "hook", pad=0.5)
    shots = [eps["peg", True], eps["socket", True], eps["cans", True]]
    per = p.n // len(shots)
    for j, e in enumerate(shots):
        st = make_studio(e.kind, W, H)
        span = {"peg": 3.0, "socket": 5.0, "cans": 6.0}[e.kind]
        if e.kind == "cans":   # lifting the first can
            t_lift = next((t for t, n, d in e.ep.job.rob.events if n == "step" and "Lift" in d.get("title", "")), e.times[0])
            a = t_lift
        else:
            a = e.times[-1] - span - 1.0
        ts = np.linspace(a, a + span, per)
        for k in range(per):
            kk = j * per + k
            cv = Canvas(render(st, e, ts[k], k / per, portrait=True))
            ta = ease(kk / FPS / 0.4)
            cv.panel(0, 0, W, 230, 0.55, r=0)
            cv.text((W // 2, 92), "No force sensor.", 84, "b", WHITE, ta, "mm")
            cv.text((W // 2, 172), "It feels through its motors.", 44, "s", (225, 225, 225), ease((kk / FPS - 0.8) / 0.4), "mm")
            caption(cv, p.caption_at(kk), H - 150)
            write_faded(wr, cv.done(), kk, p.n, 1, 1)
        st.r.close()
    for kk in range(per * len(shots), p.n):   # round off to the part's length on the last frame
        wr.write(wr.prev)


def idea(wr: Writer, snd: Sound, eps: dict) -> None:
    p = Part(wr, snd, "idea", pad=0.8, min_s=6.0)
    e = eps["whiteboard", True]
    st = make_studio("whiteboard", W, H)
    t_w = next((t for t, n, d in e.ep.job.rob.events if n == "step" and d.get("title", "").startswith("Wipe")), e.times[0])
    ts = np.linspace(t_w - 1.0, t_w - 1.0 + 2.2 * p.seconds, p.n)    # 2x speed through the wipe
    for k in range(p.n):
        cv = Canvas(render(st, e, ts[k], k / p.n, portrait=True))
        cv.panel(0, 0, W, 150, 0.55, r=0)
        cv.text((W // 2, 75), "What it feels vs the real force", 52, "b", WHITE, 1.0, "mm")
        force_plot(cv, e, ts[k], (80, H - 520, W - 220, 230), big=True)
        caption(cv, p.caption_at(k), H - 140)
        write_faded(wr, cv.done(), k, p.n)
    st.r.close()


def stacked(wr: Writer, snd: Sound, eps: dict, kind: str) -> None:
    title, sub, _, _ = JOBS[kind]
    p = Part(wr, snd, kind, pad=1.6, min_s=8.0)
    bad, ok = eps[kind, False], eps[kind, True]
    st = make_studio(kind, W, PH)
    hold = int(1.5 * FPS)
    play = p.n - hold
    warps = {}
    for e, fail in ((bad, True), (ok, False)):
        a, b = e.window(fail)
        sel = (e.times >= a) & (e.times <= b)
        warps[fail] = time_warp(e.times[sel], e.contact[sel], play / FPS, slow=2.0)
    for k in range(p.n):
        frame = np.full((H, W, 3), BG, np.uint8)
        for e, fail, y0 in ((bad, True, TOP_Y), (ok, False, BOT_Y)):
            ts = warps[fail]
            t = ts[min(k, len(ts) - 1)]
            cv = Canvas(render(st, e, t, min(k, play) / play))
            chip(cv, e.ok, e.text if k >= play else None)
            if 0 < k < len(ts) and k < play:
                speed_tag(cv, (ts[k] - ts[k - 1]) * FPS)
            if kind == "whiteboard":
                force_plot(cv, e, t, (W - 470, PH - 150, 380, 110))
            elif kind == "cans":
                weights(cv, e, t)
            if k == play:
                snd.put(wr.now, soft_ok() if e.ok else soft_fail(), 0.25 if e.ok else 0.35)
            frame[y0:y0 + PH] = cv.done()
        cv = Canvas(frame)
        cv.text((W // 2, 48), title, 52, "b", WHITE, 1.0, "mm")
        cv.text((W // 2, 97), sub, 28, "r", GREY, 1.0, "mm")
        caption(cv, p.caption_at(k), CAP_Y, size=38)
        write_faded(wr, cv.done(), k, p.n)
    st.r.close()
    for e in (bad, ok):
        print(f"  {kind} {e.seed}: worst penetration {-e.worst * 1e3:.2f} mm", flush=True)


def results(wr: Writer, snd: Sound) -> None:
    p = Part(wr, snd, "results", pad=1.4, min_s=6.5)
    rates = {j: json.load(open(RESULTS / "jobs" / f"{j}.json"))["summary"]["success"] for j in JOBS}
    rows = [("Insert a peg", rates["peg"]), ("Plug in a charger", rates["socket"]),
            ("Erase a whiteboard", rates["whiteboard"]), ("Weigh cans", rates["cans"])]
    for k in range(p.n):
        tt = k / FPS
        cv = Canvas(np.full((H, W, 3), BG, np.uint8))
        cv.text((W // 2, 150), "50 trials per job", 60, "b", WHITE, ease(tt / 0.4), "mm")
        cv.text((W // 2, 222), "motor imperfections switched on at random", 32, "r", GREY, ease(tt / 0.4), "mm")
        for r, (name, v) in enumerate(rows):
            a = ease((tt - 0.3 - 0.2 * r) / 0.4)
            y = 340 + 120 * r
            cv.text((80, y), name, 40, "s", WHITE, a, "lm")
            x0, w = 80, 760
            cv.panel(x0, y + 34, w, 30, 0.9 * a, r=15, color=(52, 48, 46))
            fill = int(w * v * ease((tt - 0.4 - 0.2 * r) / 0.7))
            if fill > 30:
                cv.panel(x0, y + 34, fill, 30, a, r=15, color=GREEN)
            cv.text((x0 + w + 30, y + 49), f"{v:.0%}", 44, "b", WHITE, a, "lm")
        a = ease((tt - 1.6) / 0.5)
        cv.text((W // 2, 870), "Felt-force error halved: 3.6 N → 1.8 N", 38, "s", ACCENT, a, "mm")
        cv.text((W // 2, 925), "Still hard: joint friction", 32, "r", GREY, a, "mm")
        b = ease((tt - 2.4) / 0.5)
        cv.panel(110, 1000, W - 220, 150, 0.9 * b, r=24, color=(40, 36, 34))
        cv.text((W // 2, 1050), REPO, 38, "s", WHITE, b, "mm")
        cv.text((W // 2, 1105), f"{NAME}  ·  simulation only (OpenArm v2, MuJoCo)", 26, "r", GREY, b, "mm")
        caption(cv, p.caption_at(k), H - 110)
        write_faded(wr, cv.done(), k, p.n, 4, 10)


# ----------------------------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "blind_touch_short.mp4"))
    ap.add_argument("--only", default=None, help="hook | idea | peg | socket | whiteboard | cans | results")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.only:
        out = out.with_name(f"{out.stem}_{args.only}.mp4")
    need = {"hook": [("peg", True), ("socket", True), ("cans", True)], "idea": [("whiteboard", True)],
            "results": [], **{j: [(j, False), (j, True)] for j in JOBS}}
    keys = sorted({k for part, ks in need.items() if args.only in (None, part) for k in ks})
    eps = {}
    for kind, ok in keys:
        _, _, good, bad = JOBS[kind]
        seed, ranges = good if ok else bad
        eps[kind, ok] = Ep(kind, seed, ranges)
    silent, wav = out.with_name(out.stem + "_silent.mp4"), out.with_name(out.stem + ".wav")
    wr, snd = Writer(silent, (W, H)), Sound()
    parts = {"hook": lambda: hook(wr, snd, eps), "idea": lambda: idea(wr, snd, eps),
             **{j: (lambda j=j: stacked(wr, snd, eps, j)) for j in JOBS}, "results": lambda: results(wr, snd)}
    for name, fn in parts.items():
        if args.only in (None, name):
            fn()
            print(f"{name} done at {wr.now:.1f} s", flush=True)
    wr.close()
    sec = wr.n / FPS
    wavfile.write(wav, SR, (np.repeat(snd.render(sec)[:, None], 2, 1) * 32767).astype(np.int16))
    import imageio_ffmpeg

    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(wav),
                    "-t", f"{sec:.3f}", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)], check=True)
    silent.unlink()
    wav.unlink()
    sn = wr.snaps
    rows = [np.hstack(sn[i:i + 6] + [np.zeros_like(sn[0])] * (6 - len(sn[i:i + 6]))) for i in range(0, len(sn), 6)]
    cv2.imwrite(str(out.with_name(out.stem + "_sheet.jpg")), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("wrote", out, f"{sec:.1f} s", flush=True)


if __name__ == "__main__":
    main()
