"""Drawing, sound and video-writing helpers shared by the showcase video.

* :class:`Canvas` -- a BGR frame plus queued anti-aliased text (PIL), rounded translucent panels, icons;
* generated sounds (ticks, taps, marimba, bell, a calm music bed) and a :class:`Mixer` that lays them out in time;
* :class:`Writer` -- H.264 frames via PyAV, with a contact sheet of snapshots.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import butter, sosfilt

FPS, W, H, SR = 30, 1920, 1080, 48000
FONTS = Path("C:/Windows/Fonts")
# colours, BGR for cv2 / RGB for PIL text
GREEN, RED = (94, 197, 34), (68, 68, 239)
GREY, WHITE, PANEL = (150, 150, 150), (245, 245, 245), (24, 21, 18)
ACCENT = (250, 165, 96)
rgb = lambda c: (c[2], c[1], c[0])

# ----------------------------------------------------------------------------------------------- drawing
_fonts: dict = {}


def font(size: int, weight: str = "r"):
    f = {"r": "segoeui.ttf", "s": "seguisb.ttf", "b": "segoeuib.ttf", "l": "segoeuil.ttf", "y": "seguisym.ttf"}[weight]
    if (f, size) not in _fonts:   # Segoe UI on Windows; PIL's own scalable font elsewhere
        path = FONTS / f
        _fonts[(f, size)] = ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default(size)
    return _fonts[(f, size)]


class Canvas:
    """A BGR frame plus queued text (drawn with PIL at the end, so fades and anti-aliased fonts are cheap)."""

    def __init__(self, img: np.ndarray):
        self.img = img
        self.texts: list = []

    def panel(self, x, y, w, h, alpha=0.82, r=18, color=PANEL):
        x, y, w, h = int(x), int(y), int(w), int(h)
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, self.img.shape[1]), min(y + h, self.img.shape[0])
        if x1 <= x0 or y1 <= y0:
            return
        roi = self.img[y0:y1, x0:x1]
        mask = np.zeros(roi.shape[:2], np.uint8)
        ox, oy = x - x0, y - y0
        cv2.rectangle(mask, (ox + r, oy), (ox + w - r, oy + h), 255, -1)
        cv2.rectangle(mask, (ox, oy + r), (ox + w, oy + h - r), 255, -1)
        for cx, cy in ((ox + r, oy + r), (ox + w - r, oy + r), (ox + r, oy + h - r), (ox + w - r, oy + h - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1, cv2.LINE_AA)
        a = (mask.astype(np.float32) / 255 * alpha)[..., None]
        roi[:] = (roi * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)

    def text(self, xy, s, size=28, weight="r", color=WHITE, alpha=1.0, anchor="la"):
        if alpha > 0.01:
            self.texts.append((xy, s, size, weight, color, alpha, anchor))

    def done(self) -> np.ndarray:
        if not self.texts:
            return self.img
        base = Image.fromarray(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        ov = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        for xy, s, size, weight, color, alpha, anchor in self.texts:
            d.text(xy, s, font=font(size, weight), fill=(*rgb(color), int(255 * min(alpha, 1))), anchor=anchor)
        return cv2.cvtColor(np.asarray(Image.alpha_composite(base, ov).convert("RGB")), cv2.COLOR_RGB2BGR)


def text_w(s, size, weight="r") -> int:
    return int(font(size, weight).getlength(s))


def check_icon(img, c, s, color, th=5):
    x, y = c
    cv2.line(img, (x - s, y), (x - s // 3, y + s * 2 // 3), color, th, cv2.LINE_AA)
    cv2.line(img, (x - s // 3, y + s * 2 // 3), (x + s, y - s * 2 // 3), color, th, cv2.LINE_AA)


def cross_icon(img, c, s, color, th=5):
    x, y = c
    cv2.line(img, (x - s, y - s), (x + s, y + s), color, th, cv2.LINE_AA)
    cv2.line(img, (x - s, y + s), (x + s, y - s), color, th, cv2.LINE_AA)


def ease(x):
    x = float(np.clip(x, 0, 1))
    return x * x * (3 - 2 * x)


# ----------------------------------------------------------------------------------------------- sound
def _t(d):
    return np.arange(int(d * SR)) / SR


def _lp(x, f):
    return sosfilt(butter(2, f, "low", fs=SR, output="sos"), x)


def tick():
    t = _t(0.06)
    return np.sin(2 * np.pi * 1500 * t) * np.exp(-t / 0.010) + 0.3 * np.sin(2 * np.pi * 3000 * t) * np.exp(-t / 0.005)


def tap(force):
    t = _t(0.2)
    n = _lp(np.random.default_rng(int(force * 100)).normal(0, 1, len(t)), 1800)
    x = np.sin(2 * np.pi * 170 * t) * np.exp(-t / 0.05) + 0.5 * np.sin(2 * np.pi * 510 * t) * np.exp(-t / 0.015) + 0.4 * n * np.exp(-t / 0.006)
    return x * (0.5 + 0.5 * min(force / 15, 1))


def marimba(f):
    t = _t(0.9)
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.3) + 0.25 * np.sin(2 * np.pi * 4 * f * t) * np.exp(-t / 0.05)


def bell(f):
    t = _t(3.0)
    x = sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / tau) for r, a, tau in ((1, 1, 1.4), (2.0, 0.45, 0.8), (2.76, 0.3, 0.5), (5.4, 0.12, 0.2)))
    return x * (1 - np.exp(-t / 0.002))


def soft_fail():
    out = np.zeros(int(0.9 * SR))
    for k, f in enumerate((330.0, 262.0)):
        t = _t(0.5)
        x = (np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t)) * np.exp(-t / 0.18) * (1 - np.exp(-t / 0.01))
        i = int(k * 0.22 * SR)
        out[i:i + len(x)] += x
    return out


def midi(n):
    return 440.0 * 2 ** ((n - 69) / 12)


def music(sec: float) -> np.ndarray:
    """Calm pad (Cmaj7 - Am7 - Fmaj7 - G6, 4 s each) with a soft bass and a sparse plucked arpeggio."""
    n = int(sec * SR) + SR
    out = np.zeros(n)
    chords = [[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 64]]
    bar = 4.0
    rng = np.random.default_rng(3)
    for b in range(int(sec / bar) + 1):
        ch = chords[b % 4]
        t = _t(bar + 1.5)
        env = np.minimum(1, t / 0.9) * np.clip((bar + 1.5 - t) / 1.2, 0, 1)
        seg = np.zeros(len(t))
        for m in ch:
            f = midi(m - 12)
            for det in (1.0, 1.004):
                seg += (np.sin(2 * np.pi * f * det * t) + 0.25 * np.sin(4 * np.pi * f * det * t)) * 0.12
        seg += 0.35 * np.sin(2 * np.pi * midi(ch[0] - 24) * t)
        i = int(b * bar * SR)
        j = min(i + len(t), n)
        out[i:j] += (seg * env)[: j - i]
        for k, m in enumerate([0, 2, 1, 3, 2, 1, 3, 2]):   # eighth notes, every other one skipped at random
            if rng.random() < 0.35:
                continue
            tt = _t(0.6)
            p = np.sin(2 * np.pi * midi(ch[m] + 12) * tt) * np.exp(-tt / 0.22) * (1 - np.exp(-tt / 0.004)) * 0.16
            s = int((b * bar + k * 0.5) * SR)
            if s >= n:
                break
            e = min(s + len(p), n)
            out[s:e] += p[: e - s]
    out = _lp(out, 2600)
    return out[: int(sec * SR)]


class Mixer:
    def __init__(self):
        self.ev: list = []      # (t, signal, gain)
        self.duck: list = []    # times of bells

    def put(self, t, x, g):
        self.ev.append((t, x, g))

    def render(self, sec: float) -> np.ndarray:
        n = int(sec * SR)
        fx = np.zeros(n + 4 * SR)
        for t, x, g in self.ev:
            i = int(t * SR)
            if 0 <= i < n:
                fx[i:i + len(x)] += x * g
        fx = fx[:n]
        mus = music(sec)
        mus *= 0.055 / (np.sqrt(np.mean(mus ** 2)) + 1e-9)
        tt = np.arange(n) / SR
        gain = np.minimum(1, tt / 2.0) * np.clip((sec - tt) / 3.0, 0, 1)
        for t0 in self.duck:
            gain *= 1 - 0.45 * np.exp(-np.clip(tt - t0, 0, None) / 0.8) * (tt >= t0 - 0.05)
        out = mus * gain + fx
        return out * (0.85 / np.abs(out).max())


# ----------------------------------------------------------------------------------------------- video writer
class Writer:
    def __init__(self, path):
        import av

        self.c = av.open(str(path), mode="w")
        self.s = self.c.add_stream("libx264", rate=FPS)
        self.s.width, self.s.height, self.s.pix_fmt = W, H, "yuv420p"
        self.s.options = {"crf": "16", "preset": "medium"}
        self.n = 0
        self.snaps: list = []
        self.prev: np.ndarray | None = None

    @property
    def now(self) -> float:
        return self.n / FPS

    def write(self, img):
        import av

        if self.n % 60 == 30:
            self.snaps.append(cv2.resize(img, (480, 270)))
        self.prev = img
        for p in self.s.encode(av.VideoFrame.from_ndarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), format="rgb24")):
            self.c.mux(p)
        self.n += 1

    def close(self):
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()
