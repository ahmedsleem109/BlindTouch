"""Anticogging: map each motor's torque ripple against its joint angle once, then cancel it.

Torque ripple (cogging, current-sensing errors) is a periodic torque in the motor angle; through a 9:1 to 40:1
gearbox it repeats 20-90 times per joint radian, adds up to ~0.6 Nm, and neither averages out nor shows in a
nearby air pass. Like Piccoli & Yim (RSS 2014, "Anticogging"), the robot maps it with the encoders it already
has: each joint is swept slowly back and forth in free space (the others held), the momentum-observer residual
(torque the model cannot explain) is recorded against the joint angle and fitted with

    r_j(q) = c0 + c1 dq + c2 dq^2 + s sign(qd) + sum_k a_k sin(w_k q) + b_k cos(w_k q)

(smooth gravity-model error, Coulomb friction, ripple harmonics k = 1, 2, 6 of the motor's electrical
frequency, each w_k searched within +-12 % of pole pairs x gear ratio -- the design values). The ripple part is
then fed forward: subtracted from the motor command (the motors deliver what the controller asked for) and
added to the reported torque the observer uses (the felt force no longer contains it).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blindtouch.sensing.observer import MomentumObserver
from blindtouch.sim.control import JointImpedance
from blindtouch.sim.env import Env
from blindtouch.sim.realism import RIPPLE_CYCLES_PER_RAD

SWEEP = np.array([0.10, 0.10, 0.05, 0.05, 0.12, 0.12, 0.12])     # +- rad per joint (several ripple periods)
SPEED = np.array([0.08, 0.08, 0.04, 0.04, 0.10, 0.10, 0.10])     # rad/s (ripple stays well inside the observer bandwidth)
HARM = (0, 1, 2)                                                 # rows of RIPPLE_CYCLES_PER_RAD (k = 1, 2, 6)


@dataclass
class CoggingMap:
    w: np.ndarray       # (3, 7) angular frequencies (rad^-1)
    a: np.ndarray       # (3, 7) sin coefficients (Nm)
    b: np.ndarray       # (3, 7) cos coefficients (Nm)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        return np.sum(self.a * np.sin(self.w * q) + self.b * np.cos(self.w * q), axis=0)

    def amplitude(self) -> np.ndarray:
        return np.hypot(self.a, self.b)


def _features(q, qd, q0, w):
    dq = q - q0
    cols = [np.ones_like(q), dq, dq ** 2, np.sign(qd)]
    for wk in w:
        cols += [np.sin(wk * q), np.cos(wk * q)]
    return np.stack(cols, 1)


def fit_joint(q: np.ndarray, qd: np.ndarray, r: np.ndarray, q0: float, w_nom: np.ndarray, span: float = 0.12, n_grid: int = 61):
    """Least-squares fit of one joint's residual; harmonic frequencies by a sequential grid search, then refined."""
    w = []
    for wn in w_nom:  # strongest (1x) first: each harmonic searched with the ones already chosen in the model
        best = (np.inf, wn)
        for wk in wn * np.linspace(1 - span, 1 + span, n_grid):
            X = _features(q, qd, q0, w + [wk])
            coef, *_ = np.linalg.lstsq(X, r, rcond=None)
            e = float(np.sum((X @ coef - r) ** 2))
            if e < best[0]:
                best = (e, wk)
        w.append(best[1])
    for _ in range(2):  # coordinate refinement with all harmonics present
        for i, wn in enumerate(w_nom):
            best = (np.inf, w[i])
            for wk in w[i] * np.linspace(0.985, 1.015, 31):
                ww = list(w)
                ww[i] = wk
                X = _features(q, qd, q0, ww)
                coef, *_ = np.linalg.lstsq(X, r, rcond=None)
                e = float(np.sum((X @ coef - r) ** 2))
                if e < best[0]:
                    best = (e, wk)
            w[i] = best[1]
    X = _features(q, qd, q0, w)
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    ab = coef[4:].reshape(-1, 2)
    rms_before = float(np.sqrt(np.mean((r - np.mean(r)) ** 2)))
    rms_after = float(np.sqrt(np.mean((X @ coef - r) ** 2)))
    return np.array(w), ab[:, 0], ab[:, 1], rms_before, rms_after


def calibrate_cogging(env: Env, log=None) -> tuple[CoggingMap, dict]:
    """Sweep every joint in free space from the current posture and fit its ripple. Leaves the arm where it
    started. Returns the map and a report (residual RMS before / after the fit, per joint; duration)."""
    dyn = env.dyn
    jc = JointImpedance(dyn, kp=np.array([150, 150, 100, 100, 25, 25, 25.0]), kd=np.array([8, 8, 5, 5, 0.8, 0.8, 0.8]))
    obs = MomentumObserver()
    q0 = env.last.q.copy()
    t0 = env.t
    w_nom = 2 * np.pi * RIPPLE_CYCLES_PER_RAD[list(HARM)]          # (3, 7)
    W, A, B = np.zeros((3, 7)), np.zeros((3, 7)), np.zeros((3, 7))
    rep = {"rms_before": [], "rms_after": []}

    def step_to(qt, qdt):
        s = env.last
        dyn.update(s.q, s.qd)
        jc.q_target, jc.qd_target = qt, qdt
        s = env.apply_torque(jc.torque(s.q, s.qd))
        dyn.update(s.q, s.qd)
        return s, obs.update(dyn.M, dyn.bias, dyn.damping, s.qd, s.tau)

    for _ in range(150):  # settle the observer at rest (0.3 s)
        step_to(q0, np.zeros(7))
    lo, hi = dyn.q_lo, dyn.q_hi
    for j in range(7):
        # never sweep into an end stop (its torque would be fitted as ripple): stay 0.03 rad short of the limits
        A_j, v = min(SWEEP[j], q0[j] - lo[j] - 0.03, hi[j] - q0[j] - 0.03), SPEED[j]
        if A_j < 0.02:
            if log:
                log(f"joint {j + 1}: at its end stop, not calibrated")
            rep["rms_before"].append(float("nan"))
            rep["rms_after"].append(float("nan"))
            continue
        period = 4 * A_j / v
        n = int(round(period / 0.002))
        qs, qds, rs = [], [], []
        for cyc in range(2):  # two triangle cycles, the first quarter of the first one discarded
            for k in range(n):
                ph = k / n
                tri = 4 * ph if ph < 0.25 else (2 - 4 * ph if ph < 0.75 else 4 * ph - 4)   # 0 -> 1 -> -1 -> 0
                qt = q0.copy()
                qt[j] += A_j * tri
                qdt = np.zeros(7)
                qdt[j] = v * (1 if (ph < 0.25 or ph >= 0.75) else -1)
                s, r = step_to(qt, qdt)
                if cyc > 0 or ph > 0.25:
                    qs.append(s.q[j])
                    qds.append(s.qd[j])
                    rs.append(r[j])
        for _ in range(100):
            step_to(q0, np.zeros(7))
        w, a, b, rb, ra = fit_joint(np.array(qs), np.array(qds), np.array(rs), q0[j], w_nom[:, j])
        W[:, j], A[:, j], B[:, j] = w, a, b
        rep["rms_before"].append(rb)
        rep["rms_after"].append(ra)
        if log:
            log(f"joint {j + 1}: residual RMS {rb:.3f} -> {ra:.3f} Nm, ripple amplitudes {np.round(np.hypot(a, b), 3)}")
    rep["seconds"] = env.t - t0
    return CoggingMap(W, A, B), rep
