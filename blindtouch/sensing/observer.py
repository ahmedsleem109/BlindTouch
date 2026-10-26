"""Generalized momentum observer (De Luca & Mattone 2003; Haddadin et al. 2017).

With p = M(q) qd the robot obeys

    dp/dt = tau + tau_ext - D qd + C(q, qd)^T qd - g(q).

MuJoCo gives the bias b = C qd + g, and C^T qd = Mdot qd - C qd, so

    dp/dt = tau + tau_ext - D qd + Mdot qd - b.

The observer integrates the model side and compares with the measured momentum:

    r = K [ p - p0 - integral( tau - D qd + Mdot qd - b + r ) dt ],

so dr/dt = K (tau_ext - r): r is tau_ext through a first-order low-pass with
bandwidth K, and no joint acceleration is ever needed. Mdot is a finite
difference of M between control steps.

Everything here uses the *nominal* model and *measured* signals. Unmodelled
friction, torque ripple and model error therefore show up in r -- which is
exactly what the fixes (disturbance observer, anticogging) have to remove.
"""

from __future__ import annotations

import numpy as np


class MomentumObserver:
    def __init__(self, gain: float | np.ndarray = 150.0, dt: float = 0.002) -> None:
        self.K = np.asarray(gain, dtype=float)  # scalar or per-joint
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self.r = 0.0
        self._integral = 0.0
        self._p0 = None
        self._M_prev = None

    def update(self, M: np.ndarray, bias: np.ndarray, damping: np.ndarray, qd: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """One step with nominal M(q), b(q, qd), D and measured qd, tau; returns tau_ext estimate."""
        p = M @ qd
        if self._p0 is None:
            self._p0 = p.copy()
            self._M_prev = M.copy()
        Mdot_qd = (M - self._M_prev) @ qd / self.dt
        self._M_prev = M.copy()
        self._integral += (tau - damping * qd + Mdot_qd - bias + self.r) * self.dt
        self.r = self.K * (p - self._p0 - self._integral)
        return self.r.copy()
