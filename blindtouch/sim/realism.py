"""The realism layer: make the simulator lie the way a real arm does.

Each effect is switched independently from :class:`RealismConfig` so every
ablation is one flag. All randomness comes from the episode RNG
so a seed reproduces an episode exactly.

Effects, and where they act:

friction     true plant only: per-episode Coulomb (``dof_frictionloss``) and
             extra viscous (``dof_damping``). The nominal model never knows.
noise        measurements: Gaussian noise on q, qd and measured torque, plus
             encoder quantisation of q.
ripple       actuation: applied torque = command + periodic errors in motor
             angle, seen through the gearbox: the 1x and 2x electrical harmonics
             (current-sensor offset and gain mismatch in the FOC drive) and the 6x
             harmonic (cogging / back-EMF), at frequencies derived from each joint's
             motor (pole pairs x gear ratio, ``RIPPLE_CYCLES_PER_RAD``).
latency      measurements are delayed ``sensor_delay`` control steps and the
             actuator applies the command from ``act_delay`` steps ago.
model_error  nominal model only: wrong link masses and centres of mass, so the
             model-based observer and the gravity compensation are off.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, fields

import mujoco
import numpy as np

from blindtouch.sim.robot import LINK_BODIES

# Datasheet-ish Coulomb friction of the three DM motor sizes (the values the
# original OpenArm model used as frictionloss), per joint.
NOMINAL_COULOMB = np.array([0.2, 0.2, 0.1, 0.1, 0.04, 0.04, 0.04])
NOMINAL_DAMPING = np.array([1.0, 1.0, 0.9, 0.9, 0.9, 0.9, 0.9])


@dataclass
class RealismConfig:
    friction: bool = False
    noise: bool = False
    ripple: bool = False
    latency: bool = False
    model_error: bool = False
    # magnitudes
    coulomb_scale: tuple = (0.5, 2.0)      # x NOMINAL_COULOMB
    viscous_extra: tuple = (0.0, 0.5)      # x NOMINAL_DAMPING added on top
    q_noise: float = 2e-4                  # rad
    qd_noise: float = 0.02                 # rad/s
    tau_noise: float = 0.01                # fraction of the joint's torque limit
    encoder_bits: int = 14
    ripple_amp: float = 0.015              # fraction of the joint's torque limit (RMS-equal to the old single sinusoid)
    ripple_weights: tuple = (0.6, 0.3, 0.1)  # amplitude share of the 1x / 2x / 6x electrical harmonics (assumed)
    ripple_freq_jitter: float = 0.1        # +- relative spread of each harmonic's frequency per episode
    sensor_delay: tuple = (1, 3)           # control steps (2 ms each)
    act_delay: tuple = (0, 2)
    mass_error: float = 0.15               # relative, per link
    com_error: float = 0.01                # metres, per link

    @classmethod
    def all_on(cls, **kw) -> RealismConfig:
        return cls(friction=True, noise=True, ripple=True, latency=True, model_error=True, **kw)

    @classmethod
    def only(cls, name: str) -> RealismConfig:
        if name in ("none", "clean"):
            return cls()
        if name == "all":
            return cls.all_on()
        if name not in EFFECTS:
            raise ValueError(f"unknown realism effect {name!r}")
        return cls(**{name: True})

    @classmethod
    def from_dict(cls, d: dict | None) -> RealismConfig:
        d = dict(d or {})
        names = {f.name for f in fields(cls)}
        return cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names})

    def label(self) -> str:
        on = [e for e in EFFECTS if getattr(self, e)]
        return "+".join(on) if on else "clean"


EFFECTS = ("friction", "noise", "ripple", "latency", "model_error")

# Ripple harmonics in cycles per joint radian: k * pole_pairs * gear_ratio / (2 pi), k = 1, 2, 6.
# Vendor listings (no official Damiao datasheet found): DM-J8009 9:1, 21 pole pairs (joints 1-2);
# DM-J4340 40:1, 14 pole pairs (joints 3-4); DM-J4310 10:1, 14 pole pairs (joints 5-7).
_PP_N = np.array([21 * 9, 21 * 9, 14 * 40, 14 * 40, 14 * 10, 14 * 10, 14 * 10], dtype=float)
RIPPLE_HARMONICS = (1, 2, 6)
RIPPLE_CYCLES_PER_RAD = np.stack([k * _PP_N / (2 * np.pi) for k in RIPPLE_HARMONICS])  # (3, 7)


@dataclass
class EpisodeRealism:
    """The per-episode draw of every realism parameter."""

    coulomb: np.ndarray = field(default_factory=lambda: np.zeros(7))
    viscous_extra: np.ndarray = field(default_factory=lambda: np.zeros(7))
    ripple_amp: np.ndarray = field(default_factory=lambda: np.zeros((3, 7)))    # per harmonic, per joint
    ripple_freq: np.ndarray = field(default_factory=lambda: np.zeros((3, 7)))   # rad^-1 (angular: 2 pi x cycles)
    ripple_phase: np.ndarray = field(default_factory=lambda: np.zeros((3, 7)))
    sensor_delay: int = 0
    act_delay: int = 0
    mass_scale: np.ndarray = field(default_factory=lambda: np.ones(len(LINK_BODIES)))
    com_shift: np.ndarray = field(default_factory=lambda: np.zeros((len(LINK_BODIES), 3)))


class Realism:
    def __init__(self, cfg: RealismConfig, tau_max: np.ndarray) -> None:
        self.cfg = cfg
        self.tau_max = tau_max
        self.ep = EpisodeRealism()
        self._sens: deque = deque()
        self._act: deque = deque()

    def sample(self, rng: np.random.Generator) -> EpisodeRealism:
        c, ep = self.cfg, EpisodeRealism()
        if c.friction:
            ep.coulomb = NOMINAL_COULOMB * rng.uniform(*c.coulomb_scale, size=7)
            ep.viscous_extra = NOMINAL_DAMPING * rng.uniform(*c.viscous_extra, size=7)
        if c.ripple:
            w = np.asarray(c.ripple_weights, dtype=float)[:, None]
            w = w / np.sqrt(np.sum(w**2))  # same RMS as one sinusoid of amplitude ripple_amp
            ep.ripple_amp = w * c.ripple_amp * self.tau_max * rng.uniform(0.5, 1.0, size=(3, 7))
            jit = rng.uniform(1 - c.ripple_freq_jitter, 1 + c.ripple_freq_jitter, size=(3, 7))
            ep.ripple_freq = 2 * np.pi * RIPPLE_CYCLES_PER_RAD * jit
            ep.ripple_phase = rng.uniform(0, 2 * np.pi, size=(3, 7))
        if c.latency:
            ep.sensor_delay = int(rng.integers(c.sensor_delay[0], c.sensor_delay[1] + 1))
            ep.act_delay = int(rng.integers(c.act_delay[0], c.act_delay[1] + 1))
        if c.model_error:
            ep.mass_scale = 1.0 + rng.uniform(-c.mass_error, c.mass_error, size=len(LINK_BODIES))
            ep.com_shift = rng.uniform(-c.com_error, c.com_error, size=(len(LINK_BODIES), 3))
        self.ep = ep
        self._sens.clear()
        self._act.clear()
        return ep

    # -- model edits ---------------------------------------------------------
    def apply_to_plant(self, model: mujoco.MjModel, arm, base_damping: np.ndarray) -> None:
        model.dof_frictionloss[arm.dof_idx] = self.ep.coulomb
        model.dof_damping[arm.dof_idx] = base_damping + self.ep.viscous_extra

    def apply_to_nominal(
        self, model: mujoco.MjModel, base_mass: np.ndarray, base_ipos: np.ndarray, base_inertia: np.ndarray
    ) -> None:
        """Distort the nominal (believed) model; ``base_*`` are the true values per link."""
        for k, name in enumerate(LINK_BODIES):
            b = model.body(name).id
            model.body_mass[b] = base_mass[k] * self.ep.mass_scale[k]
            model.body_inertia[b] = base_inertia[k] * self.ep.mass_scale[k]
            model.body_ipos[b] = base_ipos[k] + self.ep.com_shift[k]

    # -- signals -------------------------------------------------------------
    def actuate(self, tau_cmd: np.ndarray, q_true: np.ndarray) -> np.ndarray:
        """Torque the motors actually produce for a command."""
        tau = tau_cmd
        if self.cfg.latency and self.ep.act_delay > 0:
            self._act.append(tau_cmd.copy())
            if len(self._act) > self.ep.act_delay:
                tau = self._act.popleft()
            else:
                tau = self._act[0]
        if self.cfg.ripple:
            tau = tau + np.sum(self.ep.ripple_amp * np.sin(self.ep.ripple_freq * q_true + self.ep.ripple_phase), axis=0)
        return np.clip(tau, -self.tau_max, self.tau_max)

    def measure(
        self, q: np.ndarray, qd: np.ndarray, tau: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """What the drivers report: delayed, noisy, quantised (q, qd, tau)."""
        c = self.cfg
        if c.latency and self.ep.sensor_delay > 0:
            self._sens.append((q.copy(), qd.copy(), tau.copy()))
            if len(self._sens) > self.ep.sensor_delay:
                q, qd, tau = self._sens.popleft()
            else:
                q, qd, tau = self._sens[0]
        if c.noise:
            q = q + rng.normal(0, c.q_noise, size=7)
            res = 2 * np.pi / (2**c.encoder_bits)
            q = np.round(q / res) * res
            qd = qd + rng.normal(0, c.qd_noise, size=7)
            tau = tau + rng.normal(0, c.tau_noise * self.tau_max)
        return q, qd, tau
