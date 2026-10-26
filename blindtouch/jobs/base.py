"""Contact jobs on the shared touch layer.

A job: picks its scene, randomises the world per seed (positions, angles, contents) and the realism effects,
runs a scripted procedure on :class:`blindtouch.skills.touch_robot.TouchRobot` -- logging each plain-language step with
``rob.log("step", title=...)`` -- and judges success from simulator truth (never shown to the procedure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blindtouch.sim.env import Env
from blindtouch.sim.realism import EFFECTS, RealismConfig
from blindtouch.skills.touch_robot import TouchRobot


@dataclass
class Result:
    success: bool
    time: float
    peak_force: float          # true contact force, 20 ms average, max over the episode (N)
    touches: int
    reflexes: int
    info: dict

    def row(self, seed: int) -> dict:
        return {"seed": seed, "success": bool(self.success), "time": float(self.time), "peak_force": float(self.peak_force),
                "touches": int(self.touches), "reflexes": int(self.reflexes), **self.info}


class Job:
    name = "job"
    scene = "peg"
    title = ""          # plain-language name for the video and tables
    max_time = 40.0
    calibrate = True    # anticogging map once per robot (episode), before the job's clock starts
    force_budget = 25.0  # N: peak net contact force (20 ms average); a run above it does not count as a success
    F_CAP, M_CAP = 15.0, 0.5   # controller caps on the commanded push (N) and twist (Nm) -- blindtouch.jobs.force_audit
    LIMIT_MARGIN = 0.25        # rad: null-space push away from joint end stops (blindtouch.sim.control.ImpedanceGains)
    KI_Z_FREE = 1000.0         # N/(m s): vertical integral in free space (blindtouch.sim.control.ImpedanceGains.ki_z_free)
    memory = None       # blindtouch.skills.experience.Experience: what was learned from earlier mistakes (shared by all jobs)
    CALIB_LIFT = 0.04   # m above the start pose for the calibration sweeps: clear of the fixture, and not so high
                        # that holding the tool vertical takes the wrist pitch (joint 7) onto its end stop

    # the world's random ranges the procedure was developed on (eval/robustness.py widens them, one at a time)
    RANGES: dict = {}

    def __init__(self, **ranges) -> None:
        self.env: Env | None = None
        self.ranges = {**self.RANGES, **ranges}

    def make_env(self) -> Env:
        if self.env is None:
            self.env = Env(self.scene)
        return self.env

    def realism(self, rng: np.random.Generator, p: float) -> RealismConfig:
        return RealismConfig(**{e: bool(rng.random() < p) for e in EFFECTS})

    def setup(self, env: Env, rng: np.random.Generator, seed: int) -> dict:
        """Randomise the world and reset the env; returns the hidden truth for judging."""
        raise NotImplementedError

    def procedure(self, rob: TouchRobot, rng: np.random.Generator) -> dict:
        """The scripted job, using only what the robot feels. Returns what it concluded."""
        raise NotImplementedError

    def judge(self, env: Env, truth: dict, out: dict) -> tuple[bool, dict]:
        raise NotImplementedError

    def run(self, seed: int, realism_p: float = 0.5, hooks: list | None = None) -> Result:
        env = self.make_env()
        rng = np.random.default_rng(seed)
        env.realism.cfg = self.realism(rng, realism_p)
        truth = self.setup(env, rng, seed)
        env.ctrl.cogging = None
        g = env.ctrl.g
        caps = (g.f_cap, g.m_cap, g.limit_margin, g.ki_z_free)
        g.limit_margin, g.ki_z_free = self.LIMIT_MARGIN, self.KI_Z_FREE   # from the start: the calibration moves must stay off the end stops too
        calib = {}
        if self.calibrate:
            calib = self.calibrate_motors(env)
        from blindtouch.jobs.force_audit import ForceAudit

        g.f_cap, g.m_cap = self.F_CAP, self.M_CAP
        audit = ForceAudit(env.model)
        lo, hi = env.dyn.q_lo, env.dyn.q_hi
        margin = [np.inf, -1, ""]

        def limits(e, s):   # smallest distance of any joint to its end stop (reported: value, joint, step)
            mg = np.minimum(s.q - lo, hi - s.q)
            if mg.min() < margin[0]:
                margin[:] = [float(mg.min()), int(np.argmin(mg)) + 1, audit._step()]

        env.hooks[:] = list(hooks or []) + [audit, limits]
        rob = TouchRobot(env)
        rob.memory = self.memory
        rob.context = {"job": self.name, "seed": int(seed)}
        audit.rob = rob
        self.rob, self.audit = rob, audit
        t0 = env.t
        try:
            out = self.procedure(rob, rng)
        except TimeoutError as e:
            out = {"aborted": str(e)}
        ok, info = self.judge(env, truth, out)
        g.f_cap, g.m_cap, g.limit_margin, g.ki_z_free = caps
        info.update(realism=env.realism.cfg.label(), task_ok=bool(ok), joint_margin_deg=round(float(np.degrees(margin[0])), 1),
                    joint_margin_at=f"joint {margin[1]} during '{margin[2]}'",
                    **audit.report(), **calib)
        ok = ok and audit.peak_avg <= self.force_budget
        return Result(ok, env.t - t0, audit.peak_avg, rob.touches, rob.reflexes, info)


    def calibrate_motors(self, env: Env) -> dict:
        """Lift clear, sweep every joint to map its torque ripple (blindtouch.sensing.anticogging), switch the anticogging on,
        and return to the start pose. Done once per robot; not part of the job's time."""
        from blindtouch.sensing.anticogging import calibrate_cogging

        r0 = TouchRobot(env)
        start, yaw = r0.tip(), r0.yaw()
        r0.yaw_cmd = yaw
        r0.move_to(start + [0, 0, self.CALIB_LIFT], speed=0.10)
        r0.wait(0.3)
        cmap, rep = calibrate_cogging(env)
        env.ctrl.cogging = cmap
        env.ctrl.dob.reset()
        env.ctrl.i_force[:] = 0.0
        r1 = TouchRobot(env)
        r1.yaw_cmd = yaw
        r1.move_to(start, speed=0.10)
        r1.wait(0.5)
        return {"calib_s": round(float(rep["seconds"]), 1),
                "ripple_rms_after_Nm": [round(x, 3) for x in rep["rms_after"]]}


def step(rob: TouchRobot, title: str, **data) -> None:
    """Announce a procedure step (shown in the video's checklist and kept in the event log)."""
    rob.log("step", title=title, **data)


def budget(rob: TouchRobot, t_end: float) -> None:
    if rob.t > t_end:
        raise TimeoutError("time limit")
