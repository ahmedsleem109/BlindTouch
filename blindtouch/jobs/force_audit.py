"""Force pipeline, part of every job run:

1. a force cap in the controller (``ImpedanceGains.f_cap`` / ``m_cap``): no skill can command more push or twist;
2. this audit: during a run, the true contact force (simulator truth, evaluation only) is broken down by contact
   pair and by procedure step, so a spike names its cause ("finger vs strip during 'Push until the plug seats'");
3. a force budget in the success test (``Job.force_budget``).
"""

from __future__ import annotations

from collections import defaultdict, deque

import mujoco
import numpy as np

from blindtouch.sim.robot import LINK_BODIES

GRIP = {"finger_inner", "finger_outer", "hand"}


class ForceAudit:
    """Env hook: per control step, every contact between the robot (arm, fingers, held tool) and the world,
    labelled by part, with the procedure step that was running. Grip contacts (tool vs fingers) are internal and
    skipped. Also the 20 ms average of the net force on the robot (as F/T sensors and the jobs' peak report)."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.m = model
        names = [model.body(b).name for b in range(model.nbody)]
        self.robot = {b for b, n in enumerate(names) if n in LINK_BODIES or n in GRIP}
        self.tool = {b for b, n in enumerate(names) if n == "peg"}
        self.label = {}
        for g in range(model.ngeom):
            b = int(model.geom_bodyid[g])
            n = names[b]
            if b in self.tool:
                self.label[g] = model.geom(g).name or "tool"
            elif n in GRIP:
                self.label[g] = "finger" if n.startswith("finger") else "hand"
            elif b in self.robot:
                self.label[g] = "arm"
            else:
                self.label[g] = n if n != "world" else (model.geom(g).name or "world")
        self.rob = None
        self.reset()

    def reset(self) -> None:
        self.pair_peak: dict = defaultdict(float)
        self.step_peak: dict = defaultdict(float)
        self.worst = (0.0, "", "", 0.0)   # (force, pair, step, t)
        self._win: deque = deque(maxlen=10)
        self.peak_avg = 0.0

    def _step(self) -> str:
        if self.rob is None:
            return "setup"
        for t, name, d in reversed(self.rob.events):
            if name == "step":
                return d["title"]
        return "start"

    def __call__(self, env, s) -> None:
        d = env.data
        self._win.append(s.truth.force.copy())
        self.peak_avg = max(self.peak_avg, float(np.linalg.norm(np.mean(self._win, axis=0))))
        f6 = np.zeros(6)
        step = None
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot or b1 in self.tool, b2 in self.robot or b2 in self.tool
            if r1 == r2:   # robot-robot (the grip) or world-world
                continue
            mujoco.mj_contactForce(self.m, d, i, f6)
            f = float(np.linalg.norm(f6[:3]))
            a, b = (self.label[c.geom1], self.label[c.geom2]) if r1 else (self.label[c.geom2], self.label[c.geom1])
            pair = f"{a} vs {b}"
            if f > self.pair_peak[pair]:
                self.pair_peak[pair] = f
            if f > self.worst[0] or f > 5.0:
                step = step or self._step()
                if f > self.step_peak[step]:
                    self.step_peak[step] = f
                if f > self.worst[0]:
                    self.worst = (f, pair, step, env.t)

    def report(self, top: int = 3) -> dict:
        pairs = sorted(self.pair_peak.items(), key=lambda kv: -kv[1])[:top]
        return {"peak_contact_N": round(self.worst[0], 1), "peak_pair": self.worst[1], "peak_step": self.worst[2],
                "top_pairs": [(p, round(f, 1)) for p, f in pairs]}
