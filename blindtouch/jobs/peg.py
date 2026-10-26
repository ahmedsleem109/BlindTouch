"""Job 4: insert a peg into a hole the robot cannot see (scene "peg").

A 24 mm round peg, held in the gripper by friction, goes into a 27 mm square hole with a 6 mm chamfer, 30 mm deep:
1.5 mm of clearance. The robot knows only roughly where the hole is (+-8 mm). Procedure:

1. Lower the peg until it touches the plate.
2. If it landed on the flat top, spiral on the plate, pressing lightly, until the peg drops into the hole.
3. Push the peg down to the bottom, letting the felt sideways force slide it into the centre.
4. If it is felt stuck on the rim, back off and search around a little (a fix from the experience memory).

Success: the peg tip is at least 25 mm into the hole.
"""

from __future__ import annotations

import numpy as np

from blindtouch.jobs.base import Job, budget, step
from blindtouch.scenes.build import GEOM
from blindtouch.skills.touch_robot import TouchRobot

POS_RANGE = 0.008           # m: the hole is anywhere within +-8 mm of where the robot expects it
DEPTH_OK = 0.025            # m into the hole that counts as inserted
COMPLY = 0.002              # m per N per s: sideways admittance while pushing in
STUCK = 0.020               # m: felt depth below which the peg is caught on the rim, not in the bore


class PegJob(Job):
    name, scene, title = "peg", "peg", "Insert a peg into a hidden hole"
    max_time = 40.0
    RANGES = {"pos": POS_RANGE}

    def setup(self, env, rng, seed):
        hole = np.asarray(GEOM.hole_nominal) + rng.uniform(-self.ranges["pos"], self.ranges["pos"], 2)
        start = np.r_[np.asarray(GEOM.hole_nominal), GEOM.plate_top + 0.04]
        env.reset(seed=seed, hole_xy=hole, tip_start=start)
        return {"hole": hole}

    def procedure(self, rob: TouchRobot, rng):
        t_end = rob.t + self.max_time
        nominal = np.asarray(GEOM.hole_nominal)
        top = GEOM.plate_top

        step(rob, "Lower the peg until it touches the plate")
        rob.move_to(np.r_[nominal, top + 0.015], speed=0.06)
        if rob.touch_down(speed=0.015, press=3.0).event != "contact":
            return {"stage": "touch"}
        if rob.tip()[2] > top - 0.003:   # resting on the flat top, not in the chamfer
            step(rob, "Spiral on the plate until the peg drops into the hole")
            if rob.spiral(nominal, pitch=0.005, speed=0.02, max_r=0.016, press=3.0, drop=0.004).event != "drop":
                return {"stage": "spiral"}
        budget(rob, t_end)

        step(rob, "Push the peg to the bottom")
        self.push_in(rob)
        if top - rob.tip()[2] < STUCK:
            step(rob, "Mistake (push: stuck on the rim): try a fix learned from experience")
            rob.recover("push: stuck on the rim", {
                "lift and search around": lambda: self.lift_and_search(rob, 0.003),
                "lift higher and search around": lambda: self.lift_and_search(rob, 0.006)}, **rob.context)
        return {"stage": "done", "felt_depth_mm": round(1e3 * (top - rob.tip()[2]), 1)}

    def push_in(self, rob: TouchRobot) -> None:
        rob.push(force=10.0, speed=0.015, depth=0.035, click_drop=1.0, comply=COMPLY)
        rob.wait(0.3)

    def lift_and_search(self, rob: TouchRobot, lift: float) -> bool:
        """Back off ``lift``, spiral a little around where it caught, push again. Felt success: the peg went deep."""
        top = GEOM.plate_top
        here = rob.tip()
        rob.release_contact()
        rob.move_to(here + [0, 0, lift], speed=0.02, tol=0.001)
        rob.settle_free(0.2)
        rob.hold_contact(snapshot=False)
        rob.spiral(here[:2], pitch=0.002, speed=0.01, max_r=0.004, press=3.0, drop=0.004)
        self.push_in(rob)
        return top - rob.tip()[2] >= STUCK

    def judge(self, env, truth, out):
        depth = GEOM.plate_top - env.tip_pos()[2]
        off = np.abs(env.tip_pos()[:2] - truth["hole"]).max()
        info = {"stage": out.get("stage", out.get("aborted")), "depth_mm": round(1e3 * depth, 1),
                "hole_off_mm": [round(1e3 * float(x), 1) for x in truth["hole"] - GEOM.hole_nominal],
                "tip_vs_hole_mm": round(1e3 * float(off), 1)}
        return bool(depth >= DEPTH_OK and off < GEOM.hole_half), info
