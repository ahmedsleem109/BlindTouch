"""Job 3: which cans are still full? (blindtouch.scenes.job_scenes kind "cans", empty hand).

After a party: three identical 330 ml cans stand in a row, one full (343 g), one half full (178 g), one empty
(13 g), in random order and each placed within +-3 mm -- no camera can tell them apart. Procedure per can, as on
the design sheet: reach over it with the open hand, freeze the free-space estimate and feel the empty hand moving
gently up and down (joint friction flips sign and averages out); lower around the can, close on it at mid-height,
lift 4 cm and repeat the up-down reading at the same hand position -- the difference of the felt vertical force is
-m g. Then lift it over the others, carry it to the spot for its weight (recycling / sink / fridge tray) and set it
down, feeling the table take the weight.
Success: every can stands upright on the right spot.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from blindtouch.scenes import props as P
from blindtouch.scenes.job_scenes import TABLE
from blindtouch.jobs.base import Job, budget, step
from blindtouch.skills.touch_robot import TouchRobot

D = P.DIMS["can330"]
FILL = (0.0, 0.5, 1.0)              # share of 330 g of liquid: empty, half, full
NAMES = ("empty", "half full", "full")
SPOTS = ("recycling", "sink", "fridge tray")
GRIP_Z = 0.045                      # fingertips this high above the table: the pads hold the can around mid-height
WEIGH_LIFT = 0.04                   # lift this far off the table to weigh (the stroke goes 2 cm lower)
STROKE = 0.020                      # +- m of the up-down weighing stroke
CARRY_Z = D["h"] + 0.035            # fingertips this high to carry a can over the others (their tops at 115 mm)
OPEN = 0.45                         # rad: fingers open to ~81 mm around the 66 mm can
SQUEEZE = 0.025                     # rad the servo target lies inside the can: ~15 N squeeze, not a crushing 75 N
YAW = math.radians(112)             # fingers close (nearly) across the row, not into the 24 mm gaps between the cans;
                                    # yaws 105-120 deg keep every joint >= 8 deg from its stops here (scenes.check)


class CansJob(Job):
    name, scene, title = "cans", "cans", "Which cans are still full?"
    RANGES = {"jitter": 0.003, "can_mu": (0.4, 0.7)}
    max_time = 120.0

    def make_env(self):
        if self.env is not None:
            return self.env
        env = super().make_env()
        m = env.model
        self.bid = [m.body(f"can{i}").id for i in range(3)]
        self.qadr = [int(m.jnt_qposadr[m.joint(f"can{i}_free").id]) for i in range(3)]
        self.gid = [m.geom(f"can{i}").id for i in range(3)]
        return env

    def setup(self, env, rng, seed):
        fills = rng.permutation(3)
        m = env.model
        for k, b in enumerate(self.bid):
            mass = D["empty"] + FILL[fills[k]] * D["liquid"]
            m.body_mass[b] = mass
            r, h = D["d"] / 2, D["h"]
            m.body_inertia[b] = [mass * (3 * r * r + h * h) / 12] * 2 + [mass * r * r / 2]   # solid cylinder
            m.geom_friction[self.gid[k]][0] = rng.uniform(*self.ranges["can_mu"])      # wet or dry aluminium
        x0, y0 = P.CAN_SLOTS[0]
        env.reset(seed=seed, tip_start=np.array([x0, y0, TABLE + CARRY_Z]), tip_yaw=YAW)
        for k, qa in enumerate(self.qadr):   # placement error of the cans (+-3 mm)
            env.data.qpos[qa:qa + 2] = np.asarray(P.CAN_SLOTS[k]) + rng.uniform(-self.ranges["jitter"], self.ranges["jitter"], 2)
            env.data.qpos[qa + 2] = TABLE + 0.0005
            env.data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(env.model, env.data)
        return {"fills": [int(f) for f in fills], "mass": [float(env.model.body_mass[b]) for b in self.bid]}

    def procedure(self, rob: TouchRobot, rng):
        est, classes = [], []
        t_end = rob.t + self.max_time
        close = -(math.asin(D["d"] / 2 / 0.093) - SQUEEZE)
        grip_id = self.env.model.actuator("grip").id
        for i, (x, y) in enumerate(P.CAN_SLOTS):
            step(rob, f"Reach over can {i + 1}", can=i)
            rob.move_to([x, y, TABLE + CARRY_Z], speed=0.10, yaw=YAW)
            rob.grip(False, 0.4, open_angle=OPEN)
            rob.move_to([x, y, TABLE + GRIP_Z + WEIGH_LIFT], speed=0.06)
            step(rob, "Feel the empty hand (gentle up and down)", can=i)
            rob.settle_free(0.3)
            rob.freeze()                              # from here on, the felt force is what the can adds
            rob.hold_posture(True)
            w0 = rob.dither_wrench(amplitude=STROKE, cycles=4)
            ref = rob.last_dither_tip                 # weigh the can at exactly this hand position
            step(rob, "Grip the can, lift and weigh", can=i)
            rob.move_to([x, y, TABLE + GRIP_Z], speed=0.03, settle=0.3)
            self.env.data.ctrl[grip_id] = close
            rob.log("grip")
            rob.wait(0.6)
            rob.move_to([x, y, TABLE + GRIP_Z + WEIGH_LIFT], speed=0.03, settle=0.3)
            w = rob.dither_wrench(amplitude=STROKE, cycles=4, center=ref) - w0
            rob.hold_posture(False)
            m = float(-w[2] / 9.81)
            k = int(np.argmin([abs(m - (D["empty"] + f * D["liquid"])) for f in FILL]))   # known product
            est.append(m)
            classes.append(k)
            rob.log("weighed", can=i, mass=m, cls=NAMES[k])
            zx, zy = P.CAN_SPOTS[k]
            step(rob, f"It weighs {1e3 * m:.0f} g: {NAMES[k]}. Carry it to the {SPOTS[k]}", can=i, cls=k, mass=m)
            rob.move_to([x, y, TABLE + CARRY_Z + 0.06], speed=0.06)
            rob.move_to([zx, zy, TABLE + CARRY_Z + 0.06], speed=0.08)
            rob.move_to([zx, zy, TABLE + GRIP_Z + 0.02], speed=0.06)
            step(rob, "Set it down gently: feel the table take the weight", can=i)
            rob.touch_down(speed=0.02, f_contact=-m * 9.81 + 1.5, press=-m * 9.81 + 0.5, max_travel=0.05)
            rob.grip(False, 0.5, open_angle=OPEN)
            rob.release_contact()
            rob.move_to([zx, zy, TABLE + CARRY_Z + 0.06], speed=0.06)
            budget(rob, t_end)
        return {"mass": est, "cls": classes}

    def judge(self, env, truth, out):
        if "cls" not in out:
            return False, {"aborted": out.get("aborted")}
        placed = []
        for k, qa in enumerate(self.qadr):
            pos = env.data.qpos[qa:qa + 3]
            quat = env.data.qpos[qa + 3:qa + 7]
            tilt = 2 * np.degrees(np.arccos(np.clip(np.hypot(quat[0], quat[3]), -1, 1)))
            spot = np.asarray(P.CAN_SPOTS[truth["fills"][k]])
            placed.append(bool(np.max(np.abs(pos[:2] - spot)) < 0.025 and tilt < 10 and pos[2] < TABLE + 0.005))
        correct = [int(c == f) for c, f in zip(out["cls"], truth["fills"])]
        err = [abs(m - t) for m, t in zip(out["mass"], truth["mass"])]
        return all(placed) and all(correct), {"placed": placed, "classified": correct,
                                               "mass_err_g": [round(1e3 * e, 1) for e in err]}
