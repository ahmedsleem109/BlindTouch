"""Job 1: plug a charger in behind the TV (blindtouch.scenes.job_scenes kind "socket", Europlug in hand).

A 3-way power strip (Schuko sockets, CEE 7/3: round recess 39 mm wide and 17.5 mm deep) lies on the desk behind a
TV; the camera cannot see it. The robot knows roughly where it is (+-2 cm) and that it lies along the desk edge
(turned 120 +- 15 deg); it holds a Europlug (CEE 7/16: pins 4 mm x 19 mm, 18.5 mm apart). Procedure, as on the
design sheet:

1. Lower the plug until the pin tips touch the strip.
2. Spiral on the strip's top until the pins drop into a socket's recess.
3. Lift the pins 4 mm off the recess floor and touch its wall both ways along the pins and both ways across them
   (in the air: the floor's holes and friction cannot fake a wall). Along the plug's own axes its footprint is
   symmetric, so the midpoint of each pair is the centre exactly, however far off the probe line starts.
4. Turn one way, then the other, until the pins drop into their holes (aligned within ~3 deg and ~0.6 mm).
5. Push until the plug seats.

When a step fails (the pins never drop, or sink only half-way, or no wall is found) the robot names the mistake
and tries a fix chosen from its experience memory (blindtouch.skills.experience): what worked before for the same mistake, in
this job or any other.

Any free socket of the strip will do, as in real life. Success: the pins are at least 15 mm into their holes.
"""

from __future__ import annotations

import math

import numpy as np

from blindtouch.scenes import props as P
from blindtouch.scenes.build import GEOM
from blindtouch.scenes.job_scenes import TABLE
from blindtouch.jobs.base import Job, budget, step
from blindtouch.sim.robot import wrap
from blindtouch.skills.touch_robot import TouchRobot

STRIP = P.socket_strip()[2]            # top 40 mm, recess floor 22.5 mm above the table
TOP = TABLE + STRIP["top"]
RECESS = STRIP["top"] - STRIP["recess_floor"]   # 17.5 mm
PIN_LEN = P.DIMS["europlug"]["pin_len"]         # 19 mm
STRIP_X = (-0.065, 0.0, 0.065)                  # socket centres along the strip (blindtouch.scenes.props.socket_strip)
STRIP_HOLES = P.DIMS["schuko"]["hole_spacing"]  # 19 mm between a socket's two pin holes
YAW0 = math.radians(120)    # the strip's usual turn; tool yaws 105-135 deg keep the wrist off its stops
YAW_RANGE = math.radians(15)
POS_RANGE = 0.02
TURN = math.radians(25)     # search this far each way for the pin holes
YAW0_ROB = YAW0             # the plug's yaw at the start: turns are measured from here
TURN_PRESS = 1.0           # N on the recess floor while turning: lighter = less friction dragging the plug off centre
TURN_SPEED = 0.12          # rad/s
COMPLY = 0.004              # m/(N s): admittance while pressing in -- the recess rim and round pins centre the plug
SEATED = 0.018               # m: pin tips this far below the recess floor = seated (pins 19 mm; the plug can
                            # slip ~1 mm up in the fingers under the push, so judge close to full depth)
SEAT_FORCE = 15.0          # N: the contact springs grip the pins; 10 N left some plugs stuck 10 mm in (budget 25 N)
LIFT = 0.004                # m: pins lifted off the recess floor to probe its wall (the recess is 17.5 mm deep)


class SocketJob(Job):
    name, scene, title = "socket", "socket", "Plug in a charger behind the TV"
    max_time = 120.0   # with up to two recoveries
    RANGES = {"pos": POS_RANGE, "yaw": YAW_RANGE, "pin_mu": (0.2, 0.4)}

    def setup(self, env, rng, seed):
        m = env.model
        R = self.ranges
        xy = np.asarray(GEOM.hole_nominal) + rng.uniform(-R["pos"], R["pos"], 2)
        yaw = YAW0 + rng.uniform(-R["yaw"], R["yaw"])
        for g in range(m.ngeom):          # plastic-on-brass friction varies with wear (pins 0.2-0.4)
            if (m.geom(g).name or "").startswith("plug_pin"):
                m.geom_friction[g][0] = rng.uniform(*R["pin_mu"])
        start = np.r_[np.asarray(GEOM.hole_nominal), TOP + 0.04]
        env.reset(seed=seed, tip_start=start, tip_yaw=YAW0, fixture=(xy[0], xy[1], yaw))
        return {"xy": xy, "yaw": yaw}

    def stage(self, rob: TouchRobot) -> str:
        """Where the plug is, from how deep the pin tips are below the strip top (the strip's height is known)."""
        d = TOP - rob.tip()[2]
        if d < 0.006:
            return "on top"
        if d < RECESS + 0.004:
            return "in the recess"
        if d > RECESS + PIN_LEN + 0.002:   # deeper than seated pins can go: slid off the strip's edge onto the desk
            return "off the strip"
        return "in the holes"

    def procedure(self, rob: TouchRobot, rng):
        t_end = rob.t + self.max_time
        nominal = np.asarray(GEOM.hole_nominal)
        rob.yaw_cmd = YAW0

        step(rob, "Lower the plug until it touches the power strip")
        rob.move_to(np.r_[nominal, TOP + 0.015], speed=0.06, yaw=YAW0)
        if rob.touch_down(speed=0.015, press=2.5).event != "contact":
            return {"stage": "touch"}
        if self.stage(rob) == "on top":
            step(rob, "Spiral on the strip until the pins drop into a socket")
            if rob.spiral(nominal, pitch=0.006, speed=0.025, max_r=0.032, press=2.5, drop=0.008).event != "drop":
                return {"stage": "spiral"}
        rob.push(force=3.0, speed=0.02, depth=0.040, click_drop=1.0, comply=COMPLY)   # deep enough to tell the desk
        budget(rob, t_end)
        if self.stage(rob) == "off the strip":
            edge = rob.tip()[:2].copy()
            step(rob, "Mistake (spiral: off the strip): try a fix learned from experience")
            if not rob.recover("spiral: off the strip", {
                    "back on top, nearer the middle": lambda: self.back_on_top(rob, edge, nominal)}, **rob.context):
                return {"stage": "off the strip"}
        if self.stage(rob) == "on top":
            return {"stage": "recess"}
        centre = None
        if self.stage(rob) == "in the recess":
            step(rob, "Lift the pins off the floor, touch the socket wall all round, move to its centre")
            centre = self.centre(rob)
            if centre is None and not rob.recover("centre: no wall", {
                    "lift higher and probe again": lambda: (c := self.centre(rob, lift=2 * LIFT)) is not None and self._set(c),
                    "probe slower": lambda: (c := self.centre(rob, speed=0.006)) is not None and self._set(c)},
                    **rob.context):
                return {"stage": "centre"}
            centre = centre if centre is not None else self._centre
            budget(rob, t_end)
        if self.stage(rob) == "in the recess":
            step(rob, "Turn until the pins drop into their holes")
            if not self.turn_in(rob, centre, TURN):
                failure = "turn: half in" if self.sunk(rob) > 0.001 else "turn: no drop"
                step(rob, f"Mistake ({failure}): try a fix learned from experience")
                fixes = {"small spirals at stepped angles": lambda: self.stepped_search(rob, centre),
                         "re-centre and turn again": lambda: self.recentre_and_turn(rob),
                         "turn wider": lambda: self.turn_in(rob, centre, 1.6 * TURN)}
                if failure == "turn: half in":   # pushing only helps when the pins have started to go in
                    fixes = {"push in": lambda: self.push_in(rob), **fixes}
                if not rob.recover(failure, fixes, **rob.context):
                    return {"stage": "turn"}
        step(rob, "Push until the plug seats")
        rob.push(force=SEAT_FORCE, speed=0.008, depth=0.03, click_drop=1.0, comply=COMPLY)
        rob.wait(0.3)
        if not self.seated(rob):
            step(rob, "Mistake (seat: stuck): try a fix learned from experience")
            if not rob.recover("seat: stuck", {
                    "wiggle while pushing": lambda: self.wiggle_in(rob),
                    "back off and push again": lambda: self.repush(rob)}, **rob.context):
                return {"stage": "seat"}
        return {"stage": "done"}

    # ------------------------------------------------------------------ skills and recoveries (felt outcomes)
    def _set(self, c) -> bool:
        self._centre = c
        return True

    def sunk(self, rob: TouchRobot) -> float:
        """How far the pin tips are below the recess floor (felt: from the known strip height)."""
        return (TOP - RECESS) - rob.tip()[2]

    def centre(self, rob: TouchRobot, lift: float = LIFT, speed: float = 0.012):
        """Lift the pins off the recess floor and centre on the recess by touching its wall (along and across the
        pins); back down on the floor. Returns the centre (x, y) or None."""
        floor_z = TOP - RECESS
        rob.release_contact()
        rob.move_to(np.r_[rob.tip()[:2], floor_z + lift], speed=0.02, settle=0.4)
        rob.settle_free(0.3)
        y = rob.yaw()   # probe along the plug's own axes: there its two-pin footprint is symmetric
        axes = ((math.cos(y), math.sin(y)), (-math.sin(y), math.cos(y)))
        ev = rob.center_by_touch(axes=axes, max_dist=0.045, speed=speed)
        if ev.event != "centred":
            return None
        if rob.touch_down(speed=0.01, press=1.5).event != "contact":
            return None
        self._centre = ev.data["center"]
        return self._centre

    def turn_in(self, rob: TouchRobot, centre, turn: float) -> bool:
        """Turn to +turn, then to -turn (from the start yaw) about the centre until the pins drop into their holes
        (felt: the drop)."""
        for target in (YAW0_ROB + turn, YAW0_ROB - turn):   # absolute: never past the wrist's comfortable range
            ev = rob.rotate(wrap(target - rob.yaw_cmd), speed=TURN_SPEED, press=TURN_PRESS, center=centre, partial=0.0012,
                            drop=0.002)   # aim only 4 mm below the floor: less press, less tilting of the plug in the fingers
            if ev.event == "drop" or self.stage(rob) == "in the holes":
                return True
        return False

    def stepped_search(self, rob: TouchRobot, centre, step_deg: float = 4.0, span_deg: float = 20.0) -> bool:
        """Classic search when the plug cannot be placed precisely (it tilts in the fingers): at yaws 0, +-4, +-8 ...
        from the start, a small spiral (2.5 mm) about the centre, pressing lightly, until the pins drop."""
        c = np.asarray(centre if centre is not None else rob.tip()[:2])
        angles = [0.0] + [s_ * k * step_deg for k in range(1, int(span_deg / step_deg) + 1) for s_ in (1, -1)]
        for a in angles:
            rob.rotate(wrap(YAW0_ROB + math.radians(a) - rob.yaw_cmd), speed=TURN_SPEED, press=TURN_PRESS,
                       center=c, drop=0.002)
            if self.stage(rob) == "in the holes":
                return True
            ev = rob.spiral(c, pitch=0.0012, speed=0.008, max_r=0.0025, press=TURN_PRESS, drop=0.002)
            if ev.event == "drop" or self.stage(rob) == "in the holes":
                return True
        return False

    def back_on_top(self, rob: TouchRobot, edge, start) -> bool:
        """The pins slid off the strip's long edge at ``edge``: the strip's middle line lies about half its width
        (28 mm) inward, across the strip's usual direction. Lift clear, spiral again from there."""
        across = np.array([-math.sin(YAW0), math.cos(YAW0)])
        side = np.sign((edge - start) @ across) or 1.0
        c = edge - side * 0.024 * across
        rob.release_contact()
        rob.move_to(np.r_[rob.tip()[:2], TOP + 0.015], speed=0.04, settle=0.3)
        rob.move_to(np.r_[c, TOP + 0.015], speed=0.04, settle=0.3)
        if rob.touch_down(speed=0.015, press=2.5).event != "contact" or self.stage(rob) == "off the strip":
            return False
        if self.stage(rob) == "on top" and rob.spiral(c, pitch=0.006, speed=0.025, max_r=0.032, press=2.5,
                                                      drop=0.008).event != "drop":
            return False   # (landing straight in a recess is fine)
        rob.push(force=3.0, speed=0.02, depth=0.040, click_drop=1.0, comply=COMPLY)
        return self.stage(rob) in ("in the recess", "in the holes")

    def seated(self, rob: TouchRobot) -> bool:
        """Felt: the pin tips went (nearly) their full length below the recess floor (the strip height is known)."""
        return self.sunk(rob) > SEATED

    def wiggle_in(self, rob: TouchRobot, deg: float = 1.5, cycles: int = 3) -> bool:
        """As a person does with a stuck plug: small turns either way while pressing, so the pins un-wedge and the
        fingers re-centre the plug, then push again."""
        y0 = rob.yaw_cmd
        for _ in range(cycles):
            for a in (deg, -deg):
                rob.rotate(wrap(y0 + math.radians(a) - rob.yaw_cmd), speed=0.1, press=SEAT_FORCE / 2, drop=0.03)
        rob.rotate(wrap(y0 - rob.yaw_cmd), speed=0.1, press=SEAT_FORCE / 2, drop=0.03)
        rob.push(force=SEAT_FORCE, speed=0.008, depth=0.03, click_drop=1.0, comply=COMPLY)
        rob.wait(0.3)
        return self.seated(rob)

    def repush(self, rob: TouchRobot) -> bool:
        """Ease off (the plug straightens in the fingers), then push again more slowly."""
        rob.release_contact()
        rob.move_to(rob.tip() + [0, 0, 0.002], speed=0.01, settle=0.5)
        rob.hold_contact(snapshot=False)
        rob.push(force=SEAT_FORCE, speed=0.004, depth=0.03, click_drop=1.0, comply=COMPLY)
        rob.wait(0.3)
        return self.seated(rob)

    def push_in(self, rob: TouchRobot) -> bool:
        rob.push(force=6.0, speed=0.008, depth=0.02, click_drop=1.0, comply=COMPLY)
        return self.stage(rob) == "in the holes"

    def recentre_and_turn(self, rob: TouchRobot) -> bool:
        c = self.centre(rob)
        return c is not None and self.turn_in(rob, c, TURN)

    def judge(self, env, truth, out):
        m, d = env.model, env.data
        floor = TABLE + STRIP["recess_floor"]
        tips = []
        for k in range(2):
            g = m.geom(f"plug_pin{k}").id
            axis = d.geom_xmat[g].reshape(3, 3)[:, 2]
            half = m.geom_size[g][1] + m.geom_size[g][0]
            tips.append(min(d.geom_xpos[g][2] - abs(axis[2]) * half, d.geom_xpos[g][2] + abs(axis[2]) * half))
        depth = floor - max(tips)          # the shallower pin counts
        # ... and each pin must be in a hole of a socket (a plug pushed beside the strip is not plugged in)
        c, s_ = math.cos(truth["yaw"]), math.sin(truth["yaw"])
        holes = [np.asarray(truth["xy"]) + (x0 + sx * STRIP_HOLES / 2) * np.array([c, s_])
                 for x0 in STRIP_X for sx in (-1, 1)]
        in_holes = all(min(np.linalg.norm(d.geom_xpos[m.geom(f"plug_pin{k}").id][:2] - h) for h in holes) < 0.003
                       for k in range(2))
        if not in_holes:
            depth = min(depth, 0.0)
        return depth >= 0.015, {"in_holes": bool(in_holes), "pin_depth_mm": round(1e3 * depth, 1), "stage": out.get("stage", out.get("aborted")),
                                "yaw_off_deg": round(float(np.degrees(truth["yaw"] - YAW0)), 1)}
