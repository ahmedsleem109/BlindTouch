"""The shared touch layer: one robot interface that every contact job is built on.

What the robot *feels* comes only from motor signals: the generalized-momentum residual of the measured joint
state and torque against the nominal model (``blindtouch.sim.control.DisturbanceObserver``), minus its free-space
disturbance estimate, mapped to a wrench at the tool tip. No force sensor, no camera.

Primitives are blocking: each runs the simulation until its own stop condition and returns what it felt.
They run at 100 Hz (5 control steps of 2 ms); the impedance controller underneath runs at 500 Hz.

    rob = TouchRobot(env)
    rob.move_to([x, y, z])
    hit = rob.touch_down()                      # lower until a contact is felt
    ev = rob.slide([-1, 0], press=3.0)          # slide on a surface until it drops away (an edge or a hole)
    ev = rob.rotate(+0.6, press=4.0)            # turn the held tool until it drops in (pins, a key)
    ev = rob.push(force=12.0, depth=0.02)       # push down; reports a click (force rise then sudden drop)

Every primitive appends to ``rob.events`` (time, name, data) -- the procedure log the video and the
evaluation read. Truth (true contact force) is recorded for evaluation only and never used to decide.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from blindtouch.sim.env import Env
from blindtouch.sim.robot import tool_yaw, wrap

TICK = 5            # control steps per primitive tick (10 ms)
KP = 600.0          # N/m: impedance stiffness (blindtouch.sim.control.ImpedanceGains.kp_pos), offset -> force
DROP = 0.004        # m: the tip sank this far below the surface it was pressing on -> "drop"


def sink(press: float, drop: float) -> float:
    """Target depth below the surface while searching: the press force's spring offset, but always 2 mm deeper
    than the drop we look for -- the impedance target bounds how far the tool can fall into an opening."""
    return max(press / KP, drop + 0.002)


@dataclass
class Felt:
    """Result of a primitive: why it stopped, where, and what was felt."""

    event: str                     # e.g. "contact", "drop", "edge", "click", "stall", "limit", "timeout", "reached"
    tip: np.ndarray                # measured tip position when it stopped
    yaw: float = 0.0
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    data: dict = field(default_factory=dict)


class TouchRobot:
    def __init__(self, env: Env, f_limit: float = 25.0) -> None:
        self.env = env
        self.f_limit = f_limit              # reflex: felt force above this -> back off
        self.events: list[tuple[float, str, dict]] = []
        self.cmd = None                     # commanded tip target (world, m)
        self.yaw_cmd: float | None = None
        self._fwin = deque(maxlen=10)       # 20 ms window of true force (evaluation only)
        self.peak_true = 0.0
        self.touches = 0
        self.reflexes = 0
        self.trace: list[tuple] = []        # (t, tip, felt force, felt torque z, yaw) at 100 Hz, for plots / video
        self._dhist: deque = deque(maxlen=30)   # free-space disturbance estimates of the last 0.3 s
        self.contact_z: float | None = None     # height where the held contact began (None: free space)

    # ------------------------------------------------------------------ sensing (motor signals only)
    def tip(self) -> np.ndarray:
        e = self.env
        e.dyn.update(e.last.q, e.last.qd)
        return e.measured_tip_pos()

    def yaw(self) -> float:
        e = self.env
        e.dyn.update(e.last.q, e.last.qd)
        return tool_yaw(e.dyn.data.site_xmat[e.dyn.arm.tip_site])

    def wrench(self) -> np.ndarray:
        """Felt wrench at the tool tip [Fx, Fy, Fz, Mx, My, Mz] (N, Nm): force the world exerts on the tool."""
        e = self.env
        dob = e.ctrl.dob
        e.dyn.update(e.last.q, e.last.qd)
        J = e.dyn.tip_jac()
        return np.linalg.pinv(J.T, rcond=1e-3) @ (dob.rf - dob.d)

    def force(self) -> np.ndarray:
        return self.wrench()[:3]

    @property
    def t(self) -> float:
        return self.env.t

    def log(self, name: str, **data) -> None:
        self.events.append((self.env.t, name, data))

    # ------------------------------------------------------------------ deliberate contact
    def hold_contact(self, snapshot: bool = True) -> None:
        """We are touching on purpose: freeze the observer's free-space estimate at its value from 0.15 s ago
        (before the contact force started to leak into it), so the felt force stays the whole contact force
        while the arm slides, turns or pushes."""
        dob = self.env.ctrl.dob
        if snapshot and not dob.hold and len(self._dhist) >= 15:
            dob.d = self._dhist[-15].copy()
        dob.hold = True
        self.contact_z = float(self.tip()[2])

    def freeze(self) -> None:
        """Freeze the free-space estimate *now* (in free space, before grasping): from here on the felt force is
        whatever the world adds -- a grasped object's weight, then the table when it is set down. Stays frozen
        through lifting until :meth:`release_contact`."""
        self.env.ctrl.dob.hold = True
        self.contact_z = None

    def release_contact(self) -> None:
        dob = self.env.ctrl.dob
        dob.hold, dob.contact, dob._quiet, dob.q_freeze = False, False, 0.0, None
        self.contact_z = None

    # ------------------------------------------------------------------ low level
    def tick(self) -> None:
        e = self.env
        if self.cmd is None:
            self.cmd = self.tip()
        tip = self.tip()
        f = self.force()
        if self.contact_z is not None and np.linalg.norm(f) > self.f_limit:  # reflex (deliberate contact only): halve the push
            self.cmd = tip + 0.5 * (self.cmd - tip)
            self.reflexes += 1
        e.ctrl.target_pos = self.cmd.copy()
        e.ctrl.target_yaw = self.yaw_cmd
        if not e.ctrl.dob.hold:
            self._dhist.append(e.ctrl.dob.d.copy())
        for _ in range(TICK):
            s = e.control_step()
            self._fwin.append(s.truth.force.copy())
            self.peak_true = max(self.peak_true, float(np.linalg.norm(np.mean(self._fwin, axis=0))))
        w = self.wrench()
        self.trace.append((e.t, tip, w[:3].copy(), float(w[5]), self.yaw()))

    def wait(self, seconds: float) -> None:
        for _ in range(int(round(seconds / (TICK * 0.002)))):
            self.tick()

    def grip(self, close: bool, seconds: float = 0.6, open_angle: float = 0.30) -> None:
        e = self.env
        e.data.ctrl[e.model.actuator("grip").id] = 0.0 if close else -open_angle
        self.log("grip" if close else "release")
        self.wait(seconds)

    # ------------------------------------------------------------------ motion primitives
    def move_to(self, goal, speed: float = 0.06, yaw: float | None = None, yaw_speed: float = 0.8, tol: float = 0.003,
                settle: float = 1.0) -> Felt:
        """Move the commanded target in a straight line at ``speed`` (and turn to ``yaw``), then settle (at most
        ``settle`` s). Leaving a held contact upward releases it."""
        goal = np.asarray(goal, float)
        if self.cmd is None:
            self.cmd = self.tip()
        if yaw is not None and self.yaw_cmd is None:
            self.yaw_cmd = self.yaw()
        t0, step = self.t, speed * TICK * 0.002
        timeout = np.linalg.norm(goal - self.cmd) / speed + (0 if yaw is None else abs(wrap(yaw - self.yaw_cmd)) / yaw_speed) + settle
        while self.t - t0 < timeout:
            d = goal - self.cmd
            n = np.linalg.norm(d)
            self.cmd = goal.copy() if n <= step else self.cmd + d * step / n
            if yaw is not None:
                dy = wrap(yaw - self.yaw_cmd)
                self.yaw_cmd += float(np.clip(dy, -yaw_speed * TICK * 0.002, yaw_speed * TICK * 0.002))
            self.tick()
            if self.contact_z is not None and self.tip()[2] > self.contact_z + 0.004 and self.force()[2] < 1.0:
                self.release_contact()
            if n <= step and np.linalg.norm(self.tip() - goal) < tol and (yaw is None or abs(wrap(yaw - self.yaw())) < 0.03):
                return Felt("reached", self.tip(), self.yaw(), self.force())
        return Felt("timeout", self.tip(), self.yaw(), self.force())

    def touch_down(self, speed: float = 0.015, f_contact: float = 2.5, max_travel: float = 0.10, press: float = 2.0) -> Felt:
        """Lower slowly until a contact is felt (upward felt force), then keep pressing with ``press`` N.

        The descent starts in free space, so the observer's free-space estimate is frozen right away: a slowly
        rising contact force would otherwise be absorbed as a slow disturbance and never be felt."""
        if self.cmd is None:
            self.cmd = self.tip()
        self.settle_free(0.15)
        self.hold_contact(snapshot=False)
        z0, step = self.cmd[2], speed * TICK * 0.002
        while z0 - self.cmd[2] < max_travel:
            self.cmd = self.cmd + np.array([0, 0, -step])
            self.tick()
            f = self.force()
            if f[2] > f_contact:
                tip = self.tip()
                self.contact_z = float(tip[2])
                self.cmd = np.r_[tip[:2], tip[2] - press / KP]
                self.touches += 1
                self.log("contact", z=float(tip[2]), f=float(f[2]))
                return Felt("contact", tip, self.yaw(), f)
        self.release_contact()
        return Felt("nothing", self.tip(), self.yaw(), self.force())

    def touch_along(self, normal, speed: float = 0.015, f_contact: float = 2.5, max_travel: float = 0.08,
                    press: float = 2.0, stopped: float = 0.3, confirm: int = 3) -> Felt:
        """:meth:`touch_down` along ``-normal`` (a sloped surface with the tool tilted to it): approach until the felt
        force along ``normal`` exceeds ``f_contact``, then keep pressing with ``press`` N along it.

        ``stopped``: a contact must also stop the tool -- over the last 40 ms the tip approached at less than this
        share of the commanded speed. In free motion the tip keeps the command's speed (joint friction or model
        error only add a steady lag), so a felt force while still moving is the observer drifting, not a contact;
        it is ignored. So is a stop before the tool ever moved freely: joint stiction at the start of a move
        holds the tool back and shows as a felt force at the same time."""
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        if self.cmd is None:
            self.cmd = self.tip()
        self.settle_free(0.15)
        self.hold_contact(snapshot=False)
        c0, step = self.cmd.copy(), speed * TICK * 0.002
        hist = deque(maxlen=5)
        moving = False      # the tool has followed the command freely at least once (joint stiction let go)
        held = 0            # ticks in a row with force up and tool stopped (a stick-slip pause is shorter)
        while (c0 - self.cmd) @ n < max_travel:
            self.cmd = self.cmd - n * step
            self.tick()
            f = float(self.force() @ n)
            hist.append(float(self.tip() @ n))
            full = len(hist) == hist.maxlen
            progress = (hist[0] - hist[-1]) / (step * (hist.maxlen - 1)) if full else 0.0
            moving = moving or progress > 0.7
            # starting against the wall the tool never moves: then a clearly risen force (2x) is enough
            held = held + 1 if (f > f_contact and (moving or f > 2 * f_contact) and full and progress < stopped) else 0
            if held >= confirm:
                tip = self.tip()
                self.contact_z = float(tip[2])
                self.cmd = tip - n * press / KP
                self.touches += 1
                self.log("contact", z=float(tip[2]), f=f)
                return Felt("contact", tip, self.yaw(), self.force())
        self.release_contact()
        return Felt("nothing", self.tip(), self.yaw(), self.force())

    def settle_free(self, seconds: float) -> None:
        """Hold still in free space so the observer's free-space estimate catches up with the current posture."""
        if self.env.ctrl.dob.hold:
            return
        self.wait(seconds)

    def slide(self, direction, speed: float = 0.03, press: float = 3.0, max_dist: float = 0.12, drop: float = DROP,
              stop=None, surface_z: float | None = None) -> Felt:
        """Slide on a surface along ``direction`` (2D) pressing with ``press`` N until it drops away (edge or hole),
        ``stop(rob)`` returns an event name, or ``max_dist`` is covered."""
        u = np.asarray(direction, float)
        u = u / np.linalg.norm(u)
        tip0 = self.tip()
        z_s = tip0[2] if surface_z is None else surface_z
        self.cmd = np.r_[tip0[:2], z_s - sink(press, drop)]
        step = speed * TICK * 0.002
        while np.linalg.norm(self.cmd[:2] - tip0[:2]) < max_dist:
            self.cmd = self.cmd + np.r_[u * step, 0.0]
            self.tick()
            tip = self.tip()
            if tip[2] < z_s - drop:
                self.log("drop", xy=tip[:2].copy(), depth=float(z_s - tip[2]))
                return Felt("drop", tip, self.yaw(), self.force(), {"surface_z": z_s})
            if stop is not None and (ev := stop(self)):
                return Felt(ev, tip, self.yaw(), self.force(), {"surface_z": z_s})
            if self.force()[2] < 0.5 and np.linalg.norm(self.cmd[:2] - tip0[:2]) > 0.01:
                z_s = min(z_s, tip[2])  # lost the surface without sinking: follow it down gently
        return Felt("end", self.tip(), self.yaw(), self.force(), {"surface_z": z_s})

    def rotate(self, dyaw: float, speed: float = 0.35, press: float = 3.0, drop: float = DROP, stop=None,
               comply: float = 0.0, partial: float | None = None, stall_lag: float | None = None,
               center=None) -> Felt:
        """Turn the held tool about its vertical axis by up to ``dyaw`` rad while pressing down; stops when it
        drops in (pins into holes, a key into its slot), ``stop(rob)`` fires, or -- with ``stall_lag`` -- the
        tool stops following the commanded turn by more than that angle (it hit a stop: the command is then
        backed off to where the tool is, so the wrist does not keep twisting against it).

        ``comply``: admittance in x, y (as in :meth:`push`). ``partial``: on a partial sink of this depth the
        turn pauses and the tool is pressed compliantly for 0.4 s, so a beveled entrance can guide it in.
        ``center``: hold the turn at this (x, y) -- e.g. a centre found by touch -- instead of wherever the tip is."""
        tip0 = self.tip()
        z_s = tip0[2]
        if self.yaw_cmd is None:
            self.yaw_cmd = self.yaw()
        y_end = self.yaw_cmd + dyaw
        xy = tip0[:2] if center is None else np.asarray(center, float)[:2]
        self.cmd = np.r_[xy, z_s - sink(press, drop)]
        step = np.sign(dyaw) * speed * TICK * 0.002
        paused = False
        while abs(wrap(y_end - self.yaw_cmd)) > abs(step):
            tip = self.tip()
            if partial is not None and not paused and tip[2] < z_s - partial:
                paused = True   # something started to give: hold the angle and let compliance centre it
                for _ in range(40):
                    self.cmd[:2] += comply * self.force()[:2] * TICK * 0.002
                    self.tick()
                    if self.tip()[2] < z_s - drop:
                        break
            else:
                self.yaw_cmd += step
                if comply > 0:
                    self.cmd[:2] += comply * self.force()[:2] * TICK * 0.002
                self.tick()
            tip = self.tip()
            if tip[2] < z_s - drop:
                self.log("drop", yaw=self.yaw(), depth=float(z_s - tip[2]))
                return Felt("drop", tip, self.yaw(), self.force())
            if partial is not None and paused and tip[2] > z_s - partial / 2:
                paused = False   # it did not go in: carry on turning
            if stall_lag is not None and abs(wrap(self.yaw_cmd - self.yaw())) > stall_lag:
                self.yaw_cmd = self.yaw()
                self.log("stop", yaw=self.yaw(), torque=float(self.wrench()[5]))
                return Felt("stop", tip, self.yaw(), self.force(), {"torque": float(self.wrench()[5])})
            if stop is not None and (ev := stop(self)):
                return Felt(ev, tip, self.yaw(), self.force())
        return Felt("end", self.tip(), self.yaw(), self.force())

    def push(self, force: float = 10.0, speed: float = 0.01, depth: float = 0.03, click_drop: float = 0.35, stop=None,
             comply: float = 0.0) -> Felt:
        """Push straight down until the felt force reaches ``force`` (resistance), the tip went ``depth`` deeper,
        or a *click*: the felt force rose and then fell by ``click_drop`` of its peak while moving on (1.0: off).

        ``comply`` > 0: admittance in x, y -- the felt sideways force moves the target (m per N per s), so a sloped
        rim or a rounded pin can slide the tool into the centre instead of the stiff controller holding it on
        the rim."""
        tip0 = self.tip()
        z_goal = tip0[2] - depth
        step = speed * TICK * 0.002
        fmax, fz_hist = 0.0, []
        self.cmd = np.r_[tip0[:2], min(self.cmd[2], tip0[2])]
        while True:
            self.cmd = self.cmd + np.array([0, 0, -step])
            if comply > 0:
                fxy = self.force()[:2]
                self.cmd[:2] += comply * fxy * TICK * 0.002
            self.tick()
            tip, fz = self.tip(), float(self.force()[2])
            fz_hist.append(fz)
            fmax = max(fmax, fz)
            if click_drop < 1.0 and fmax > 2.0 and fz < (1 - click_drop) * fmax:   # 1.0: no click (felt force dips below 0 when sliding freely)
                self.log("click", peak=fmax, z=float(tip[2]))
                return Felt("click", tip, self.yaw(), self.force(), {"peak": fmax, "fz": fz_hist})
            if fz > force:
                self.log("resist", f=fz, z=float(tip[2]))
                return Felt("resist", tip, self.yaw(), self.force(), {"peak": fmax, "fz": fz_hist})
            if tip[2] < z_goal:
                return Felt("depth", tip, self.yaw(), self.force(), {"peak": fmax, "fz": fz_hist})
            if stop is not None and (ev := stop(self)):
                return Felt(ev, tip, self.yaw(), self.force(), {"peak": fmax, "fz": fz_hist})
            if self.cmd[2] < tip[2] - force / KP - 0.01:  # stalled far behind the command: blocked
                return Felt("stall", tip, self.yaw(), self.force(), {"peak": fmax, "fz": fz_hist})

    def mean_wrench(self, seconds: float = 0.4) -> np.ndarray:
        """Hold still and average the felt wrench (used for weighing)."""
        ws = []
        for _ in range(int(round(seconds / (TICK * 0.002)))):
            self.tick()
            ws.append(self.wrench())
        return np.mean(ws[len(ws) // 3:], axis=0)

    def dither_wrench(self, amplitude: float = 0.010, period: float = 1.0, cycles: int = 3,
                      center: np.ndarray | None = None) -> np.ndarray:
        """Average the felt wrench while moving gently up and down at constant speed (triangle wave, +-``amplitude``).

        Joint (Coulomb) friction holds an unknown share of the load whenever the arm stands still, so a still
        reading can be off by a newton or more. At constant speed, friction has the same size going up and going
        down and flips sign, so the average over whole cycles cancels it; every height of the stroke is sampled
        equally, which also averages torque ripple (periodic in the motor angles). ``center``: measured tip
        position to centre the stroke on -- the target is first nudged until the tip sits there, so a reading
        with a load (the hand sags under it) is taken at the same joint angles as the reading without.
        Returns the mean over the cycles after the first."""
        c = self.cmd.copy() if self.cmd is not None else self.tip()
        if center is not None:  # remove the sag: same joint angles as the reference reading
            for _ in range(60):
                self.cmd = self.cmd + 0.3 * (np.asarray(center) - self.tip())
                self.tick()
            c = self.cmd.copy()
        n = int(round(period / (TICK * 0.002)))
        ws, tips = [], []
        for k in range(n * cycles):
            ph = (k % n) / n
            tri = 4 * ph - 1 if ph < 0.5 else 3 - 4 * ph     # -1 -> 1 -> -1
            self.cmd = c + np.array([0, 0, amplitude * tri])
            self.tick()
            if k >= n:
                ws.append(self.wrench())
                tips.append(self.tip())
        self.cmd = c
        self.wait(0.1)
        self.last_dither_tip = np.mean(tips, axis=0)
        return np.mean(ws, axis=0)

    def air_pass(self, path, duration: float, offset, keep_start: bool = False) -> callable:
        """Trace ``path(s)`` (s in [0, 1] over ``duration`` s) shifted by ``offset`` (clear of any contact) and
        record the felt force along it. Returns ``felt_free(s)``: what the arm feels at s with no contact (joint
        friction, ripple and model error for exactly this motion), to subtract during the real pass."""
        n = int(round(duration / (TICK * 0.002)))
        off = np.asarray(offset, float)
        self.move_to(path(0.0) + off, speed=0.05)
        prof = []
        for k in range(n + 1):
            s = k / n
            self.cmd = path(s) + off
            self.tick()
            prof.append(self.force())
        prof = np.array(prof)
        # first 10 % hold the start-up transient from standstill; use the steady part for the start too -- unless
        # the real pass will also start from standstill after a fresh freeze (``keep_start``): then it has the same
        if not keep_start:
            k0 = max(1, n // 10)
            prof[:k0] = prof[k0]
        ss = np.linspace(0, 1, n + 1)
        return lambda s: np.array([np.interp(s, ss, prof[:, i]) for i in range(3)])

    def hold_posture(self, on: bool, kp: float = 30.0) -> None:
        """Stiffen the arm's spare (null-space) posture at the current joint angles, or restore the default.
        A load can otherwise swing the elbow without moving the tip, changing the joint angles between two
        readings (torque ripple depends on them)."""
        c = self.env.ctrl
        if on:
            self._posture = (c.q_rest.copy(), c.g.kp_null)
            c.q_rest, c.g.kp_null = self.env.last.q.copy(), kp
        elif getattr(self, "_posture", None) is not None:
            c.q_rest, c.g.kp_null = self._posture
            self._posture = None

    def spiral(self, center, pitch: float = 0.006, speed: float = 0.025, max_r: float = 0.03, press: float = 3.0,
               drop: float = DROP) -> Felt:
        """Press on a surface and slide along an outward spiral r = pitch * theta / 2 pi around ``center`` until the
        tool drops into something (an opening) or ``max_r`` is reached (classic industrial search)."""
        c = np.asarray(center, float)[:2]
        tip0 = self.tip()
        z_s = tip0[2]
        theta = 0.0
        self.cmd = np.r_[c, z_s - sink(press, drop)]
        dt = TICK * 0.002
        while True:
            r = pitch * theta / (2 * np.pi)
            if r > max_r:
                return Felt("end", self.tip(), self.yaw(), self.force())
            theta += speed * dt / max(np.hypot(r, pitch / (2 * np.pi)), 1e-4)
            r = pitch * theta / (2 * np.pi)
            self.cmd = np.r_[c + r * np.array([np.cos(theta), np.sin(theta)]), z_s - sink(press, drop)]
            self.tick()
            tip = self.tip()
            if tip[2] < z_s - drop:
                self.log("drop", xy=tip[:2].copy(), depth=float(z_s - tip[2]), r=float(r))
                return Felt("drop", tip, self.yaw(), self.force(), {"surface_z": z_s, "r": r})

    def recover(self, failure: str, options: dict, tries: int = 2, **context) -> bool:
        """Something went wrong (``failure``: a skill-level signature). Pick a recovery from ``options``
        ({name: callable returning True if the robot *felt* it worked}) with the experience memory
        (blindtouch.skills.experience), run it, record the outcome; up to ``tries`` different recoveries. No memory attached:
        options are tried in the given order."""
        mem = getattr(self, "memory", None)
        left = dict(options)
        for _ in range(min(tries, len(left))):
            name = mem.choose(failure, list(left)) if mem is not None else next(iter(left))
            self.log("recover", failure=failure, recovery=name)
            ok = bool(left.pop(name)())
            if mem is not None:
                mem.record(failure, name, ok, **context)
            self.log("recovered" if ok else "recovery failed", failure=failure, recovery=name)
            if ok:
                return True
        return False

    def center_by_touch(self, axes=((1.0, 0.0), (0.0, 1.0)), max_dist: float = 0.03, speed: float = 0.012,
                        f_contact: float = 3.0) -> Felt:
        """Find the middle of an opening the tool hangs in, clear of its floor: along each axis, touch the wall one
        way and the other (:meth:`touch_along` sideways -- the force must rise *and* the tool stop), go back to the
        midpoint. Probing in the air avoids what sliding on a floor feels on the way (friction, holes, seams).
        The midpoint is exact for a round opening when ``axes`` are the tool's own symmetry axes (e.g. along and
        across a plug's two pins); along other axes an off-centre tool touches at unequal distances."""
        z = self.tip()[2]
        for ax in axes:
            u = np.r_[np.asarray(ax, float) / np.linalg.norm(ax), 0.0]
            start = self.tip()
            hits = []
            for sgn in (1, -1):
                ev = self.touch_along(-sgn * u, speed=speed, f_contact=f_contact, max_travel=max_dist, press=0.5)
                self.release_contact()
                if ev.event != "contact":
                    self.move_to(start, speed=0.02, settle=0.3)
                    return Felt("no wall", self.tip(), self.yaw(), self.force())
                hits.append(ev.tip.copy())
                self.log("wall", xy=ev.tip[:2].copy())
                self.move_to(start, speed=0.02, settle=0.3)
                self.settle_free(0.2)
            mid = (hits[0] + hits[1]) / 2
            self.move_to(np.r_[mid[:2], z], speed=0.02, settle=0.6, tol=0.0005)
            self.settle_free(0.2)
        # the centre is the computed midpoint (the tip may still lag it under joint friction)
        return Felt("centred", self.tip(), self.yaw(), self.force(), {"center": mid[:2].copy()})

    def center_between_walls(self, axes=((1.0, 0.0), (0.0, 1.0)), press: float = 2.5, f_wall: float = 2.5,
                             max_dist: float = 0.025, speed: float = 0.015, drop: float = 0.008) -> Felt:
        """With part of the tool hanging inside an opening, slide along each axis until it meets a wall (felt
        sideways push-back), then the other way; move to the midpoint. Stops early if the tool drops deeper
        (it fell into the opening on the way)."""
        for ax in axes:
            u = np.asarray(ax, float)
            hits = []
            for sgn in (1, -1):
                d = sgn * u
                p0 = self.tip()[:2].copy()
                base = {}

                def wall(r, d=d, p0=p0, base=base):
                    """A wall: the push-back against the motion rises above its steady sliding level (taken once
                    1.5 mm into the slide, when leaving a previous wall's push is over)."""
                    if np.linalg.norm(r.tip()[:2] - p0) < 0.0015:
                        return None
                    f = -(r.force()[:2] @ d)
                    if "f" not in base:
                        base["f"] = f
                    return "wall" if f - base["f"] > f_wall else None

                ev = self.slide(d, speed=speed, press=press, max_dist=max_dist, drop=drop, stop=wall)
                if ev.event == "drop":
                    return ev
                hits.append(ev.tip[:2].copy())
                self.log("wall", xy=ev.tip[:2].copy())
            mid = (hits[0] + hits[1]) / 2
            ev = self.slide(mid - self.tip()[:2] if np.linalg.norm(mid - self.tip()[:2]) > 1e-4 else u, speed=speed,
                            press=press, max_dist=float(np.linalg.norm(mid - self.tip()[:2])), drop=drop)
            if ev.event == "drop":
                return ev
        return Felt("centred", self.tip(), self.yaw(), self.force())
