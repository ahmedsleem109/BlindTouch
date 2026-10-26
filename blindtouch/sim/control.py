"""Low-level torque control on the nominal model.

The controller sees only what a real controller sees: measured joint state and
the nominal (possibly wrong) dynamics model. It returns joint torques.

:class:`NominalDynamics` evaluates the believed model once per control step;
the controller and the momentum observer share that evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from blindtouch.sim.robot import PEG_AXIS_DOWN, Arm, tool_yaw, wrap


class NominalDynamics:
    """Mass matrix, bias forces and Jacobians of the nominal model at measured (q, qd)."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        self.data = mujoco.MjData(model)
        self.arm = Arm(model)
        self.M = np.zeros((7, 7))
        self.bias = np.zeros(7)
        self.damping = model.dof_damping[self.arm.dof_idx].copy()
        self._Mfull = np.zeros((model.nv, model.nv))
        self._last = None
        jids = model.dof_jntid[self.arm.dof_idx]
        self.q_lo, self.q_hi = model.jnt_range[jids, 0].copy(), model.jnt_range[jids, 1].copy()

    def update(self, q: np.ndarray, qd: np.ndarray) -> None:
        if self._last is not None and np.array_equal(q, self._last[0]) and np.array_equal(qd, self._last[1]):
            return  # already evaluated at this measurement
        self._last = (q.copy(), qd.copy())
        a = self.arm
        self.data.qpos[a.qpos_idx] = q
        self.data.qvel[a.dof_idx] = qd
        m, d = self.model, self.data
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_crb(m, d)
        mujoco.mj_comVel(m, d)
        mujoco.mj_rne(m, d, 0, d.qfrc_bias)  # Coriolis + centrifugal + gravity
        mujoco.mj_fullM(m, d, self._Mfull)
        self.M = self._Mfull[np.ix_(a.dof_idx, a.dof_idx)].copy()
        self.bias = d.qfrc_bias[a.dof_idx].copy()

    def tip_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.arm.site_pose(self.data, self.arm.tip_site)

    def tip_jac(self) -> np.ndarray:
        return self.arm.site_jac(self.data, self.arm.tip_site)


@dataclass
class ImpedanceGains:
    kp_pos: np.ndarray = field(default_factory=lambda: np.array([600.0, 600.0, 600.0]))
    kd_pos: np.ndarray = field(default_factory=lambda: np.array([45.0, 45.0, 45.0]))
    kp_rot: float = 25.0
    kd_rot: float = 0.3  # 1.0 made the low-inertia wrist (joints 5, 7) oscillate at 1.2 rad/s with 6-10 ms latency
    kp_yaw: float = 8.0    # Nm/rad: turn about the vertical tool axis, only when CartesianImpedance.target_yaw is set
    # force cap (the touch layer turns it on): the spring term may never command more than f_cap N of push or
    # m_cap Nm of twist, whatever target a skill sets -- a stuck tool then meets a bounded force, not a winding
    # spring. None = off.
    f_cap: float | None = None
    m_cap: float | None = None
    kp_null: float = 4.0
    kd_null: float = 0.3
    # joint-limit avoidance in the null space: within limit_margin (rad) of an end stop, a spring of kp_limit
    # (Nm/rad) pushes the joint away, projected so the tip pose is not disturbed. A joint resting on its stop adds a
    # torque no model explains -- the observer reads it as a contact force. None = off.
    limit_margin: float | None = None
    kp_limit: float = 20.0
    # slow, leaky, clamped integral on the lateral (x, y) tip error (with the DOB, the best setting for every search
    # skill in a skill benchmark during development). At standstill torque ripple
    # and gravity-model error act like constant tip forces of 1-2 N, i.e. 2-3 mm of steady lateral error at
    # kp 600 N/m -- more than the 1.5 mm hole clearance. z is left alone: there the offset is a force command.
    ki_pos: np.ndarray = field(default_factory=lambda: np.array([1000.0, 1000.0, 0.0]))  # N/(m s)
    i_max: float = 3.0     # N: the integral force never exceeds this (it cannot hide a real contact force)
    i_leak: float = 0.5    # 1/s: forgets old error, so a held contact force is not fought indefinitely
    # anti-windup against a constraint: when the estimated lateral contact force (DOB) opposes the integral, the
    # environment is pushing back -- a bore wall or a chamfer -- so the integral bleeds off fast instead of
    # pressing the peg into the wall (it jammed the v2 demonstrator in the bore: 12/12 -> 6/12 clean)
    # vertical integral in free space only (needs the DOB): joint stiction can hold the tip ~3 mm short of a lift
    # (the socket job probed its wall with the pins 1 mm above the floor instead of 4). In contact -- held on
    # purpose or detected -- the z offset is a force command, so the integral bleeds off at i_wall_leak instead
    # (keeping it in contact was tried: light presses, 1 N while turning a plug, then went wrong -- 10/12 -> 7/12).
    ki_z_free: float = 0.0     # N/(m s); 0 = off; the jobs switch it on (blindtouch.jobs.base.Job)
    i_wall_force: float = 1.0  # N: lateral contact force that counts as pushing back
    i_wall_leak: float = 15.0  # 1/s
    # contact-gated disturbance observer (DisturbanceObserver): cancels the unmodelled joint torque (gravity-model
    # error, torque ripple, joint friction) estimated in free space, frozen while in contact
    dob: bool = True  # on by default since the v2 skill bench (near-miss spiral, all effects: 43 -> 80 %)
    dob_tau: float = 0.08      # s: low-pass time constant of the free-space disturbance estimate
    dob_detect_tau: float = 0.03  # s: low-pass of the residual for the contact test (raw torque noise maps to > 2.5 N)
    dob_on: float = 2.5        # N: |tip force of (residual - estimate)| above this -> contact, freeze
    dob_off: float = 1.2       # N: below this for dob_release seconds -> free space again
    dob_release: float = 0.3   # s
    dob_moved: float = 0.05    # rad: joints moved this far while "in contact" -> it was a model-error transient
                               # (e.g. inertia error in a fast move), not a contact: re-adapt (else the frozen
                               # estimate goes stale as the posture changes and never releases)


class DisturbanceObserver:
    """Contact-gated disturbance observer on the generalized-momentum residual.

    The residual r of a momentum observer on the nominal model is the joint torque the model does not explain:
    r = tau_dist + J^T F_contact, where tau_dist (gravity/mass error, torque ripple, joint friction) varies
    slowly with the posture. In free space r = tau_dist, so a low-pass estimate d of r, fed forward as -d,
    removes the controller's steady-state error. A contact shows up as a jump of r - d; the estimate is then
    frozen (hysteresis on the tip force of r - d), so a real contact force is never cancelled -- the target
    offset stays a force command. Uses only measured q, qd, tau and the nominal model.
    """

    def __init__(self, g: ImpedanceGains, dt: float = 0.002) -> None:
        from blindtouch.sensing.observer import MomentumObserver

        self.g, self.dt = g, dt
        self.obs = MomentumObserver(dt=dt)
        self.reset()

    def reset(self) -> None:
        self.obs.reset()
        self.d = np.zeros(7)
        self.rf = np.zeros(7)
        self.f_contact = np.zeros(3)  # tip force of (residual - estimate): the contact force estimate (N)
        self.q_freeze = None
        self.contact = False
        self.hold = False  # set by a caller that *knows* it is in contact (blindtouch.skills.touch_robot): keep d frozen, whatever moves
        self._quiet = 0.0
        self._n = 0

    def update(self, dyn: NominalDynamics, qd: np.ndarray, tau_meas: np.ndarray, J: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
        g = self.g
        r = self.obs.update(dyn.M, dyn.bias, dyn.damping, qd, tau_meas)
        self._n += 1
        self.rf += (r - self.rf) * min(1.0, self.dt / g.dob_detect_tau)
        if self._n < 25:  # observer start-up transient (50 ms)
            return self.d
        Jt_pinv = np.linalg.pinv(J[:3].T, rcond=1e-3)  # joint torque -> tip force (translational part)
        self.f_contact = Jt_pinv @ (self.rf - self.d)
        f = np.linalg.norm(self.f_contact)
        if self.hold:
            self.contact = True
            return self.d
        if self.contact:
            self._quiet = self._quiet + self.dt if f < g.dob_off else 0.0
            moved = q is not None and self.q_freeze is not None and np.max(np.abs(q - self.q_freeze)) > g.dob_moved
            if self._quiet >= g.dob_release or moved:
                self.contact = False
        elif f > g.dob_on:
            self.contact, self._quiet = True, 0.0
            self.q_freeze = None if q is None else q.copy()
        if not self.contact:
            self.d += (self.rf - self.d) * min(1.0, self.dt / g.dob_tau)
        return self.d


class CartesianImpedance:
    """Peg-tip Cartesian impedance with gravity compensation and a posture null space.

    In contact the steady-state force is ``kp_pos * (target - tip)``, so the
    target offset is a force command; the reflex layer relies on that.
    """

    def __init__(self, dyn: NominalDynamics, gains: ImpedanceGains | None = None, q_rest: np.ndarray | None = None) -> None:
        self.dyn = dyn
        self.g = gains or ImpedanceGains()
        self.q_rest = q_rest
        self.target_pos = None
        self.target_axis = PEG_AXIS_DOWN.copy()
        self.target_yaw: float | None = None  # turn about the tool axis (rad); None = free, only damped (the peg)
        self.ff_force = np.zeros(3)  # optional feed-forward tip force (e.g. wiping)
        self.i_force = np.zeros(3)   # state of the lateral integral term
        self.dt = 0.002              # control period (blindtouch.sim.env.CTRL_DT)
        self.dob = DisturbanceObserver(self.g, self.dt)
        # anticogging (blindtouch.sensing.anticogging.CoggingMap): measured-angle -> ripple torque; fed forward on the command
        # and added to the reported torque the observer sees. None = off.
        self.cogging = None

    def reset(self) -> None:
        self.target_pos = None
        self.i_force[:] = 0.0
        self.dob.reset()

    def torque(self, q: np.ndarray, qd: np.ndarray, tau_meas: np.ndarray | None = None) -> np.ndarray:
        """Torque command; assumes ``dyn.update(q, qd)`` was called this step.

        ``tau_meas`` (the drivers' measured torque) feeds the disturbance observer when ``gains.dob`` is on.
        """
        dyn, g = self.dyn, self.g
        pos, quat = dyn.tip_pose()
        J = dyn.tip_jac()
        v = J @ qd
        if self.target_pos is None:
            self.target_pos = pos.copy()
            self.i_force[:] = 0.0
        err = self.target_pos - pos
        if g.f_cap is not None and g.kp_pos[0] * np.linalg.norm(err) > g.f_cap:
            err = err * g.f_cap / (g.kp_pos[0] * np.linalg.norm(err))
        if np.any(g.ki_pos):
            leak = g.i_leak
            if g.dob and self.dob.contact:
                fl = self.dob.f_contact[:2]
                if np.linalg.norm(fl) > g.i_wall_force and fl @ self.i_force[:2] < 0:
                    leak = g.i_wall_leak
            self.i_force[:2] += (g.ki_pos[:2] * err[:2] - leak * self.i_force[:2]) * self.dt
            self.i_force = np.clip(self.i_force, -g.i_max, g.i_max)
        if g.ki_z_free > 0 and g.dob:
            free = not (self.dob.hold or self.dob.contact)
            self.i_force[2] += ((g.ki_z_free * err[2] if free else 0.0)
                                - (g.i_leak if free else g.i_wall_leak) * self.i_force[2]) * self.dt
            self.i_force[2] = np.clip(self.i_force[2], -g.i_max, g.i_max)
        f = g.kp_pos * err - g.kd_pos * v[:3] + self.ff_force + self.i_force
        # align the peg axis (hand -z) with target_axis; yaw is left free and only damped
        peg_axis = -dyn.data.site_xmat[dyn.arm.tip_site].reshape(3, 3)[:, 2]
        m = g.kp_rot * np.cross(peg_axis, self.target_axis) - g.kd_rot * v[3:]
        if self.target_yaw is not None:  # keys, plugs, nuts: hold or turn the tool about its (vertical) axis
            tw = g.kp_yaw * wrap(self.target_yaw - tool_yaw(dyn.data.site_xmat[dyn.arm.tip_site]))
            if g.m_cap is not None:
                tw = float(np.clip(tw, -g.m_cap, g.m_cap))
            m += tw * self.target_axis * -1.0
        tau = J.T @ np.concatenate([f, m]) + dyn.bias
        cog = self.cogging(q) if self.cogging is not None else 0.0
        if g.dob and tau_meas is not None:
            tau -= self.dob.update(dyn, qd, tau_meas + cog, J, q)
        tau = tau - cog
        if self.q_rest is not None:
            Jt = J
            if self.target_yaw is None and g.limit_margin is not None:
                # the tool's turn about its axis is free: leave it to the null space too (2-D), so the posture
                # terms can roll the wrist off an end stop
                a = self.target_axis / np.linalg.norm(self.target_axis)
                Jt = np.vstack([J[:3], (np.eye(3) - np.outer(a, a)) @ J[3:]])
            Jpinv = np.linalg.pinv(Jt, rcond=1e-3)
            N = np.eye(7) - Jt.T @ Jpinv.T
            t_null = g.kp_null * (self.q_rest - q) - g.kd_null * qd
            if g.limit_margin is not None:
                lo, hi = self.dyn.q_lo, self.dyn.q_hi
                t_null = t_null + g.kp_limit * (np.maximum(0.0, g.limit_margin - (q - lo)) - np.maximum(0.0, g.limit_margin - (hi - q)))
            tau += N @ t_null
        return np.clip(tau, -dyn.arm.tau_max, dyn.arm.tau_max)


class JointImpedance:
    """Joint PD + nominal gravity compensation (used by the poke data generator)."""

    def __init__(self, dyn: NominalDynamics, kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        self.dyn = dyn
        self.kp = np.array([60, 60, 40, 40, 10, 10, 10.0]) if kp is None else kp
        self.kd = np.array([4, 4, 2.5, 2.5, 0.5, 0.5, 0.5]) if kd is None else kd
        self.q_target = None
        self.qd_target = np.zeros(7)

    def torque(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        if self.q_target is None:
            self.q_target = q.copy()
        tau = self.kp * (self.q_target - q) + self.kd * (self.qd_target - qd) + self.dyn.bias
        return np.clip(tau, -self.dyn.arm.tau_max, self.dyn.arm.tau_max)
