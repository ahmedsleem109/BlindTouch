"""The Blind Touch environment: true plant, nominal model, realism, ground truth.

Rates
-----
physics   1 kHz   (``model.opt.timestep``)
control   500 Hz  torque update, measurement and momentum observer

Everything the robot may use is the *measured* joint state and motor torque produced by
:class:`blindtouch.sim.realism.Realism`; ground truth lives in ``Sample.truth`` and is for evaluation only.

The true plant holds its tool (the "peg" body: a peg, a plug, an eraser) as a free body in the real gripper,
held by friction only; the nominal model the controller and the observers use has the tool rigidly attached
(:func:`blindtouch.scenes.build.scene_path` with ``nominal=True``). Measured tip positions therefore assume the
tool sits where it was grasped -- if it slips or tilts in the fingers, that is an error the robot has to live with.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from blindtouch.scenes.build import GEOM, scene_path
from blindtouch.sim.contacts import ContactTruth, arm_contacts
from blindtouch.sim.control import CartesianImpedance, ImpedanceGains, NominalDynamics
from blindtouch.sim.realism import Realism, RealismConfig
from blindtouch.sim.robot import LINK_BODIES, PEG_AXIS_DOWN, Arm

PHYS_DT = 0.001
CTRL_SUBSTEPS = 2
CTRL_DT = PHYS_DT * CTRL_SUBSTEPS

# Peg vertical, tip 8 cm above the nominal hole (IK, farthest from joint limits).
Q_HOME = np.array([0.9878, 0.8302, -1.2986, 1.4257, -0.4867, -0.5138, 1.3167])
GRIP_OPEN = 0.30  # rad: finger angle of the open empty hand (fingertips ~48 mm from the centre: clear of the neck and of neighbours)


@dataclass
class Sample:
    """One control step: what the robot measured, what it commanded, what was true."""

    t: float
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray          # measured (reported) motor torque
    tau_cmd: np.ndarray      # exact command (for diagnostics only)
    truth: ContactTruth


class Env:
    def __init__(
        self,
        scene: str = "peg",
        realism: RealismConfig | None = None,
        seed: int = 0,
        gains: ImpedanceGains | None = None,
    ) -> None:
        self.scene = scene
        self.model = mujoco.MjModel.from_xml_path(str(scene_path(scene)))
        self.data = mujoco.MjData(self.model)
        self.nominal_model = mujoco.MjModel.from_xml_path(str(scene_path(scene, nominal=True)))
        self.arm = Arm(self.model)
        m = self.model
        self.finger_qpos = m.jnt_qposadr[[m.joint(n).id for n in ("finger_inner_joint", "finger_outer_joint")]]
        # the held tool (a free body named "peg" in every scene that has one; the empty hand has none)
        has_tool = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "peg_free") >= 0
        self.peg_qpos = int(m.jnt_qposadr[m.joint("peg_free").id]) if has_tool else None
        self.peg_dof = int(m.jnt_dofadr[m.joint("peg_free").id]) if has_tool else None
        from blindtouch.scenes.job_scenes import KINDS, grip_angle

        self.grip_angle = grip_angle(scene) if scene in KINDS else GEOM.finger_angle   # fingers on this tool
        self.dyn = NominalDynamics(self.nominal_model)
        self.ctrl = CartesianImpedance(self.dyn, gains, q_rest=Q_HOME.copy())
        self.realism = Realism(realism or RealismConfig(), self.arm.tau_max)
        self.rng = np.random.default_rng(seed)
        self._base_damping = self.model.dof_damping[self.arm.dof_idx].copy()
        ids = [self.model.body(n).id for n in LINK_BODIES]
        self._base_mass = self.model.body_mass[ids].copy()
        self._base_ipos = self.model.body_ipos[ids].copy()
        self._base_inertia = self.model.body_inertia[ids].copy()
        self.hole_mocap = self.model.body("hole").mocapid[0] if self._has_body("hole") else -1
        # job scenes (scenes/jobs.py): the task furniture on a movable mocap body "fixture"
        self.fixture_mocap = self.model.body("fixture").mocapid[0] if self._has_body("fixture") else -1
        self.fixture_z = float(self.model.body("fixture").pos[2]) if self.fixture_mocap >= 0 else 0.0
        self.t = 0.0
        self.step_count = 0
        self.tau_cmd = np.zeros(7)
        self.last: Sample | None = None
        self.hooks: list = []  # callables(env, sample) run every control step

    def _has_body(self, name: str) -> bool:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0

    # ------------------------------------------------------------------
    def reset(
        self,
        seed: int | None = None,
        hole_xy: np.ndarray | None = None,
        q0: np.ndarray | None = None,
        tip_start: np.ndarray | None = None,
        tip_yaw: float | None = None,
        fixture: tuple | None = None,
    ) -> Sample:
        """``fixture`` = (x, y, yaw) of the job scenes' fixture; ``tip_yaw`` also fixes the hand's turn about the
        vertical tool axis in the start IK (default: free, as the peg needs)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.realism.sample(self.rng)
        self.realism.apply_to_plant(self.model, self.arm, self._base_damping)
        self.realism.apply_to_nominal(self.nominal_model, self._base_mass, self._base_ipos, self._base_inertia)
        self.dyn.damping = self.nominal_model.dof_damping[self.arm.dof_idx].copy()
        self.dyn._last = None  # the nominal model may have changed
        if self.hole_mocap >= 0:
            xy = np.asarray(GEOM.hole_nominal) if hole_xy is None else np.asarray(hole_xy)
            self.data.mocap_pos[self.hole_mocap] = [xy[0], xy[1], GEOM.plate_top]
        if self.fixture_mocap >= 0:
            fx = (*GEOM.hole_nominal, 0.0) if fixture is None else fixture
            self.data.mocap_pos[self.fixture_mocap] = [fx[0], fx[1], self.fixture_z]
            self.data.mocap_quat[self.fixture_mocap] = [np.cos(fx[2] / 2), 0.0, 0.0, np.sin(fx[2] / 2)]
        q = Q_HOME.copy() if q0 is None else np.asarray(q0, dtype=float)
        if tip_start is not None:  # IK on the rigid nominal hand: the plant's peg tip is not attached to the arm
            q, _ = self.dyn.arm.ik(self.dyn.data, tip_start, q0=q, rng=self.rng, yaw=tip_yaw)
        self.arm.set_q(self.data, q)
        self._grasp(q)
        self.t = 0.0
        self.step_count = 0
        self.tau_cmd = np.zeros(7)
        self.ctrl.target_pos = None
        self.ctrl.target_axis = PEG_AXIS_DOWN.copy()
        self.ctrl.target_yaw = None
        self.ctrl.ff_force[:] = 0
        self.ctrl.i_force[:] = 0
        self.ctrl.dob.reset()
        self.last = self._measure()
        self.dyn.update(self.last.q, self.last.qd)
        self.ctrl.target_pos = self.dyn.tip_pose()[0]
        return self.last

    def _grasp(self, q: np.ndarray, settle_s: float = 0.3) -> None:
        """Put the peg between the fingers where the nominal hand holds it, then let the grip settle.

        The arm is held at ``q`` while the finger servo squeezes and gravity
        loads the grasp; the peg ends up wherever friction holds it.
        """
        d = self.data
        d.ctrl[:] = 0.0
        if self.peg_qpos is None:  # empty hand: start with the fingers open
            d.qpos[self.finger_qpos] = -GRIP_OPEN
            d.ctrl[self.model.actuator("grip").id] = -GRIP_OPEN
        else:
            nd = mujoco.MjData(self.nominal_model)
            nd.qpos[self.dyn.arm.qpos_idx] = q
            mujoco.mj_kinematics(self.nominal_model, nd)
            peg = self.nominal_model.body("peg").id
            d.qpos[self.peg_qpos : self.peg_qpos + 3] = nd.xpos[peg]
            d.qpos[self.peg_qpos + 3 : self.peg_qpos + 7] = nd.xquat[peg]
            d.qvel[self.peg_dof : self.peg_dof + 6] = 0.0
            d.qpos[self.finger_qpos] = -self.grip_angle
            d.ctrl[self.model.actuator("grip").id] = 0.0  # squeeze toward fully closed
        self.dyn.update(q, np.zeros(7))
        d.ctrl[self.arm.act_idx] = self.dyn.bias  # gravity compensation: the arm really stands still
        for _ in range(int(settle_s / PHYS_DT)):
            mujoco.mj_step(self.model, d)
            self.arm.set_q(d, q)
        mujoco.mj_forward(self.model, d)
        self.dyn._last = None

    # ------------------------------------------------------------------
    def _measure(self) -> Sample:
        q, qd = self.arm.q(self.data), self.arm.qd(self.data)
        qm, qdm, taum = self.realism.measure(q, qd, self.tau_cmd.copy(), self.rng)
        truth = arm_contacts(self.model, self.data, self.arm)
        return Sample(self.t, qm, qdm, taum, self.tau_cmd.copy(), truth)

    def apply_torque(self, tau_cmd: np.ndarray) -> Sample:
        """Advance one control step with an explicit torque command."""
        self.tau_cmd = np.clip(tau_cmd, -self.arm.tau_max, self.arm.tau_max)
        for _ in range(CTRL_SUBSTEPS):
            tau = self.realism.actuate(self.tau_cmd, self.arm.q(self.data))
            self.data.ctrl[self.arm.act_idx] = tau
            mujoco.mj_step(self.model, self.data)
        self.t += CTRL_DT
        self.step_count += 1
        self.last = self._measure()
        for h in self.hooks:
            h(self, self.last)
        return self.last

    def control_step(self) -> Sample:
        """One control step of the Cartesian impedance controller on measured state."""
        s = self.last
        self.dyn.update(s.q, s.qd)
        tau = self.ctrl.torque(s.q, s.qd, s.tau)
        return self.apply_torque(tau)

    def run(self, n: int) -> list[Sample]:
        return [self.control_step() for _ in range(n)]

    # -- convenience (ground truth, for setting up and judging jobs) --
    def tip_pos(self) -> np.ndarray:
        """True peg-tip position (simulator truth)."""
        return self.data.site_xpos[self.arm.tip_site].copy()

    def measured_tip_pos(self) -> np.ndarray:
        """Tool-tip position from measured joints through the nominal model (what the robot may use)."""
        return self.dyn.tip_pose()[0]

    def hole_xy(self) -> np.ndarray:
        return self.data.mocap_pos[self.hole_mocap][:2].copy()
