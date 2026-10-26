"""Index tables and kinematics for the single OpenArm v2 arm.

Everything here works on *any* MjModel built from the arm template, so the same
code serves the true plant and the nominal model the estimator believes.
"""

from __future__ import annotations

import mujoco
import numpy as np

JOINTS = tuple(f"joint{i}" for i in range(1, 8))
# Contact-link classes. 0 is "no contact"; the hand class lumps the hand, the
# frozen fingers and the peg, which are rigidly attached to joint 7.
LINK_BODIES = ("link1", "link2", "link3", "link4", "link5", "link6", "hand")
LINK_NAMES = ("none",) + LINK_BODIES
N_LINKS = len(LINK_BODIES)
N_CLASSES = N_LINKS + 1
HAND_EXTRA_BODIES = ("finger_inner", "finger_outer", "peg")

# Peg pointing straight down: the hand frame's orientation at q = 0.
DOWN_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
PEG_AXIS_DOWN = np.array([0.0, 0.0, -1.0])


def quat_err(q_target: np.ndarray, q_current: np.ndarray) -> np.ndarray:
    """World-frame rotation vector taking ``q_current`` to ``q_target``."""
    err = np.empty(4)
    conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    mujoco.mju_mulQuat(err, q_target, conj)
    if err[0] < 0:
        err = -err
    v = np.empty(3)
    mujoco.mju_quat2Vel(v, err, 1.0)
    return v


def wrap(a: float) -> float:
    """Angle wrapped to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))


def tool_yaw(R: np.ndarray) -> float:
    """Turn of a (vertical) tool about world z: heading of its frame's x axis."""
    R = np.asarray(R).reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def mat2quat(mat: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, dtype=np.float64).reshape(9))
    return q


class Arm:
    """Name-resolved indices plus kinematic helpers for one model."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        jids = [model.joint(n).id for n in JOINTS]
        self.qpos_idx = model.jnt_qposadr[jids].astype(int)
        self.dof_idx = model.jnt_dofadr[jids].astype(int)
        self.act_idx = np.array([model.actuator(n).id for n in JOINTS])
        self.q_lo = model.jnt_range[jids, 0].copy()
        self.q_hi = model.jnt_range[jids, 1].copy()
        self.tau_max = model.actuator_ctrlrange[self.act_idx, 1].copy()
        self.ee_site = model.site("ee").id
        self.tip_site = model.site("peg_tip").id
        self.link_body = np.array([model.body(n).id for n in LINK_BODIES])
        # body id -> link class (1..7), 0 for anything that is not the arm
        self.body_class = np.zeros(model.nbody, dtype=int)
        for k, b in enumerate(self.link_body):
            self.body_class[b] = k + 1
        for n in HAND_EXTRA_BODIES:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) >= 0:  # the empty hand holds no peg
                self.body_class[model.body(n).id] = N_LINKS
        self.n = 7
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # -- state access -------------------------------------------------------
    def q(self, data: mujoco.MjData) -> np.ndarray:
        return data.qpos[self.qpos_idx].copy()

    def qd(self, data: mujoco.MjData) -> np.ndarray:
        return data.qvel[self.dof_idx].copy()

    def set_q(self, data: mujoco.MjData, q: np.ndarray, qd: np.ndarray | None = None) -> None:
        data.qpos[self.qpos_idx] = q
        data.qvel[self.dof_idx] = 0.0 if qd is None else qd

    # -- kinematics (call after mj_kinematics / mj_forward) ------------------
    def site_pose(self, data: mujoco.MjData, site: int) -> tuple[np.ndarray, np.ndarray]:
        return data.site_xpos[site].copy(), mat2quat(data.site_xmat[site])

    def site_jac(self, data: mujoco.MjData, site: int) -> np.ndarray:
        """6x7 [linear; angular] Jacobian of a site w.r.t. the arm joints."""
        mujoco.mj_jacSite(self.model, data, self._jacp, self._jacr, site)
        return np.vstack([self._jacp[:, self.dof_idx], self._jacr[:, self.dof_idx]])

    def point_jac(self, data: mujoco.MjData, point: np.ndarray, body: int) -> np.ndarray:
        """3x7 translational Jacobian of a world point rigidly attached to ``body``."""
        mujoco.mj_jac(self.model, data, self._jacp, None, point, body)
        return self._jacp[:, self.dof_idx].copy()

    def ik(
        self,
        data: mujoco.MjData,
        pos: np.ndarray,
        axis: np.ndarray = PEG_AXIS_DOWN,
        q0: np.ndarray | None = None,
        site: int | None = None,
        restarts: int = 8,
        rng: np.random.Generator | None = None,
        tol: float = 1e-4,
        yaw: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Bounded least-squares IK for a tip position with the peg along ``axis``.

        Only the peg axis is constrained -- the peg is a cylinder, so yaw is
        free -- which matters on OpenArm: its wrist pitch is limited to +-45 deg.
        Starts from ``q0`` (default: current q), then random restarts; among
        solutions it returns the one farthest from the joint limits.
        Returns (q, position error in metres).
        """
        from scipy.optimize import least_squares

        site = self.tip_site if site is None else site
        scratch = mujoco.MjData(self.model)
        axis = np.asarray(axis, dtype=float)
        pos = np.asarray(pos, dtype=float)

        def resid(q):
            scratch.qpos[self.qpos_idx] = q
            mujoco.mj_kinematics(self.model, scratch)
            R = scratch.site_xmat[site].reshape(3, 3)
            # hand +z points opposite to the peg (the peg hangs along -z)
            r = np.concatenate([10.0 * (pos - scratch.site_xpos[site]), np.cross(-R[:, 2], axis)])
            if yaw is not None:  # turn about the vertical tool axis (heading of the hand's x axis)
                r = np.r_[r, wrap(tool_yaw(R) - yaw)]
            return r

        rng = rng or np.random.default_rng(0)
        lo, hi = self.q_lo + 1e-3, self.q_hi - 1e-3
        starts = [np.clip(self.q(data) if q0 is None else np.asarray(q0, float), lo, hi)]
        starts += [rng.uniform(lo + 0.05, hi - 0.05) for _ in range(restarts)]
        best, best_key = starts[0], (np.inf, 0.0)
        q_ref = starts[0]
        for k, s0 in enumerate(starts):
            if k == 0:
                # warm start: the tip pose leaves two free directions (redundancy + peg yaw); a weak pull
                # toward q0 keeps them near q0 instead of drifting to the wrist limits (costs << 0.1 mm)
                r = least_squares(lambda q: np.concatenate([resid(q), 0.01 * (q - q_ref)]), s0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
                r.fun = r.fun[: 6 if yaw is None else 7]
            else:
                r = least_squares(resid, s0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
            err = float(np.linalg.norm(r.fun[:3]) / 10.0 + np.linalg.norm(r.fun[3:]))
            margin = float(np.min(np.minimum(r.x - self.q_lo, self.q_hi - r.x)))
            ok = err < tol
            key = (0.0 if ok else err, -margin)
            if key < best_key:
                best, best_key = r.x, key
            if k == 0 and ok and margin > 0.1:
                # the warm start is a comfortable solution in the same arm configuration (branch) as q0;
                # random restarts can land in a flipped branch that hits joint 1's limit when the hand moves
                break
        scratch.qpos[self.qpos_idx] = best
        mujoco.mj_kinematics(self.model, scratch)
        return best, float(np.linalg.norm(pos - scratch.site_xpos[site]))
