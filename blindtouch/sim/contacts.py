"""Ground truth: the contact wrench MuJoCo actually applied to the arm.

Used for evaluation only -- never as an input to anything the robot decides.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from blindtouch.sim.robot import N_LINKS, Arm


@dataclass
class ContactTruth:
    force: np.ndarray        # (3,) total world-frame force on the arm from the environment
    link: int                # dominant contact link class (0 = no contact)
    point: np.ndarray        # (3,) world point of the dominant contact (force-weighted)
    link_force: np.ndarray   # (N_LINKS,) |force| per link
    tau_ext: np.ndarray      # (7,) joint torques produced by the contacts, sum J^T f
    n_contacts: int
    n_links_touched: int


_F6 = np.zeros(6)


def arm_contacts(model: mujoco.MjModel, data: mujoco.MjData, arm: Arm, min_force: float = 1e-6) -> ContactTruth:
    """Sum every contact on the arm: total force, dominant link and point, joint torques."""
    force = np.zeros(3)
    link_f = np.zeros((N_LINKS, 3))
    link_pt = np.zeros((N_LINKS, 3))
    link_w = np.zeros(N_LINKS)
    tau = np.zeros(7)
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.efc_address < 0:
            continue
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        k1, k2 = arm.body_class[b1], arm.body_class[b2]
        if (k1 > 0) == (k2 > 0):
            continue  # not arm-vs-world
        mujoco.mj_contactForce(model, data, i, _F6)
        frame = c.frame.reshape(3, 3)
        # mj_contactForce gives the force geom1 exerts on geom2, in the contact
        # frame whose first row is the normal from geom1 to geom2.
        f_on_2 = frame.T @ _F6[:3]
        if k2 > 0:
            f, k = f_on_2, k2
        else:
            f, k = -f_on_2, k1
        mag = float(np.linalg.norm(f))
        if mag < min_force:
            continue
        n += 1
        force += f
        link_f[k - 1] += f
        link_pt[k - 1] += mag * c.pos
        link_w[k - 1] += mag
        # torque through the arm joints: the fingers and the held peg pass their load to the hand,
        # so use the link's own body (a free peg's Jacobian has no arm columns)
        tau += arm.point_jac(data, c.pos, arm.link_body[k - 1]).T @ f
    link_norm = np.linalg.norm(link_f, axis=1)
    if n == 0:
        return ContactTruth(force, 0, np.zeros(3), link_norm, tau, 0, 0)
    k = int(np.argmax(link_w))
    return ContactTruth(
        force=force,
        link=k + 1,
        point=link_pt[k] / link_w[k],
        link_force=link_norm,
        tau_ext=tau,
        n_contacts=n,
        n_links_touched=int(np.sum(link_w > 0)),
    )
