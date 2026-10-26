"""Simulation, ground truth and realism switches."""

import numpy as np
import pytest

from blindtouch.scenes.build import GEOM
from blindtouch.sim.env import Env
from blindtouch.sim.realism import EFFECTS, RealismConfig


def test_contact_truth_matches_constraint_torque():
    env = Env("peg", seed=0)
    hx, hy = GEOM.hole_nominal
    env.reset()
    env.ctrl.target_pos = np.array([hx + 0.05, hy, GEOM.plate_top - 0.01])  # on the flat plate top, clear of the chamfer
    env.run(750)
    tr = env.last.truth
    assert tr.link == 7
    # pressing 1 cm past the surface with 600 N/m: ~6 N upward on the arm
    assert 4.0 < tr.force[2] < 8.0
    # the grip passes the contact *and* the held peg's own weight into the arm joints
    m, d = env.model, env.data
    peg = m.body("peg").id
    weight = m.body_mass[peg] * m.opt.gravity
    tau_weight = env.arm.point_jac(d, d.xipos[peg], env.arm.link_body[-1]).T @ weight
    np.testing.assert_allclose(tr.tau_ext + tau_weight, d.qfrc_constraint[env.arm.dof_idx], atol=0.02)


def test_nominal_insertion_reaches_bottom():
    env = Env("peg", seed=0)
    hx, hy = GEOM.hole_nominal
    env.reset()
    env.ctrl.target_pos = np.array([hx, hy, GEOM.plate_top + 0.01])
    env.run(500)
    env.ctrl.target_pos = np.array([hx, hy, GEOM.plate_top - 0.04])
    env.run(1000)
    assert GEOM.plate_top - env.tip_pos()[2] > GEOM.plate_thick - 0.002


def test_seeded_episodes_are_reproducible():
    def run():
        env = Env("peg", realism=RealismConfig.all_on(), seed=0)
        env.reset(seed=5)
        env.ctrl.target_pos = env.ctrl.target_pos + np.array([0.02, 0, -0.03])
        return np.array([s.q for s in env.run(200)])

    np.testing.assert_array_equal(run(), run())


@pytest.mark.parametrize("effect", EFFECTS)
def test_each_effect_changes_measurements(effect):
    def run(cfg):
        env = Env("peg", realism=cfg, seed=0)
        env.reset(seed=3)
        env.ctrl.target_pos = env.ctrl.target_pos + np.array([0.03, 0.0, -0.02])
        return np.array([np.r_[s.q, s.tau] for s in env.run(150)])

    clean, on = run(RealismConfig()), run(RealismConfig.only(effect))
    assert not np.allclose(clean, on)


def test_realism_config_parsing():
    assert RealismConfig.only("clean").label() == "clean"
    assert RealismConfig.only("all").label() == "+".join(EFFECTS)
    c = RealismConfig.from_dict({"friction": True, "sensor_delay": [3, 5]})
    assert c.friction and c.sensor_delay == (3, 5)
