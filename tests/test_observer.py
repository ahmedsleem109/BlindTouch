"""The momentum observer converges to known external torques."""

import mujoco
import numpy as np
import pytest

from blindtouch.sensing.observer import MomentumObserver
from blindtouch.sim.env import Env

PENDULUM = """
<mujoco>
  <option timestep="0.001" integrator="implicitfast"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.02" mass="1.0"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge"/></actuator>
</mujoco>
"""


def test_pendulum_converges_to_applied_torque():
    m = mujoco.MjModel.from_xml_string(PENDULUM)
    d = mujoco.MjData(m)
    obs = MomentumObserver(gain=100.0, dt=m.opt.timestep)
    tau_ext = 0.8
    M = np.zeros((1, 1))
    for k in range(3000):
        tau = np.array([1.5 * np.sin(0.003 * k)])  # arbitrary motor torque
        d.ctrl[0] = tau[0]
        d.qfrc_applied[0] = tau_ext if k > 500 else 0.0
        mujoco.mj_forward(m, d)
        mujoco.mj_fullM(m, d, M)
        r = obs.update(M, d.qfrc_bias.copy(), m.dof_damping.copy(), d.qvel.copy(), tau)
        mujoco.mj_step(m, d)
    assert r.shape == (1,) and abs(r[0] - tau_ext) < 0.02


def test_arm_observer_recovers_known_force():
    env = Env("peg", seed=0)
    env.reset()
    obs = MomentumObserver(gain=150.0)
    F = np.array([4.0, -6.0, 3.0])
    body = env.model.body("link5").id
    point_local = np.array([0.0, 0.03, -0.06])
    for k in range(1500):
        s = env.last
        env.dyn.update(s.q, s.qd)
        tau = env.ctrl.torque(s.q, s.qd)
        R = env.data.xmat[body].reshape(3, 3)
        p = env.data.xpos[body] + R @ point_local
        env.data.qfrc_applied[:] = 0
        if k > 300:
            mujoco.mj_applyFT(env.model, env.data, F, np.zeros(3), p, body, env.data.qfrc_applied)
        true_tau = env.data.qfrc_applied[env.arm.dof_idx].copy()
        s = env.apply_torque(tau)
        env.dyn.update(s.q, s.qd)
        r = obs.update(env.dyn.M, env.dyn.bias, env.dyn.damping, s.qd, s.tau)
    np.testing.assert_allclose(r, true_tau, atol=0.05)


@pytest.mark.parametrize("gain", [50.0, 150.0])
def test_observer_is_zero_without_contact(gain):
    env = Env("peg", seed=1)
    env.reset()
    obs = MomentumObserver(gain=gain)
    rs = []
    for _ in range(500):
        s = env.control_step()
        env.dyn.update(s.q, s.qd)
        rs.append(obs.update(env.dyn.M, env.dyn.bias, env.dyn.damping, s.qd, s.tau))
    assert np.abs(np.array(rs[100:])).max() < 0.05
