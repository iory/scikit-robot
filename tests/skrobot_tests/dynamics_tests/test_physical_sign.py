"""Pin the dynamics functions to physics on a pendulum worked out by hand.

A 1 kg point mass sits 0.5 m out along +x from a joint turning about +y, with
gravity -10 m/s^2 along z. Gravity turns +x towards -z, which is a positive
rotation about +y, so holding the arm horizontal takes tau = -m g l = -5 N m
(= dU/dq), and accelerating it at +1 rad/s^2 takes +m l^2 = +0.25 N m.
"""

import unittest

import numpy as np

from skrobot.pycompat import HAS_JAX


MASS = 1.0
LENGTH = 0.5
GRAVITY = np.array([0.0, 0.0, -10.0])
HOLD = -MASS * 10.0 * LENGTH
ACCELERATE = MASS * LENGTH ** 2

PENDULUM = """<robot name="pendulum">
  <link name="base"/>
  <link name="arm">
    <inertial>
      <origin xyz="0.5 0 0"/><mass value="1.0"/>
      <inertia ixx="0" iyy="0" izz="0" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>
  <joint name="shoulder" type="revolute">
    <parent link="base"/><child link="arm"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>
  {tip}
</robot>"""

# A second joint at the end of the arm moving an empty link.
EMPTY_TIP = """<link name="tip"/>
  <joint name="wrist" type="revolute">
    <parent link="arm"/><child link="tip"/><origin xyz="1.0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-3" upper="3" effort="1" velocity="1"/>
  </joint>"""


def requires_jax(test_func):
    return unittest.skipUnless(HAS_JAX, 'JAX not available')(test_func)


def _pendulum(tip=''):
    from skrobot.models.urdf import RobotModelFromURDF
    robot = RobotModelFromURDF(urdf=PENDULUM.format(tip=tip))
    robot.angle_vector(np.zeros(len(robot.joint_list)))
    return robot


class TestHoldingTorqueSign(unittest.TestCase):

    def test_robot_model_methods(self):
        robot = _pendulum()
        np.testing.assert_allclose(robot.torque_vector(gravity=GRAVITY), [HOLD])
        np.testing.assert_allclose(robot.inverse_dynamics(gravity=GRAVITY), [HOLD])

    def test_inverse_dynamics_fn_numpy(self):
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        fn = build_inverse_dynamics_fn(_pendulum(), backend=get_backend('numpy'))
        np.testing.assert_allclose(fn(np.zeros(1), gravity=GRAVITY), [HOLD])

    def test_gravity_module(self):
        from skrobot.backend import get_backend
        from skrobot.dynamics.differentiable import extract_inverse_dynamics_parameters
        from skrobot.dynamics.gravity import build_gravity_torque_function
        from skrobot.dynamics.gravity import compute_gravity_torque
        robot = _pendulum()
        np.testing.assert_allclose(
            compute_gravity_torque(robot, np.zeros(1), gravity=GRAVITY), [HOLD])
        params = extract_inverse_dynamics_parameters(robot, [robot.link_list[1]])
        fn = build_gravity_torque_function(params, backend=get_backend('numpy'))
        np.testing.assert_allclose(fn(np.zeros(1), GRAVITY), [HOLD])

    @requires_jax
    def test_jax_gravity_builders(self):
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_gravity_torque_fn_vectorized
        from skrobot.dynamics import build_inverse_dynamics_fn
        from skrobot.dynamics import build_optimized_gravity_torque_fn
        from skrobot.dynamics.differentiable import build_rnea_gravity_fn
        from skrobot.dynamics.differentiable import build_rnea_serial_gravity_fn
        robot = _pendulum()
        builders = {
            'inverse_dynamics_jax': lambda r: build_inverse_dynamics_fn(
                r, backend=get_backend('jax')),
            'optimized': build_optimized_gravity_torque_fn,
            'vectorized': build_gravity_torque_fn_vectorized,
            'rnea': build_rnea_gravity_fn,
            'rnea_serial': build_rnea_serial_gravity_fn,
        }
        for name, build in builders.items():
            with self.subTest(name):
                tau = np.asarray(build(robot)(np.zeros(1), gravity=GRAVITY))
                np.testing.assert_allclose(tau, [HOLD], rtol=1e-5)

    def test_gravity_is_the_gradient_of_potential_energy(self):
        robot = _pendulum()
        q = 0.7
        h = 1e-6

        def potential(angle):
            robot.angle_vector(np.array([angle]))
            arm = robot.link_list[1]
            com = arm.worldpos() + arm.worldrot() @ np.asarray(arm.centroid)
            return -MASS * GRAVITY @ com

        expected = (potential(q + h) - potential(q - h)) / (2 * h)
        robot.angle_vector(np.array([q]))
        np.testing.assert_allclose(robot.torque_vector(gravity=GRAVITY), [expected],
                                   rtol=1e-6)


class TestInertialTorqueSign(unittest.TestCase):

    def test_accelerating_takes_positive_torque(self):
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        fn = build_inverse_dynamics_fn(_pendulum(), backend=get_backend('numpy'))
        tau = fn(np.zeros(1), qd=np.zeros(1), qdd=np.ones(1), gravity=np.zeros(3))
        np.testing.assert_allclose(tau, [ACCELERATE])


class TestMasslessLinks(unittest.TestCase):

    def test_a_moving_link_of_zero_mass_weighs_nothing(self):
        robot = _pendulum(EMPTY_TIP)
        robot.link_list[-1].mass = 0.0
        np.testing.assert_allclose(robot.torque_vector(gravity=GRAVITY), [HOLD, 0.0],
                                   atol=1e-12)

    def test_zeroed_mass_weighs_nothing(self):
        robot = _pendulum()
        robot.link_list[1].mass = 0.0
        np.testing.assert_allclose(robot.torque_vector(gravity=GRAVITY), [0.0],
                                   atol=1e-12)


if __name__ == '__main__':
    unittest.main()
