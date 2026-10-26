"""torque_vector on a branched robot whose joint_list is not in depth order.

The inverse dynamics sort the links so that parents come first. Three
branches of unequal length (3, 1 and 2 joints), listed branch by branch,
sort into a permutation that is not its own inverse, so applying the
permutation the wrong way round shows up: torques land on joints of other
branches. Two branches of equal length sort into a plain swap, which hid
the bug.

Both references come from skrobot's own forward kinematics: J^T F by
central differences of the tip position, and the gravity torque as the
gradient of the potential energy sum(m g z_com).
"""

import unittest

import numpy as np

from skrobot.pycompat import HAS_JAX


GRAVITY = np.array([0.0, 0.0, -9.80665])
# skrobot's FK keeps parts of a joint's motion in float32, which rounds a
# step of 1e-5 rad by ~0.1 %; at 1e-3 the rounding and the O(EPS^2)
# truncation of the central difference are both ~1e-5 relative.
EPS = 1e-3
RTOL = 1e-4


def _link(name, mass, com):
    return """<link name="{n}">
    <inertial>
      <origin xyz="{c}"/><mass value="{m}"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>""".format(n=name, m=mass, c=com)


def _joint(name, jtype, parent, child, xyz, axis):
    return """<joint name="{n}" type="{t}">
    <parent link="{p}"/><child link="{c}"/>
    <origin xyz="{o}" rpy="0.1 0.2 0.3"/><axis xyz="{a}"/>
    <limit lower="-2" upper="2" effort="1" velocity="1"/>
  </joint>""".format(n=name, t=jtype, p=parent, c=child, o=xyz, a=axis)


def _branched_urdf(a2_type='prismatic'):
    parts = ['<robot name="branched">', '<link name="base"/>']
    # (joint, type, parent, child, origin, axis, mass, com)
    spec = [
        ('a0', 'revolute', 'base', 'la0', '0 0.2 0.1', '0 0 1', 1.0, '0.1 0 0'),
        ('a1', 'revolute', 'la0', 'la1', '0.2 0 0', '0 1 0', 0.8, '0.1 0.02 0'),
        ('a2', a2_type, 'la1', 'la2', '0.2 0 0', '1 0 0', 0.5, '0.05 0 0.01'),
        ('b0', 'revolute', 'base', 'lb0', '0 0 0.3', '1 0 0', 0.7, '0 0 0.1'),
        ('c0', 'revolute', 'base', 'lc0', '0 -0.2 0.1', '0 1 0', 0.9, '0.1 0 0'),
        ('c1', 'revolute', 'lc0', 'lc1', '0.2 0 0', '1 0 1', 0.6, '0.1 0 -0.02'),
    ]
    for name, jtype, parent, child, xyz, axis, mass, com in spec:
        parts.append(_link(child, mass, com))
        parts.append(_joint(name, jtype, parent, child, xyz, axis))
    parts.append('</robot>')
    return '\n'.join(parts)


def _robot(a2_type='prismatic'):
    from skrobot.models.urdf import RobotModelFromURDF
    robot = RobotModelFromURDF(urdf=_branched_urdf(a2_type))
    robot.angle_vector(np.array([0.3, -0.5, 0.1, 0.4, -0.7, 0.6]))
    return robot


def _central_difference(robot, value_fn):
    """d value_fn / dq_i, moving one joint at a time (hooks off)."""
    grad = np.zeros(len(robot.joint_list))
    for i, joint in enumerate(robot.joint_list):
        angle = joint.joint_angle()
        joint.joint_angle(angle + EPS, enable_hook=False)
        plus = value_fn()
        joint.joint_angle(angle - EPS, enable_hook=False)
        minus = value_fn()
        joint.joint_angle(angle, enable_hook=False)
        grad[i] = (plus - minus) / (2 * EPS)
    return grad


def _potential_energy(robot):
    g = -GRAVITY[2]
    return sum(
        link.mass * g * link.worldcoords().transform_vector(link.centroid)[2]
        for link in robot.link_list
        if getattr(link, 'mass', None))


class TestBranchedTorqueVector(unittest.TestCase):

    def test_link_order_is_not_an_involution(self):
        """The fixture must exercise a non-trivial permutation."""
        from skrobot.dynamics.differentiable import extract_inverse_dynamics_parameters
        robot = _robot()
        params = extract_inverse_dynamics_parameters(
            robot, [j.child_link for j in robot.joint_list])
        perm = params['sorted_to_original']
        self.assertFalse(np.array_equal(perm[perm], np.arange(len(perm))))

    def test_external_force_is_jacobian_transpose(self):
        robot = _robot()
        force = np.array([0.3, -1.0, 2.0])
        for tip in (robot.la2, robot.lc1):
            tau = robot.torque_vector(
                force_list=[force], moment_list=[np.zeros(3)],
                target_coords=[tip], gravity=np.zeros(3))
            jtf = _central_difference(robot, lambda: tip.worldpos() @ force)
            # The joints supply the opposite of the external force's torque.
            np.testing.assert_allclose(tau, -jtf, rtol=RTOL, atol=1e-6)
            # Only the branch carrying the tip feels the force.
            self.assertTrue(np.any(np.abs(jtf) > 1e-2))

    def test_gravity_is_energy_gradient(self):
        robot = _robot()
        tau = robot.torque_vector(gravity=GRAVITY)
        grad = _central_difference(robot, lambda: _potential_energy(robot))
        np.testing.assert_allclose(tau, grad, rtol=RTOL, atol=1e-6)

    @unittest.skipUnless(HAS_JAX, 'JAX not available')
    def test_jax_gravity_functions(self):
        from skrobot.backend import get_backend
        from skrobot.dynamics import build_inverse_dynamics_fn
        from skrobot.dynamics import build_optimized_gravity_torque_fn
        from skrobot.dynamics.differentiable import build_rnea_gravity_fn

        def inverse_dynamics_jax(robot, link_list):
            return build_inverse_dynamics_fn(
                robot, link_list, backend=get_backend('jax'))

        # The gravity-only builders take revolute joints only.
        for a2_type, builders in (
                ('prismatic', (inverse_dynamics_jax,)),
                ('revolute', (inverse_dynamics_jax,
                              build_optimized_gravity_torque_fn,
                              build_rnea_gravity_fn))):
            robot = _robot(a2_type)
            q = robot.angle_vector().copy()
            grad = _central_difference(
                robot, lambda: _potential_energy(robot))
            for build in builders:
                fn = build(robot, [j.child_link for j in robot.joint_list])
                # JAX runs in float32 unless x64 is enabled.
                np.testing.assert_allclose(
                    np.asarray(fn(q, gravity=GRAVITY)), grad,
                    rtol=1e-3, atol=1e-4,
                    err_msg='{} ({})'.format(build.__name__, a2_type))


if __name__ == '__main__':
    unittest.main()
