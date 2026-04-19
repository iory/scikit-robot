import copy

import numpy as np
from numpy import testing

from skrobot.model import FloatingJoint
from skrobot.model import Link
from skrobot.model import PlanarJoint
from skrobot.model import RotationalJoint
from skrobot.model.robot_model import RobotModel


def _build_two_link_robot(base_joint_cls):
    """Build a tiny RobotModel: world_link -> base_joint -> root_link
    -> rotational_joint -> end_link."""
    world_link = Link(name='world')
    root_link = Link(name='root')
    end_link = Link(name='end')
    end_link.translate((0.3, 0, 0))

    world_link.assoc(root_link)
    root_link.assoc(end_link)

    base_joint = base_joint_cls(
        name='base_joint', parent_link=world_link, child_link=root_link)
    world_link.add_child_link(root_link)
    root_link.add_parent_link(world_link)
    root_link.add_joint(base_joint)

    arm_joint = RotationalJoint(
        name='arm_joint', parent_link=root_link, child_link=end_link,
        axis='z')
    root_link.add_child_link(end_link)
    end_link.add_parent_link(root_link)
    end_link.add_joint(arm_joint)

    robot = RobotModel(
        link_list=[root_link, end_link],
        joint_list=[base_joint, arm_joint],
        root_link=world_link)
    return robot, base_joint, arm_joint, root_link, end_link


def _jacobian_vs_finite_difference(base_joint_cls, base_dof, q0):
    robot, base_joint, arm_joint, root_link, end_link = \
        _build_two_link_robot(base_joint_cls)
    link_list = [root_link, end_link]

    def set_q(q):
        base_joint.joint_angle(q[:base_dof])
        arm_joint.joint_angle(q[base_dof])

    n_dof = base_dof + 1
    set_q(q0)
    pos0 = end_link.worldpos().copy()

    jac_num = np.zeros((3, n_dof))
    eps = 1e-7
    for idx in range(n_dof):
        q1 = copy.copy(q0)
        q1[idx] += eps
        set_q(q1)
        jac_num[:, idx] = (end_link.worldpos() - pos0) / eps

    set_q(q0)
    world_link = root_link.parent_link
    jac_analytic = robot.calc_jacobian_from_link_list(
        end_link, link_list,
        rotation_mask=False, transform_coords=world_link)
    testing.assert_almost_equal(jac_num, jac_analytic, decimal=5)


def test_planar_joint_dof_and_type():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = PlanarJoint(parent_link=parent, child_link=child)
    assert j.joint_dof == 3
    assert j.type == 'planar'


def test_planar_joint_forward_kinematics():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = PlanarJoint(parent_link=parent, child_link=child)

    j.joint_angle(np.array([1.0, 2.0, np.pi / 2]))
    testing.assert_almost_equal(child.worldpos(), [1.0, 2.0, 0.0])
    expected_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    testing.assert_almost_equal(child.worldrot(), expected_rot)


def test_planar_joint_angle_round_trip():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = PlanarJoint(parent_link=parent, child_link=child)
    q = np.array([0.5, -0.2, 0.3])
    j.joint_angle(q)
    testing.assert_almost_equal(j.joint_angle(), q)


def test_planar_joint_jacobian_matches_finite_difference():
    q0 = np.array([0.1, -0.2, 0.3, 0.4])
    _jacobian_vs_finite_difference(PlanarJoint, 3, q0)


def test_floating_joint_dof_and_type():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = FloatingJoint(parent_link=parent, child_link=child)
    assert j.joint_dof == 6
    assert j.type == 'floating'


def test_floating_joint_forward_kinematics_translation_only():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = FloatingJoint(parent_link=parent, child_link=child)
    j.joint_angle(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
    testing.assert_almost_equal(child.worldpos(), [1.0, 2.0, 3.0])
    testing.assert_almost_equal(child.worldrot(), np.eye(3))


def test_floating_joint_forward_kinematics_rotation_only():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = FloatingJoint(parent_link=parent, child_link=child)
    j.joint_angle(np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2]))
    expected_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    testing.assert_almost_equal(child.worldrot(), expected_rot)


def test_floating_joint_angle_round_trip():
    parent = Link()
    child = Link()
    parent.assoc(child)
    j = FloatingJoint(parent_link=parent, child_link=child)
    q = np.array([0.5, -0.2, 0.3, 0.1, -0.1, 0.2])
    j.joint_angle(q)
    testing.assert_almost_equal(j.joint_angle(), q)


def test_floating_joint_jacobian_matches_finite_difference():
    q0 = np.array([0.1, -0.2, 0.3, 0.05, -0.08, 0.1, 0.4])
    _jacobian_vs_finite_difference(FloatingJoint, 6, q0)
