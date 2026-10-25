import copy
import pickle

import numpy as np
from numpy import testing
import pytest

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import rotation_matrix
from skrobot.model import LinearJoint
from skrobot.model import Link
from skrobot.model import RotationalJoint
from skrobot.model.robot_model import RobotModel


def _build_arm():
    """A chain: base -> 6 rotational joints (axes z, y, x, ...) with offsets
    between them -> a linear joint along x -> tip."""
    base = Link(name='base')
    links, joints = [], []
    parent = base
    axes = ['z', 'y', 'x', 'z', 'y', 'x']
    for i, axis in enumerate(axes):
        link = Link(name='link{}'.format(i))
        link.translate((0.05, 0.02 * (i % 2), 0.15))
        link.rotate(0.3 * (i + 1), 'x')
        parent.assoc(link, relative_coords='local')
        joint = RotationalJoint(name='joint{}'.format(i), parent_link=parent, child_link=link, axis=axis,
                                min_angle=-np.pi, max_angle=np.pi)
        parent.add_child_link(link)
        link.add_parent_link(parent)
        link.add_joint(joint)
        links.append(link)
        joints.append(joint)
        parent = link
    tip = Link(name='tip')
    tip.translate((0.1, 0.0, 0.05))
    parent.assoc(tip, relative_coords='local')
    slide = LinearJoint(name='slide', parent_link=parent, child_link=tip, axis='x',
                        min_angle=-0.5, max_angle=0.5)
    parent.add_child_link(tip)
    tip.add_parent_link(parent)
    tip.add_joint(slide)
    links.append(tip)
    joints.append(slide)
    return RobotModel(link_list=links, joint_list=joints, root_link=base)


def _poses(robot):
    return np.array([link.worldcoords().T() for link in robot.link_list])


def _random_angles(robot, n, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.array([j.min_angle for j in robot.joint_list])
    hi = np.array([j.max_angle for j in robot.joint_list])
    return rng.uniform(lo, hi, (n, len(lo)))


def test_no_drift_with_float32_angles():
    # Turning a link by each change of angle accumulated rounding: with
    # float32 angle vectors the links drifted ~1e-6 m in a few thousand
    # calls.  Set from the zero pose, they are where a fresh robot puts them.
    robot, fresh = _build_arm(), _build_arm()
    for av in _random_angles(robot, 3000).astype(np.float32):
        robot.angle_vector(av)
    for av in _random_angles(robot, 20, seed=1).astype(np.float32):
        robot.angle_vector(av)
        fresh.angle_vector(av)
        testing.assert_allclose(_poses(robot), _poses(fresh), rtol=0, atol=1e-12)
        for link in robot.link_list:
            R = link.worldrot()
            testing.assert_allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-12)


def test_child_link_moved_directly_is_kept():
    # A child link moved by something else than its joint (here, before the
    # joint first moves, and between changes) keeps the move, as it did when
    # the joint turned it by each change of angle.
    robot = _build_arm()
    j, link = robot.joint_list[2], robot.link_list[2]      # axis x
    link.rotate(0.2, 'y')
    link.translate((0.01, -0.02, 0.005))
    R_e, p_e = link.rotation.copy(), link.translation.copy()
    j.joint_angle(0.4)
    testing.assert_allclose(link.rotation, R_e @ rotation_matrix(0.4, [1, 0, 0]), atol=1e-14)
    testing.assert_allclose(link.translation, p_e, atol=1e-14)

    j.joint_angle(-0.3)
    link.newcoords(Coordinates(pos=[0.05, 0.0, 0.1]).rotate(0.3, 'z'))
    R_e, p_e = link.rotation.copy(), link.translation.copy()
    j.joint_angle(0.9)
    testing.assert_allclose(link.rotation, R_e @ rotation_matrix(0.9 + 0.3, [1, 0, 0]), atol=1e-14)
    testing.assert_allclose(link.translation, p_e, atol=1e-14)

    # and from there on, without moves, as a fresh robot moved the same way
    for a in np.linspace(-1, 1, 7):
        j.joint_angle(a)
    j.joint_angle(0.9)
    testing.assert_allclose(link.rotation, R_e @ rotation_matrix(0.9 + 0.3, [1, 0, 0]), atol=1e-14)

    # a linear joint: the move kept, the slide along its axis in the moved frame
    s, tip = robot.joint_list[-1], robot.link_list[-1]
    s.joint_angle(0.1)
    tip.rotate(0.25, 'z')
    tip.translate((0.0, 0.01, 0.0))
    R_e, p_e = tip.rotation.copy(), tip.translation.copy()
    s.joint_angle(0.3)
    testing.assert_allclose(tip.rotation, R_e, atol=1e-14)
    testing.assert_allclose(tip.translation, p_e + R_e @ np.array([0.2, 0, 0]), atol=1e-14)


def test_written_pose_is_read_only():
    robot = _build_arm()
    robot.angle_vector(_random_angles(robot, 1)[0])
    link = robot.link_list[3]
    with pytest.raises(ValueError):
        link.translation[0] += 1.0
    with pytest.raises(ValueError):
        link.rotation[0, 0] = 1.0
    # through the property setter: kept by the next change of angle
    v = link.translation + np.array([0.0, 0.0, 0.03])
    link.translation = v
    robot.joint_list[3].joint_angle(robot.joint_list[3].joint_angle() + 0.2)
    testing.assert_allclose(link.translation, v, atol=1e-14)


@pytest.mark.parametrize('clone', [copy.deepcopy, lambda r: pickle.loads(pickle.dumps(r))])
def test_copied_robot(clone):
    robot = _build_arm()
    robot.angle_vector(_random_angles(robot, 1)[0])
    other = clone(robot)
    fresh = _build_arm()
    for av in _random_angles(robot, 5, seed=2):
        other.angle_vector(av)
        fresh.angle_vector(av)
        testing.assert_allclose(_poses(other), _poses(fresh), rtol=0, atol=1e-12)
