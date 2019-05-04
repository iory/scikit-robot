import unittest

import numpy as np
from numpy import testing

from skrobot.coordinates import make_cascoords
from skrobot.coordinates import make_coords
from skrobot.handeye_calibration.handeye_calibration_using_dual_quaternion \
    import compute_handeye_calibration_using_dual_quaternion
from skrobot.robot_models import Kuka


class TestHandyeyeCalibrationUsingDualQuaternion(unittest.TestCase):

    def test_compute_handeye_calibration_using_dual_quaternion(self):
        robot = Kuka()
        camera = make_coords(pos=[0.1, 0.1, 0.1])

        marker_attached_link = robot.lbr_iiwa_with_wsg50__lbr_iiwa_link_7
        robot.marker_coords = make_cascoords(
            pos=[0, 0, 0.1],
            parent=marker_attached_link,
            name='marker').rotate(np.pi / 2.0, 'y')

        max_angles = robot.rarm.joint_max_angles
        min_angles = robot.rarm.joint_min_angles

        # create random pose
        base_to_ee_dq_vec = []
        camera_to_marker_dq_vec = []
        for i in range(100):
            av = np.random.uniform(min_angles, max_angles)
            robot.rarm.angle_vector(av)

            base_to_ee_dq_vec.append(
                marker_attached_link.copy_worldcoords().dual_quaternion)
            camera_to_marker_dq_vec.append(
                camera.copy_worldcoords().transformation(
                    robot.marker_coords.copy_worldcoords()).
                copy_worldcoords().dual_quaternion)

        result = compute_handeye_calibration_using_dual_quaternion(
            base_to_ee_dq_vec, camera_to_marker_dq_vec)
        testing.assert_almost_equal(
            result['pose'],
            [0, 0, 0.1, 0, 7.0710678e-01, 0, 7.0710678e-01])

        # calculating base to camera transform
        base_to_ee_dq_vec = []
        camera_to_marker_dq_vec = []
        for i in range(100):
            av = np.random.uniform(min_angles, max_angles)
            robot.rarm.angle_vector(av)

            base_to_ee_dq_vec.append(
                marker_attached_link.copy_worldcoords().
                dual_quaternion.inverse)
            camera_to_marker_dq_vec.append(
                camera.copy_worldcoords().transformation(
                    robot.marker_coords.copy_worldcoords()).
                copy_worldcoords().dual_quaternion.inverse)

        result = compute_handeye_calibration_using_dual_quaternion(
            base_to_ee_dq_vec, camera_to_marker_dq_vec)
        testing.assert_almost_equal(
            result['pose'],
            [0.1, 0.1, 0.1, 0, 0, 0, 1])
