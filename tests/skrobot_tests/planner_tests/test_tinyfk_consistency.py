import copy
import unittest

import numpy as np
from numpy import testing
import tinyfk

import skrobot
from skrobot.planner.utils import forward_kinematics_multi
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config

def jacobian_test_util(func, x0, decimal=5):
    f0, jac = func(x0)
    n_dim = len(x0)

    eps = 1e-7
    jac_numerical = np.zeros(jac.shape)
    for idx in range(n_dim):
        x1 = copy.copy(x0)
        x1[idx] += eps
        f1, _ = func(x1)
        jac_numerical[:, idx] = (f1 - f0) / eps
    testing.assert_almost_equal(jac, jac_numerical, decimal=decimal)

class TestTinyfkConsistency(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        robot_model = skrobot.models.PR2()
        fksolver = tinyfk.RobotModel(robot_model.urdf_path)

        link_idx_table = {}
        for link_idx in range(len(robot_model.link_list)):
            name = robot_model.link_list[link_idx].name
            link_idx_table[name] = link_idx

        link_names = ["r_shoulder_pan_link", "r_shoulder_lift_link",
                      "r_upper_arm_roll_link", "r_elbow_flex_link",
                      "r_forearm_roll_link", "r_wrist_flex_link",
                      "r_wrist_roll_link"]

        link_list = [robot_model.link_list[link_idx_table[lname]]
                     for lname in link_names]
        joint_list = [l.joint for l in link_list]
        link_ids = fksolver.get_link_ids([l.name for l in link_list])
        joint_ids = fksolver.get_joint_ids([j.name for j in joint_list])

        cls.robot_model = robot_model
        cls.fksolver = fksolver
        cls.link_ids = link_ids
        cls.joint_ids = joint_ids

        av = np.array([0.4, 0.6] + [-0.7] * 5)
        cls.av = np.array([0.4, 0.6] + [-0.7] * 5)
        cls.av_with_base = np.hstack((av, [0.1, 0.0, 0.3]))

    def test_forwardkinematics(self):
        fksolver = self.fksolver 
        link_ids = self.link_ids
        joint_ids = self.joint_ids

        av_init = self.av
        av_with_base_init = self.av_with_base
        link_ids = [link_ids[5]]
        n_feature = len(link_ids)

        def fk_jac_test(av, with_base, with_rot):
            pose_arr, jac_arr = fksolver.solve_forward_kinematics([av], link_ids, joint_ids, 
                    with_rot=with_rot, with_base=with_base, with_jacobian=True)

            pose_flatten = pose_arr.flatten()  # (pose_dim * n_feature)
            n_dof = len(joint_ids) + (3 if with_base else 0) 
            pose_dim = 6 if with_rot else 3
            jac_flatten = jac_arr.reshape(n_feature * pose_dim, n_dof)
            return pose_flatten, jac_flatten

        # checking jacobian
        jacobian_test_util(
            lambda av: fk_jac_test(av, False, False), av_init)
        jacobian_test_util(
            lambda av: fk_jac_test(av, True, False), av_with_base_init)
        jacobian_test_util(
            lambda av: fk_jac_test(av, False, True), av_init)
        jacobian_test_util(
            lambda av: fk_jac_test(av, True, True), av_with_base_init)
