import copy
import unittest

import numpy as np
from numpy import testing

import skrobot
from skrobot.planner.utils import forward_kinematics_multi
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config


def jacobian_test_util(func, x0, decimal=5):
    # test jacobian by comparing the resulting and numerical jacobian
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


class TestPlannerUtils(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        robot_model = skrobot.models.PR2()

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

        cls.robot_model = robot_model
        cls.link_list = link_list
        cls.joint_list = joint_list

        av = np.array([0.4, 0.6] + [-0.7] * 5)
        cls.av = np.array([0.4, 0.6] + [-0.7] * 5)
        cls.av_with_base = np.hstack((av, [0.1, 0.0, 0.3]))

    def test_set_and_get_robot_config(self):
        robot_model = self.robot_model
        joint_list = self.joint_list
        av = self.av
        av_with_base = self.av_with_base

        with self.assertRaises(AssertionError):
            set_robot_config(robot_model, joint_list, av, with_base=True)
        set_robot_config(robot_model, joint_list, av, with_base=False)

        with self.assertRaises(AssertionError):
            set_robot_config(robot_model, joint_list,
                             av_with_base, with_base=False)
        set_robot_config(robot_model, joint_list, av_with_base, with_base=True)

        testing.assert_almost_equal(
            av,
            get_robot_config(robot_model, joint_list, with_base=False)
        )
        testing.assert_almost_equal(
            av_with_base,
            get_robot_config(robot_model, joint_list, with_base=True)
        )

    def test_forward_kinematics_multi(self):
        robot_model = self.robot_model
        link_list = self.link_list
        joint_list = self.joint_list
        av_init = self.av
        av_with_base_init = self.av_with_base

        move_target_list = link_list
        n_feature = len(move_target_list)

        def fk_fun_simple(av, with_base, with_rot, with_jacobian):
            pose_arr, jac_arr = forward_kinematics_multi(
                robot_model, joint_list, av, move_target_list,
                with_rot, with_base, with_jacobian)
            return pose_arr, jac_arr

        # checking returning types and shapes:
        for with_base, av in [(False, av_init), (True, av_with_base_init)]:
            for with_rot, n_pose in [(False, 3), (True, 3 + 4)]:
                for with_jacobian in [False, True]:
                    p, jac = fk_fun_simple(
                        av, with_base, with_rot, with_jacobian)
                    self.assertEqual(p.shape, (n_feature, n_pose))
                    n_dof = len(av)
                    if with_jacobian:
                        self.assertEqual(jac.shape, (n_feature, n_pose, n_dof))
                    else:
                        self.assertEqual(jac, None)

        def fk_jac_test(av, with_base, with_rot):
            n_dof = len(av)
            pose_arr, jac_arr = fk_fun_simple(av, with_base, with_rot, True)
            pose_flatten = pose_arr.flatten()  # (pose_dim * n_feature)
            pose_dim = 7 if with_rot else 3
            jac_flatten = jac_arr.reshape(n_feature * pose_dim, n_dof)
            return pose_flatten, jac_flatten

        # checking jacobian
        jacobian_test_util(
            lambda av: fk_jac_test(av, False, False), av_init)
        jacobian_test_util(
            lambda av: fk_jac_test(av, False, True), av_init)
        jacobian_test_util(
            lambda av: fk_jac_test(av, True, False), av_with_base_init)
        jacobian_test_util(
            lambda av: fk_jac_test(av, True, True), av_with_base_init)
