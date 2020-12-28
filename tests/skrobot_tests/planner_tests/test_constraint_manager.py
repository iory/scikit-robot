import os
import copy
import json
import unittest
import numpy as np
from numpy import testing
import tinyfk

import skrobot
from skrobot.planner import ConstraintManager
from skrobot.planner.utils import get_robot_config
from skrobot.planner.constraint_manager import ConfigurationConstraint
from skrobot.planner.constraint_manager import EqualityConstraint
from skrobot.planner.constraint_manager import PoseConstraint

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

    print(jac.T)
    print(jac_numerical.T)
    testing.assert_almost_equal(jac, jac_numerical, decimal=decimal)


class TestConstraintManager(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        n_wp = 5
        urdf_path = skrobot.data.pr2_urdfpath()
        fksolver = tinyfk.RobotModel(urdf_path)
        joint_names = ["r_shoulder_pan_joint", "r_shoulder_lift_joint", "r_upper_arm_roll_joint", "r_elbow_flex_joint", "r_forearm_roll_joint", "r_wrist_flex_joint", "r_wrist_roll_joint"]

        n_dof = len(joint_names) + 3
        cm = ConstraintManager(n_wp, joint_names, fksolver, True)

        cls.n_wp = n_wp
        cls.n_dof = n_dof
        cls.cm = cm

    def test_check_func(self):
        cm, n_wp, n_dof = self.cm, self.n_wp, self.n_dof

        def dummy_shitty_func(xi):
            n_wp, n_dof = xi.shape
            f = np.zeros(n_wp)
            jac = np.zeros((n_wp+1, n_wp * n_dof-1))
            return f, jac

        def dummy_good_func(xi):
            f = np.zeros(5)
            jac = np.zeros((5, n_wp * n_dof))
            return f, jac

        cons = EqualityConstraint(n_wp, n_dof, 2, "who")
        cons._check_func(dummy_good_func)

        with self.assertRaises(AssertionError):
            cons._check_func(dummy_shitty_func)

    def test_configuration_constraint(self):
        cm, n_wp, n_dof = self.cm, self.n_wp, self.n_dof
        idx_mid = 1
        av_desired = np.random.randn(n_dof)
        ceq_config = ConfigurationConstraint(n_wp, n_dof, idx_mid, av_desired)
        func = ceq_config.gen_func()

        dummy_av_seq = np.random.randn(n_wp, n_dof) 
        f, jac = func(dummy_av_seq)
        f_expected = dummy_av_seq[idx_mid] - av_desired
        testing.assert_equal(f, f_expected)

        jacobian_test_util(lambda xi: func(xi.reshape(n_wp, n_dof)), dummy_av_seq.flatten())

    def test_pose_constraint(self):
        cm, n_wp, n_dof = copy.copy(self.cm), self.n_wp, self.n_dof
        n_dof_all = n_wp * n_dof

        position_desired = np.array([0.8, -0.6, 0.7, 0, 0, 0])
        with_base = True
        cons = PoseConstraint(n_wp, n_dof, 2, "r_gripper_tool_frame", position_desired,
                cm.fksolver, cm.joint_ids, with_base)
        func = cons.gen_func()
        # TODO test generated function
        dummy_av_seq = np.random.randn(n_wp, n_dof) 
        jacobian_test_util(lambda xi: func(xi.reshape(n_wp, n_dof)), dummy_av_seq.flatten())


    def test_add_eq_configuration(self):
        # TODO must be deepcopied
        cm, n_wp, n_dof = copy.copy(self.cm), self.n_wp, self.n_dof
        n_dof_all = n_wp * n_dof
        idx_mid = 3

        av_start = np.random.randn(n_dof)
        av_mid = np.random.randn(n_dof)

        with self.assertRaises(AssertionError):
            cm.add_eq_configuration(n_wp, av_start)

        cm.add_eq_configuration(0, av_start)
        cm.add_eq_configuration(idx_mid, av_mid)

        fun_eq = cm.gen_combined_constraint_func()

        dummy_av_seq = np.random.randn(n_wp * n_dof) 
        av1 = dummy_av_seq[:n_dof]
        av2 = dummy_av_seq[n_dof * idx_mid : n_dof * (idx_mid+1)]
        f_expected = np.hstack((av1 - av_start, av2 - av_mid))
        f, _ = fun_eq(dummy_av_seq)
        testing.assert_equal(f, f_expected)
        jacobian_test_util(fun_eq, dummy_av_seq)
