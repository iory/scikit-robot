import unittest

import numpy as np
from numpy import testing

import skrobot
from skrobot.model.primitives import Box
from skrobot.planner import tinyfk_sqp_plan_trajectory
from skrobot.planner import TinyfkSweptSphereSdfCollisionChecker
from skrobot.planner.utils import get_robot_config
from skrobot.planner.utils import set_robot_config


class Test_sqp_based_planner(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        robot_model = skrobot.models.PR2()
        robot_model.init_pose()

        # setting up box sdf
        box_center = np.array([0.9, -0.2, 0.9])
        box = Box(extents=[0.7, 0.5, 0.6], with_sdf=True)
        box.translate(box_center)

        # defining control joints
        link_list = [
            robot_model.r_shoulder_pan_link, robot_model.r_shoulder_lift_link,
            robot_model.r_upper_arm_roll_link, robot_model.r_elbow_flex_link,
            robot_model.r_forearm_roll_link, robot_model.r_wrist_flex_link,
            robot_model.r_wrist_roll_link]
        joint_list = [link.joint for link in link_list]

        # defining collision links
        coll_link_list = [
            robot_model.r_upper_arm_link, robot_model.r_forearm_link,
            robot_model.r_gripper_palm_link,
            robot_model.r_gripper_r_finger_link,
            robot_model.r_gripper_l_finger_link]

        # collision checker setup
        sscc = TinyfkSweptSphereSdfCollisionChecker(
            lambda X: box.sdf(X), robot_model)
        for link in coll_link_list:
            sscc.add_collision_link(link)

        # by solving inverse kinematics, setting up av_goal
        coef = 3.1415 / 180.0
        joint_angles = [coef * e for e in [-60, 74, -70, -120, -20, -30, 180]]
        set_robot_config(robot_model, joint_list, joint_angles)

        robot_model.inverse_kinematics(
            target_coords=skrobot.coordinates.Coordinates(
                [0.8, -0.6, 0.8], [0, 0, 0]),
            link_list=link_list,
            move_target=robot_model.rarm_end_coords, rotation_axis=True)
        av_goal = get_robot_config(robot_model, joint_list, with_base=False)

        cls.joint_list = joint_list
        cls.av_start = np.array([0.564, 0.35, -0.74, -0.7, -0.7, -0.17, -0.63])
        cls.av_goal = av_goal
        cls.sscc = sscc

    def test_sqp_plan_trajectory(self):
        _av_start = self.av_start
        _av_goal = self.av_goal
        joint_list = self.joint_list
        sscc = self.sscc
        n_wp = 10

        for with_base in [False, True]:
            if with_base:
                av_start = np.hstack((_av_start, [0, 0, 0]))
                av_goal = np.hstack((_av_goal, [0, 0, 0]))
            else:
                av_start, av_goal = _av_start, _av_goal

            av_seq = tinyfk_sqp_plan_trajectory(
                sscc, av_start, av_goal, joint_list, n_wp,
                safety_margin=1e-2, with_base=with_base)
            # check equality (terminal) constraint
            testing.assert_almost_equal(av_seq[0], av_start)
            testing.assert_almost_equal(av_seq[-1], av_goal)

            # check inequality constraint,
            # meaning that in any waypoint, rarm does not collide
            batch_dists, _ = sscc.compute_batch_sd_vals(
                joint_list, av_seq, with_base=with_base, with_jacobian=False)
            self.assertTrue(np.all(batch_dists > -1e-4))
