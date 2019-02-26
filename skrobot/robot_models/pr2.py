from skrobot.robot_model import RobotModel


class PR2(RobotModel):
    """
    PR2 Robot Model
    """

    def __init__(self, *args, **kwargs):
        RobotModel.__init__(self, *args, **kwargs)

    def reset_manip_pose(self):
        self.torso_lift_joint.joint_angle(300)
        self.l_shoulder_pan_joint.joint_angle(75)
        self.l_shoulder_lift_joint.joint_angle(50)
        self.l_upper_arm_roll_joint.joint_angle(110)
        self.l_elbow_flex_joint.joint_angle(-110)
        self.l_forearm_roll_joint.joint_angle(-20)
        self.l_wrist_flex_joint.joint_angle(-10)
        self.l_wrist_roll_joint.joint_angle(-10)
        self.r_shoulder_pan_joint.joint_angle(-75)
        self.r_shoulder_lift_joint.joint_angle(50)
        self.r_upper_arm_roll_joint.joint_angle(-110)
        self.r_elbow_flex_joint.joint_angle(-110)
        self.r_forearm_roll_joint.joint_angle(20)
        self.r_wrist_flex_joint.joint_angle(-10)
        self.r_wrist_roll_joint.joint_angle(-10)
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(50)
        return self.angle_vector()

    def reset_pose(self):
        self.torso_lift_joint.joint_angle(50)
        self.l_shoulder_pan_joint.joint_angle(60)
        self.l_shoulder_lift_joint.joint_angle(74)
        self.l_upper_arm_roll_joint.joint_angle(70)
        self.l_elbow_flex_joint.joint_angle(-120)
        self.l_forearm_roll_joint.joint_angle(20)
        self.l_wrist_flex_joint.joint_angle(-30)
        self.l_wrist_roll_joint.joint_angle(180)
        self.r_shoulder_pan_joint.joint_angle(-60)
        self.r_shoulder_lift_joint.joint_angle(74)
        self.r_upper_arm_roll_joint.joint_angle(-70)
        self.r_elbow_flex_joint.joint_angle(-120)
        self.r_forearm_roll_joint.joint_angle(-20)
        self.r_wrist_flex_joint.joint_angle(-30)
        self.r_wrist_roll_joint.joint_angle(180)
        self.head_pan_joint.joint_angle(0)
        self.head_tilt_joint.joint_angle(0)
        return self.angle_vector()
