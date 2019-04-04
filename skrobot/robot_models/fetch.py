from cached_property import cached_property
from skrobot.coordinates import CascadedCoords
from skrobot.data import fetch_urdfpath
from skrobot.robot_model import RobotModel


class Fetch(RobotModel):
    """Fetch Robot Model.

    http://docs.fetchrobotics.com/robot_hardware.html
    """

    def __init__(self, urdf_path=None, *args, **kwargs):
        super(Fetch, self).__init__(*args, **kwargs)
        if urdf_path is None:
            urdf_path = fetch_urdfpath()
        self.urdf_path = urdf_path
        self.load_urdf(urdf_path)

        self.rarm_end_coords = CascadedCoords(parent=self.gripper_link,
                                              name='rarm_end_coords')
        self.rarm_end_coords.translate([0, 0, 0])
        self.rarm_end_coords.rotate(0, axis='z')
        self.end_coords = [self.rarm_end_coords]

        self.link_list = [self.base_link,
                          self.torso_lift_link,
                          self.shoulder_pan_link,
                          self.shoulder_lift_link,
                          self.upperarm_roll_link,
                          self.elbow_flex_link,
                          self.forearm_roll_link,
                          self.wrist_flex_link,
                          self.wrist_roll_link,
                          self.head_pan_link,
                          self.head_tilt_link]

    def reset_pose(self):
        self.torso_lift_joint.joint_angle(0)
        self.shoulder_pan_joint.joint_angle(75.6304)
        self.shoulder_lift_joint.joint_angle(80.2141)
        self.upperarm_roll_joint.joint_angle(-11.4592)
        self.elbow_flex_joint.joint_angle(98.5487)
        self.forearm_roll_joint.joint_angle(0.0)
        self.wrist_flex_joint.joint_angle(95.111)
        self.wrist_roll_joint.joint_angle(0.0)
        self.head_pan_joint.joint_angle(0.0)
        self.head_tilt_joint.joint_angle(0.0)
        return self.angle_vector()

    def reset_manip_pose(self):
        self.torso_lift_joint.joint_angle(0)
        self.shoulder_pan_joint.joint_angle(0)
        self.shoulder_lift_joint.joint_angle(0)
        self.upperarm_roll_joint.joint_angle(0)
        self.elbow_flex_joint.joint_angle(90)
        self.forearm_roll_joint.joint_angle(0.0)
        self.wrist_flex_joint.joint_angle(-90)
        self.wrist_roll_joint.joint_angle(0)
        self.head_pan_joint.joint_angle(0.0)
        self.head_tilt_joint.joint_angle(0.0)
        return self.angle_vector()

    @cached_property
    def rarm(self):
        rarm_links = [self.shoulder_pan_link,
                      self.shoulder_lift_link,
                      self.upperarm_roll_link,
                      self.elbow_flex_link,
                      self.forearm_roll_link,
                      self.wrist_flex_link,
                      self.wrist_roll_link]
        rarm_joints = []
        for link in rarm_links:
            rarm_joints.append(link.joint)
        r = RobotModel(link_list=rarm_links,
                       joint_list=rarm_joints)
        r.end_coords = self.rarm_end_coords
        return r
