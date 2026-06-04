from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import aero_urdfpath
from skrobot.model import RobotModel


class Aero(RobotModel):

    """SEED-Noid / Aero upper-body humanoid on a wheeled lifter base.

    The robot has two 8-DOF arms, a 3-DOF neck, a 3-DOF waist and a 2-DOF
    parallel-link lifter (``ankle_joint`` and ``knee_joint``) that raises and
    lowers the whole upper body while keeping it level.

    Parameters
    ----------
    use_hand : bool
        If True (default), load the model with the SEED hand, whose finger
        joints are part of ``angle_vector``.  If False, load the hand-less
        model whose arms end at the built-in ``*_eef_*`` frames.

    Examples
    --------
    >>> from skrobot.models import Aero
    >>> robot = Aero()                 # with hand
    >>> robot_nohand = Aero(use_hand=False)
    >>> robot.reset_pose()
    >>> robot.rarm.inverse_kinematics(
    ...     robot.rarm.end_coords.copy_worldcoords().translate([0.03, 0, -0.05]),
    ...     rotation_axis=True)
    """

    def __init__(self, use_hand=True, *args, **kwargs):
        super(Aero, self).__init__(*args, **kwargs)
        self.use_hand = use_hand
        self.load_urdf_file(aero_urdfpath(use_hand=use_hand),
                            include_mimic_joints=False)

        # Built-in end-effector frames (present in both hand / no-hand URDFs).
        self.rarm_end_coords = CascadedCoords(
            parent=self.r_eef_grasp_link, name='rarm_end_coords')
        self.larm_end_coords = CascadedCoords(
            parent=self.l_eef_grasp_link, name='larm_end_coords')
        self.torso_end_coords = CascadedCoords(
            parent=self.leg_base_link, name='torso_end_coords')
        self.torso_end_coords.translate([-0.032, 0.0, -0.3507])
        self.head_end_coords = CascadedCoords(
            parent=self.head_link, name='head_end_coords')
        self.head_end_coords.translate([0.079, 0.0, 0.1035]).rotate(
            1.7453275066650775, 'y')

        self.end_coords = [self.rarm_end_coords,
                           self.larm_end_coords,
                           self.torso_end_coords]

    def reset_pose(self):
        self.r_shoulder_p_joint.joint_angle(np.deg2rad(-14.0))
        self.r_shoulder_r_joint.joint_angle(np.deg2rad(-1.5))
        self.r_shoulder_y_joint.joint_angle(np.deg2rad(-17.0))
        self.r_elbow_joint.joint_angle(np.deg2rad(-135.0))
        self.r_wrist_y_joint.joint_angle(0.0)
        self.r_wrist_p_joint.joint_angle(0.0)
        self.r_wrist_r_joint.joint_angle(0.0)
        self.r_hand_y_joint.joint_angle(0.0)
        self.l_shoulder_p_joint.joint_angle(np.deg2rad(-14.0))
        self.l_shoulder_r_joint.joint_angle(np.deg2rad(1.5))
        self.l_shoulder_y_joint.joint_angle(np.deg2rad(17.0))
        self.l_elbow_joint.joint_angle(np.deg2rad(-135.0))
        self.l_wrist_y_joint.joint_angle(0.0)
        self.l_wrist_p_joint.joint_angle(0.0)
        self.l_wrist_r_joint.joint_angle(0.0)
        self.l_hand_y_joint.joint_angle(0.0)
        self.neck_y_joint.joint_angle(0.0)
        self.neck_p_joint.joint_angle(np.deg2rad(25.0))
        self.neck_r_joint.joint_angle(0.0)
        self.waist_y_joint.joint_angle(0.0)
        self.waist_p_joint.joint_angle(0.0)
        self.waist_r_joint.joint_angle(0.0)
        self.ankle_joint.joint_angle(np.deg2rad(30.0))
        self.knee_joint.joint_angle(np.deg2rad(-30.0))
        return self.angle_vector()

    def _limb(self, links, end_coords):
        joint_list = [link.joint for link in links if link.joint is not None]
        r = RobotModel(link_list=links, joint_list=joint_list)
        r.end_coords = end_coords
        return r

    def _arm_links(self, prefix):
        return [getattr(self, prefix + suffix) for suffix in (
            'shoulder_link', 'shoulder_center', 'upperarm_link', 'elbow_link',
            'forearm_link', 'wrist_center', 'hand_yaw_link', 'hand_link')]

    @cached_property
    def head(self):
        return self._limb(
            [self.neck_link, self.head_base_link, self.head_link],
            self.head_end_coords)

    @cached_property
    def rarm(self):
        return self._limb(self._arm_links('r_'), self.rarm_end_coords)

    @cached_property
    def larm(self):
        return self._limb(self._arm_links('l_'), self.larm_end_coords)

    @cached_property
    def _lifter_links(self):
        # The leg parallel linkage that carries the upper body.  The passive
        # (mimic) links must stay in the chain so the IK Jacobian accounts for
        # the linkage; the mimic hooks keep them consistent automatically.
        return [self.leg_shank_link,
                self.leg_knee_link,
                self.leg_thigh_link,
                self.leg_base_link,
                self.hip_sphere_link,
                self.hip_center]

    @cached_property
    def rarm_whole_body(self):
        return self._limb(
            self._lifter_links + self._arm_links('r_'),
            self.rarm_end_coords)

    @cached_property
    def larm_whole_body(self):
        return self._limb(
            self._lifter_links + self._arm_links('l_'),
            self.larm_end_coords)

    @cached_property
    def torso(self):
        return self._limb(
            self._lifter_links + [self.body_link],
            self.torso_end_coords)
