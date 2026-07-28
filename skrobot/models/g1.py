from cached_property import cached_property

from skrobot.coordinates import CascadedCoords
from skrobot.data import g1_urdfpath
from skrobot.models.urdf import RobotModelFromURDF


class G1(RobotModelFromURDF):
    """Unitree G1 humanoid.

    ~1.3 m / ~35 kg humanoid. By default the 23-DOF locomotion model is
    loaded (6 + 6 leg, 1 waist-yaw and 5 + 5 arm joints); pass ``dof=29`` for
    the extended model (waist roll/pitch and wrist pitch/yaw). The URDF +
    meshes are fetched on first use from
    `isri-aist/g1_description
    <https://github.com/isri-aist/g1_description>`_, derived from Unitree's
    official ``g1_description`` (BSD-3-Clause).
    """

    def __init__(self, dof=23, urdf=None, urdf_file=None):
        self._dof = dof
        super(G1, self).__init__(urdf=urdf, urdf_file=urdf_file)

        # Foot sole frames, ~0.03 m below the ankle-roll link origin.
        self.rleg_end_coords = CascadedCoords(
            parent=self.right_ankle_roll_link,
            pos=[0.0, 0.0, -0.03],
            name='rleg_end_coords')
        self.lleg_end_coords = CascadedCoords(
            parent=self.left_ankle_roll_link,
            pos=[0.0, 0.0, -0.03],
            name='lleg_end_coords')

        # Hand frames (23-DOF model ends at wrist_roll; 29-DOF adds wrist
        # pitch/yaw). Use whichever end link the loaded model exposes.
        self.rarm_end_coords = CascadedCoords(
            parent=self._arm_end_link('right'), name='rarm_end_coords')
        self.larm_end_coords = CascadedCoords(
            parent=self._arm_end_link('left'), name='larm_end_coords')

        self.end_coords = [self.rarm_end_coords, self.larm_end_coords]

        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return g1_urdfpath(dof=self._dof)

    def _arm_end_link(self, side):
        for name in ('{}_wrist_yaw_link'.format(side),
                     '{}_wrist_roll_rubber_hand'.format(side),
                     '{}_wrist_roll_link'.format(side),
                     '{}_rubber_hand'.format(side)):
            link = getattr(self, name, None)
            if link is not None:
                return link
        raise AttributeError('no arm end link found for side {}'.format(side))

    def reset_pose(self):
        """A statically-stable half-crouch: hips/knees/ankles flexed so the
        soles sit flat and the CoM is inside the support polygon."""
        angles = {
            'left_hip_pitch_joint': -0.10,
            'right_hip_pitch_joint': -0.10,
            'left_knee_joint': 0.30,
            'right_knee_joint': 0.30,
            'left_ankle_pitch_joint': -0.20,
            'right_ankle_pitch_joint': -0.20,
            'left_shoulder_roll_joint': 0.20,
            'right_shoulder_roll_joint': -0.20,
            'left_elbow_joint': 0.30,
            'right_elbow_joint': 0.30,
        }
        for name, ang in angles.items():
            joint = getattr(self, name, None)
            if joint is not None:
                joint.joint_angle(float(ang))
        return self.angle_vector()
