from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import jaxon_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class JaxonJVRC(RobotModelFromURDF):
    """JAXON humanoid (JVRC variant).

    1.7 m / ~133 kg humanoid with 6 + 6 leg, 3 torso, 2 head and
    8 + 8 arm joints. The URDF + meshes are fetched on first use
    from
    `robot-descriptions/jaxon_description
    <https://github.com/robot-descriptions/jaxon_description>`_,
    licensed under CC BY-SA 4.0.
    """

    def __init__(self, *args, **kwargs):
        super(JaxonJVRC, self).__init__(*args, **kwargs)

        self.rarm_end_coords = CascadedCoords(
            parent=self.RARM_LINK7,
            pos=[0.0, 0.0, -0.217],
            name='rarm_end_coords')
        self.rarm_end_coords.rotate(np.deg2rad(90), 'y')

        self.larm_end_coords = CascadedCoords(
            parent=self.LARM_LINK7,
            pos=[0.0, 0.0, -0.217],
            name='larm_end_coords')
        self.larm_end_coords.rotate(np.deg2rad(90), 'y')

        self.rleg_end_coords = CascadedCoords(
            parent=self.RLEG_LINK5,
            pos=[0.0, 0.0, -0.1],
            name='rleg_end_coords')

        self.lleg_end_coords = CascadedCoords(
            parent=self.LLEG_LINK5,
            pos=[0.0, 0.0, -0.1],
            name='lleg_end_coords')

        self.head_end_coords = CascadedCoords(
            parent=self.HEAD_LINK1,
            pos=[0.1, 0.0, 0.1],
            name='head_end_coords')
        self.head_end_coords.rotate(np.deg2rad(90), 'y')

        self.end_coords = [self.rarm_end_coords, self.larm_end_coords]

        self.reset_pose()

    @cached_property
    def default_urdf_path(self):
        return jaxon_urdfpath()

    def _set_pose_deg(self, rleg, lleg, torso, head, rarm, larm):
        groups = (
            ('RLEG_JOINT', rleg), ('LLEG_JOINT', lleg),
            ('CHEST_JOINT', torso), ('HEAD_JOINT', head),
            ('RARM_JOINT', rarm), ('LARM_JOINT', larm),
        )
        for prefix, vals in groups:
            for i, v in enumerate(vals):
                joint = getattr(self, '{}{}'.format(prefix, i))
                joint.joint_angle(np.deg2rad(float(v)))
        return self.angle_vector()

    def reset_pose(self):
        return self._set_pose_deg(
            rleg=[0, 0, -20, 40, -20, 0],
            lleg=[0, 0, -20, 40, -20, 0],
            torso=[0, 0, 0],
            head=[0, 0],
            rarm=[0, 40, -20, -5, -80, 0, 0, -20],
            larm=[0, 40, 20, 5, -80, 0, 0, -20],
        )

    def reset_manip_pose(self):
        return self._set_pose_deg(
            rleg=[0, 0, -20, 40, -20, 0],
            lleg=[0, 0, -20, 40, -20, 0],
            torso=[0, 0, 0],
            head=[0, 30],
            rarm=[0, 55, -20, -15, -100, -25, 0, -45],
            larm=[0, 55, 20, 15, -100, 25, 0, -45],
        )

    def collision_free_init_pose(self):
        """Mostly-zero pose with the shoulder roll moved just inside its limit."""
        return self._set_pose_deg(
            rleg=[0, 0, 0, 0, 0, 0],
            lleg=[0, 0, 0, 0, 0, 0],
            torso=[0, 0, 0],
            head=[0, 0],
            rarm=[0, 0, -16, 0, 0, 0, 0, 0],
            larm=[0, 0, 16, 0, 0, 0, 0, 0],
        )

    def _make_subrobot(self, link_names, end_coords=None):
        links = [getattr(self, n) for n in link_names]
        joints = [link.joint for link in links]
        sub = RobotModel(link_list=links, joint_list=joints)
        if end_coords is not None:
            sub.end_coords = end_coords
        return sub

    @cached_property
    def rarm(self):
        return self._make_subrobot(
            ['RARM_LINK{}'.format(i) for i in range(8)],
            self.rarm_end_coords)

    @cached_property
    def larm(self):
        return self._make_subrobot(
            ['LARM_LINK{}'.format(i) for i in range(8)],
            self.larm_end_coords)

    @cached_property
    def rleg(self):
        return self._make_subrobot(
            ['RLEG_LINK{}'.format(i) for i in range(6)],
            self.rleg_end_coords)

    @cached_property
    def lleg(self):
        return self._make_subrobot(
            ['LLEG_LINK{}'.format(i) for i in range(6)],
            self.lleg_end_coords)

    @cached_property
    def head(self):
        return self._make_subrobot(
            ['HEAD_LINK{}'.format(i) for i in range(2)],
            self.head_end_coords)

    @cached_property
    def torso(self):
        return self._make_subrobot(
            ['CHEST_LINK{}'.format(i) for i in range(3)])

    @property
    def right_arm(self):
        return self.rarm

    @property
    def left_arm(self):
        return self.larm

    @property
    def right_leg(self):
        return self.rleg

    @property
    def left_leg(self):
        return self.lleg

    @property
    def right_arm_end_coords(self):
        return self.rarm_end_coords

    @property
    def left_arm_end_coords(self):
        return self.larm_end_coords
