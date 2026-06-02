from cached_property import cached_property

from skrobot.coordinates import CascadedCoords
from skrobot.data import hydrus_urdfpath
from skrobot.model import RobotModel
from skrobot.models.urdf import RobotModelFromURDF


class Hydrus(RobotModelFromURDF):
    """JSK HYDRUS transformable multirotor.

    HYDRUS is a planar multilink aerial robot: four rotor links
    (``link1`` .. ``link4``) connected in series by the revolute joints
    ``joint1`` .. ``joint3``, so the airframe can change its 2D shape in
    flight.  ``leg5`` is a fixed frame at the tip of ``link4`` that is
    handy as an end-effector for planar tasks, and ``fc`` is the flight
    controller frame near the center of the body.

    Kinematic chain::

        root --[root_joint: fixed]-- link1 --[joint1: revolute]-- link2
             --[joint2: revolute]-- link3 --[joint3: revolute]-- link4
             --[link42leg5: fixed]-- leg5

    Only the revolute joints ``joint1`` .. ``joint3`` are movable, so the
    ``arm`` limb's ``link_list`` is their child links ``link2``, ``link3``,
    ``link4``.  ``link1`` is rigidly fixed to ``root`` and ``leg5`` is
    rigidly fixed to ``link4``, so neither carries a degree of freedom and
    neither belongs in ``link_list``.

    The serial chain ``joint1`` -> ``joint3`` is exposed as the ``arm``
    limb with ``arm_end_coords`` at ``leg5``, so end-effector IK can be
    solved with ``robot.arm.inverse_kinematics(target_coords)`` just like
    the manipulators of other models::

        robot = Hydrus()
        robot.arm.inverse_kinematics(target_coords, rotation_axis=False)

    The URDF and meshes are fetched on first use and cached under
    ``~/.skrobot/hydrus_description/``.
    """

    @cached_property
    def default_urdf_path(self):
        return hydrus_urdfpath()

    @cached_property
    def arm_end_coords(self):
        """End-effector frame at the tip of the arm (``leg5``)."""
        return CascadedCoords(parent=self.leg5, name='arm_end_coords')

    @cached_property
    def arm(self):
        """Serial multilink arm from joint1 to joint3, tip at leg5."""
        arm_links = [self.link2, self.link3, self.link4]
        arm_joints = [link.joint for link in arm_links]
        r = RobotModel(link_list=arm_links, joint_list=arm_joints)
        r.end_coords = self.arm_end_coords
        return r
