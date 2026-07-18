"""Generate a FollowJointTrajectory action server that drives Isaac Sim.

Isaac Sim's ROS 2 bridge subscribes to a single ``/joint_command``
(sensor_msgs/JointState) and drives an ArticulationController; it has no
joint_trajectory_controller and no action interface. Anything that speaks
control_msgs/FollowJointTrajectory -- MoveIt, or skrobot's ROS2RobotInterface --
needs one. :func:`generate_fjt_server` emits a standalone node that closes the
gap: one FJT action server per controller, streaming the time-interpolated
trajectory into ``/joint_command`` at a fixed rate.

The controller-to-joint mapping is the only robot-specific input, so this
follows the same generate-from-config pattern as ``skrobot.urdf.ros_config``.
"""

from pathlib import Path
import pprint
from string import Template

from skrobot.data import data_dir


TEMPLATES_DIR = Path(data_dir) / 'ros_config_templates'
_FJT_SERVER_TEMPLATE = 'isaac_fjt_server.py.template'


def _load_template(name):
    """Load a template file from skrobot's ROS config template directory.

    Parameters
    ----------
    name : str
        Template filename.

    Returns
    -------
    Template
        string.Template instance.
    """
    with open(TEMPLATES_DIR / name) as f:
        return Template(f.read())


def generate_fjt_server(controllers, node_name='isaac_fjt_server',
                        joint_command_topic='/joint_command',
                        joint_states_topic='/joint_states',
                        publish_hz=100.0, goal_tolerance=0.01,
                        settle_timeout=3.0):
    """Return the source of a standalone FollowJointTrajectory -> Isaac bridge.

    The node exposes ``/<name>/follow_joint_trajectory`` for each controller and
    streams the interpolated trajectory (position + velocity feedforward) into
    ``joint_command_topic``, holding the goal until every owned joint is within
    ``goal_tolerance``. It runs OUTSIDE the Isaac container as a plain ROS 2 node
    (needs ``rclpy`` + ``control_msgs``, not Isaac).

    Parameters
    ----------
    controllers : dict[str, list[str]]
        Controller name -> the joint names it owns. Joints not listed by any
        controller are left untouched, so a gripper driver can share
        ``joint_command_topic``.
    node_name : str
        ROS 2 node name.
    joint_command_topic, joint_states_topic : str
        Topics the sim's ROS 2 bridge exposes.
    publish_hz : float
        Rate the command table is streamed at.
    goal_tolerance : float
        Per-joint error (rad or m) the goal waits for before succeeding.
    settle_timeout : float
        Seconds (sim clock) to wait for the tolerance before returning
        GOAL_TOLERANCE_VIOLATED.

    Returns
    -------
    str
        Python source for the server. Write it out and run it with ``python3``.
    """
    if not controllers or not all(controllers.values()):
        raise ValueError('controllers must map each name to a non-empty '
                         'list of joint names')

    return _load_template(_FJT_SERVER_TEMPLATE).substitute(
        node_name=repr(node_name),
        joint_command_topic=joint_command_topic,
        joint_command_topic_repr=repr(joint_command_topic),
        joint_states_topic_repr=repr(joint_states_topic),
        publish_hz=repr(float(publish_hz)),
        goal_tolerance=repr(float(goal_tolerance)),
        settle_timeout=repr(float(settle_timeout)),
        controllers=pprint.pformat(dict(controllers), width=76),
    )


def write_fjt_server(controllers, path, **kwargs):
    """Write :func:`generate_fjt_server`'s output to ``path``. Returns ``path``."""
    with open(path, 'w') as f:
        f.write(generate_fjt_server(controllers, **kwargs))
    return path
