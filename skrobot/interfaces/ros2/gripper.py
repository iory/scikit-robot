import threading

import control_msgs.action
import rclpy
from rclpy.action import ActionClient


def _wait_for_future(node, future, timeout_sec=None):
    """Block until ``future`` completes, whoever is spinning ``node``.

    ``rclpy.spin_until_future_complete`` spins the node itself. If an external
    executor is already spinning that same node -- which is how this interface
    is meant to be used, and the only way ``/joint_states`` keeps arriving --
    that makes two spinners share one wait set, and rclpy fails with
    ``wait set index for status subscription is out of bounds``. Wait on a
    callback instead whenever someone else is driving the node.

    Parameters
    ----------
    node : rclpy.node.Node
        Node the future belongs to.
    future : rclpy.task.Future
        Future to wait on.
    timeout_sec : float or None
        Optional timeout. ``None`` waits indefinitely.

    Returns
    -------
    bool
        True if the future completed, False on timeout.
    """
    if node.executor is None:
        rclpy.spin_until_future_complete(node, future, timeout_sec=timeout_sec)
        return future.done()
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    return done.wait(timeout=timeout_sec)


class GripperCommandActionClient(object):
    """Action client wrapper for ``control_msgs/action/GripperCommand``.

    Parameters
    ----------
    node : rclpy.node.Node
        Node used for action communication and logging.
    action_name : str
        Action namespace of ``GripperCommand``.
    width_scale : float, default 0.5
        Conversion factor from jaw gap width (meters) to the commanded
        joint value. ``command.position = width * width_scale``.
        For common symmetric two-finger grippers controlled by one
        joint this is 0.5 because each finger tracks half the jaw gap.
    max_effort : float, default 0.0
        Default effort limit forwarded to ``command.max_effort``.

    Notes
    -----
    ``move`` always accepts jaw gap width in meters to keep a consistent
    external API, regardless of controller joint units.
    """

    def __init__(self,
                 node,
                 action_name,
                 width_scale=0.5,
                 max_effort=0.0):
        self._node = node
        self._width_scale = width_scale
        self._max_effort = max_effort

        self._action_client = ActionClient(
            node,
            control_msgs.action.GripperCommand,
            action_name)

        self._goal_future = None
        self._goal_handle = None
        self._goal_rejected_logged = False

    @property
    def available(self):
        """Whether the action server is currently available."""
        return self._action_client.wait_for_server(timeout_sec=0.0)

    def _store_goal_handle(self, future):
        goal_handle = future.result()
        self._goal_handle = goal_handle
        if goal_handle is None or not goal_handle.accepted:
            self._node.get_logger().warn(
                "GripperCommand goal was rejected by action server.")
            self._goal_rejected_logged = True

    def move(self, width, max_effort=None, wait=True):
        """Send a gripper command using jaw gap width in meters.

        Parameters
        ----------
        width : float
            Target jaw gap width in meters.
        max_effort : float or None
            Effort limit for this goal. If None, uses the constructor
            default.
        wait : bool, default True
            If True, wait for completion and return action result.
            If False, return the goal future.
        """
        goal = control_msgs.action.GripperCommand.Goal()
        goal.command.position = width * self._width_scale
        goal.command.max_effort = (
            self._max_effort if max_effort is None else max_effort)

        self._goal_rejected_logged = False
        self._goal_future = self._action_client.send_goal_async(goal)
        self._goal_future.add_done_callback(self._store_goal_handle)

        if wait:
            _wait_for_future(self._node, self._goal_future)
            goal_handle = self._goal_future.result()
            if goal_handle is None or not goal_handle.accepted:
                if not self._goal_rejected_logged:
                    self._node.get_logger().warn(
                        "GripperCommand goal was rejected by action server.")
                return
            self._goal_handle = goal_handle

            result_future = goal_handle.get_result_async()
            _wait_for_future(self._node, result_future)
            return result_future.result()

        return self._goal_future

    def cancel(self, wait=True):
        """Cancel the in-flight gripper goal if any."""
        if self._goal_handle is None and self._goal_future is not None:
            _wait_for_future(self._node, self._goal_future)
            self._goal_handle = self._goal_future.result()

        goal_handle = self._goal_handle
        if goal_handle is None:
            self._node.get_logger().warn("No GripperCommand goal to cancel.")
            return

        cancel_future = goal_handle.cancel_goal_async()
        if wait:
            _wait_for_future(self._node, cancel_future)
            self._goal_handle = None
            return cancel_future.result()

        self._goal_handle = None
        return cancel_future
