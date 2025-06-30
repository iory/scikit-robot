"""
Visualization utilities for skrobot inverse kinematics.
"""

from contextlib import contextmanager
import os
from threading import local
import time


# Thread-local storage for visualization state
_state = local()


def _get_global_enabled():
    """Get global visualization enabled state."""
    if not hasattr(_state, 'enabled'):
        # Check environment variable
        env_disabled = os.environ.get('SKROBOT_DISABLE_IK_VISUALIZATION', '').lower()
        _state.enabled = env_disabled not in ('1', 'true', 'yes')
    return _state.enabled


def _get_current_context():
    """Get current visualization context."""
    if not hasattr(_state, 'context_stack'):
        _state.context_stack = []
    return _state.context_stack[-1] if _state.context_stack else None


def set_ik_visualization_enabled(enabled):
    """
    Set global inverse kinematics visualization state.

    Parameters
    ----------
    enabled : bool
        Whether to enable IK visualization globally

    Examples
    --------
    >>> from skrobot.utils.visualization import set_ik_visualization_enabled
    >>>
    >>> # Disable visualization globally
    >>> set_ik_visualization_enabled(False)
    >>>
    >>> # All subsequent IK calls will not show visualization
    >>> robot.inverse_kinematics(target, inverse_kinematics_hook=auto_ik_hook())
    """
    _state.enabled = enabled


def get_ik_visualization_enabled():
    """
    Get current inverse kinematics visualization state.

    Returns
    -------
    bool
        Whether IK visualization is currently enabled
    """
    context = _get_current_context()
    if context is not None:
        return context.get('enabled', True)
    return _get_global_enabled()


@contextmanager
def ik_visualization(viewer=None, sleep_time=0.05, enabled=True):
    """
    Context manager for controlling inverse kinematics visualization.

    Parameters
    ----------
    viewer : skrobot.viewers.TrimeshSceneViewer or skrobot.viewers.PyrenderViewer, optional
        The viewer object to use for visualization
    sleep_time : float, optional
        Sleep time in seconds between iterations (default: 0.05)
    enabled : bool, optional
        Whether visualization is enabled in this context (default: True)

    Examples
    --------
    >>> import skrobot
    >>> from skrobot.utils.visualization import ik_visualization, auto_ik_hook
    >>>
    >>> robot = skrobot.models.PR2()
    >>> viewer = skrobot.viewers.TrimeshSceneViewer()
    >>> viewer.add(robot)
    >>> viewer.show()
    >>>
    >>> # Enable visualization within context
    >>> with ik_visualization(viewer, sleep_time=0.1):
    ...     robot.inverse_kinematics(
    ...         target,
    ...         link_list=robot.rarm.link_list,
    ...         move_target=robot.rarm.end_coords,
    ...         inverse_kinematics_hook=auto_ik_hook()
    ...     )
    >>>
    >>> # Disable visualization temporarily
    >>> with ik_visualization(enabled=False):
    ...     robot.inverse_kinematics(target, inverse_kinematics_hook=auto_ik_hook())
    """
    if not hasattr(_state, 'context_stack'):
        _state.context_stack = []

    context = {
        'viewer': viewer,
        'sleep_time': sleep_time,
        'enabled': enabled,
        'type': 'ik'
    }

    _state.context_stack.append(context)
    try:
        yield
    finally:
        _state.context_stack.pop()


@contextmanager
def trajectory_visualization(viewer=None, robot_model=None, joint_list=None,
                             sleep_time=0.1, enabled=True, with_base=False,
                             update_every_n_iterations=1, debug=False,
                             waypoint_mode='goal'):
    """
    Context manager for controlling trajectory optimization visualization.

    Parameters
    ----------
    viewer : skrobot.viewers.TrimeshSceneViewer or skrobot.viewers.PyrenderViewer, optional
        The viewer object to use for visualization
    robot_model : skrobot.model.RobotModel, optional
        Robot model to update during optimization
    joint_list : list[skrobot.model.Joint], optional
        List of joints to control
    sleep_time : float, optional
        Sleep time in seconds between iterations (default: 0.1)
    enabled : bool, optional
        Whether visualization is enabled in this context (default: True)
    with_base : bool, optional
        Whether base coordinates are included (default: False)
    update_every_n_iterations : int, optional
        Update visualization every N optimizer iterations (default: 1)
    debug : bool, optional
        Enable debug output for troubleshooting (default: False)
    waypoint_mode : str, optional
        Which waypoint to visualize ('goal', 'middle', 'cycle') (default: 'goal')

    Examples
    --------
    >>> import skrobot
    >>> from skrobot.utils.visualization import trajectory_visualization
    >>> from skrobot.planner import sqp_plan_trajectory
    >>>
    >>> robot = skrobot.models.PR2()
    >>> viewer = skrobot.viewers.TrimeshSceneViewer()
    >>> viewer.add(robot)
    >>> viewer.show()
    >>>
    >>> # Enable trajectory optimization visualization
    >>> with trajectory_visualization(viewer, robot, joint_list, sleep_time=0.2):
    ...     trajectory = sqp_plan_trajectory(
    ...         collision_checker, av_start, av_goal, joint_list, n_wp
    ...     )
    """
    if not hasattr(_state, 'context_stack'):
        _state.context_stack = []

    context = {
        'viewer': viewer,
        'robot_model': robot_model,
        'joint_list': joint_list,
        'sleep_time': sleep_time,
        'enabled': enabled,
        'with_base': with_base,
        'update_every_n_iterations': update_every_n_iterations,
        'debug': debug,
        'waypoint_mode': waypoint_mode,
        'iteration_count': 0,
        'type': 'trajectory'
    }

    _state.context_stack.append(context)
    try:
        yield
    finally:
        _state.context_stack.pop()


def auto_ik_hook():
    """
    Create an automatic inverse kinematics hook based on current context.

    This function automatically creates appropriate visualization hooks
    based on the current visualization context set by ik_visualization()
    or global settings.

    Returns
    -------
    list
        A list of hook functions for inverse_kinematics_hook parameter.
        Returns empty list if visualization is disabled.

    Examples
    --------
    >>> import skrobot
    >>> from skrobot.utils.visualization import ik_visualization, auto_ik_hook
    >>>
    >>> with ik_visualization(viewer):
    ...     # Automatically uses the viewer from context
    ...     robot.inverse_kinematics(target, inverse_kinematics_hook=auto_ik_hook())
    >>>
    >>> # Or with global settings
    >>> robot.inverse_kinematics(target, inverse_kinematics_hook=auto_ik_hook())
    """
    if not get_ik_visualization_enabled():
        return []

    context = _get_current_context()
    if context and context.get('type') == 'ik' and context.get('viewer'):
        return create_ik_visualization_hook(
            context['viewer'],
            sleep_time=context.get('sleep_time', 0.05),
            enabled=True
        )

    # No context or viewer, return empty hooks
    return []


def get_trajectory_optimization_callback():
    """
    Get trajectory optimization callback based on current context.

    Returns a callback function that can be passed to scipy.optimize.minimize
    for trajectory optimization visualization.

    Returns
    -------
    callable or None
        Callback function for trajectory optimization, or None if disabled

    Examples
    --------
    >>> from skrobot.utils.visualization import trajectory_visualization, get_trajectory_optimization_callback
    >>>
    >>> with trajectory_visualization(viewer, robot_model, joint_list):
    ...     callback = get_trajectory_optimization_callback()
    ...     # Use callback in optimization
    """
    if not get_ik_visualization_enabled():
        return None

    context = _get_current_context()
    if not context or context.get('type') != 'trajectory' or not context.get('enabled'):
        return None

    viewer = context.get('viewer')
    robot_model = context.get('robot_model')
    joint_list = context.get('joint_list')
    sleep_time = context.get('sleep_time', 0.1)
    with_base = context.get('with_base', False)
    update_every_n = context.get('update_every_n_iterations', 1)

    if not all([viewer, robot_model, joint_list]):
        return None

    def trajectory_callback(xi):
        """Callback function for trajectory optimization visualization."""
        # Increment iteration counter
        context['iteration_count'] += 1

        # Skip if not time to update
        if context['iteration_count'] % update_every_n != 0:
            return

        try:
            from skrobot.planner.utils import set_robot_config

            # Reshape trajectory and get current best waypoint
            n_dof = len(joint_list) + (3 if with_base else 0)
            n_wp = len(xi) // n_dof
            av_seq = xi.reshape(n_wp, n_dof)

            # Debug output
            if context.get('debug', False):
                print("Iteration {}: n_wp={}, n_dof={}, av_seq shape={}".format(
                    context['iteration_count'], n_wp, n_dof, av_seq.shape))

            # Select waypoint based on visualization mode
            waypoint_mode = context.get('waypoint_mode', 'goal')
            if waypoint_mode == 'goal':
                # Show final waypoint (goal) to see convergence to target
                waypoint_idx = n_wp - 1
            elif waypoint_mode == 'middle':
                # Show middle waypoint to see trajectory evolution
                waypoint_idx = n_wp // 2
            elif waypoint_mode == 'cycle':
                # Cycle through different waypoints
                waypoint_idx = (context['iteration_count'] // update_every_n) % n_wp
            else:
                waypoint_idx = n_wp - 1  # Default to goal

            if waypoint_idx < len(av_seq):
                current_av = av_seq[waypoint_idx]
                set_robot_config(robot_model, joint_list, current_av, with_base)

                # Force update of robot mesh in viewer
                if hasattr(viewer, 'update'):
                    viewer.update()
                viewer.redraw()

                time.sleep(sleep_time)

        except Exception as e:
            # If visualization fails, don't break optimization but show error if debug
            if context.get('debug', False):
                print("Visualization error: {}".format(e))
            pass

    return trajectory_callback


def create_ik_visualization_hook(viewer, sleep_time=0.05, enabled=None):
    """
    Create a visualization hook function for inverse kinematics.

    This function creates a hook that can be passed to inverse_kinematics_hook
    parameter to visualize the robot's movement during IK solving.

    Parameters
    ----------
    viewer : skrobot.viewers.TrimeshSceneViewer or skrobot.viewers.PyrenderViewer
        The viewer object to redraw for visualization
    sleep_time : float, optional
        Sleep time in seconds between iterations (default: 0.05)
    enabled : bool, optional
        Whether to enable the hook. If None, uses global/context settings.

    Returns
    -------
    list
        A list of hook functions that can be passed to inverse_kinematics_hook.
        Returns empty list if visualization is disabled.

    Examples
    --------
    >>> import skrobot
    >>> from skrobot.utils.visualization import create_ik_visualization_hook
    >>>
    >>> robot = skrobot.models.PR2()
    >>> viewer = skrobot.viewers.TrimeshSceneViewer()
    >>> viewer.add(robot)
    >>> viewer.show()
    >>>
    >>> # Create visualization hook
    >>> ik_hook = create_ik_visualization_hook(viewer, sleep_time=0.1)
    >>>
    >>> # Use in inverse kinematics
    >>> target = skrobot.coordinates.Coordinates([0.5, 0, 0.8])
    >>> robot.inverse_kinematics(
    ...     target,
    ...     link_list=robot.rarm.link_list,
    ...     move_target=robot.rarm.end_coords,
    ...     inverse_kinematics_hook=ik_hook
    ... )
    >>>
    >>> # Create disabled hook
    >>> ik_hook = create_ik_visualization_hook(viewer, enabled=False)  # Returns []
    """
    if enabled is None:
        enabled = get_ik_visualization_enabled()

    if not enabled:
        return []

    def redraw_hook():
        viewer.redraw()

    def sleep_hook():
        time.sleep(sleep_time)

    return [redraw_hook, sleep_hook]


def create_custom_ik_hook(*hook_functions, **kwargs):
    """
    Create a custom inverse kinematics hook from multiple functions.

    Parameters
    ----------
    *hook_functions : callable
        Functions to be called during each IK iteration
    enabled : bool, optional
        Whether to enable the hook. If None, uses global/context settings.
        Pass as keyword argument: enabled=True/False

    Returns
    -------
    list
        A list of hook functions that can be passed to inverse_kinematics_hook.
        Returns empty list if visualization is disabled.

    Examples
    --------
    >>> def print_hook():
    ...     print("IK iteration")
    >>>
    >>> def custom_redraw():
    ...     viewer.redraw()
    ...     time.sleep(0.1)
    >>>
    >>> ik_hook = create_custom_ik_hook(print_hook, custom_redraw)
    >>> robot.inverse_kinematics(target, inverse_kinematics_hook=ik_hook)
    >>>
    >>> # Create disabled hook
    >>> ik_hook = create_custom_ik_hook(print_hook, enabled=False)  # Returns []
    """
    enabled = kwargs.get('enabled', None)
    if enabled is None:
        enabled = get_ik_visualization_enabled()

    if not enabled:
        return []

    return list(hook_functions)
