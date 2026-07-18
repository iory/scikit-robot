from skrobot.export.contact_links import contact_links
from skrobot.export.usd import urdf_to_usd


def export_for_task(robot, trajectory, objects, out_path,
                    threshold=0.1, **usd_kwargs):
    """Export a robot to USD, decomposing only the links its task touches.

    Ties :func:`contact_links` to :func:`urdf_to_usd`: run the task kinematically
    (no dynamics) to find which links come near the manipulated objects, then
    convert to USD with those -- and only those -- given accurate CoACD
    colliders. The rest keep the cheap single convex hull. On a robot where only
    the gripper touches anything, that decomposes a handful of links instead of
    the whole arm.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Robot loaded from a URDF (its ``urdf_path`` / ``urdf_robot_model`` is
        reused for the conversion). Its configuration is changed during the call.
    trajectory : iterable of array_like
        Joint configurations of the task (each a full angle-vector).
    objects : array_like
        World-frame points the robot interacts with, shape (3,) or (M, 3).
    out_path : str
        Destination ``.usd`` / ``.usdc`` path.
    threshold : float
        Distance [m] within which a link is considered to touch an object and so
        needs an accurate collider.
    **usd_kwargs
        Forwarded to :func:`urdf_to_usd` (e.g. ``home_positions``,
        ``coacd_params``). ``decompose_links`` is set from the task and cannot be
        overridden here.

    Returns
    -------
    stage : Usd.Stage
        The written USD stage.

    See Also
    --------
    contact_links : the link selection this runs, if you want the names too.
    """
    usd_kwargs.pop('decompose_links', None)   # decided from the task
    links = contact_links(robot, trajectory, objects, threshold=threshold)
    urdf = getattr(robot, 'urdf_path', None) or robot.urdf_robot_model
    return urdf_to_usd(urdf, out_path, decompose_links=links, **usd_kwargs)
