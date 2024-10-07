import geometry_msgs.msg

from skrobot.coordinates import Coordinates


def coords_to_geometry_pose(coords):
    """Convert Coordinates to geometry_msgs.msg.Pose

    Parameters
    ----------
    coords : skrobot.coordinates.Coordinates
        coordinates pose.

    Returns
    -------
    pose : geometry_msgs.msg.Pose
        converted geometry_msgs pose.
    """
    pose = geometry_msgs.msg.Pose()
    pose.position.x = coords.translation[0]
    pose.position.y = coords.translation[1]
    pose.position.z = coords.translation[2]
    q = coords.quaternion_wxyz
    pose.orientation.w = q[0]
    pose.orientation.x = q[1]
    pose.orientation.y = q[2]
    pose.orientation.z = q[3]
    return pose


def coords_to_tf_pose(coords):
    """Convert Coordinates to geometry_msgs.msg.Transform

    Parameters
    ----------
    coords : skrobot.coordinates.Coordinates
        coordinates pose.

    Returns
    -------
    pose : geometry_msgs.msg.Transform
        converted transform pose.
    """
    pose = geometry_msgs.msg.Transform()
    pose.translation.x = coords.translation[0]
    pose.translation.y = coords.translation[1]
    pose.translation.z = coords.translation[2]
    q = coords.quaternion_wxyz
    pose.rotation.w = q[0]
    pose.rotation.x = q[1]
    pose.rotation.y = q[2]
    pose.rotation.z = q[3]
    return pose


def tf_pose_to_coords(tf_pose):
    """Convert TransformStamped to Coordinates

    Parameters
    ----------
    tf_pose : geometry_msgs.msg.Transform or geometry_msgs.msg.TransformStamped
        transform pose.

    Returns
    -------
    ret : skrobot.coordinates.Coordinates
        converted coordinates.
    """
    if isinstance(tf_pose, geometry_msgs.msg.Transform):
        transform = tf_pose
    elif isinstance(tf_pose, geometry_msgs.msg.TransformStamped):
        transform = tf_pose.transform
    else:
        raise TypeError('{} not supported'.format(type(tf_pose)))
    if transform.rotation.w == 0.0 and \
       transform.rotation.x == 0.0 and \
       transform.rotation.y == 0.0 and \
       transform.rotation.z == 0.0:
        transform.rotation.w = 1.0
    return Coordinates(pos=[transform.translation.x,
                            transform.translation.y,
                            transform.translation.z],
                       rot=[transform.rotation.w, transform.rotation.x,
                            transform.rotation.y, transform.rotation.z])


def geometry_pose_to_coords(tf_pose):
    """Convert geometry_msgs.msg.Pose to Coordinates

    Parameters
    ----------
    tf_pose : geometry_msgs.msg.Pose or geometry_msgs.msg.PoseStamped
        pose.

    Returns
    -------
    ret : skrobot.coordinates.Coordinates
        converted coordinates.
    """
    if isinstance(tf_pose, geometry_msgs.msg.Pose):
        pose = tf_pose
    elif isinstance(tf_pose, geometry_msgs.msg.PoseStamped):
        pose = tf_pose.pose
    else:
        raise TypeError('{} not supported'.format(type(tf_pose)))
    if pose.orientation.w == 0.0 and \
       pose.orientation.x == 0.0 and \
       pose.orientation.y == 0.0 and \
       pose.orientation.z == 0.0:
        pose.orientation.w = 1.0
    return Coordinates(pos=[pose.position.x, pose.position.y, pose.position.z],
                       rot=[pose.orientation.w, pose.orientation.x,
                            pose.orientation.y, pose.orientation.z])
