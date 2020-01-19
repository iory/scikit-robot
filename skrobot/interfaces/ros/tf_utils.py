import geometry_msgs.msg

from skrobot.coordinates import Coordinates


def coords_to_geometry_pose(coords):
    """Convert Coordinates to geometry_msgs.msg.Pose

    """
    pose = geometry_msgs.msg.Pose()
    pose.position.x = coords.translation[0]
    pose.position.y = coords.translation[1]
    pose.position.z = coords.translation[2]
    q = coords.quaternion
    pose.orientation.w = q[0]
    pose.orientation.x = q[1]
    pose.orientation.y = q[2]
    pose.orientation.z = q[3]
    return pose


def tf_pose_to_coords(tf_pose):
    """Convert TransformStamped to Coordinates

    """
    if tf_pose.transform.rotation.w == 0.0 and \
       tf_pose.transform.rotation.x == 0.0 and \
       tf_pose.transform.rotation.y == 0.0 and \
       tf_pose.transform.rotation.z == 0.0:
        tf_pose.transform.rotation.w = 1.0
    return Coordinates(pos=[tf_pose.transform.translation.x,
                            tf_pose.transform.translation.y,
                            tf_pose.transform.translation.z],
                       rot=[tf_pose.transform.rotation.w,
                            tf_pose.transform.rotation.x,
                            tf_pose.transform.rotation.y,
                            tf_pose.transform.rotation.z])


def geometry_pose_to_coords(tf_pose):
    """Convert geometry_msgs.msg.Pose to Coordinates

    """
    if tf_pose.pose.orientation.w == 0.0 and \
       tf_pose.pose.orientation.x == 0.0 and \
       tf_pose.pose.orientation.y == 0.0 and \
       tf_pose.pose.orientation.z == 0.0:
        tf_pose.pose.orientation.w = 1.0
    return Coordinates(pos=[tf_pose.pose.position.x,
                            tf_pose.pose.position.y,
                            tf_pose.pose.position.z],
                       rot=[tf_pose.pose.orientation.w,
                            tf_pose.pose.orientation.x,
                            tf_pose.pose.orientation.y,
                            tf_pose.pose.orientation.z])
