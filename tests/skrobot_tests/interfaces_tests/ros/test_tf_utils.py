import unittest

from numpy import testing

from skrobot.coordinates import Coordinates
try:
    ModuleNotFoundError
except NameError:
    ModuleNotFoundError = ImportError
try:
    import geometry_msgs.msg

    from skrobot.interfaces.ros.tf_utils import coords_to_geometry_pose
    from skrobot.interfaces.ros.tf_utils import coords_to_tf_pose
    from skrobot.interfaces.ros.tf_utils import geometry_pose_to_coords
    from skrobot.interfaces.ros.tf_utils import tf_pose_to_coords
    _ros_available = True
except ModuleNotFoundError:
    _ros_available = False


class TestTFUtils(unittest.TestCase):

    @unittest.skipUnless(_ros_available, 'ROS is not available.')
    def test_coords_to_geometry_pose(self):
        c = Coordinates(pos=(1, 2, 3))
        pose = coords_to_geometry_pose(c)
        testing.assert_equal(
            [pose.position.x, pose.position.y, pose.position.z], (1, 2, 3))
        testing.assert_equal(
            [pose.orientation.w, pose.orientation.x,
             pose.orientation.y, pose.orientation.z], (1, 0, 0, 0))

    @unittest.skipUnless(_ros_available, 'ROS is not available.')
    def test_coords_to_tf_pose(self):
        c = Coordinates(pos=(1, 2, 3))
        pose = coords_to_tf_pose(c)
        testing.assert_equal(
            [pose.translation.x, pose.translation.y, pose.translation.z],
            (1, 2, 3))
        testing.assert_equal(
            [pose.rotation.w, pose.rotation.x,
             pose.rotation.y, pose.rotation.z], (1, 0, 0, 0))

    @unittest.skipUnless(_ros_available, 'ROS is not available.')
    def test_tf_pose_to_coords(self):
        pose = geometry_msgs.msg.Transform()
        c = tf_pose_to_coords(pose)
        testing.assert_equal(c.translation, (0, 0, 0))
        testing.assert_equal(c.quaternion, (1, 0, 0, 0))

        pose_stamped = geometry_msgs.msg.TransformStamped()
        c = tf_pose_to_coords(pose_stamped)
        testing.assert_equal(c.translation, (0, 0, 0))
        testing.assert_equal(c.quaternion, (1, 0, 0, 0))

    @unittest.skipUnless(_ros_available, 'ROS is not available.')
    def test_geometry_pose_to_coords(self):
        pose = geometry_msgs.msg.Pose()
        c = geometry_pose_to_coords(pose)
        testing.assert_equal(c.translation, (0, 0, 0))
        testing.assert_equal(c.quaternion, (1, 0, 0, 0))

        pose_stamped = geometry_msgs.msg.Pose()
        c = geometry_pose_to_coords(pose_stamped)
        testing.assert_equal(c.translation, (0, 0, 0))
        testing.assert_equal(c.quaternion, (1, 0, 0, 0))
