import geometry_msgs.msg
import rospy
import tf
import tf2_ros


class TransformListener(object):

    """ROS TF Listener class

    """

    def __init__(self, use_tf2=True):
        if use_tf2:
            try:
                self.tf_listener = tf2_ros.BufferClient("/tf2_buffer_server")
                ok = self.tf_listener.wait_for_server(rospy.Duration(10))
                if not ok:
                    raise Exception(
                        "timed out: wait_for_server for 10.0 seconds")
            except Exception as e:
                rospy.logerr("Failed to initialize tf2 client: %s" % str(e))
                rospy.logwarn("Fallback to tf client")
                use_tf2 = False
        if not use_tf2:
            self.tf_listener = tf.TransformListener()
        self.use_tf2 = use_tf2

    def _wait_for_transform_tf1(self,
                                target_frame, source_frame,
                                time, timeout):
        try:
            self.tf_listener.waitForTransform(
                target_frame, source_frame, time, timeout)
            return True
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
                rospy.exceptions.ROSTimeMovedBackwardsException):
            return False

    def _wait_for_transform_tf2(self,
                                target_frame, source_frame,
                                time, timeout):
        try:
            ret = self.tf_listener.can_transform(
                target_frame, source_frame, time, timeout, True)
            if ret[0] > 0:
                return True
            else:
                raise Exception(ret[1])
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TransformException,
                rospy.exceptions.ROSTimeMovedBackwardsException):
            return False

    def wait_for_transform(self,
                           target_frame, source_frame,
                           time, timeout=rospy.Duration(0)):
        if self.use_tf2:
            ret = self._wait_for_transform_tf2(
                target_frame, source_frame, time, timeout)
        else:
            ret = self._wait_for_transform_tf1(
                target_frame, source_frame, time, timeout)
        return ret

    def _lookup_transform_tf1(self, target_frame, source_frame, time, timeout):
        self._wait_for_transform_tf1(target_frame, source_frame, time, timeout)
        try:
            res = self.tf_listener.lookupTransform(
                target_frame, source_frame, time)
            if time.is_zero():
                time = self.tf_listener.getLatestCommonTime(
                    target_frame, source_frame)
        except (tf.LookupException,
                tf.ConnectivityException,
                tf.ExtrapolationException,
                rospy.exceptions.ROSTimeMovedBackwardsException):
            return False
        ret = geometry_msgs.msg.TransformStamped()
        ret.header.frame_id = target_frame
        ret.header.stamp = time
        ret.child_frame_id = source_frame
        ret.transform.translation.x = res[0][0]
        ret.transform.translation.y = res[0][1]
        ret.transform.translation.z = res[0][2]
        ret.transform.rotation.x = res[1][0]
        ret.transform.rotation.y = res[1][1]
        ret.transform.rotation.z = res[1][2]
        ret.transform.rotation.w = res[1][3]
        return ret

    def _lookup_transform_tf2(self, target_frame, source_frame, time, timeout):
        try:
            return self.tf_listener.lookup_transform(
                target_frame, source_frame, time, timeout)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TransformException,
                rospy.exceptions.ROSTimeMovedBackwardsException):
            return False

    def lookup_transform(self,
                         target_frame,
                         source_frame,
                         time=rospy.Time(0),
                         timeout=rospy.Duration(0)):
        if self.use_tf2:
            ret = self._lookup_transform_tf2(
                target_frame, source_frame, time, timeout)
        else:
            ret = self._lookup_transform_tf1(
                target_frame, source_frame, time, timeout)
        return ret
