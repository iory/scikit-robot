import pybullet as p
import numpy as np

from robot.robot_model import RobotModel
from robot.math import quaternion2rpy


class PybulletRobotInterface(RobotModel):

    def __init__(self, urdf_path, *args, **kwargs):
        super(PybulletRobotInterface, self).__init__(
            *args, **kwargs)
        self.load_urdf(urdf_path)
        self.robot_id = None
        self.realtime_simualtion = False

    def load_bullet(self, p, robot_id):
        joint_num = p.getNumJoints(robot_id)

        self.robot_id = robot_id
        joint_ids = [None] * joint_num
        for i in range(len(joint_ids)):
            joint_name = p.getJointInfo(robot_id, i)[1]
            try:
                idx = self.joint_names.index(joint_name.decode('utf-8'))
            except ValueError:
                continue
            if idx != -1:
                joint_ids[idx] = i
        self.joint_ids = joint_ids

    def send_angle_vector(self, angle_vector=None, realtime_simualtion=None):
        if realtime_simualtion is not None and isinstance(realtime_simualtion, bool):
            self.realtime_simualtion = realtime_simualtion

        if self.robot_id is None:
            return self.angle_vector()
        if angle_vector is None:
            angle_vector = self.angle_vector()

        for idx, angle in zip(self.joint_ids, angle_vector):
            if idx is None:
                continue

            if self.realtime_simualtion:
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=np.deg2rad(angle),
                                        targetVelocity=0.01,
                                        force=100,
                                        positionGain=0.01,
                                        velocityGain=1)
            else:
                p.resetJointState(self.robot_id, idx, np.deg2rad(angle))

        return angle_vector

    def wait_interpolation(self):
        while True:
            p.stepSimulation()
            wait = False
            for idx in self.joint_ids:
                if idx is None:
                    continue
                _, velocity, _, _ = p.getJointState(self.robot_id,
                                                    idx)
                if velocity > 0.05:
                    wait = True
            if wait is False:
                break
        return True

    def sync(self):
        if self.robot_id is None:
            return self.angle_vector()

        for idx, joint in zip(self.joint_ids, self.joint_list):
            if idx is None:
                continue
            joint_state = p.getJointState(self.robot_id,
                                          idx)
            joint.joint_angle(np.rad2deg(joint_state[0]))
        pos, orientation = p.getBasePositionAndOrientation(self.robot_id)
        rpy, _ = quaternion2rpy([orientation[3], orientation[0],
                                 orientation[1], orientation[2]])
        self.root_link.newcoords(np.array([rpy[0], rpy[1], rpy[2]]),
                                 pos=pos)
        return self.angle_vector()
