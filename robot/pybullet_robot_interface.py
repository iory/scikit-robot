import pybullet as p
import numpy as np

from robot.math import quaternion2rpy


class PybulletRobotInterface(object):

    def __init__(self, robot, urdf_path=None, *args, **kwargs):
        super(PybulletRobotInterface, self).__init__(*args, **kwargs)
        if urdf_path is None:
            if robot.urdf_path is not None:
                urdf_path = robot.urdf_path
            else:
                raise ValueError('urdf_path should be given.')
        self.robot = robot
        self.robot_id = p.loadURDF(urdf_path, [0, 0, 0])
        self.load_bullet()
        self.realtime_simualtion = False

    def load_bullet(self):
        joint_num = p.getNumJoints(self.robot_id)
        joint_ids = [None] * joint_num
        for i in range(len(joint_ids)):
            joint_name = p.getJointInfo(self.robot_id, i)[1]
            try:
                idx = self.robot.joint_names.index(joint_name.decode('utf-8'))
            except ValueError:
                continue
            if idx != -1:
                joint_ids[idx] = i
        self.joint_ids = joint_ids

        self.force = 200
        self.max_velcity = 1.0
        self.position_gain = 0.1
        self.target_velocity = 0.0
        self.velocity_gain = 0.1

    def angle_vector(self, angle_vector=None, realtime_simualtion=None):
        if realtime_simualtion is not None and isinstance(realtime_simualtion, bool):
            self.realtime_simualtion = realtime_simualtion

        if self.robot_id is None:
            return self.robot.angle_vector()
        if angle_vector is None:
            angle_vector = self.robot.angle_vector()

        for idx, angle in zip(self.joint_ids, angle_vector):
            if idx is None:
                continue

            if self.realtime_simualtion:
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=np.deg2rad(angle),
                                        targetVelocity=self.target_velocity,
                                        force=self.force,
                                        positionGain=self.position_gain,
                                        velocityGain=self.velocity_gain,
                                        maxVelocity=self.max_velcity)
            else:
                p.resetJointState(self.robot_id, idx, np.deg2rad(angle))

        return angle_vector

    def wait_interpolation(self, thresh=0.05):
        while True:
            p.stepSimulation()
            wait = False
            for idx in self.joint_ids:
                if idx is None:
                    continue
                _, velocity, _, _ = p.getJointState(self.robot_id,
                                                    idx)
                if velocity > thresh:
                    wait = True
            if wait is False:
                break
        return True

    def sync(self):
        if self.robot_id is None:
            return self.angle_vector()

        for idx, joint in zip(self.joint_ids, self.robot.joint_list):
            if idx is None:
                continue
            joint_state = p.getJointState(self.robot_id,
                                          idx)
            joint.joint_angle(np.rad2deg(joint_state[0]))
        pos, orientation = p.getBasePositionAndOrientation(self.robot_id)
        rpy, _ = quaternion2rpy([orientation[3], orientation[0],
                                 orientation[1], orientation[2]])
        self.robot.root_link.newcoords(np.array([rpy[0], rpy[1], rpy[2]]),
                                       pos=pos)
        return self.angle_vector()
