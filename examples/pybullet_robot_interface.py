#!/usr/bin/env python

import time

import numpy as np
import pybullet

import skrobot


def main():
    # initialize robot
    robot = skrobot.models.Kuka()
    interface = skrobot.interfaces.PybulletRobotInterface(robot)
    pybullet.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-45,
        cameraTargetPosition=(0, 0, 0.5),
    )
    print('==> Initialized Kuka Robot on PyBullet')
    for _ in range(100):
        pybullet.stepSimulation()
    time.sleep(3)

    # reset pose
    print('==> Moving to Reset Pose')
    robot.reset_manip_pose()
    interface.angle_vector(robot.angle_vector(), realtime_simulation=True)
    interface.wait_interpolation()
    time.sleep(3)

    # ik
    print('==> Solving Inverse Kinematics')
    target_coords = skrobot.coordinates.Coordinates(
        pos=[0.5, 0, 0]
    ).rotate(np.pi / 2.0, 'y', 'local')
    skrobot.interfaces.pybullet.draw(target_coords)
    robot.inverse_kinematics(
        target_coords,
        link_list=robot.rarm.link_list,
        move_target=robot.rarm_end_coords,
        rotation_axis=True,
        stop=1000,
    )
    interface.angle_vector(robot.angle_vector(), realtime_simulation=True)
    interface.wait_interpolation()

    # wait
    while pybullet.isConnected():
        time.sleep(0.01)


if __name__ == '__main__':
    main()
