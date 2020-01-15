# scikit-robot: A Flexible Framework for Robot Control in Python

[![Build Status](https://travis-ci.com/iory/scikit-robot.svg?token=zM5rExyvuRoJThsnqHAF&branch=master)](https://travis-ci.com/iory/scikit-robot)

Scikit-Robot is a lightweight pure-Python library for robotic kinematics,
motion planning, visualization and control.

## Installation

```bash
pip install scikit-robot
```

## Features

- [x] Loading robot model from URDF ([examples/robot_models.py](examples/robot_models.py))
- [x] Forward and inverse kinematics ([examples/trimesh_scene_viewer.py](examples/trimesh_scene_viewer.py))
- [x] Collision detection
- [x] Interactive viewer ([examples/trimesh_scene_viewer.py](examples/trimesh_scene_viewer.py))
- [x] Pybullet and ROS command interface ([examples/pybullet_robot_interface.py](examples/pybullet_robot_interface.py))
- [ ] Forward and inverse dynamics
- [ ] Path planning
