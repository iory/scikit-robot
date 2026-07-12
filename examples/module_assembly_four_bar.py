#!/usr/bin/env python
"""Assemble a closed-loop four-bar linkage from bare link modules.

Builds a four-bar from four bare bar modules (no movable internal
joints): the hinges are movable CONNECTIONS
(``connect(joint_type='revolute')``) and the loop is closed with a cut
edge (``connect(loop=True)``).  Two variants are shown:

* a parallelogram -- ``build()`` certifies the geometry and writes
  exact ``<mimic>`` tags, so the loop stays closed in any URDF viewer;
* a general four-bar -- the coupling is nonlinear, so the loop is
  closed at runtime with ``skrobot.kinematics.LoopClosureSolver``
  (the ``loop_closures.yaml`` sidecar carries the same constraint for
  ROS-side consumers).

Usage:
    python examples/module_assembly_four_bar.py
    python examples/module_assembly_four_bar.py --viewer
"""

import argparse
import os
import tempfile
import time

import numpy as np

from skrobot.kinematics import LoopClosureSolver
from skrobot.urdf import RobotAssembly
from skrobot.urdf import RobotModule


_GROUND_URDF = """<?xml version="1.0"?>
<robot name="ground">
  <link name="base_link"/>
  <link name="g1"/>
  <link name="g2"/>
  <joint name="fg1" type="fixed">
    <parent link="base_link"/><child link="g1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="fg2" type="fixed">
    <parent link="base_link"/><child link="g2"/>
    <origin xyz="{g2_xyz}" rpy="0 0 0"/>
  </joint>
</robot>
"""

_BAR_URDF = """<?xml version="1.0"?>
<robot name="bar">
  <link name="base_link">
    <visual>
      <origin xyz="{half_xyz}" rpy="0 0 0"/>
      <geometry><box size="{box_size}"/></geometry>
    </visual>
  </link>
  <link name="tip"/>
  <joint name="ext" type="fixed">
    <parent link="base_link"/><child link="tip"/>
    <origin xyz="{tip_xyz}" rpy="0 0 0"/>
  </joint>
</robot>
"""


def write_bar(directory, name, length, along):
    """Write a single-link bar module and parse it.

    Parameters
    ----------
    directory : str
        Where to write the URDF file.
    name : str
        Module name (also the file stem).
    length : float
        Bar length in metres.
    along : str
        ``'x'`` or ``'y'``: the axis the bar extends along.

    Returns
    -------
    skrobot.urdf.RobotModule
        The parsed module.
    """
    if along == 'x':
        tip = (length, 0.0, 0.0)
        box = (length, 0.01, 0.01)
    else:
        tip = (0.0, length, 0.0)
        box = (0.01, length, 0.01)
    half = tuple(v / 2.0 for v in tip)
    path = os.path.join(directory, name + '.urdf')
    with open(path, 'w') as f:
        f.write(_BAR_URDF.format(
            tip_xyz=' '.join(str(v) for v in tip),
            half_xyz=' '.join(str(v) for v in half),
            box_size=' '.join(str(v) for v in box)))
    return RobotModule.from_urdf(name, path)


def assemble_four_bar(directory, crank2_length, g2_xyz):
    """Assemble ground + two cranks + coupler, loop-closed at the tips.

    With ``crank2_length=0.1`` and ``g2_xyz='0.2 0 0'`` the hinge
    quadrilateral is a parallelogram; other values give a general
    four-bar (as long as the zero pose still closes).

    Parameters
    ----------
    directory : str
        Where the module URDFs are written.
    crank2_length : float
        Length of the second crank in metres.
    g2_xyz : str
        Ground pivot of the second crank ("x y z").

    Returns
    -------
    skrobot.urdf.RobotAssembly
        The assembled linkage.
    """
    ground_path = os.path.join(directory, 'ground.urdf')
    with open(ground_path, 'w') as f:
        f.write(_GROUND_URDF.format(g2_xyz=g2_xyz))
    ground = RobotModule.from_urdf('ground', ground_path)
    crank1 = write_bar(directory, 'crank1', 0.1, 'y')
    crank2 = write_bar(directory, 'crank2', crank2_length, 'y')
    coupler = write_bar(directory, 'coupler', 0.2, 'x')

    assembly = RobotAssembly('fourbar')
    assembly.add_module_instance('g', ground)
    assembly.add_module_instance('c1', crank1)
    assembly.add_module_instance('c2', crank2)
    assembly.add_module_instance('cp', coupler)
    hinge = {'joint_type': 'revolute', 'lower': -3.0, 'upper': 3.0}
    assembly.connect('g', 'g1', 'c1', 'base_link', **hinge)
    assembly.connect('g', 'g2', 'c2', 'base_link', **hinge)
    assembly.connect('c1', 'tip', 'cp', 'base_link', **hinge)
    # the cut edge: never a URDF joint, exported as a closure constraint
    assembly.connect('cp', 'tip', 'c2', 'tip', loop=True)
    assembly.set_root('g', 'base_link')
    return assembly


def main():
    parser = argparse.ArgumentParser(
        description='Closed-loop module assembly demo')
    parser.add_argument('--viewer', action='store_true',
                        help='animate the general four-bar in a viewer')
    parser.add_argument('--no-interactive', action='store_true',
                        help='run in non-interactive mode (CI testing)')
    args = parser.parse_args()
    if args.no_interactive:
        args.viewer = False

    tmp = tempfile.mkdtemp(prefix='fourbar_demo_')

    print('=== parallelogram: exact <mimic>, no solver needed ===')
    parallelogram = assemble_four_bar(tmp, 0.1, '0.2 0 0')
    urdf_path = parallelogram.build(
        output_path=os.path.join(tmp, 'parallelogram.urdf'))
    print(f'built: {urdf_path}')
    print(f'sidecar: {os.path.join(tmp, "loop_closures.yaml")}')
    print(f'closure config: {parallelogram.loop_closures}')

    print()
    print('=== general four-bar: closed by LoopClosureSolver ===')
    general = assemble_four_bar(tmp, 0.05, '0.2 0.05 0')
    robot = general.build_robot_model()
    solver = LoopClosureSolver(robot, general.loop_closures)
    driver = robot.g_g1_to_c1_base_link_joint
    for angle in np.linspace(0.0, 0.45, 4):
        driver.joint_angle(angle)
        error = solver.solve()
        follower = robot.g_g2_to_c2_base_link_joint.joint_angle()
        print(f'crank {angle:+.3f} rad -> rocker {follower:+.3f} rad '
              f'(closure error {error:.2e} m)')

    if args.viewer:
        from skrobot.viewers import TrimeshSceneViewer
        viewer = TrimeshSceneViewer()
        viewer.add(robot)
        viewer.show()
        for angle in np.tile(np.concatenate([
                np.linspace(0.0, 0.45, 30),
                np.linspace(0.45, 0.0, 30)]), 10):
            driver.joint_angle(angle)
            solver.solve()
            viewer.redraw()
            time.sleep(0.03)


if __name__ == '__main__':
    main()
