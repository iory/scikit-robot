import os
import tempfile
import unittest

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
  <link name="base_link"/>
  <link name="arm"/>
  <link name="tip"/>
  <joint name="hinge" type="revolute">
    <parent link="base_link"/><child link="arm"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.0" upper="3.0" effort="1" velocity="1"/>
  </joint>
  <joint name="tipj" type="fixed">
    <parent link="arm"/><child link="tip"/>
    <origin xyz="{tip_xyz}" rpy="0 0 0"/>
  </joint>
</robot>
"""


def _four_bar(tmp, parallelogram):
    """Ground + two cranks + coupler, loop-closed at the coupler tip.

    ``parallelogram=False`` shortens the second crank and shifts its
    ground pivot (still closing at the zero pose), so the coupling is
    nonlinear and no exact ``<mimic>`` is emitted -- the solver is the
    only thing that can close it.
    """
    def write(name, content):
        path = os.path.join(tmp, name + '.urdf')
        with open(path, 'w') as f:
            f.write(content)
        return RobotModule.from_urdf(name, path)

    g2_xyz = '0.2 0 0' if parallelogram else '0.2 0.05 0'
    crank2_tip = '0 0.1 0' if parallelogram else '0 0.05 0'
    ground = write('ground', _GROUND_URDF.format(g2_xyz=g2_xyz))
    crank = write('crank', _BAR_URDF.format(tip_xyz='0 0.1 0'))
    crank2 = write('crank2', _BAR_URDF.format(tip_xyz=crank2_tip))
    coupler = write('coupler', _BAR_URDF.format(tip_xyz='0.2 0 0'))
    assembly = RobotAssembly('fourbar')
    assembly.add_module_instance('g', ground)
    assembly.add_module_instance('c1', crank)
    assembly.add_module_instance('c2', crank2)
    assembly.add_module_instance('cp', coupler)
    assembly.connect('g', 'g1', 'c1', 'base_link')
    assembly.connect('g', 'g2', 'c2', 'base_link')
    assembly.connect('c1', 'tip', 'cp', 'base_link')
    assembly.connect('cp', 'tip', 'c2', 'tip', loop=True)
    assembly.set_root('g', 'base_link')
    return assembly


class TestLoopClosureSolver(unittest.TestCase):

    def test_general_four_bar_closes_over_a_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = _four_bar(tmp, parallelogram=False)
            robot = assembly.build_robot_model()
        solver = LoopClosureSolver(robot, assembly.loop_closures)
        coupler_length = 0.2
        # this linkage's short crank can follow the driver up to
        # theta ~ 0.49 rad; sweep inside that reachable arc
        for angle in np.linspace(0.0, 0.45, 6):
            robot.c1_hinge.joint_angle(angle)
            error = solver.solve()
            self.assertLess(error, 1e-9)
            # independent invariant: the loop is closed exactly when the
            # two crank tips stay one coupler length apart
            span = np.linalg.norm(robot.c1_tip.worldpos()
                                  - robot.c2_tip.worldpos())
            self.assertAlmostEqual(span, coupler_length, places=9)
        # the coupling is nonlinear, so the followers must have moved to
        # values a linear mimic could not produce
        self.assertNotAlmostEqual(robot.cp_hinge.joint_angle(), -0.45,
                                  places=3)
        self.assertGreater(abs(robot.cp_hinge.joint_angle()), 1e-3)

    def test_parallelogram_agrees_with_injected_mimic(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = _four_bar(tmp, parallelogram=True)
            robot = assembly.build_robot_model()
        solver = LoopClosureSolver(robot, assembly.loop_closures)
        robot.c1_hinge.joint_angle(0.4)
        error = solver.solve()
        self.assertLess(error, 1e-8)
        self.assertAlmostEqual(robot.cp_hinge.joint_angle(), -0.4, places=7)
        self.assertAlmostEqual(robot.c2_hinge.joint_angle(), 0.4, places=7)

    def test_unreachable_driver_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = _four_bar(tmp, parallelogram=False)
            robot = assembly.build_robot_model()
        solver = LoopClosureSolver(robot, assembly.loop_closures)
        # the short crank cannot follow the long one this far around
        robot.c1_hinge.joint_angle(2.5)
        with self.assertRaisesRegex(ValueError, 'did not converge'):
            solver.solve()

    def test_from_yaml_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = _four_bar(tmp, parallelogram=False)
            assembly.build(output_path=os.path.join(tmp, 'fourbar.urdf'))
            robot = assembly.build_robot_model()
            solver = LoopClosureSolver.from_yaml(
                robot, os.path.join(tmp, 'loop_closures.yaml'))
            robot.c1_hinge.joint_angle(0.3)
            self.assertLess(solver.solve(), 1e-8)

    def test_failed_solve_does_not_poison_warm_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = _four_bar(tmp, parallelogram=False)
            robot = assembly.build_robot_model()
        solver = LoopClosureSolver(robot, assembly.loop_closures)
        robot.c1_hinge.joint_angle(0.3)
        self.assertLess(solver.solve(), 1e-9)
        robot.c1_hinge.joint_angle(2.5)
        with self.assertRaises(ValueError):
            solver.solve()
        # the failed target must not have become the warm start: a
        # reachable target afterwards still solves
        robot.c1_hinge.joint_angle(0.3)
        self.assertLess(solver.solve(), 1e-9)

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            assembly = _four_bar(tmp, parallelogram=False)
            robot = assembly.build_robot_model()
        config = assembly.loop_closures
        with self.assertRaisesRegex(ValueError, 'no closure config'):
            LoopClosureSolver(robot, None)
        with self.assertRaises(ValueError):
            LoopClosureSolver(robot, {'closures': []})
        broken = dict(config, dependent=['nope'])
        with self.assertRaisesRegex(ValueError, 'do not exist'):
            LoopClosureSolver(robot, broken)
        solver = LoopClosureSolver(robot, config)
        with self.assertRaises(ValueError):
            solver.solve(max_step=0.0)


if __name__ == '__main__':
    unittest.main()
