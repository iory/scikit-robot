import math
import os
import tempfile
import unittest

from skrobot.collision import is_fcl_available
from skrobot.collision import link_meshes
from skrobot.collision import SelfCollision
from skrobot.collision import sweep_limits


# A bar hinged 0.1 m up on a pedestal, with a separate pillar standing on the
# pedestal under the bar's reach.  Parent/child pairs are baseline-excluded,
# so the limit comes from the NON-adjacent bar-vs-pillar contact: the bar's
# lower edge (half thickness 0.01) meets the pillar top (z = 0.06) at
#   tan(theta) = (0.1 - 0.06 - 0.01) / 0.25  ->  theta ~ 6.8 deg.
# Tilting up is free (the pillar is below), so only the upper direction hits.
_URDF = """<?xml version="1.0"?>
<robot name="hinged_bar">
  <link name="base_link">
    <visual>
      <geometry><box size="0.1 0.1 0.1"/></geometry>
    </visual>
  </link>
  <link name="pillar">
    <visual>
      <origin xyz="0.25 0 0.03" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.06"/></geometry>
    </visual>
  </link>
  <link name="bar">
    <visual>
      <origin xyz="0.15 0 0" rpy="0 0 0"/>
      <geometry><box size="0.3 0.02 0.02"/></geometry>
    </visual>
  </link>
  <joint name="mount" type="fixed">
    <parent link="base_link"/>
    <child link="pillar"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="pitch" type="revolute">
    <parent link="base_link"/>
    <child link="bar"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _load_robot():
    from skrobot.models.urdf import RobotModelFromURDF
    with tempfile.NamedTemporaryFile('w', suffix='.urdf',
                                     delete=False) as f:
        f.write(_URDF)
        path = f.name
    try:
        return RobotModelFromURDF(urdf_file=path)
    finally:
        os.unlink(path)


@unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
class TestSelfCollision(unittest.TestCase):

    def test_no_new_pairs_at_home(self):
        robot = _load_robot()
        sc = SelfCollision(robot, link_meshes(robot))
        self.assertEqual(sc.new_pairs(), set())

    def test_collision_detected_when_bar_dips(self):
        robot = _load_robot()
        meshes = link_meshes(robot)
        self.assertEqual(sorted(meshes), ['bar', 'base_link', 'pillar'])
        sc = SelfCollision(robot, meshes, hull=False)
        # adjacent (parent/child) pairs are baseline-excluded by design
        self.assertIn(frozenset(('base_link', 'bar')), sc.baseline)
        robot.pitch.joint_angle(math.radians(15.0))
        self.assertIn(frozenset(('bar', 'pillar')), sc.new_pairs())


@unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
class TestSweepLimits(unittest.TestCase):

    def test_limit_found_near_analytic_angle(self):
        robot = _load_robot()
        result = sweep_limits(robot, link_meshes(robot),
                              step_deg=2.0, margin_deg=1.0)
        self.assertIn('pitch', result)
        limits = result['pitch']
        # the bar meets the pillar top at ~6.8 deg; the reported limit backs
        # off by the margin, so expect it inside (2, 6.8 + tol) degrees
        expected = math.atan((0.1 - 0.06 - 0.01) / 0.25)
        self.assertGreater(limits['upper'], math.radians(2.0))
        self.assertLess(limits['upper'], expected + math.radians(0.5))
        self.assertEqual(limits['hit_upper'], ('bar', 'pillar'))
        self.assertFalse(limits['continuous'])
        # the free direction swings far past the down-side limit
        self.assertLess(limits['lower'], -math.radians(90.0))
        # the sweep must restore the pre-call pose
        self.assertAlmostEqual(float(robot.pitch.joint_angle()), 0.0)

    def test_parts_based_model_is_rejected(self):
        robot = _load_robot()
        meshes = link_meshes(robot)
        sc = SelfCollision(robot, meshes,
                           parts={'pillar': [meshes['pillar']]})
        with self.assertRaises(ValueError):
            sweep_limits(robot, meshes, sc=sc)


if __name__ == '__main__':
    unittest.main()
