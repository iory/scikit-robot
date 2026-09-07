import os
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from skrobot.collision import collision_model as collision_model_module
from skrobot.collision import is_fcl_available
from skrobot.collision import kinematic_distance
from skrobot.collision import RobotCollisionModel


# A column fixed on a base carries a bar on a pitch joint. A pad is fixed to
# the column but sits overlapping the base, so base/pad is a NON-adjacent
# pair (base - column - pad) that touches by design at the default pose and
# must be excluded. The bar is clear of everything at rest and only reaches
# the base when pitched down, so base/bar must be kept. Every other pair is
# parent/child.
_URDF = """<?xml version="1.0"?>
<robot name="column_bar">
  <link name="base_link">
    <collision>
      <geometry><box size="0.2 0.2 0.1"/></geometry>
    </collision>
  </link>
  <link name="column">
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 0.3"/></geometry>
    </collision>
  </link>
  <link name="pad">
    <collision>
      <geometry><box size="0.04 0.04 0.04"/></geometry>
    </collision>
  </link>
  <link name="bar">
    <collision>
      <origin xyz="0.175 0 0" rpy="0 0 0"/>
      <geometry><box size="0.35 0.02 0.02"/></geometry>
    </collision>
  </link>
  <joint name="mount" type="fixed">
    <parent link="base_link"/>
    <child link="column"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>
  <joint name="pad_mount" type="fixed">
    <parent link="column"/>
    <child link="pad"/>
    <origin xyz="0.08 0 0" rpy="0 0 0"/>
  </joint>
  <joint name="pitch" type="revolute">
    <parent link="column"/>
    <child link="bar"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
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


class TestKinematicDistance(unittest.TestCase):

    def test_distances_along_the_tree(self):
        robot = _load_robot()
        self.assertEqual(kinematic_distance(robot.base_link, robot.column), 1)
        self.assertEqual(kinematic_distance(robot.column, robot.bar), 1)
        self.assertEqual(kinematic_distance(robot.base_link, robot.pad), 2)
        self.assertEqual(kinematic_distance(robot.pad, robot.bar), 2)
        self.assertEqual(kinematic_distance(robot.base_link, robot.bar), 2)
        self.assertEqual(kinematic_distance(robot.bar, robot.bar), 0)


class TestRobotCollisionModelPairs(unittest.TestCase):

    def setUp(self):
        self.robot = _load_robot()
        self.tmpdir = tempfile.mkdtemp()

    def _build(self, **kwargs):
        kwargs.setdefault('cache_dir', self.tmpdir)
        return RobotCollisionModel(self.robot, **kwargs)

    def test_collision_links_are_those_with_a_mesh(self):
        model = self._build()
        self.assertEqual(
            [link.name for link in model.link_list],
            ['base_link', 'column', 'pad', 'bar'])

    def test_adjacent_pairs_are_never_checked(self):
        model = self._build()
        pairs = set(model.self_collision_pair_names)
        for a, b in pairs:
            self.assertGreaterEqual(
                kinematic_distance(getattr(self.robot, a),
                                   getattr(self.robot, b)), 2)
        self.assertEqual(
            set(model.excluded['adjacent']),
            {('base_link', 'column'), ('bar', 'column'), ('column', 'pad')})

    @unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
    def test_pair_touching_at_default_pose_is_excluded_and_recorded(self):
        model = self._build()
        self.assertEqual(model.method, 'fcl')
        self.assertNotIn(('base_link', 'pad'), model.self_collision_pair_names)
        self.assertIn(('base_link', 'pad'), model.excluded['default_pose'])

    @unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
    def test_pair_that_can_collide_is_kept(self):
        model = self._build()
        self.assertIn(('bar', 'base_link'), model.self_collision_pair_names)
        self.assertEqual(model.excluded['always'], [])

    def test_robot_pose_is_restored_after_derivation(self):
        self.robot.pitch.joint_angle(0.3)
        before = self.robot.angle_vector().copy()
        self._build()
        np.testing.assert_allclose(self.robot.angle_vector(), before)

    @unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
    def test_checker_reflects_the_pairs_and_the_pose(self):
        model = self._build()
        checker = model.checker
        self.assertGreater(len(checker.self_collision_pairs), 0)

        self.robot.pitch.joint_angle(0.0)
        self.assertTrue(checker.is_collision_free())
        # Pitched straight down (R_y(+pi/2) sends +x to -z), the bar runs
        # through the base.
        self.robot.pitch.joint_angle(np.pi / 2)
        self.assertFalse(checker.is_collision_free())

    @unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
    def test_in_self_collision_answers_on_the_meshes(self):
        model = self._build()
        self.robot.pitch.joint_angle(0.0)
        self.assertFalse(model.in_self_collision())
        self.assertEqual(model.self_colliding_pairs(), [])
        self.robot.pitch.joint_angle(np.pi / 2)
        self.assertTrue(model.in_self_collision())
        self.assertEqual(model.self_colliding_pairs(),
                         [('bar', 'base_link')])

    def test_exact_query_without_fcl_is_a_clear_error(self):
        with mock.patch.object(collision_model_module, 'is_fcl_available',
                               return_value=False):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = self._build()
            with self.assertRaises(RuntimeError) as ctx:
                model.in_self_collision()
        self.assertIn('python-fcl', str(ctx.exception))

    @unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
    def test_derivation_is_persisted_and_read_back(self):
        first = self._build()
        self.assertFalse(first.from_cache)
        second = self._build()
        self.assertTrue(second.from_cache)
        self.assertEqual(second.self_collision_pair_names,
                         first.self_collision_pair_names)
        self.assertEqual(second.excluded, first.excluded)
        self.assertEqual(second.method, first.method)
        self.assertEqual(second.proxy_overlap_pairs,
                         first.proxy_overlap_pairs)
        self.assertIsInstance(first.proxy_overlap_pairs, list)

        # Different settings are a different entry.
        other = self._build(n_samples=7)
        self.assertFalse(other.from_cache)

        forced = self._build(use_cache=False)
        self.assertFalse(forced.from_cache)

    def test_without_fcl_the_primitive_path_warns_and_says_so(self):
        with mock.patch.object(collision_model_module, 'is_fcl_available',
                               return_value=False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = self._build()
        self.assertEqual(model.method, 'primitive')
        self.assertTrue(any('python-fcl' in str(w.message) for w in caught))
        # The structural rule still holds.
        self.assertEqual(model.excluded['always'], [])
        for a, b in model.self_collision_pair_names:
            self.assertGreaterEqual(
                kinematic_distance(getattr(self.robot, a),
                                   getattr(self.robot, b)), 2)

    @unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
    def test_gridsdf_data_uses_the_derived_pairs(self):
        model = self._build()
        data = model.gridsdf_self_data(dim_grid=12, n_surface=8)
        n_pairs = len(model.self_collision_pair_names)
        # Each unordered pair is looked up in both directions.
        self.assertEqual(len(data['pairs_a']), 2 * n_pairs)
        self.assertEqual(len(data['pairs_b']), 2 * n_pairs)
        self.assertEqual(data['grids'].shape[0], len(model.link_list))
        # Cached per resolution.
        self.assertIs(model.gridsdf_self_data(dim_grid=12, n_surface=8), data)


@unittest.skipUnless(is_fcl_available(), 'python-fcl is not installed')
class TestRobotCollisionModelFetch(unittest.TestCase):
    """A real robot whose links touch by design at rest."""

    @classmethod
    def setUpClass(cls):
        import skrobot
        cls.robot = skrobot.models.Fetch()
        cls.model = cls.robot.build_collision_model(
            cache_dir=tempfile.mkdtemp())

    def test_links_touching_by_design_are_excluded(self):
        self.assertEqual(self.model.method, 'fcl')
        self.assertGreater(len(self.model.excluded['default_pose']), 0)
        for pair in self.model.excluded['default_pose']:
            self.assertNotIn(pair, self.model.self_collision_pair_names)

    def test_no_adjacent_pair_is_checked(self):
        for a, b in self.model.self_collision_pair_names:
            self.assertGreaterEqual(
                kinematic_distance(getattr(self.robot, a),
                                   getattr(self.robot, b)), 2)

    def test_reset_pose_is_self_collision_free_on_the_meshes(self):
        self.robot.reset_pose()
        self.assertFalse(self.model.in_self_collision())

    def test_folding_the_arm_into_the_torso_is_detected(self):
        self.robot.reset_pose()
        # Drive the elbow hard into its limit with the shoulder lifted: the
        # forearm ends up inside the torso.
        self.robot.shoulder_lift_joint.joint_angle(
            self.robot.shoulder_lift_joint.max_angle)
        self.robot.elbow_flex_joint.joint_angle(
            self.robot.elbow_flex_joint.max_angle)
        self.assertTrue(self.model.in_self_collision())

    def test_proxies_are_reported_as_conservative(self):
        # Fetch's bulky base and torso are approximated by a few spheres,
        # so the proxy checker over-reports; the model must say so rather
        # than hide it.
        self.assertGreater(len(self.model.proxy_overlap_pairs), 0)
        self.assertIn('proxies', self.model.describe())


class TestRobotModelIntegration(unittest.TestCase):

    def test_collision_model_is_lazy_and_cached_on_the_robot(self):
        robot = _load_robot()
        tmpdir = tempfile.mkdtemp()
        self.assertIsNone(robot._collision_model)
        model = robot.build_collision_model(cache_dir=tmpdir)
        self.assertIs(robot.collision_model, model)
        rebuilt = robot.build_collision_model(cache_dir=tmpdir)
        self.assertIsNot(rebuilt, model)
        self.assertIs(robot.collision_model, rebuilt)

    def test_collision_model_builds_on_first_access(self):
        robot = _load_robot()
        with mock.patch.object(
                collision_model_module, '_default_cache_dir',
                return_value=tempfile.mkdtemp()):
            model = robot.collision_model
        self.assertIsInstance(model, RobotCollisionModel)
        self.assertIn('self-collision pairs', model.describe())


if __name__ == '__main__':
    unittest.main()
