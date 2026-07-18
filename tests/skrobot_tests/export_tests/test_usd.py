import os
import shutil
import tempfile
import unittest
import warnings

import numpy as np


try:
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False

import skrobot
from skrobot.export.usd import urdf_to_usd


KUKA_URDF = os.path.join(
    os.path.dirname(skrobot.__file__),
    'data', 'kuka_description', 'kuka.urdf')
KUKA_PRIMITIVE_LINKS = [
    'lbr_iiwa_with_wsg50__base_link',
    'lbr_iiwa_with_wsg50__left_finger',
    'lbr_iiwa_with_wsg50__right_finger',
]


def _write_text(path, text):
    with open(path, 'w') as f:
        f.write(text)


def _link_collision_prims(stage, link_name):
    prefix = '/robot/{}/'.format(link_name)
    return [p for p in stage.Traverse()
            if p.GetPath().pathString.startswith(prefix)
            and p.HasAPI(UsdPhysics.CollisionAPI)]


def _link_visual_prims(stage, link_name):
    prefix = '/robot/{}/'.format(link_name)
    return [p for p in stage.Traverse()
            if p.GetPath().pathString.startswith(prefix)
            and p.GetName().startswith('visual_')]


def _world_extent(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise AssertionError('missing prim at path {}'.format(prim_path))
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=False,
        ignoreVisibility=True)
    return np.asarray(
        bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange().GetSize(),
        dtype=float)


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestUrdfToUsd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.out = os.path.join(cls.tmpdir, 'kuka.usdc')
        cls.stage = urdf_to_usd(KUKA_URDF, cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_writes_a_stage(self):
        self.assertTrue(os.path.exists(self.out))
        self.assertIsNotNone(Usd.Stage.Open(self.out))

    def test_articulation_root(self):
        root = self.stage.GetPrimAtPath('/robot')
        self.assertTrue(root.IsValid())
        self.assertTrue(root.HasAPI(UsdPhysics.ArticulationRootAPI))

    def test_base_is_welded_to_the_world(self):
        # A fixed base means a FixedJoint whose body1 is the base link and whose
        # body0 (the world) is left empty.
        fj = UsdPhysics.FixedJoint(
            self.stage.GetPrimAtPath('/robot/joints/base_fixed'))
        self.assertTrue(fj.GetPrim().IsValid())

    def test_movable_joints_become_drives(self):
        revolute = [p for p in self.stage.Traverse()
                    if p.GetTypeName() == 'PhysicsRevoluteJoint']
        self.assertGreater(len(revolute), 0)
        # every movable joint gets a position drive
        for j in revolute:
            self.assertTrue(j.HasAPI(UsdPhysics.DriveAPI)
                            or j.GetAttribute(
                                'drive:angular:physics:stiffness').IsValid())

    def test_home_positions_set_the_drive_target(self):
        joint = 'lbr_iiwa_with_wsg50__J1'
        stage = urdf_to_usd(KUKA_URDF,
                            os.path.join(self.tmpdir, 'kuka_home.usdc'),
                            home_positions={joint: 0.5})
        j = stage.GetPrimAtPath('/robot/joints/' + joint)
        self.assertTrue(j.IsValid())
        target = j.GetAttribute('drive:angular:physics:targetPosition').Get()
        # USD angular targets are in DEGREES
        self.assertAlmostEqual(float(target), np.rad2deg(0.5), places=3)

    def test_sanitized_prim_paths(self):
        # names with URDF-legal but USD-illegal characters must not appear raw
        for prim in self.stage.Traverse():
            self.assertNotIn('-', prim.GetName())
            self.assertNotIn('.', prim.GetName())

    def test_kuka_primitive_links_have_visual_and_collision_prims(self):
        for link in KUKA_PRIMITIVE_LINKS:
            self.assertGreater(len(_link_visual_prims(self.stage, link)), 0)
            self.assertGreater(len(_link_collision_prims(self.stage, link)), 0)

    def test_kuka_fingers_have_collision_api(self):
        for link in ('lbr_iiwa_with_wsg50__left_finger',
                     'lbr_iiwa_with_wsg50__right_finger'):
            self.assertGreater(len(_link_collision_prims(self.stage, link)), 0)


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestSilentWrongSceneGuards(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_urdf(self, name, body):
        path = os.path.join(self.tmpdir, name)
        _write_text(path, body)
        return path

    def test_primitive_shapes_emit_with_expected_extents(self):
        urdf = """\
<robot name="primitive_shapes">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.4 0.6"/>
      </geometry>
    </visual>
    <visual>
      <origin xyz="1.0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </visual>
    <visual>
      <origin xyz="-1.0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.2 0.4 0.6"/>
      </geometry>
    </collision>
    <collision>
      <origin xyz="1.0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.3"/>
      </geometry>
    </collision>
    <collision>
      <origin xyz="-1.0 0 0" rpy="0 0 0"/>
      <geometry>
        <sphere radius="0.07"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            stage = urdf_to_usd(
                self._write_urdf('primitive_shapes.urdf', urdf),
                os.path.join(self.tmpdir, 'primitive_shapes.usdc'))
        msgs = [str(w.message) for w in records]
        self.assertFalse(any('primitive' in m and 'omitted' in m for m in msgs))

        self.assertEqual(
            stage.GetPrimAtPath('/robot/base_link/visual_0').GetTypeName(),
            'Cube')
        self.assertEqual(
            stage.GetPrimAtPath('/robot/base_link/visual_1').GetTypeName(),
            'Cylinder')
        self.assertEqual(
            stage.GetPrimAtPath('/robot/base_link/visual_2').GetTypeName(),
            'Sphere')
        for i in range(3):
            prim = stage.GetPrimAtPath('/robot/base_link/collision_{}'.format(i))
            self.assertTrue(prim.IsValid())
            self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        np.testing.assert_allclose(
            _world_extent(stage, '/robot/base_link/collision_0'),
            np.array([0.2, 0.4, 0.6]), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(
            _world_extent(stage, '/robot/base_link/collision_1'),
            np.array([0.1, 0.1, 0.3]), atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(
            _world_extent(stage, '/robot/base_link/collision_2'),
            np.array([0.14, 0.14, 0.14]), atol=1e-6, rtol=0.0)

    def test_unsupported_joint_type_warns_and_is_welded(self):
        urdf = """\
<robot name="unsupported_joint">
  <link name="base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <link name="child">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="j_planar" type="planar">
    <parent link="base"/>
    <child link="child"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""
        with self.assertWarns(RuntimeWarning) as ctx:
            stage = urdf_to_usd(
                self._write_urdf('unsupported_joint.urdf', urdf),
                os.path.join(self.tmpdir, 'unsupported_joint.usdc'))
        self.assertIn('j_planar', str(ctx.warning))
        self.assertIn('planar', str(ctx.warning))
        self.assertIn('fixed weld', str(ctx.warning))
        j = stage.GetPrimAtPath('/robot/joints/j_planar')
        self.assertTrue(j.IsValid())
        self.assertEqual(j.GetTypeName(), 'PhysicsFixedJoint')

    def test_warns_when_inertia_recompute_requested_but_fails(self):
        from skrobot.export import usd as usd_mod
        original = usd_mod._inertia_from_mesh

        def _mock_failure(link, mass, return_reason=False):
            del link, mass
            if return_reason:
                return None, 'synthetic mesh measurement failure'
            return None

        usd_mod._inertia_from_mesh = _mock_failure
        self.addCleanup(setattr, usd_mod, '_inertia_from_mesh', original)

        urdf = """\
<robot name="inertia_recompute_warning">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
</robot>
"""
        with self.assertWarns(RuntimeWarning) as ctx:
            urdf_to_usd(
                self._write_urdf('inertia_warn.urdf', urdf),
                os.path.join(self.tmpdir, 'inertia_warn.usdc'),
                recompute_inertia=True)
        msg = str(ctx.warning)
        self.assertIn('recompute_inertia=True', msg)
        self.assertIn('base_link', msg)
        self.assertIn('synthetic mesh measurement failure', msg)

    def test_no_inertia_warning_for_meshless_links(self):
        from skrobot.export import usd as usd_mod
        original = usd_mod._inertia_from_mesh

        def _mock_no_mesh(link, mass, return_reason=False):
            del link, mass
            if return_reason:
                return None, usd_mod._NO_MESH_INERTIA_REASON
            return None

        usd_mod._inertia_from_mesh = _mock_no_mesh
        self.addCleanup(setattr, usd_mod, '_inertia_from_mesh', original)

        urdf = """\
<robot name="inertia_no_mesh">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
  </link>
</robot>
"""
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter('always')
            urdf_to_usd(
                self._write_urdf('inertia_no_mesh.urdf', urdf),
                os.path.join(self.tmpdir, 'inertia_no_mesh.usdc'),
                recompute_inertia=True)
        msgs = [str(w.message) for w in records]
        self.assertFalse(any('recompute_inertia=True' in m for m in msgs))


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
