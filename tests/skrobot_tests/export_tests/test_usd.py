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
class TestCoacdDecomposition(unittest.TestCase):

    def test_decompose_links_emits_multiple_colliders(self):
        try:
            import coacd  # noqa: F401
        except ImportError:
            self.skipTest('coacd not installed')
        tmpdir = tempfile.mkdtemp()
        try:
            stage = urdf_to_usd(KUKA_URDF, os.path.join(tmpdir, 'kuka.usdc'),
                                decompose_links=['finger'])
            parts = [p for p in stage.Traverse()
                     if p.HasAPI(UsdPhysics.CollisionAPI)
                     and '_part_' in p.GetName()]
            # a decomposed link yields more than one convex collider
            self.assertGreater(len(parts), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_coacd_params_control_fidelity(self):
        try:
            import coacd  # noqa: F401
        except ImportError:
            self.skipTest('coacd not installed')
        tmpdir = tempfile.mkdtemp()
        try:
            # a coarse budget yields fewer parts than a generous one
            coarse = urdf_to_usd(
                KUKA_URDF, os.path.join(tmpdir, 'coarse.usdc'),
                decompose_links=['finger'],
                coacd_params={'quality': 'balanced'})
            fine = urdf_to_usd(
                KUKA_URDF, os.path.join(tmpdir, 'fine.usdc'),
                decompose_links=['finger'],
                coacd_params={'threshold': 0.05, 'max_convex_hull': 32})

            def n_parts(stage):
                return len([p for p in stage.Traverse()
                            if p.HasAPI(UsdPhysics.CollisionAPI)
                            and '_part_' in p.GetName()])
            self.assertGreaterEqual(n_parts(fine), n_parts(coarse))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_hull_inflation_separates_convex_from_concave(self):
        import trimesh

        from skrobot.export.usd import _hull_inflation
        box = trimesh.creation.box((0.1, 0.1, 0.1))
        thin = trimesh.creation.annulus(r_min=0.045, r_max=0.05, height=0.02)
        conv = _hull_inflation(box.vertices.tolist(),
                               box.faces.flatten().tolist())
        conc = _hull_inflation(thin.vertices.tolist(),
                               thin.faces.flatten().tolist())
        self.assertAlmostEqual(conv, 1.0, places=2)   # a box IS its hull
        self.assertGreater(conc, 2.0)                  # a thin ring is not

    def test_decompose_links_true_skips_convex_links(self):
        try:
            import coacd  # noqa: F401
        except ImportError:
            self.skipTest('coacd not installed')
        # decompose_links=True decomposes a link iff its hull inflation exceeds
        # the threshold, so a convex link must fall below it.
        import trimesh

        from skrobot.export.usd import _DECOMPOSE_RATIO
        from skrobot.export.usd import _hull_inflation
        box = trimesh.creation.box((0.1, 0.1, 0.1))
        self.assertLess(
            _hull_inflation(box.vertices.tolist(), box.faces.flatten().tolist()),
            _DECOMPOSE_RATIO)


class TestHullFallbackWarns(unittest.TestCase):
    """A requested decomposition that degrades to a hull must say so.

    A single convex hull fills every cavity, so a scanned room or a gripper
    silently becomes a solid block. Nothing downstream can detect that, which
    is why these paths warn instead of returning None quietly.
    """

    def setUp(self):
        from skrobot.export import usd as usd_mod
        self.usd_mod = usd_mod
        usd_mod._COACD_CACHE.clear()
        self.addCleanup(usd_mod._COACD_CACHE.clear)
        import trimesh
        box = trimesh.creation.box((0.1, 0.1, 0.1))
        self.pts = box.vertices.tolist()
        self.counts = [3] * len(box.faces)
        self.idx = box.faces.flatten().tolist()

    def test_warns_when_coacd_is_missing(self):
        original = self.usd_mod.is_coacd_available
        self.usd_mod.is_coacd_available = lambda: False
        self.addCleanup(setattr, self.usd_mod, 'is_coacd_available', original)
        with self.assertWarns(RuntimeWarning) as ctx:
            parts = self.usd_mod._coacd_parts(
                self.pts, self.counts, self.idx, None)
        self.assertIsNone(parts)
        self.assertIn('coacd', str(ctx.warning))
        self.assertIn('CONVEX HULL', str(ctx.warning))

    def test_warns_when_mesh_is_not_triangulated(self):
        with self.assertWarns(RuntimeWarning):
            parts = self.usd_mod._coacd_parts(
                self.pts, [4] * 6, self.idx, None)
        self.assertIsNone(parts)

    def test_warns_again_on_a_cached_failure(self):
        # the result is cached per mesh; a second link reusing the same mesh
        # must not degrade silently just because the first one already warned
        original = self.usd_mod.is_coacd_available
        self.usd_mod.is_coacd_available = lambda: True
        self.addCleanup(setattr, self.usd_mod, 'is_coacd_available', original)
        original_decompose = self.usd_mod.convex_decomposition

        def _boom(*args, **kwargs):
            raise RuntimeError('coacd exploded')

        self.usd_mod.convex_decomposition = _boom
        self.addCleanup(
            setattr, self.usd_mod, 'convex_decomposition', original_decompose)
        with self.assertWarns(RuntimeWarning) as first:
            self.usd_mod._coacd_parts(self.pts, self.counts, self.idx, None)
        self.assertIn('coacd exploded', str(first.warning))
        with self.assertWarns(RuntimeWarning) as second:
            parts = self.usd_mod._coacd_parts(
                self.pts, self.counts, self.idx, None)
        self.assertIsNone(parts)
        self.assertIn('already failed', str(second.warning))


class TestCoacdCacheKey(unittest.TestCase):

    def setUp(self):
        from skrobot.export import usd as usd_mod
        self.usd_mod = usd_mod
        usd_mod._COACD_CACHE.clear()
        self.addCleanup(usd_mod._COACD_CACHE.clear)

    def test_cache_key_includes_face_topology(self):
        # Same vertices and same triangle count, but different face topology.
        verts = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
        faces_a = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ], dtype=int)
        faces_b = faces_a.copy()
        faces_b[0] = [0, 1, 3]
        faces_b[1] = [1, 2, 3]
        counts = [3] * len(faces_a)

        original_available = self.usd_mod.is_coacd_available
        self.usd_mod.is_coacd_available = lambda: True
        self.addCleanup(
            setattr, self.usd_mod, 'is_coacd_available', original_available)
        original_decompose = self.usd_mod.convex_decomposition
        calls = []

        class _DummyPart(object):

            def __init__(self, marker):
                self.vertices = np.array(
                    [[0.0 + marker, 0.0, 0.0],
                     [1.0 + marker, 0.0, 0.0],
                     [0.0 + marker, 1.0, 0.0],
                     [0.0 + marker, 0.0, 1.0]], dtype=float)
                self.faces = np.array(
                    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=int)
                self.is_watertight = True

        def _fake_decompose(mesh, **kwargs):
            del kwargs
            calls.append(np.asarray(mesh.faces, dtype=int).copy())
            return [_DummyPart(float(len(calls)))]

        self.usd_mod.convex_decomposition = _fake_decompose
        self.addCleanup(
            setattr, self.usd_mod, 'convex_decomposition', original_decompose)

        out_a = self.usd_mod._coacd_parts(
            verts, counts, faces_a.flatten().tolist(), None)
        out_b = self.usd_mod._coacd_parts(
            verts, counts, faces_b.flatten().tolist(), None)
        self.assertIsNotNone(out_a)
        self.assertIsNotNone(out_b)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.usd_mod._COACD_CACHE), 2)
        self.assertNotEqual(out_a[0][0][0][0], out_b[0][0][0][0])


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
                os.path.join(self.tmpdir, 'primitive_shapes.usdc'),
                decompose_links=['base_link'])
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
        self.assertEqual(
            len([p for p in stage.Traverse()
                 if p.HasAPI(UsdPhysics.CollisionAPI)
                 and '_part_' in p.GetName()]),
            0)

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
