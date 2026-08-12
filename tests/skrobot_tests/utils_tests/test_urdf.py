import os
import sys
import tempfile
import unittest
from unittest import mock

from lxml import etree
import numpy as np
import pytest

from skrobot.utils import mesh as mesh_utils
from skrobot.utils import urdf as urdf_utils
from skrobot.utils import urdf_mesh
from skrobot.utils.package import is_package_installed


class TestLoadMeshesNoneGuard(unittest.TestCase):

    def test_raises_file_not_found_with_helpful_message(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            urdf_mesh._load_meshes(None)
        msg = str(excinfo.value)
        # The user has to know the cause and the typical fix.
        assert 'package://' in msg
        assert 'install/setup.bash' in msg or 'ROS_PACKAGE_PATH' in msg


class TestDracoCompressedMeshDetection(unittest.TestCase):
    """A Draco-compressed glTF/GLB must be reported when DracoPy is absent.

    Without DracoPy, trimesh does not raise; it silently returns degenerate
    (all-zero) geometry. The loader must detect the Draco extension up front
    and skip the mesh with a clear warning (returning no meshes) instead of
    returning broken geometry. It must NOT raise: a single Draco mesh would
    otherwise abort the entire URDF load.
    """

    def _make_draco_glb(self, path):
        import json
        import struct
        gltf = {
            'asset': {'version': '2.0'},
            'extensionsUsed': ['KHR_draco_mesh_compression'],
            'extensionsRequired': ['KHR_draco_mesh_compression'],
        }
        json_bytes = json.dumps(gltf).encode('utf-8')
        json_bytes += b' ' * ((4 - len(json_bytes) % 4) % 4)
        with open(path, 'wb') as f:
            f.write(struct.pack('<4sII', b'glTF', 2,
                                12 + 8 + len(json_bytes)))
            f.write(struct.pack('<II', len(json_bytes), 0x4E4F534A))
            f.write(json_bytes)

    def test_gltf_uses_draco_detects_extension(self):
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            path = tmp.name
        try:
            self._make_draco_glb(path)
            self.assertTrue(urdf_mesh._gltf_uses_draco(path))
        finally:
            os.remove(path)

    def test_load_skips_and_warns_when_draco_and_no_dracopy(self):
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            path = tmp.name
        try:
            self._make_draco_glb(path)
            # Reset the one-time hint flag so the verbose hint is emitted.
            urdf_mesh._DRACO_MISSING_HINT_SHOWN = False
            with mock.patch(
                    'skrobot.utils.draco.is_dracopy_available',
                    return_value=False):
                with self.assertLogs(
                        'skrobot.utils.urdf_mesh', level='WARNING') as logs:
                    meshes = urdf_mesh._load_meshes(path)
            # The mesh is skipped (no broken geometry returned) but not raised,
            # so the rest of a URDF can still load.
            self.assertEqual(meshes, [])
            joined = '\n'.join(logs.output)
            assert 'DracoPy' in joined
            assert path in joined
        finally:
            os.remove(path)

    def test_plain_glb_not_flagged_as_draco(self):
        import trimesh
        box = trimesh.creation.box(extents=[1, 1, 1])
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            path = tmp.name
        try:
            box.export(path)
            self.assertFalse(urdf_mesh._gltf_uses_draco(path))
        finally:
            os.remove(path)


class TestGetPathWithCacheResolverOrder(unittest.TestCase):
    """get_path_with_cache should respect ROS_VERSION when both resolvers are present."""

    def setUp(self):
        urdf_mesh.get_path_with_cache.cache_clear()
        # Save and clear ROS_VERSION so each test sets it explicitly.
        self._saved_ros_version = os.environ.pop('ROS_VERSION', None)

    def tearDown(self):
        urdf_mesh.get_path_with_cache.cache_clear()
        if self._saved_ros_version is None:
            os.environ.pop('ROS_VERSION', None)
        else:
            os.environ['ROS_VERSION'] = self._saved_ros_version

    def test_ament_is_tried_first_by_default(self):
        with mock.patch.object(urdf_mesh, '_try_ament',
                               return_value='/ament/foo') as ament, \
             mock.patch.object(urdf_mesh, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_mesh.get_path_with_cache('foo') == '/ament/foo'
            ament.assert_called_once_with('foo')
            rospkg_.assert_not_called()

    def test_ros_version_2_prefers_ament(self):
        os.environ['ROS_VERSION'] = '2'
        with mock.patch.object(urdf_mesh, '_try_ament',
                               return_value='/ament/foo'), \
             mock.patch.object(urdf_mesh, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_mesh.get_path_with_cache('foo') == '/ament/foo'
            rospkg_.assert_not_called()

    def test_ros_version_1_prefers_rospkg(self):
        """Hybrid env where the user explicitly asks for ROS 1 must keep using rospkg."""
        os.environ['ROS_VERSION'] = '1'
        with mock.patch.object(urdf_mesh, '_try_ament',
                               return_value='/ament/foo') as ament, \
             mock.patch.object(urdf_mesh, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_mesh.get_path_with_cache('foo') == '/rospkg/foo'
            rospkg_.assert_called_once_with('foo')
            ament.assert_not_called()

    def test_falls_back_when_first_resolver_returns_none(self):
        with mock.patch.object(urdf_mesh, '_try_ament',
                               return_value=None) as ament, \
             mock.patch.object(urdf_mesh, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_mesh.get_path_with_cache('foo') == '/rospkg/foo'
            ament.assert_called_once()
            rospkg_.assert_called_once()

    def test_lookup_error_when_both_resolvers_fail(self):
        with mock.patch.object(urdf_mesh, '_try_ament', return_value=None), \
             mock.patch.object(urdf_mesh, '_try_rospkg', return_value=None):
            # rospkg variable still pointing at the real (or None) module is
            # fine — we just need the resolvers to claim "not found".
            with pytest.raises((LookupError, ImportError)):
                urdf_mesh.get_path_with_cache('does_not_exist')

    def test_import_error_when_no_resolver_installed(self):
        with mock.patch.dict(sys.modules,
                             {'ament_index_python': None,
                              'ament_index_python.packages': None}):
            with mock.patch.object(urdf_mesh, 'rospkg', None):
                with pytest.raises(ImportError) as excinfo:
                    urdf_mesh.get_path_with_cache('whatever')
                assert 'ament_index_python' in str(excinfo.value)
                assert 'rospkg' in str(excinfo.value)


class TestResolveFilepathWithoutRospkg(unittest.TestCase):
    """package:// URIs must resolve via ament when rospkg is missing.

    Regression test for the previous ``if rospkg and parsed_url.scheme == 'package':``
    guard which silently skipped the entire package:// branch on rospkg-less
    environments such as ``apt install ros-<distro>-desktop`` without ROS 1.
    """

    def setUp(self):
        urdf_mesh.get_path_with_cache.cache_clear()
        self._saved_ros_version = os.environ.pop('ROS_VERSION', None)

    def tearDown(self):
        urdf_mesh.get_path_with_cache.cache_clear()
        if self._saved_ros_version is None:
            os.environ.pop('ROS_VERSION', None)
        else:
            os.environ['ROS_VERSION'] = self._saved_ros_version

    def test_resolves_via_ament_when_rospkg_missing(self):
        with tempfile.TemporaryDirectory() as pkg_dir:
            mesh_dir = os.path.join(pkg_dir, 'meshes')
            os.makedirs(mesh_dir)
            mesh_path = os.path.join(mesh_dir, 'foo.dae')
            with open(mesh_path, 'w') as f:
                f.write('<dummy/>')

            with mock.patch.object(urdf_mesh, '_try_ament',
                                    return_value=pkg_dir):
                with mock.patch.object(urdf_mesh, 'rospkg', None):
                    result = urdf_mesh.resolve_filepath(
                        '/tmp', 'package://my_pkg/meshes/foo.dae')
            assert result == mesh_path

    def test_falls_back_to_search_up_when_no_ros_installed(self):
        """package:// must resolve via search_up on machines without ROS at all.

        Mirrors the typical scikit-robot data layout under ~/.skrobot/, where a
        downloaded URDF references its sibling meshes via package://<name>/...
        even though no ROS distro is present.
        """
        with tempfile.TemporaryDirectory() as data_root:
            # Layout: <data_root>/pr2_description/{pr2.urdf, meshes/foo.dae}
            pkg_dir = os.path.join(data_root, 'pr2_description')
            mesh_dir = os.path.join(pkg_dir, 'meshes')
            os.makedirs(mesh_dir)
            urdf_path = os.path.join(pkg_dir, 'pr2.urdf')
            mesh_path = os.path.join(mesh_dir, 'foo.dae')
            for p in (urdf_path, mesh_path):
                with open(p, 'w') as f:
                    f.write('<dummy/>')

            with mock.patch.object(urdf_mesh, '_try_ament',
                                    return_value=None), \
                 mock.patch.object(urdf_mesh, '_try_rospkg',
                                    return_value=None), \
                 mock.patch.dict(sys.modules,
                                 {'ament_index_python': None,
                                  'ament_index_python.packages': None}), \
                 mock.patch.object(urdf_mesh, 'rospkg', None):
                # base_path is the URDF's directory; resolver should walk up
                # one level and find sibling 'pr2_description/meshes/foo.dae'.
                result = urdf_mesh.resolve_filepath(
                    pkg_dir, 'package://pr2_description/meshes/foo.dae')
            assert result == mesh_path


class TestGlbExportPreservesMaterialColor(unittest.TestCase):
    """GLB export must bake a material-only color into per-vertex colors.

    ``TextureVisuals.to_color`` can return a color array whose length does
    not match the vertex count when the mesh has a material color but no UV
    coordinates. The GLB exporter then drops the mismatched colors and the
    mesh becomes the default gray. The exporter must broadcast the material
    color to every vertex so the original color survives.
    """

    def test_material_color_baked_into_vertex_colors(self):
        import trimesh
        from trimesh.visual.material import PBRMaterial

        color = [72, 169, 84, 255]

        def make_box():
            box = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
            box.visual = trimesh.visual.TextureVisuals(
                material=PBRMaterial(baseColorFactor=color))
            return box

        with tempfile.TemporaryDirectory() as tmpdir:
            make_box().export(os.path.join(tmpdir, 'box.dae'))
            mesh = urdf_utils.Mesh(filename='box.dae', meshes=[make_box()])
            with urdf_mesh.export_mesh_format('.glb', overwrite_mesh=True):
                mesh._to_xml(etree.Element('visual'), tmpdir)

            scene = trimesh.load(
                os.path.join(tmpdir, 'box.glb'), process=False)
            geometries = list(scene.geometry.values())
            self.assertEqual(len(geometries), 1)
            geometry = geometries[0]
            vertex_colors = np.asarray(geometry.visual.vertex_colors)
            self.assertEqual(len(vertex_colors), len(geometry.vertices))
            np.testing.assert_array_equal(
                np.unique(vertex_colors.reshape(-1, 4), axis=0),
                np.array([color], dtype=np.uint8))


class TestEnvPrefixResolver(unittest.TestCase):
    """_try_env_prefixes resolves package:// from the shell environment alone,
    so a frozen binary (no ament_index_python / rospkg) still finds meshes
    after sourcing a workspace."""

    _VARS = ('AMENT_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'CMAKE_PREFIX_PATH',
             'ROS_PACKAGE_PATH', 'ROS_VERSION')

    def setUp(self):
        urdf_mesh.get_path_with_cache.cache_clear()
        self._saved = {v: os.environ.pop(v, None) for v in self._VARS}
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        urdf_mesh.get_path_with_cache.cache_clear()
        for v, val in self._saved.items():
            if val is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = val
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _make_package(self, directory, name):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'package.xml'), 'w') as f:
            f.write('<?xml version="1.0"?>\n'
                    '<package><name>{}</name></package>\n'.format(name))
        return directory

    def test_resolves_via_ament_prefix_share_layout(self):
        prefix = os.path.join(self._tmp, 'install')
        share = self._make_package(
            os.path.join(prefix, 'share', 'my_robot'), 'my_robot')
        os.environ['AMENT_PREFIX_PATH'] = prefix
        assert os.path.samefile(
            urdf_mesh._try_env_prefixes('my_robot'), share)

    def test_resolves_via_ros_package_path_direct_child(self):
        src = os.path.join(self._tmp, 'ws', 'src')
        pkg = self._make_package(os.path.join(src, 'my_robot'), 'my_robot')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert os.path.samefile(
            urdf_mesh._try_env_prefixes('my_robot'), pkg)

    def test_resolves_via_ros_package_path_recursive_crawl(self):
        src = os.path.join(self._tmp, 'ws', 'src')
        pkg = self._make_package(
            os.path.join(src, 'nested', 'my_robot'), 'my_robot')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert os.path.samefile(
            urdf_mesh._try_env_prefixes('my_robot'), pkg)

    def test_name_in_manifest_wins_over_directory_name(self):
        # under ROS_PACKAGE_PATH the package.xml <name>, not the folder name,
        # is authoritative (the ament share/<pkg> layout is name-keyed instead)
        src = os.path.join(self._tmp, 'ws', 'src')
        pkg = self._make_package(os.path.join(src, 'pkg_dir'), 'real_name')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert urdf_mesh._try_env_prefixes('pkg_dir') is None
        assert os.path.samefile(
            urdf_mesh._try_env_prefixes('real_name'), pkg)

    def test_returns_none_when_not_found(self):
        os.environ['AMENT_PREFIX_PATH'] = self._tmp
        os.environ['ROS_PACKAGE_PATH'] = self._tmp
        assert urdf_mesh._try_env_prefixes('absent') is None

    def test_malformed_manifest_is_skipped_gracefully(self):
        # a broken package.xml must not raise; the crawl just moves on
        src = os.path.join(self._tmp, 'ws', 'src')
        broken = os.path.join(src, 'broken')
        os.makedirs(broken)
        with open(os.path.join(broken, 'package.xml'), 'w') as f:
            f.write('<package><name>oops')          # unterminated XML
        good = self._make_package(os.path.join(src, 'good'), 'good')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert urdf_mesh._manifest_package_name(broken) is None
        assert urdf_mesh._try_env_prefixes('broken') is None
        assert os.path.samefile(
            urdf_mesh._try_env_prefixes('good'), good)

    def test_get_path_with_cache_falls_back_to_env(self):
        prefix = os.path.join(self._tmp, 'install')
        share = self._make_package(
            os.path.join(prefix, 'share', 'my_robot'), 'my_robot')
        os.environ['AMENT_PREFIX_PATH'] = prefix
        # neither ROS Python resolver available -> env fallback must resolve
        with mock.patch.object(urdf_mesh, '_try_ament', return_value=None), \
             mock.patch.object(urdf_mesh, '_try_rospkg', return_value=None):
            assert os.path.samefile(
                urdf_mesh.get_path_with_cache('my_robot'), share)


class TestConfigureOrigin(unittest.TestCase):

    def test_xyzrpy_6vector(self):
        # xyz + rpy 6-vector -> 4x4 (this path used to overwrite the input
        # with np.eye(4) before reading it and crashed on unpacking)
        from skrobot.coordinates.math import rpy2matrix

        xyz = [1.0, 2.0, 3.0]
        rpy = [0.1, 0.2, 0.3]
        matrix = urdf_utils.configure_origin(list(xyz) + list(rpy))
        self.assertEqual(matrix.shape, (4, 4))
        np.testing.assert_allclose(matrix[:3, 3], xyz)
        np.testing.assert_allclose(matrix[:3, :3], rpy2matrix(*rpy))
        np.testing.assert_allclose(matrix[3], [0, 0, 0, 1])

    def test_none_and_4x4_passthrough(self):
        np.testing.assert_allclose(
            urdf_utils.configure_origin(None), np.eye(4))
        m = np.eye(4)
        m[:3, 3] = [4.0, 5.0, 6.0]
        np.testing.assert_allclose(urdf_utils.configure_origin(m), m)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            urdf_utils.configure_origin(np.zeros(5))
        with pytest.raises(TypeError):
            urdf_utils.configure_origin('not-a-matrix')


class TestForceVisualMeshOriginToZero(unittest.TestCase):
    """Every element must get its own origin baked into its own vertices.

    ``force_visual_mesh_origin_to_zero`` zeroes an element's ``<origin>``
    and moves the offset into the mesh vertices. One mesh file is commonly
    referenced by several elements with different origins -- a ``<visual>``
    and a ``<collision>`` of one link, or two links sharing a part. Baking
    only the first origin seen for a filename, or transforming a shared
    Trimesh in place, misplaces every other element's geometry.
    """

    URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="shared_mesh">
  <link name="base_link">
    <visual>
      <origin xyz="1 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="box.stl"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 2 0" rpy="0 0 0"/>
      <geometry><mesh filename="box.stl"/></geometry>
    </collision>
  </link>
  <link name="second_link">
    <visual>
      <origin xyz="0 0 3" rpy="0 0 0"/>
      <geometry><mesh filename="box.stl"/></geometry>
    </visual>
  </link>
  <joint name="fixed_joint" type="fixed">
    <parent link="base_link"/>
    <child link="second_link"/>
  </joint>
</robot>
"""

    def setUp(self):
        import trimesh

        self._tmp = tempfile.mkdtemp()
        box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        box.export(os.path.join(self._tmp, 'box.stl'))
        self._urdf_path = os.path.join(self._tmp, 'robot.urdf')
        with open(self._urdf_path, 'w') as f:
            f.write(self.URDF_TEMPLATE)
        urdf_mesh._MESH_CACHE.clear()

    def tearDown(self):
        import shutil

        urdf_mesh._MESH_CACHE.clear()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _load(self):
        with urdf_mesh.force_visual_mesh_origin_to_zero():
            return urdf_utils.URDF.load(self._urdf_path)

    def _assert_baked(self, robot):
        elements = {}
        for link in robot.links:
            for visual in link.visuals:
                elements['visual/' + link.name] = visual
            for collision in link.collisions:
                elements['collision/' + link.name] = collision
        expected = {'visual/base_link': [1.0, 0.0, 0.0],
                    'collision/base_link': [0.0, 2.0, 0.0],
                    'visual/second_link': [0.0, 0.0, 3.0]}
        self.assertEqual(set(elements), set(expected))
        for key, center in expected.items():
            element = elements[key]
            # The origin is zeroed, so the offset has to live in the vertices.
            np.testing.assert_allclose(element.origin, np.eye(4), atol=1e-9)
            vertices = np.vstack(
                [m.vertices for m in element.geometry.meshes])
            np.testing.assert_allclose(
                vertices.mean(axis=0), center, atol=1e-6,
                err_msg='{} geometry is not centered at {}'.format(
                    key, center))

    def test_each_element_keeps_its_own_origin(self):
        self._assert_baked(self._load())

    def test_reloading_the_same_urdf_bakes_again(self):
        # No module-level state may make a later load skip the baking.
        self._load()
        self._assert_baked(self._load())

    def test_shared_trimesh_objects_are_not_transformed_in_place(self):
        # enable_mesh_cache hands the very same Trimesh objects to every
        # element referencing the file.
        with urdf_mesh.enable_mesh_cache():
            self._assert_baked(self._load())


class TestBakeOriginIntoMeshes(unittest.TestCase):
    """The helper behind ``force_visual_mesh_origin_to_zero``."""

    def _geometry(self):
        import trimesh
        mesh = urdf_utils.Mesh(filename='box.stl',
                               meshes=[trimesh.creation.box(
                                   extents=(0.2, 0.2, 0.2))])
        return urdf_utils.Geometry(mesh=mesh)

    def test_an_identity_origin_leaves_the_meshes_shared(self):
        """An identity origin must not even copy: under a mesh cache the
        objects are shared, and copying them costs memory for nothing."""
        geometry = self._geometry()
        before = geometry.mesh.meshes[0]
        urdf_mesh.bake_origin_into_meshes(geometry, np.eye(4))
        self.assertIs(geometry.mesh.meshes[0], before)

    def test_a_real_origin_copies_and_transforms(self):
        geometry = self._geometry()
        before = geometry.mesh.meshes[0]
        origin = np.eye(4)
        origin[:3, 3] = [1.0, 2.0, 3.0]
        urdf_mesh.bake_origin_into_meshes(geometry, origin)
        baked = geometry.mesh.meshes[0]
        self.assertIsNot(baked, before)
        np.testing.assert_allclose(
            baked.bounding_box.centroid, [1.0, 2.0, 3.0], atol=1e-9)
        # the original is left where it was, for whoever else holds it
        np.testing.assert_allclose(
            before.bounding_box.centroid, [0.0, 0.0, 0.0], atol=1e-9)

    def test_a_primitive_geometry_is_left_alone(self):
        """A ``<box>`` has no vertices to bake into; the helper must not
        reach for a mesh that is not there."""
        geometry = urdf_utils.Geometry(box=urdf_utils.Box(size=[1, 1, 1]))
        origin = np.eye(4)
        origin[:3, 3] = [1.0, 0.0, 0.0]
        urdf_mesh.bake_origin_into_meshes(geometry, origin)
        self.assertIsNone(geometry.mesh)


class TestTransformVertexNormals(unittest.TestCase):
    """Normals do not transform like points.

    ``Trimesh.apply_transform`` multiplies them by the matrix directly, which
    only happens to be right for a pure rotation.
    """

    def test_a_rotation_rotates_them(self):
        angle = 0.7
        matrix = np.eye(4)
        matrix[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                          [np.sin(angle), np.cos(angle), 0],
                          [0, 0, 1]]
        normals = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        got = mesh_utils._transform_vertex_normals(normals, matrix)
        np.testing.assert_allclose(
            got, [[np.cos(angle), np.sin(angle), 0.0], [0.0, 0.0, 1.0]],
            atol=1e-9)

    def test_a_uniform_scale_leaves_the_direction_alone(self):
        normals = np.array([[0.0, 0.6, 0.8]])
        got = mesh_utils._transform_vertex_normals(normals, np.eye(4) * 25.4)
        np.testing.assert_allclose(got, normals, atol=1e-9)

    def test_a_non_uniform_scale_needs_the_inverse_transpose(self):
        """Squashing z by ten flattens the surface, so a 45 degree normal
        tips *towards* z -- the opposite of what multiplying by the matrix
        does, which would tip it away."""
        matrix = np.diag([1.0, 1.0, 0.1, 1.0])
        normals = np.array([[np.sqrt(0.5), 0.0, np.sqrt(0.5)]])
        got = mesh_utils._transform_vertex_normals(normals, matrix)
        self.assertLess(got[0, 0], normals[0, 0])
        self.assertGreater(got[0, 2], normals[0, 2])
        np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-9)
        # it must stay perpendicular to the surface it came from
        tangent = np.array([1.0, 0.0, -1.0]) @ matrix[:3, :3].T
        self.assertAlmostEqual(float(got[0] @ tangent), 0.0, places=9)

    def test_a_mirror_flips_them_back(self):
        """A negative determinant reverses the triangle winding, so the mapped
        normal has to be negated to keep pointing out of the same side."""
        matrix = np.diag([1.0, -1.0, 1.0, 1.0])
        normals = np.array([[0.0, 1.0, 0.0]])
        got = mesh_utils._transform_vertex_normals(normals, matrix)
        np.testing.assert_allclose(got, [[0.0, 1.0, 0.0]], atol=1e-9)

    def test_a_singular_matrix_is_passed_through(self):
        normals = np.array([[0.0, 0.0, 1.0]])
        got = mesh_utils._transform_vertex_normals(normals, np.zeros((4, 4)))
        np.testing.assert_allclose(got, normals, atol=1e-9)


class TestSceneNormalsSurviveLoading(unittest.TestCase):
    """Every step that rebuilds geometry drops trimesh's cache, and that cache
    is where normals read from the file live."""

    def _scene(self, transform=None):
        import trimesh
        mesh = trimesh.creation.box()
        mesh.vertex_normals = np.tile([0.0, 0.0, 1.0], (len(mesh.vertices), 1))
        scene = trimesh.Scene()
        scene.add_geometry(mesh, geom_name='thing', transform=transform)
        scene.units = 'meters'
        return scene

    def test_authored_normals_are_only_reported_when_really_authored(self):
        import trimesh
        plain = trimesh.Scene()
        plain.add_geometry(trimesh.creation.box(), geom_name='thing')
        plain.units = 'meters'
        self.assertEqual(mesh_utils._authored_vertex_normals(plain), {})
        self.assertIn('thing',
                      mesh_utils._authored_vertex_normals(self._scene()))

    def test_they_are_restored_after_a_units_conversion(self):
        scene = self._scene()
        snapshot = mesh_utils._authored_vertex_normals(scene)
        converted = scene.convert_units('inch')
        # convert_units rebuilds the geometry, losing the cache
        self.assertNotIn(
            'vertex_normals',
            list(converted.geometry.values())[0]._cache.cache)
        mesh_utils._restore_vertex_normals(converted, snapshot)
        np.testing.assert_allclose(
            np.asarray(list(converted.geometry.values())[0].vertex_normals),
            snapshot['thing'], atol=1e-9)

    def test_dumping_a_scene_maps_them_through_the_node_transform(self):
        transform = np.eye(4)
        # quarter turn about x, so +z becomes +y, plus a scale
        transform[:3, :3] = np.array([[1.0, 0.0, 0.0],
                                      [0.0, 0.0, -1.0],
                                      [0.0, 1.0, 0.0]]) * 25.4
        meshes = mesh_utils._dump_scene(self._scene(transform))
        self.assertEqual(len(meshes), 1)
        np.testing.assert_allclose(
            np.asarray(meshes[0].vertex_normals),
            np.tile([0.0, -1.0, 0.0], (len(meshes[0].vertices), 1)),
            atol=1e-9)


class TestConfiguredRestoresState(unittest.TestCase):
    """The export settings are module-global, so a block that leaks them
    poisons every later load and export in the same process."""

    def test_a_raising_block_still_restores(self):
        before = dict(urdf_mesh._CONFIGURABLE_VALUES)
        with pytest.raises(RuntimeError):
            with urdf_mesh.export_mesh_format('.stl',
                                               collision_mesh_format='.glb',
                                               target_triangles=100):
                raise RuntimeError('export blew up')
        self.assertEqual(urdf_mesh._CONFIGURABLE_VALUES, before)

    def test_a_raising_scale_or_origin_block_still_restores(self):
        before = dict(urdf_mesh._CONFIGURABLE_VALUES)
        for manager in (urdf_mesh.apply_scale(2.0),
                        urdf_mesh.force_visual_mesh_origin_to_zero(),
                        urdf_mesh.enable_mesh_cache(),
                        urdf_mesh.no_mesh_load_mode()):
            with pytest.raises(RuntimeError):
                with manager:
                    raise RuntimeError('boom')
            self.assertEqual(urdf_mesh._CONFIGURABLE_VALUES, before)

    def test_they_nest(self):
        with urdf_mesh.apply_scale(2.0):
            with urdf_mesh.apply_scale(3.0):
                self.assertEqual(
                    urdf_mesh._CONFIGURABLE_VALUES['scale_factor'], 3.0)
            # the outer block is still running and still means 2.0
            self.assertEqual(
                urdf_mesh._CONFIGURABLE_VALUES['scale_factor'], 2.0)


class TestResolveMeshOutputPath(unittest.TestCase):
    """An output mesh file may only ever hold geometry that matches it.

    ``force_visual_mesh_origin_to_zero`` gives each element its own baked
    copy of a shared mesh, and the simplification options rewrite every
    mesh of an export. Either way the geometry no longer matches the file
    it was read from, and naming the output after that file alone makes
    elements overwrite each other or destroy the input.
    """

    def _origin(self, xyz):
        origin = np.eye(4)
        origin[:3, 3] = xyz
        return origin

    def test_unbaked_geometry_keeps_the_plain_name(self):
        output_file, urdf_filename = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.dae', None)
        self.assertEqual(output_file, '/meshes/box.dae')
        self.assertEqual(urdf_filename, 'box.dae')

    def test_a_package_uri_is_rewritten_like_a_path(self):
        _, urdf_filename = urdf_mesh.resolve_mesh_output_path(
            '/share/pkg/meshes/box.stl', 'package://pkg/meshes/box.stl',
            '.dae', None)
        self.assertEqual(urdf_filename, 'package://pkg/meshes/box.dae')

    def test_two_baked_origins_get_two_files(self):
        first, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.dae', self._origin([0, 0, 1]))
        second, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.dae', self._origin([0, 0, 5]))
        self.assertNotEqual(first, second)

    def test_the_same_baked_origin_shares_one_file(self):
        first, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.dae', self._origin([0, 0, 1]))
        second, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.dae', self._origin([0, 0, 1]))
        self.assertEqual(first, second)

    def test_a_negative_zero_is_the_same_origin(self):
        """-0.0 and 0.0 are equal but have different bytes; hashing the
        raw bytes would hand one origin two files."""
        plus, minus = np.eye(4), np.eye(4)
        minus[0, 3] = -0.0
        self.assertEqual(urdf_mesh.geometry_variant_suffix(plus),
                         urdf_mesh.geometry_variant_suffix(minus))

    def test_baking_survives_the_format_staying_the_same(self):
        output_file, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.stl', self._origin([0, 0, 1]))
        self.assertNotEqual(output_file, '/meshes/box.stl')

    def test_processing_only_renames_when_it_would_hit_the_source(self):
        # converting the format already moves the output off the source
        converted, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.dae', None, '_deadbeef')
        self.assertEqual(converted, '/meshes/box.dae')
        # ... but keeping the format would land on the input file
        kept, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.stl', None, '_deadbeef')
        self.assertEqual(kept, '/meshes/box_deadbeef.stl')

    def test_an_empty_processing_key_lets_the_source_be_replaced(self):
        # what --overwrite-mesh asks for
        kept, _ = urdf_mesh.resolve_mesh_output_path(
            '/meshes/box.stl', 'box.stl', '.stl', None, '')
        self.assertEqual(kept, '/meshes/box.stl')


class TestMeshProcessingKey(unittest.TestCase):

    def test_it_is_empty_without_processing(self):
        self.assertEqual(urdf_mesh.mesh_processing_key(), '')

    def test_different_settings_do_not_share_a_file(self):
        with urdf_mesh.export_mesh_format(
                '.stl', simplify_vertex_clustering_voxel_size=0.01):
            coarse = urdf_mesh.mesh_processing_key()
        with urdf_mesh.export_mesh_format(
                '.stl', simplify_vertex_clustering_voxel_size=0.05):
            finer = urdf_mesh.mesh_processing_key()
        self.assertTrue(coarse)
        self.assertNotEqual(coarse, finer)


class TestConvertUrdfMeshesSharedMesh(unittest.TestCase):
    """Saving must keep apart what loading kept apart.

    ``force_visual_mesh_origin_to_zero`` bakes each element's origin into
    its own copy of the mesh. Writing those copies back out under the
    source filename hands every element one file: whichever element is
    written decides the geometry for all of them, and the others end up
    displaced by the difference between their origins.
    """

    URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="shared_mesh">
  <link name="base_link">
    <visual>
      <origin xyz="0 0 1" rpy="0 0 0"/>
      <geometry><mesh filename="box.stl"/></geometry>
    </visual>
  </link>
  <link name="second_link">
    <visual>
      <origin xyz="0 0 5" rpy="0 0 0"/>
      <geometry><mesh filename="box.stl"/></geometry>
    </visual>
  </link>
  <joint name="fixed_joint" type="fixed">
    <parent link="base_link"/>
    <child link="second_link"/>
  </joint>
</robot>
"""

    def setUp(self):
        import trimesh

        self._tmp = tempfile.mkdtemp()
        trimesh.creation.box(extents=[0.2, 0.2, 0.2]).export(
            os.path.join(self._tmp, 'box.stl'))
        self._urdf_path = os.path.join(self._tmp, 'robot.urdf')
        with open(self._urdf_path, 'w') as f:
            f.write(self.URDF_TEMPLATE)
        urdf_mesh._MESH_CACHE.clear()

    def tearDown(self):
        import shutil

        urdf_mesh._MESH_CACHE.clear()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _convert(self, mesh_format='.dae', **kwargs):
        import trimesh

        output_path = os.path.join(self._tmp, 'converted.urdf')
        urdf_utils.convert_urdf_meshes(
            self._urdf_path, output_path, mesh_format, **kwargs)
        centers = []
        for visual in etree.parse(output_path).getroot().iter('visual'):
            filename = visual.find('geometry/mesh').get('filename')
            mesh = trimesh.load(
                os.path.join(self._tmp, filename), force='mesh')
            centers.append(mesh.bounding_box.centroid[2])
        return centers

    def test_each_element_gets_the_geometry_it_was_given(self):
        np.testing.assert_allclose(
            self._convert(force_zero_visual_origin=True), [1.0, 5.0],
            atol=1e-6)

    def test_overwriting_does_not_make_one_element_win(self):
        np.testing.assert_allclose(
            self._convert(force_zero_visual_origin=True, overwrite_mesh=True),
            [1.0, 5.0], atol=1e-6)

    def test_baking_survives_the_format_staying_the_same(self):
        # Converting .stl to .stl leaves the output path equal to the
        # input path, where an "it already exists" check used to skip the
        # write and leave both elements pointing at unbaked geometry.
        np.testing.assert_allclose(
            self._convert('.stl', force_zero_visual_origin=True), [1.0, 5.0],
            atol=1e-6)

    def test_the_source_mesh_is_left_alone(self):
        import trimesh

        self._convert('.stl', force_zero_visual_origin=True)
        source = trimesh.load(os.path.join(self._tmp, 'box.stl'))
        np.testing.assert_allclose(
            source.bounding_box.centroid, [0.0, 0.0, 0.0], atol=1e-9)

    def test_saving_twice_writes_the_meshes_twice(self):
        """The record of what has already been written belongs to one
        save. A module-global one that nothing resets makes the next save
        skip meshes that are not there yet."""
        first = os.path.join(self._tmp, 'first.urdf')
        urdf_utils.convert_urdf_meshes(
            self._urdf_path, first, '.dae', force_zero_visual_origin=True)
        written = [name for name in os.listdir(self._tmp)
                   if name.endswith('.dae')]
        # one per baked origin
        self.assertEqual(len(written), 2)
        for name in written:
            os.remove(os.path.join(self._tmp, name))

        second = os.path.join(self._tmp, 'second.urdf')
        urdf_utils.convert_urdf_meshes(
            self._urdf_path, second, '.dae', force_zero_visual_origin=True)
        for name in written:
            self.assertTrue(os.path.exists(os.path.join(self._tmp, name)),
                            '{} was not written again'.format(name))

    def test_an_unbaked_shared_mesh_still_shares_one_file(self):
        # Without baking the elements really do have the same geometry;
        # giving each of them a file would be pure duplication.
        output_path = os.path.join(self._tmp, 'converted.urdf')
        urdf_utils.convert_urdf_meshes(self._urdf_path, output_path, '.dae')
        filenames = {visual.find('geometry/mesh').get('filename') for visual
                     in etree.parse(output_path).getroot().iter('visual')}
        self.assertEqual(filenames, {'box.dae'})


@pytest.mark.skipif(
    not is_package_installed('open3d'),
    reason='the simplification this exercises runs through open3d')
class TestConvertUrdfMeshesSimplification(unittest.TestCase):
    """A simplified mesh must reach the file the saved URDF points at."""

    URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="ball">
  <link name="base_link">
    <visual><geometry><mesh filename="ball.stl"/></geometry></visual>
  </link>
</robot>
"""

    def setUp(self):
        import trimesh

        self._tmp = tempfile.mkdtemp()
        self._source = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
        self._source.export(os.path.join(self._tmp, 'ball.stl'))
        self._urdf_path = os.path.join(self._tmp, 'robot.urdf')
        with open(self._urdf_path, 'w') as f:
            f.write(self.URDF_TEMPLATE)
        urdf_mesh._MESH_CACHE.clear()

    def tearDown(self):
        import shutil

        urdf_mesh._MESH_CACHE.clear()
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _faces_of_the_saved_mesh(self, **kwargs):
        import trimesh

        output_path = os.path.join(self._tmp, 'converted.urdf')
        urdf_utils.convert_urdf_meshes(
            self._urdf_path, output_path, '.stl',
            simplify_vertex_clustering_voxel_size=0.05, **kwargs)
        filename = etree.parse(output_path).getroot().find(
            'link/visual/geometry/mesh').get('filename')
        mesh = trimesh.load(os.path.join(self._tmp, filename), force='mesh')
        return len(mesh.faces)

    def test_it_is_written_even_when_the_format_does_not_change(self):
        self.assertLess(self._faces_of_the_saved_mesh(),
                        len(self._source.faces))

    def test_it_does_not_land_on_top_of_its_own_source(self):
        import trimesh

        self._faces_of_the_saved_mesh()
        source = trimesh.load(os.path.join(self._tmp, 'ball.stl'))
        self.assertEqual(len(source.faces), len(self._source.faces))

    def test_overwrite_mesh_asks_for_the_source_to_be_replaced(self):
        import trimesh

        self._faces_of_the_saved_mesh(overwrite_mesh=True)
        source = trimesh.load(os.path.join(self._tmp, 'ball.stl'))
        self.assertLess(len(source.faces), len(self._source.faces))
