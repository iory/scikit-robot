import os
import sys
import tempfile
import unittest
from unittest import mock

from lxml import etree
import numpy as np
import pytest

from skrobot.utils import urdf as urdf_utils


class TestLoadMeshesNoneGuard(unittest.TestCase):

    def test_raises_file_not_found_with_helpful_message(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            urdf_utils._load_meshes(None)
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
            self.assertTrue(urdf_utils._gltf_uses_draco(path))
        finally:
            os.remove(path)

    def test_load_skips_and_warns_when_draco_and_no_dracopy(self):
        with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as tmp:
            path = tmp.name
        try:
            self._make_draco_glb(path)
            # Reset the one-time hint flag so the verbose hint is emitted.
            urdf_utils._DRACO_MISSING_HINT_SHOWN = False
            with mock.patch(
                    'skrobot.utils.draco.is_dracopy_available',
                    return_value=False):
                with self.assertLogs(
                        'skrobot.utils.urdf', level='WARNING') as logs:
                    meshes = urdf_utils._load_meshes(path)
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
            self.assertFalse(urdf_utils._gltf_uses_draco(path))
        finally:
            os.remove(path)


class TestGetPathWithCacheResolverOrder(unittest.TestCase):
    """get_path_with_cache should respect ROS_VERSION when both resolvers are present."""

    def setUp(self):
        urdf_utils.get_path_with_cache.cache_clear()
        # Save and clear ROS_VERSION so each test sets it explicitly.
        self._saved_ros_version = os.environ.pop('ROS_VERSION', None)

    def tearDown(self):
        urdf_utils.get_path_with_cache.cache_clear()
        if self._saved_ros_version is None:
            os.environ.pop('ROS_VERSION', None)
        else:
            os.environ['ROS_VERSION'] = self._saved_ros_version

    def test_ament_is_tried_first_by_default(self):
        with mock.patch.object(urdf_utils, '_try_ament',
                               return_value='/ament/foo') as ament, \
             mock.patch.object(urdf_utils, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_utils.get_path_with_cache('foo') == '/ament/foo'
            ament.assert_called_once_with('foo')
            rospkg_.assert_not_called()

    def test_ros_version_2_prefers_ament(self):
        os.environ['ROS_VERSION'] = '2'
        with mock.patch.object(urdf_utils, '_try_ament',
                               return_value='/ament/foo'), \
             mock.patch.object(urdf_utils, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_utils.get_path_with_cache('foo') == '/ament/foo'
            rospkg_.assert_not_called()

    def test_ros_version_1_prefers_rospkg(self):
        """Hybrid env where the user explicitly asks for ROS 1 must keep using rospkg."""
        os.environ['ROS_VERSION'] = '1'
        with mock.patch.object(urdf_utils, '_try_ament',
                               return_value='/ament/foo') as ament, \
             mock.patch.object(urdf_utils, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_utils.get_path_with_cache('foo') == '/rospkg/foo'
            rospkg_.assert_called_once_with('foo')
            ament.assert_not_called()

    def test_falls_back_when_first_resolver_returns_none(self):
        with mock.patch.object(urdf_utils, '_try_ament',
                               return_value=None) as ament, \
             mock.patch.object(urdf_utils, '_try_rospkg',
                               return_value='/rospkg/foo') as rospkg_:
            assert urdf_utils.get_path_with_cache('foo') == '/rospkg/foo'
            ament.assert_called_once()
            rospkg_.assert_called_once()

    def test_lookup_error_when_both_resolvers_fail(self):
        with mock.patch.object(urdf_utils, '_try_ament', return_value=None), \
             mock.patch.object(urdf_utils, '_try_rospkg', return_value=None):
            # rospkg variable still pointing at the real (or None) module is
            # fine — we just need the resolvers to claim "not found".
            with pytest.raises((LookupError, ImportError)):
                urdf_utils.get_path_with_cache('does_not_exist')

    def test_import_error_when_no_resolver_installed(self):
        with mock.patch.dict(sys.modules,
                             {'ament_index_python': None,
                              'ament_index_python.packages': None}):
            with mock.patch.object(urdf_utils, 'rospkg', None):
                with pytest.raises(ImportError) as excinfo:
                    urdf_utils.get_path_with_cache('whatever')
                assert 'ament_index_python' in str(excinfo.value)
                assert 'rospkg' in str(excinfo.value)


class TestResolveFilepathWithoutRospkg(unittest.TestCase):
    """package:// URIs must resolve via ament when rospkg is missing.

    Regression test for the previous ``if rospkg and parsed_url.scheme == 'package':``
    guard which silently skipped the entire package:// branch on rospkg-less
    environments such as ``apt install ros-<distro>-desktop`` without ROS 1.
    """

    def setUp(self):
        urdf_utils.get_path_with_cache.cache_clear()
        self._saved_ros_version = os.environ.pop('ROS_VERSION', None)

    def tearDown(self):
        urdf_utils.get_path_with_cache.cache_clear()
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

            with mock.patch.object(urdf_utils, '_try_ament',
                                    return_value=pkg_dir):
                with mock.patch.object(urdf_utils, 'rospkg', None):
                    result = urdf_utils.resolve_filepath(
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

            with mock.patch.object(urdf_utils, '_try_ament',
                                    return_value=None), \
                 mock.patch.object(urdf_utils, '_try_rospkg',
                                    return_value=None), \
                 mock.patch.dict(sys.modules,
                                 {'ament_index_python': None,
                                  'ament_index_python.packages': None}), \
                 mock.patch.object(urdf_utils, 'rospkg', None):
                # base_path is the URDF's directory; resolver should walk up
                # one level and find sibling 'pr2_description/meshes/foo.dae'.
                result = urdf_utils.resolve_filepath(
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
            with urdf_utils.export_mesh_format('.glb', overwrite_mesh=True):
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
        urdf_utils.get_path_with_cache.cache_clear()
        self._saved = {v: os.environ.pop(v, None) for v in self._VARS}
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        urdf_utils.get_path_with_cache.cache_clear()
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
            urdf_utils._try_env_prefixes('my_robot'), share)

    def test_resolves_via_ros_package_path_direct_child(self):
        src = os.path.join(self._tmp, 'ws', 'src')
        pkg = self._make_package(os.path.join(src, 'my_robot'), 'my_robot')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert os.path.samefile(
            urdf_utils._try_env_prefixes('my_robot'), pkg)

    def test_resolves_via_ros_package_path_recursive_crawl(self):
        src = os.path.join(self._tmp, 'ws', 'src')
        pkg = self._make_package(
            os.path.join(src, 'nested', 'my_robot'), 'my_robot')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert os.path.samefile(
            urdf_utils._try_env_prefixes('my_robot'), pkg)

    def test_name_in_manifest_wins_over_directory_name(self):
        # under ROS_PACKAGE_PATH the package.xml <name>, not the folder name,
        # is authoritative (the ament share/<pkg> layout is name-keyed instead)
        src = os.path.join(self._tmp, 'ws', 'src')
        pkg = self._make_package(os.path.join(src, 'pkg_dir'), 'real_name')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert urdf_utils._try_env_prefixes('pkg_dir') is None
        assert os.path.samefile(
            urdf_utils._try_env_prefixes('real_name'), pkg)

    def test_returns_none_when_not_found(self):
        os.environ['AMENT_PREFIX_PATH'] = self._tmp
        os.environ['ROS_PACKAGE_PATH'] = self._tmp
        assert urdf_utils._try_env_prefixes('absent') is None

    def test_malformed_manifest_is_skipped_gracefully(self):
        # a broken package.xml must not raise; the crawl just moves on
        src = os.path.join(self._tmp, 'ws', 'src')
        broken = os.path.join(src, 'broken')
        os.makedirs(broken)
        with open(os.path.join(broken, 'package.xml'), 'w') as f:
            f.write('<package><name>oops')          # unterminated XML
        good = self._make_package(os.path.join(src, 'good'), 'good')
        os.environ['ROS_PACKAGE_PATH'] = src
        assert urdf_utils._manifest_package_name(broken) is None
        assert urdf_utils._try_env_prefixes('broken') is None
        assert os.path.samefile(
            urdf_utils._try_env_prefixes('good'), good)

    def test_get_path_with_cache_falls_back_to_env(self):
        prefix = os.path.join(self._tmp, 'install')
        share = self._make_package(
            os.path.join(prefix, 'share', 'my_robot'), 'my_robot')
        os.environ['AMENT_PREFIX_PATH'] = prefix
        # neither ROS Python resolver available -> env fallback must resolve
        with mock.patch.object(urdf_utils, '_try_ament', return_value=None), \
             mock.patch.object(urdf_utils, '_try_rospkg', return_value=None):
            assert os.path.samefile(
                urdf_utils.get_path_with_cache('my_robot'), share)
