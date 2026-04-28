import os
import sys
import tempfile
import unittest
from unittest import mock

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
