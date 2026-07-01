import unittest
from unittest import mock

import numpy as np

import skrobot
from skrobot.viewers import _supported_kwargs
from skrobot.viewers import create_viewer
from skrobot.viewers import PyrenderViewer
from skrobot.viewers import TrimeshSceneViewer
from skrobot.viewers import ViserViewer
from skrobot.viewers._base import _InteractiveViewerMixin


class _CameraResolver(object):
    """Minimal stand-in exposing ViserViewer's camera math without a server."""

    _linkid_to_link = {}
    _collect_world_points = ViserViewer._collect_world_points
    _resolve_camera_view = ViserViewer._resolve_camera_view


class _CountingViewer(_InteractiveViewerMixin):
    """Mixin-only viewer that counts redraw() calls (no GL backend)."""

    def __init__(self):
        self.redraw_count = 0

    def redraw(self):
        self.redraw_count += 1


class TestCreateViewer(unittest.TestCase):

    def test_unknown_viewer_raises(self):
        with self.assertRaises(ValueError):
            create_viewer('not-a-viewer')

    def test_non_string_name_raises_value_error(self):
        for bad_name in (None, 123, ['trimesh']):
            with self.assertRaises(ValueError):
                create_viewer(bad_name)

    def test_viewer_classes_registered(self):
        self.assertIs(
            skrobot.viewers._VIEWER_CLASSES['trimesh'], TrimeshSceneViewer)
        self.assertIs(
            skrobot.viewers._VIEWER_CLASSES['pyrender'], PyrenderViewer)
        self.assertIs(
            skrobot.viewers._VIEWER_CLASSES['viser'], ViserViewer)

    def test_viewer_types_constant(self):
        from skrobot.viewers import VIEWER_TYPES
        self.assertIn('mitsuba', VIEWER_TYPES)
        self.assertNotIn('notebook', VIEWER_TYPES)
        for name in VIEWER_TYPES:
            self.assertIn(name, skrobot.viewers._VIEWER_CLASSES)

    def test_name_is_case_insensitive(self):
        # Exercise create_viewer with mixed-case names. A lightweight fake
        # is registered so no real viewer (window / server) is created.
        class _FakeViewer(object):
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        with mock.patch.dict(skrobot.viewers._VIEWER_CLASSES,
                             {'trimesh': _FakeViewer}):
            for name in ('trimesh', 'TRIMESH', 'TrImEsH'):
                self.assertIsInstance(create_viewer(name), _FakeViewer)

    def test_supported_kwargs_filters_unsupported(self):
        # resolution is accepted by the trimesh / pyrender viewers ...
        self.assertEqual(
            _supported_kwargs(
                TrimeshSceneViewer, {'resolution': (1, 1), 'bogus': 1}),
            {'resolution': (1, 1)})
        # ... but not by the viser viewer, so it is dropped.
        self.assertEqual(
            _supported_kwargs(
                ViserViewer, {'resolution': (1, 1), 'enable_ik': True}),
            {'enable_ik': True})

    def test_wait_until_close_available_on_all_viewers(self):
        for cls in (TrimeshSceneViewer, PyrenderViewer, ViserViewer):
            self.assertTrue(
                hasattr(cls, 'wait_until_close'),
                '{} is missing wait_until_close'.format(cls.__name__))

    def test_pause_available_on_all_viewers(self):
        for cls in (TrimeshSceneViewer, PyrenderViewer, ViserViewer):
            self.assertTrue(
                hasattr(cls, 'pause'),
                '{} is missing pause'.format(cls.__name__))

    def test_pause_pumps_redraw(self):
        viewer = _CountingViewer()
        # Non-positive / non-finite duration triggers exactly one redraw
        # and returns immediately.
        for duration in (0, -1.0, float('nan'), float('inf')):
            viewer.redraw_count = 0
            viewer.pause(duration)
            self.assertEqual(viewer.redraw_count, 1)

        # A finite pause keeps pumping redraw more than once.
        viewer.redraw_count = 0
        viewer.pause(0.1, fps=30.0)
        self.assertGreater(viewer.redraw_count, 1)

    def test_pause_rejects_non_positive_fps(self):
        viewer = _CountingViewer()
        for bad_fps in (0, -5.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                viewer.pause(0.1, fps=bad_fps)


class TestViserSetCameraMath(unittest.TestCase):

    def setUp(self):
        self.resolver = _CameraResolver()

    def test_explicit_distance_and_center(self):
        # Camera should sit ``distance`` away from ``center`` along the
        # rotation's +Z axis (identity rotation -> +Z), looking at center.
        position, look_at, _up = self.resolver._resolve_camera_view(
            [0, 0, 0], 3.0, [1, 1, 1], None, None)
        np.testing.assert_allclose(look_at, [1, 1, 1])
        np.testing.assert_allclose(position, [1, 1, 4])
        self.assertAlmostEqual(np.linalg.norm(position - look_at), 3.0)

    def test_transform_overrides_angles(self):
        transform = np.eye(4)
        transform[:3, 3] = [1, 2, 3]
        position, _look_at, _up = self.resolver._resolve_camera_view(
            None, None, None, None, transform)
        np.testing.assert_allclose(position, [1, 2, 3])

    def test_returns_none_without_pose_info(self):
        self.assertIsNone(
            self.resolver._resolve_camera_view(None, None, None, None, None))


if __name__ == '__main__':
    unittest.main()
