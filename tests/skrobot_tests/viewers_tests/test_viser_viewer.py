from contextlib import redirect_stdout
import inspect
import io
import threading
import time
import unittest
import warnings

import numpy as np

from skrobot.model.primitives import LineString
from skrobot.viewers import _viser as viser_module
from skrobot.viewers import ViserViewer
from skrobot.viewers._base import _InteractiveViewerMixin


class _RecordingScene(object):

    def __init__(self):
        self.line_segments_calls = []

    def add_line_segments(self, name, **kwargs):
        self.line_segments_calls.append((name, kwargs))
        return object()


class _DummyServer(object):

    def __init__(self):
        self.stop_count = 0
        self.scene = _RecordingScene()

    def stop(self):
        self.stop_count += 1


def _viewer():
    # Borrow methods without starting a real viser server / browser.
    viewer = ViserViewer.__new__(ViserViewer)
    viewer._is_active = True
    viewer._server = _DummyServer()
    return viewer


class TestViserViewer(unittest.TestCase):

    def test_wait_until_close_matches_shared_signature_except_message_default(self):
        shared = inspect.signature(
            _InteractiveViewerMixin.wait_until_close).parameters
        mine = inspect.signature(ViserViewer.wait_until_close).parameters
        for name in shared:
            self.assertIn(name, mine)
            if name == 'message':
                self.assertNotEqual(shared[name].default, mine[name].default)
                self.assertIsInstance(mine[name].default, str)
                self.assertNotEqual(mine[name].default, '')
            else:
                self.assertEqual(shared[name].default, mine[name].default)

    def test_wait_until_close_message_kwarg_prints_once_and_returns(self):
        viewer = _viewer()

        def _close_later():
            time.sleep(0.05)
            viewer.close()

        close_thread = threading.Thread(target=_close_later)
        close_thread.start()
        out = io.StringIO()
        with redirect_stdout(out):
            viewer.wait_until_close(message='x', interval=0.01)
        close_thread.join(timeout=1.0)

        self.assertFalse(close_thread.is_alive())
        self.assertEqual(out.getvalue().count('x'), 1)

    def test_check_interval_alias_warns_and_waits(self):
        viewer = _viewer()

        def _close_later():
            time.sleep(0.05)
            viewer.close()

        close_thread = threading.Thread(target=_close_later)
        close_thread.start()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            viewer.wait_until_close(message=None, check_interval=0.01)
        close_thread.join(timeout=1.0)

        self.assertFalse(close_thread.is_alive())
        self.assertTrue(caught)
        self.assertEqual(caught[0].category, DeprecationWarning)
        self.assertIn('interval', str(caught[0].message))

    def test_check_interval_and_interval_together_raises_type_error(self):
        viewer = _viewer()
        with self.assertRaises(TypeError):
            viewer.wait_until_close(interval=0.2, check_interval=0.1)

    def test_has_exit_matches_not_is_active(self):
        viewer = _viewer()
        self.assertEqual(viewer.has_exit, not viewer.is_active)
        viewer._is_active = False
        self.assertEqual(viewer.has_exit, not viewer.is_active)


class TestViserLineString(unittest.TestCase):

    def setUp(self):
        self.viewer = _viewer()
        self.viewer._line_thickness = 0.004
        self.viewer._linkid_to_handle = dict()
        self.viewer._linkid_to_link = dict()
        self.viewer._obstacle_link_ids = set()
        self.viewer._obstacle_original_colors = dict()
        self.points = np.array([[0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0]])

    def _add(self, line):
        self.viewer._add_link(line)
        self.assertEqual(len(self.viewer._server.scene.line_segments_calls), 1)
        return self.viewer._server.scene.line_segments_calls[0][1]

    def test_line_string_is_added_as_line_segments(self):
        line = LineString(self.points, color=[255, 0, 0, 255])
        kwargs = self._add(line)

        segments = kwargs['points']
        self.assertEqual(segments.shape, (2, 2, 3))
        np.testing.assert_allclose(segments[0], self.points[:2])
        np.testing.assert_allclose(segments[1], self.points[1:])
        np.testing.assert_array_equal(
            kwargs['colors'],
            np.tile(np.array([255, 0, 0], dtype=np.uint8), (2, 2, 1)))
        self.assertIn(str(id(line)), self.viewer._linkid_to_handle)

    def test_line_string_uses_link_world_pose(self):
        line = LineString(self.points, pos=(0.0, 0.0, 0.5))
        kwargs = self._add(line)
        np.testing.assert_allclose(kwargs['position'], [0.0, 0.0, 0.5])

    def test_line_string_without_color_falls_back_to_default(self):
        line = LineString(self.points)
        kwargs = self._add(line)
        np.testing.assert_array_equal(
            kwargs['colors'], np.array(viser_module._DEFAULT_LINE_COLOR))

    def test_line_thickness_is_forwarded_when_supported(self):
        line = LineString(self.points)
        kwargs = self._add(line)
        if viser_module._LINE_SEGMENTS_SUPPORTS_THICKNESS:
            self.assertEqual(kwargs['thickness'], 0.004)
        else:
            self.assertNotIn('thickness', kwargs)


if __name__ == '__main__':
    unittest.main()
