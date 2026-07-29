from contextlib import redirect_stdout
import inspect
import io
import threading
import time
import unittest
import warnings

from skrobot.viewers import ViserViewer
from skrobot.viewers._base import _InteractiveViewerMixin


class _DummyServer(object):

    def __init__(self):
        self.stop_count = 0

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


if __name__ == '__main__':
    unittest.main()
