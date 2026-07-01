import os
import tempfile
import unittest

import numpy as np

import skrobot
from skrobot.viewers import MitsubaViewer


def _mitsuba_available():
    try:
        import mitsuba  # noqa: F401
    except ImportError:
        return False
    return True


class TestMitsubaViewerRegistration(unittest.TestCase):
    """These checks do not need mitsuba to be installed."""

    def test_registered(self):
        self.assertIn('mitsuba', skrobot.viewers._VIEWER_CLASSES)
        self.assertIs(
            skrobot.viewers._VIEWER_CLASSES['mitsuba'], MitsubaViewer)

    def test_drop_in_methods_exist(self):
        for name in ('add', 'delete', 'set_camera', 'render', 'save_image',
                     'redraw', 'show', 'close', 'wait_until_close', 'pause',
                     'is_active'):
            self.assertTrue(
                hasattr(MitsubaViewer, name),
                'MitsubaViewer is missing {}'.format(name))


@unittest.skipUnless(_mitsuba_available(), 'mitsuba is not installed')
class TestMitsubaViewerRender(unittest.TestCase):

    def setUp(self):
        self.robot = skrobot.models.Panda()
        # small image + low spp keeps the offscreen render fast for CI
        self.viewer = MitsubaViewer(resolution=(160, 120), spp=4)

    def test_render_returns_uint8_image(self):
        self.viewer.add(self.robot)
        self.viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        image = self.viewer.render()
        # resolution is (width, height) -> image is (height, width, 3)
        self.assertEqual(image.shape, (120, 160, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_add_primitives_and_render(self):
        self.viewer.add(self.robot)
        self.viewer.add_sphere([0.3, 0.0, 0.6], 0.05, name='ball')
        self.viewer.add_box([0.3, 0.2, 0.4], [0.1, 0.1, 0.02], name='tray')
        image = self.viewer.render()
        self.assertEqual(image.shape, (120, 160, 3))

    def test_save_image(self):
        self.viewer.add(self.robot)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'render.png')
            self.viewer.save_image(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_redraw_without_window_re_renders(self):
        # redraw() must work even if show() was never called: it simply
        # re-renders and updates nothing to display.
        self.viewer.add(self.robot)
        self.viewer.redraw()
        self.assertEqual(self.viewer._last_image.shape, (120, 160, 3))

    def test_incremental_update_matches_full_rebuild(self):
        # After moving a joint, the cached-scene incremental render must match
        # a brand-new viewer that rebuilds the scene from scratch.
        self.viewer.add(self.robot)
        self.viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        before = self.viewer.render()                 # builds + caches scene
        self.robot.rarm.joint_list[1].joint_angle(-0.5)
        self.robot.rarm.joint_list[3].joint_angle(-1.8)
        incremental = self.viewer.render()            # transform-only update

        fresh = MitsubaViewer(resolution=(160, 120), spp=4)
        fresh.add(self.robot)
        fresh.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        full = fresh.render()

        moved = np.abs(incremental.astype(int) - before.astype(int)).mean()
        self.assertGreater(moved, 1.0)                # the pose actually changed
        diff = np.abs(incremental.astype(int) - full.astype(int)).mean()
        self.assertLess(diff, 3.0)                    # matches full rebuild

    def test_pause_renders_and_is_active_without_window(self):
        self.viewer.add(self.robot)
        # no show() -> no window -> not active, but pause() still re-renders
        self.assertFalse(self.viewer.is_active)
        self.viewer.pause(0.001)
        self.assertEqual(self.viewer._last_image.shape, (120, 160, 3))

    def test_variant_via_arg_and_env(self):
        import mitsuba as mi
        target = list(mi.variants())[0]
        viewer = MitsubaViewer(resolution=(32, 32), spp=1, variant=target)
        self.assertEqual(viewer.mi.variant(), target)
        os.environ['SKROBOT_MITSUBA_VARIANT'] = target
        try:
            auto = MitsubaViewer(resolution=(32, 32), spp=1)
            self.assertEqual(auto.mi.variant(), target)
        finally:
            del os.environ['SKROBOT_MITSUBA_VARIANT']

    def test_unknown_variant_raises(self):
        with self.assertRaises(ValueError):
            MitsubaViewer(variant='definitely_not_a_variant')

    def test_orbit_keeps_target_and_distance(self):
        self.viewer.add(self.robot)
        self.viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        self.viewer._init_orbit()
        eye0, target0, _ = self.viewer._effective_camera()
        dist0 = np.linalg.norm(eye0 - target0)
        # rotate the azimuth and re-derive the camera
        self.viewer._orbit_az += 0.5
        self.viewer._apply_orbit()
        eye1, target1, _ = self.viewer._camera
        np.testing.assert_allclose(target1, [0.0, 0.0, 0.5], atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(eye1 - target1), dist0, places=5)
        self.assertFalse(np.allclose(eye0, eye1))


if __name__ == '__main__':
    unittest.main()
