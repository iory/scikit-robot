import inspect
import os
import tempfile
import time
import types
import unittest

import numpy as np

import skrobot
from skrobot.coordinates import Coordinates
from skrobot.viewers import MitsubaViewer


def _mitsuba_available():
    try:
        import mitsuba  # noqa: F401
    except ImportError:
        return False
    return True


def _transform_matrix_element(transform, i, j):
    mat = np.asarray(transform.matrix)
    return float(np.asarray(mat)[..., i, j].reshape(-1)[0])


def _transform_uniform_scale(transform):
    mat = np.asarray(transform.matrix, float)
    return float(np.linalg.norm(mat[:3, 0]))


def _count_strong_red_pixels(image, min_red=120, margin=25):
    red = image[..., 0].astype(np.int16)
    green = image[..., 1].astype(np.int16)
    blue = image[..., 2].astype(np.int16)
    return int(np.sum((red >= min_red)
                      & ((red - green) >= margin)
                      & ((red - blue) >= margin)))


def _add_box_row(viewer, span, prefix='row'):
    xs = np.linspace(-0.5 * span, 0.5 * span, 5)
    for i, x in enumerate(xs):
        viewer.add_box([float(x), 0.0, 0.2], [0.35, 0.35, 0.35],
                       color=(0.7, 0.7, 0.7),
                       name='{}_{}'.format(prefix, i))


def _capture_mitsuba_logs(callback, log_level='Warn'):
    """Capture Mitsuba logger messages while running ``callback``."""
    import mitsuba as mi

    class _Collector(mi.Appender):
        def __init__(self):
            mi.Appender.__init__(self)
            self.messages = []

        def append(self, _level, text):
            self.messages.append(str(text))

    logger = mi.logger()
    old_level = logger.log_level()
    collector = _Collector()
    logger.add_appender(collector)
    logger.set_log_level(getattr(mi.LogLevel, log_level))
    try:
        callback()
        return list(collector.messages)
    finally:
        logger.remove_appender(collector)
        logger.set_log_level(old_level)


def _ply_header_property_lines(path):
    """Return ``property ...`` lines from a PLY header."""
    props = []
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            decoded = line.decode('ascii', errors='ignore').strip()
            if decoded == 'end_header':
                break
            if decoded.startswith('property '):
                props.append(decoded)
    return props


def _key_light_matrix(viewer):
    return np.asarray(viewer._scene_dict()['key']['to_world'].matrix, float)


def _frame_non_background_fraction(image, tolerance=0):
    """Return frame non-background fraction using the top-left corner color."""
    background = image[0, 0, :].astype(np.int16)
    diff = np.abs(image.astype(np.int16) - background)
    non_background = int(np.sum(np.any(diff > int(tolerance), axis=2)))
    total = int(image.shape[0] * image.shape[1])
    return float(non_background) / float(total), non_background, total


def _look_at_camera_from_viewer(
        viewer, angles, distance=None, center=None):
    import trimesh
    points = viewer._collect_world_points()
    if points is None:
        bounds = np.zeros((2, 3), dtype=np.float64)
        if distance is None:
            distance = 1.0
    else:
        bounds = np.vstack([points.min(axis=0), points.max(axis=0)])
    x_fov, y_fov = viewer._effective_sensor_fov_xy_radians()
    pose = trimesh.scene.cameras.look_at(
        points=bounds,
        fov=np.degrees([x_fov, y_fov]),
        rotation=trimesh.transformations.euler_matrix(*angles),
        distance=distance,
        center=center)
    eye = pose[:3, 3]
    if center is not None:
        target = np.asarray(center, dtype=np.float64).reshape(3)
    else:
        target = bounds.mean(axis=0)
    forward = -pose[:3, 2]
    forward_dir = forward / max(np.linalg.norm(forward), 1e-12)
    target_vec = target - eye
    target_dir = target_vec / max(np.linalg.norm(target_vec), 1e-12)
    np.testing.assert_allclose(forward_dir, target_dir, atol=1e-7, rtol=1e-6)
    up = pose[:3, 1]
    return eye, target, up


def _scene_center_from_viewer(viewer):
    points = viewer._collect_world_points()
    if points is None:
        return np.zeros(3, dtype=np.float64)
    return points.min(axis=0) + 0.5 * np.ptp(points, axis=0)


def _render_with_legacy_opaque_mask(viewer):
    """Render a scene where opaque link BSDFs are wrapped in mask(opacity=1)."""
    scene = viewer._scene_dict()
    for key, entry in scene.items():
        if not key.startswith('m_'):
            continue
        bsdf = entry.get('bsdf')
        if isinstance(bsdf, dict) and bsdf.get('type') != 'mask':
            entry['bsdf'] = {'type': 'mask',
                             'opacity': {'type': 'rgb', 'value': 1.0},
                             'nested': bsdf}
    try:
        compiled = viewer.mi.load_dict(scene, optimize=False)
    except TypeError:
        compiled = viewer.mi.load_dict(scene)
    image = viewer.mi.render(compiled, spp=viewer.spp)
    return np.array(viewer.mi.util.convert_to_bitmap(image))[..., :3]


class TestMitsubaViewerRegistration(unittest.TestCase):
    """These checks do not need mitsuba to be installed."""

    def test_registered(self):
        self.assertIn('mitsuba', skrobot.viewers._VIEWER_CLASSES)
        self.assertIs(
            skrobot.viewers._VIEWER_CLASSES['mitsuba'], MitsubaViewer)

    def test_drop_in_methods_exist(self):
        for name in ('add', 'delete', 'set_camera', 'render', 'save_image',
                     'redraw', 'show', 'close', 'wait_until_close', 'pause',
                     'is_active', 'has_exit',
                     'add_joint_axis', 'delete_joint_axis'):
            self.assertTrue(
                hasattr(MitsubaViewer, name),
                'MitsubaViewer is missing {}'.format(name))

    def test_wait_until_close_matches_the_shared_signature(self):
        # The trimesh / pyrender viewers get wait_until_close from
        # _InteractiveViewerMixin, and examples/skeleton_visualization.py calls
        # it with message=. This viewer used to take no arguments at all, so
        # that example died with a TypeError under --viewer mitsuba.
        from skrobot.viewers._base import _InteractiveViewerMixin
        shared = inspect.signature(
            _InteractiveViewerMixin.wait_until_close).parameters
        mine = inspect.signature(MitsubaViewer.wait_until_close).parameters
        for name in shared:
            self.assertIn(name, mine)
            self.assertEqual(shared[name].default, mine[name].default)

    def test_new_camera_and_scene_options_exist(self):
        init_params = inspect.signature(MitsubaViewer.__init__).parameters
        self.assertIn('update_interval', init_params)
        self.assertIn('title', init_params)
        self.assertIn('fov', init_params)
        self.assertIn('fov_axis', init_params)
        self.assertIn('ground_height', init_params)
        self.assertIn('ground_size', init_params)
        self.assertIn('light_intensity', init_params)
        self.assertIn('light_size', init_params)
        self.assertIn('line_radius', init_params)
        self.assertIn('point_radius', init_params)
        self.assertIn('ambient_light', init_params)
        set_camera_params = inspect.signature(MitsubaViewer.set_camera).parameters
        self.assertIn('fov', set_camera_params)

    def test_save_image_signature_is_compatible_for_video_backends(self):
        expected = inspect.signature(
            skrobot.viewers.TrimeshSceneViewer.save_image).parameters
        for name in ('trimesh', 'pyrender', 'mitsuba'):
            cls = skrobot.viewers._VIEWER_CLASSES[name]
            self.assertTrue(
                hasattr(cls, 'save_image'),
                '{} backend is missing save_image'.format(name))
            params = inspect.signature(cls.save_image).parameters
            self.assertEqual(tuple(params.keys()), tuple(expected.keys()))


@unittest.skipUnless(_mitsuba_available(), 'mitsuba is not installed')
class TestMitsubaViewerRender(unittest.TestCase):

    def setUp(self):
        os.environ.setdefault(
            'SKROBOT_CACHE_DIR', os.path.join(
                tempfile.gettempdir(), 'skrobot_cache'))
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

    def test_cached_link_local_geometry_round_trips_to_world_vertices(self):
        from skrobot.model.primitives import Box

        viewer = MitsubaViewer(resolution=(96, 72), spp=1)
        box = Box(extents=[0.24, 0.16, 0.11])
        box.translate([0.37, -0.21, 0.53])
        box.rotate(np.deg2rad(31.0), 'z')
        box.rotate(np.deg2rad(-19.0), 'y')
        viewer.add(box)
        viewer.set_camera(eye=[0.95, -0.95, 0.72], target=[0.37, -0.21, 0.53])
        viewer.render()  # build scene + cache local geometry

        entries = [entry for entry in viewer._mesh_local.values()
                   if entry.get('kind') == 'link' and entry.get('link') is box]
        self.assertGreaterEqual(len(entries), 1)

        coords = box.worldcoords()
        for entry in entries:
            world_vertices = coords.transform_vector(np.asarray(entry['v'], float))
            expected_vertices = np.asarray(
                viewer._params[entry['vpk']]).reshape(-1, 3)
            np.testing.assert_allclose(
                world_vertices, expected_vertices, atol=1e-12, rtol=1e-12)

            if 'nk' not in entry:
                continue
            world_normals = coords.rotate_vector(np.asarray(entry['n'], float))
            expected_normals = np.asarray(
                viewer._params[entry['nk']]).reshape(-1, 3)
            np.testing.assert_allclose(
                world_normals, expected_normals, atol=1e-12, rtol=1e-12)

    def test_moving_named_box_marker_does_not_rebuild_scene(self):
        viewer = MitsubaViewer(resolution=(64, 48), spp=1)
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        viewer.add_box([0.45, 0.0, 0.25], [0.05, 0.05, 0.05],
                       color=(0.9, 0.2, 0.2), name='cube')
        build_calls = {'count': 0}
        build_scene = viewer._build_scene

        def wrapped_build_scene():
            build_calls['count'] += 1
            return build_scene()

        viewer._build_scene = wrapped_build_scene
        viewer.render()  # first and only scene build
        for i in range(5):
            self.robot.rarm.joint_list[1].joint_angle(-0.4 - 0.03 * i)
            self.robot.rarm.joint_list[3].joint_angle(-1.4 - 0.04 * i)
            viewer.add_box(
                [0.45 + 0.01 * np.sin(i), 0.02 * np.cos(i), 0.25],
                [0.05, 0.05, 0.05], color=(0.9, 0.2, 0.2), name='cube')
            viewer.render()
        self.assertEqual(build_calls['count'], 1)

    def test_incremental_box_marker_update_matches_full_rebuild(self):
        self.viewer.add(self.robot)
        self.viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        start = [0.30, -0.20, 0.30]
        goal = [0.52, 0.20, 0.34]
        extents = [0.08, 0.08, 0.08]
        color = (0.85, 0.15, 0.15)
        self.viewer.add_box(start, extents, color=color, name='cube')
        before = self.viewer.render()
        self.viewer.add_box(goal, extents, color=color, name='cube')
        incremental = self.viewer.render()

        fresh = MitsubaViewer(resolution=(160, 120), spp=4)
        fresh.add(self.robot)
        fresh.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        fresh.add_box(goal, extents, color=color, name='cube')
        full = fresh.render()

        moved = np.abs(incremental.astype(int) - before.astype(int)).mean()
        self.assertGreater(moved, 1.0)
        diff = np.abs(incremental.astype(int) - full.astype(int)).mean()
        self.assertLess(diff, 3.0)

    def test_readding_named_box_with_new_color_rebuilds(self):
        viewer = MitsubaViewer(resolution=(96, 72), spp=8)
        viewer.set_camera(eye=[0.9, -0.9, 0.6], target=[0.35, 0.0, 0.2])
        viewer.add_box([0.35, 0.0, 0.2], [0.18, 0.18, 0.18],
                       color=(0.9, 0.1, 0.1), name='cube')
        build_calls = {'count': 0}
        build_scene = viewer._build_scene

        def wrapped_build_scene():
            build_calls['count'] += 1
            return build_scene()

        viewer._build_scene = wrapped_build_scene
        red = viewer.render()
        viewer.add_box([0.35, 0.0, 0.2], [0.18, 0.18, 0.18],
                       color=(0.1, 0.1, 0.9), name='cube')
        blue = viewer.render()

        changed = np.abs(blue.astype(int) - red.astype(int)).mean()
        self.assertGreater(changed, 1.0)
        self.assertEqual(build_calls['count'], 2)

    def test_incremental_sphere_marker_update_matches_full_rebuild(self):
        viewer = MitsubaViewer(resolution=(160, 120), spp=4)
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        start = [0.25, -0.15, 0.30]
        goal = [0.58, 0.20, 0.36]
        radius = 0.11
        color = (0.2, 0.7, 0.3)
        viewer.add_sphere(start, radius, color=color, name='ball')
        before = viewer.render()
        viewer.add_sphere(goal, radius, color=color, name='ball')
        incremental = viewer.render()

        fresh = MitsubaViewer(resolution=(160, 120), spp=4)
        fresh.add(self.robot)
        fresh.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        fresh.add_sphere(goal, radius, color=color, name='ball')
        full = fresh.render()

        moved = np.abs(incremental.astype(int) - before.astype(int)).mean()
        self.assertGreater(moved, 1.0)
        diff = np.abs(incremental.astype(int) - full.astype(int)).mean()
        self.assertLess(diff, 3.0)

    def test_identical_geometry_links_move_independently(self):
        # Two links that share the same mesh *and* colour are exactly what
        # Mitsuba's shape-merging optimisation collapses into a single shape.
        # A merged shape exposes no per-link vertices, which used to make the
        # incremental update silently skip -- freezing such links at their
        # initial pose (e.g. a grasped box that never followed the gripper).
        # Moving one of them must still change the render.
        from skrobot.model.primitives import Box
        a = Box(extents=[0.12, 0.12, 0.12])
        a.translate([0.25, 0.0, 0.3])
        b = Box(extents=[0.12, 0.12, 0.12])
        b.translate([-0.25, 0.0, 0.3])
        self.viewer.add(a)
        self.viewer.add(b)
        self.viewer.set_camera(eye=[0.0, -1.6, 0.4], target=[0.0, 0.0, 0.3])
        before = self.viewer.render()
        b.translate([0.0, 0.0, 0.5], 'world')         # move only the second box
        after = self.viewer.render()
        moved = np.abs(after.astype(int) - before.astype(int)).mean()
        self.assertGreater(moved, 0.5)                # the moved box is not frozen

    def test_add_line_geometry_does_not_crash(self):
        # A LineString's visual mesh is a trimesh Path3D (no faces), so the
        # Mitsuba backend needs to convert it into renderable tube geometry.
        # add() must not raise and the line must be visible.
        from skrobot.model.primitives import LineString
        viewer = MitsubaViewer(resolution=(200, 150), spp=64, variant='llvm_ad_rgb')
        viewer.set_camera(eye=[0.3, -0.9, 0.35], target=[0.3, 0.0, 0.0])
        before = viewer.render()
        before_red = _count_strong_red_pixels(before)

        line = LineString(np.array([[0.0, -0.2, 0.01],
                                    [0.2, -0.1, 0.01],
                                    [0.4, 0.0, 0.01],
                                    [0.6, 0.1, 0.01]]),
                          color=[255, 0, 0, 255])
        viewer.add(line)  # must not raise
        line_entries = [k for k, v in viewer._links.items() if v[0] is line]
        self.assertGreaterEqual(len(line_entries), 1)

        image = viewer.render()
        after_red = _count_strong_red_pixels(image)
        self.assertGreater(after_red, before_red + 60)
        self.assertEqual(image.shape, (150, 200, 3))

    def test_point_cloud_geometry_is_visible(self):
        from skrobot.model.primitives import PointCloudLink
        viewer = MitsubaViewer(resolution=(200, 150), spp=64, variant='llvm_ad_rgb')
        viewer.set_camera(eye=[0.4, -1.0, 0.45], target=[0.3, 0.0, 0.0])
        before = viewer.render()
        before_red = _count_strong_red_pixels(before)

        xs = np.linspace(0.05, 0.55, 8)
        ys = np.linspace(-0.2, 0.2, 5)
        points = np.array([[x, y, 0.02] for x in xs for y in ys], dtype=float)
        cloud = PointCloudLink(points, colors=np.array([255, 0, 0, 255]))
        viewer.add(cloud)
        cloud_entries = [k for k, v in viewer._links.items() if v[0] is cloud]
        self.assertGreaterEqual(len(cloud_entries), 1)

        image = viewer.render()
        after_red = _count_strong_red_pixels(image)
        self.assertGreaterEqual(after_red, before_red + 80)

    def test_line_radius_changes_rendered_thickness(self):
        from skrobot.model.primitives import LineString
        points = np.array([[0.0, -0.2, 0.01],
                           [0.2, -0.1, 0.01],
                           [0.4, 0.0, 0.01],
                           [0.6, 0.1, 0.01]], dtype=float)
        line = LineString(points, color=[255, 0, 0, 255])
        eye = [0.3, -0.9, 0.35]
        target = [0.3, 0.0, 0.0]
        thin = MitsubaViewer(
            resolution=(200, 150), spp=64, variant='llvm_ad_rgb',
            line_radius=0.001)
        thick = MitsubaViewer(
            resolution=(200, 150), spp=64, variant='llvm_ad_rgb',
            line_radius=0.02)
        thin.set_camera(eye=eye, target=target)
        thick.set_camera(eye=eye, target=target)
        thin.add(line)
        thick.add(LineString(points, color=[255, 0, 0, 255]))
        image_thin = thin.render()
        image_thick = thick.render()
        thin_red = _count_strong_red_pixels(image_thin)
        thick_red = _count_strong_red_pixels(image_thick)
        self.assertGreater(thick_red, thin_red)

    def test_point_radius_changes_rendered_thickness(self):
        from skrobot.model.primitives import PointCloudLink
        points = np.array([[0.1, -0.1, 0.02],
                           [0.2, 0.0, 0.02],
                           [0.3, 0.1, 0.02],
                           [0.4, 0.0, 0.02],
                           [0.5, -0.1, 0.02]], dtype=float)
        eye = [0.3, -0.9, 0.35]
        target = [0.3, 0.0, 0.0]
        thin = MitsubaViewer(
            resolution=(200, 150), spp=64, variant='llvm_ad_rgb',
            point_radius=0.002)
        thick = MitsubaViewer(
            resolution=(200, 150), spp=64, variant='llvm_ad_rgb',
            point_radius=0.02)
        thin.set_camera(eye=eye, target=target)
        thick.set_camera(eye=eye, target=target)
        thin.add(PointCloudLink(points, colors=np.array([255, 0, 0, 255])))
        thick.add(PointCloudLink(points, colors=np.array([255, 0, 0, 255])))
        image_thin = thin.render()
        image_thick = thick.render()
        thin_red = _count_strong_red_pixels(image_thin)
        thick_red = _count_strong_red_pixels(image_thick)
        self.assertGreater(thick_red, thin_red)

    def test_add_line_with_repeated_vertices_does_not_raise(self):
        from skrobot.model.primitives import LineString
        line = LineString(np.array([[0.0, 0.0, 0.02],
                                    [0.0, 0.0, 0.02],
                                    [0.3, 0.1, 0.02]]),
                          color=[255, 0, 0, 255])
        viewer = MitsubaViewer(resolution=(160, 120), spp=16, variant='llvm_ad_rgb')
        viewer.set_camera(eye=[0.3, -0.8, 0.3], target=[0.15, 0.0, 0.0])
        viewer.add(line)
        image = viewer.render()
        self.assertEqual(image.shape, (120, 160, 3))
        self.assertGreaterEqual(
            len([k for k, v in viewer._links.items() if v[0] is line]), 1)

    def test_key_light_does_not_occlude_the_camera(self):
        # The key light is a rectangle above the subject and Mitsuba's area
        # emitter is one-sided: from behind it emits nothing yet still occludes.
        # A camera on its back side used to see nothing but that black
        # underside. This is the camera examples/pr2_inverse_kinematics.py asks
        # for, which rendered a completely black frame.
        self.viewer.add(self.robot)
        self.viewer.set_camera([np.deg2rad(45), 0.0, np.deg2rad(135)],
                               distance=2.5)
        image = self.viewer.render()
        self.assertGreater(image.max(), 0)         # not an all-black frame
        self.assertGreater(image.mean(), 20.0)

    def test_multicolor_mesh_keeps_its_colors(self):
        # An Axis mixes red/green/blue (and white) faces in a single mesh. It
        # must be split into one shape per colour rather than averaged into a
        # single grey, so the axes keep their colours in the render.
        from skrobot.model.primitives import Axis
        axis = Axis(axis_radius=0.02, axis_length=0.2)
        self.viewer.add(axis)
        colors = [tuple(np.round(v[2], 3)) for v in self.viewer._links.values()]
        self.assertGreaterEqual(len(set(colors)), 3)   # not averaged to one
        # a strongly red, green and blue submesh should each be present
        arr = np.array([list(c) for c in colors])
        self.assertTrue(np.any((arr[:, 0] > 0.5) & (arr[:, 1] < 0.5)))  # red
        self.assertTrue(np.any((arr[:, 1] > 0.5) & (arr[:, 0] < 0.5)))  # green
        self.assertTrue(np.any((arr[:, 2] > 0.5) & (arr[:, 0] < 0.5)))  # blue

    def test_transparent_front_panel_reveals_red_pixels_behind(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(
            resolution=(200, 150), spp=64, variant='llvm_ad_rgb')
        viewer.set_camera(eye=[0.0, -1.2, 0.15], target=[0.0, 0.0, 0.12])

        back = Box(extents=[0.3, 0.3, 0.16])
        back.translate([0.0, 0.0, 0.08], 'world')
        back.set_color([255, 0, 0, 255])
        panel = Box(extents=[1.2, 0.01, 1.2])
        panel.translate([0.0, -1.1, 0.12], 'world')

        viewer.add(back)
        viewer.add(panel)
        panel.set_color([220, 220, 220, 255])
        opaque = viewer.render()
        opaque_red = _count_strong_red_pixels(opaque)

        panel.set_color([220, 220, 220, 40])
        transparent = viewer.render()
        transparent_red = _count_strong_red_pixels(transparent)

        self.assertLessEqual(opaque_red, 5)
        self.assertGreater(transparent_red, opaque_red + 1000)

    def test_set_alpha_after_add_changes_render(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(
            resolution=(200, 150), spp=64, variant='llvm_ad_rgb')
        viewer.set_camera(eye=[0.0, -1.2, 0.15], target=[0.0, 0.0, 0.12])

        back = Box(extents=[0.3, 0.3, 0.16])
        back.translate([0.0, 0.0, 0.08], 'world')
        back.set_color([255, 0, 0, 255])
        panel = Box(extents=[1.2, 0.01, 1.2])
        panel.translate([0.0, -1.1, 0.12], 'world')
        panel.set_color([220, 220, 220, 255])

        viewer.add(back)
        viewer.add(panel)
        before = viewer.render()
        before_red = _count_strong_red_pixels(before)

        panel.set_alpha(0.3)
        after = viewer.render()
        after_red = _count_strong_red_pixels(after)
        diff = np.abs(after.astype(int) - before.astype(int)).mean()

        self.assertGreater(after_red, before_red + 1000)
        self.assertGreater(diff, 5.0)

    def test_opaque_panda_render_matches_legacy_and_uses_no_mask(self):
        import mitsuba as mi
        if 'llvm_ad_rgb' not in mi.variants():
            self.skipTest('llvm_ad_rgb is not available')

        viewer = MitsubaViewer(
            resolution=(160, 120), spp=4, ground=True, variant='llvm_ad_rgb')
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        scene = viewer._scene_dict()

        link_entries = [v for k, v in scene.items() if k.startswith('m_')]
        self.assertGreater(len(link_entries), 0)
        for entry in link_entries:
            self.assertNotEqual(entry['bsdf'].get('type'), 'mask')

        image = viewer.render().astype(np.int16)
        legacy = _render_with_legacy_opaque_mask(viewer).astype(np.int16)
        abs_diff = np.abs(image - legacy)
        # llvm_ad_rgb path tracing can differ by one LSB across runs, so exact
        # digest matching flakes; compare with a small pixel tolerance instead.
        self.assertLess(float(abs_diff.mean()), 0.05)
        self.assertLessEqual(int(abs_diff.max()), 2)

    def test_mixed_alpha_faces_are_split_into_separate_shapes(self):
        from skrobot.model.primitives import Box
        box = Box(extents=[0.2, 0.2, 0.2])
        colors = np.asarray(box.concatenated_visual_mesh.visual.face_colors).copy()
        n_faces = len(colors)
        colors[:n_faces // 2, 3] = 255
        colors[n_faces // 2:, 3] = 128
        box.concatenated_visual_mesh.visual.face_colors = colors

        viewer = MitsubaViewer(resolution=(64, 48), spp=1, variant='llvm_ad_rgb')
        viewer.add(box)

        box_entries = [v for v in viewer._links.values() if v[0] is box]
        self.assertGreaterEqual(len(box_entries), 2)
        alphas = sorted(set(float(np.round(v[2][3], 3)) for v in box_entries))
        self.assertIn(1.0, alphas)
        self.assertIn(float(np.round(128.0 / 255.0, 3)), alphas)

    def test_rgba_named_box_marker_alpha_change_rebuilds(self):
        viewer = MitsubaViewer(resolution=(96, 72), spp=8, variant='llvm_ad_rgb')
        viewer.set_camera(eye=[0.9, -0.9, 0.6], target=[0.35, 0.0, 0.2])
        viewer.add_box([0.35, 0.0, 0.2], [0.18, 0.18, 0.18],
                       color=(0.1, 0.7, 0.1, 1.0), name='cube')
        build_calls = {'count': 0}
        build_scene = viewer._build_scene

        def wrapped_build_scene():
            build_calls['count'] += 1
            return build_scene()

        viewer._build_scene = wrapped_build_scene
        opaque = viewer.render()
        viewer.add_box([0.35, 0.0, 0.2], [0.18, 0.18, 0.18],
                       color=(0.1, 0.7, 0.1, 0.3), name='cube')
        transparent = viewer.render()

        changed = np.abs(transparent.astype(int) - opaque.astype(int)).mean()
        self.assertGreater(changed, 2.0)
        self.assertEqual(build_calls['count'], 2)

    def test_pause_renders_and_is_active_without_window(self):
        self.viewer.add(self.robot)
        # no show() -> no window -> not active, but pause() still re-renders
        self.assertFalse(self.viewer.is_active)
        self.viewer.pause(0.001)
        self.assertEqual(self.viewer._last_image.shape, (120, 160, 3))

    def test_pause_without_show_honors_duration_and_validates_fps(self):
        self.viewer.add(self.robot)
        t0 = time.monotonic()
        self.viewer.pause(0.0)
        single_redraw = time.monotonic() - t0

        t0 = time.monotonic()
        self.viewer.pause(2.0)
        elapsed = time.monotonic() - t0

        t0 = time.monotonic()
        self.viewer.pause(0.001)
        short_elapsed = time.monotonic() - t0

        self.assertGreater(elapsed, 1.5)
        self.assertGreater(elapsed, single_redraw + 1.0)
        self.assertLess(short_elapsed, 1.0)
        with self.assertRaises(ValueError):
            self.viewer.pause(0.001, fps=0)
        with self.assertRaises(ValueError):
            self.viewer.pause(0.001, fps=float('nan'))

    def test_init_accepts_title_and_update_interval(self):
        viewer = MitsubaViewer(
            resolution=(320, 240), update_interval=0.1, title='x',
            spp=1)
        self.assertEqual(viewer.resolution, (320, 240))

    def test_has_exit_matches_not_is_active(self):
        viewer = MitsubaViewer(resolution=(64, 48), spp=1)
        self.assertEqual(viewer.has_exit, not viewer.is_active)
        try:
            viewer.show(block=False)
        except ImportError:
            self.skipTest('matplotlib is not installed')
        self.assertEqual(viewer.has_exit, not viewer.is_active)
        viewer.close()
        self.assertEqual(viewer.has_exit, not viewer.is_active)

    def test_add_and_delete_joint_axis(self):
        viewer = MitsubaViewer(resolution=(96, 72), spp=8)
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        joint = self.robot.rarm.joint_list[0]
        before_count = len(viewer._extra)

        viewer.add_joint_axis(joint)
        first_names = viewer._joint_axis_map[str(id(joint))]
        self.assertEqual(len(viewer._extra), before_count + 2)
        image = viewer.render()
        self.assertEqual(image.shape, (72, 96, 3))

        self.robot.rarm.joint_list[1].joint_angle(-0.3)
        viewer.add_joint_axis(joint)
        second_names = viewer._joint_axis_map[str(id(joint))]
        self.assertEqual(first_names, second_names)
        self.assertEqual(len(viewer._extra), before_count + 2)

        viewer.delete_joint_axis(joint)
        self.assertNotIn(str(id(joint)), viewer._joint_axis_map)
        self.assertEqual(len(viewer._extra), before_count)

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

    def test_fov_scalar_changes_render(self):
        eye = [1.5, -1.5, 1.0]
        target = [0.0, 0.0, 0.5]
        narrow = MitsubaViewer(resolution=(160, 120), spp=8, fov=20)
        wide = MitsubaViewer(resolution=(160, 120), spp=8, fov=60)
        narrow.add(self.robot)
        wide.add(self.robot)
        narrow.set_camera(eye=eye, target=target)
        wide.set_camera(eye=eye, target=target)
        img_narrow = narrow.render()
        img_wide = wide.render()
        diff = np.abs(img_narrow.astype(int) - img_wide.astype(int)).mean()
        self.assertGreater(diff, 1.0)

    def test_set_camera_angles_distance_sets_camera_and_changes_render(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=8)
        viewer.add(self.robot)
        auto = viewer.render()
        viewer.set_camera(angles=[np.deg2rad(30), 0, 0], distance=2.5)
        self.assertIsNotNone(viewer._camera)
        angled = viewer.render()
        diff = np.abs(auto.astype(int) - angled.astype(int)).mean()
        self.assertGreater(diff, 1.0)

    def test_set_camera_angles_explicit_distance_is_used_verbatim(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=4)
        viewer.add(self.robot)
        angles = [np.deg2rad(30), 0, np.deg2rad(45)]
        requested_distance = 2.5
        viewer.set_camera(angles=angles, distance=requested_distance)
        expected = _look_at_camera_from_viewer(
            viewer, angles, distance=requested_distance)
        np.testing.assert_allclose(viewer._camera[0], expected[0], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[1], expected[1], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[2], expected[2], atol=1e-8)

    def test_set_camera_positional_argument_is_angles(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=8)
        viewer.add(self.robot)
        viewer.set_camera([0, 0, np.pi / 2.0])
        self.assertIsNotNone(viewer._camera)
        image_a = viewer.render()
        viewer.set_camera([0, 0, 0])
        image_b = viewer.render()
        diff = np.abs(image_a.astype(int) - image_b.astype(int)).mean()
        self.assertGreater(diff, 1.0)

    def test_set_camera_angles_without_distance_frames_pr2_like_pyrender(self):
        import mitsuba as mi
        if 'llvm_ad_rgb' not in mi.variants():
            self.skipTest('llvm_ad_rgb is not available')

        viewer = MitsubaViewer(
            resolution=(640, 480), spp=16, ground=False, variant='llvm_ad_rgb')
        viewer.add(skrobot.models.PR2())
        viewer.set_camera(angles=[0, 0, np.pi / 2.0])
        image = viewer.render()
        fraction, non_background, total = _frame_non_background_fraction(
            image, tolerance=35)
        # Measured on PR2 (640x480, llvm_ad_rgb, spp=16):
        # legacy sphere-fit + scalar 45/x-axis default ~= 0.0424
        # shared look_at + default (60, 45) tuple ~= 0.0769
        self.assertGreater(
            fraction, 0.07,
            'non-background pixels: {}/{} ({:.2%})'.format(
                non_background, total, fraction))
        self.assertLess(
            fraction, 0.09,
            'non-background pixels: {}/{} ({:.2%})'.format(
                non_background, total, fraction))

    def test_set_camera_angles_without_distance_matches_trimesh_look_at(self):
        viewer = MitsubaViewer(resolution=(640, 480), spp=4, ground=False)
        viewer.add(self.robot)
        angles = [0, 0, np.pi / 2.0]
        viewer.set_camera(angles=angles)
        expected = _look_at_camera_from_viewer(viewer, angles)
        np.testing.assert_allclose(viewer._camera[0], expected[0], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[1], expected[1], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[2], expected[2], atol=1e-8)

    def test_set_camera_angles_without_distance_keeps_true_look_at_distance(self):
        import trimesh
        viewer = MitsubaViewer(resolution=(640, 480), spp=4, ground=False)
        viewer.add(skrobot.models.PR2())
        angles = [np.deg2rad(50), 0.0, np.deg2rad(30)]
        viewer.set_camera(angles=angles)
        eye, target, _up = viewer._camera
        points = viewer._collect_world_points()
        bounds = np.vstack([points.min(axis=0), points.max(axis=0)])
        scene_center = bounds.mean(axis=0)
        x_fov, y_fov = viewer._effective_sensor_fov_xy_radians()
        pose = trimesh.scene.cameras.look_at(
            points=bounds,
            fov=np.degrees([x_fov, y_fov]),
            rotation=trimesh.transformations.euler_matrix(*angles),
            distance=None,
            center=None)
        expected_distance = float(np.linalg.norm(pose[:3, 3] - scene_center))
        resolved_distance = float(np.linalg.norm(eye - target))
        np.testing.assert_allclose(target, scene_center, atol=1e-8)
        self.assertAlmostEqual(resolved_distance, expected_distance, places=7)

    def test_set_camera_angles_centers_ground_and_key_like_eye_target(self):
        angles = [np.deg2rad(50), 0.0, np.deg2rad(30)]
        from_angles = MitsubaViewer(resolution=(128, 96), spp=4, ground=True)
        from_angles.add(skrobot.models.PR2())
        from_angles.set_camera(angles=angles)
        scene_center = _scene_center_from_viewer(from_angles)
        eye, _target, up = from_angles._camera

        from_eye_target = MitsubaViewer(
            resolution=(128, 96), spp=4, ground=True)
        from_eye_target.add(skrobot.models.PR2())
        from_eye_target.set_camera(eye=eye, target=scene_center, up=up)

        scene_a = from_angles._scene_dict()
        scene_b = from_eye_target._scene_dict()
        ground_center_a = np.asarray(
            scene_a['ground']['to_world'].matrix, float)[:3, 3]
        ground_center_b = np.asarray(
            scene_b['ground']['to_world'].matrix, float)[:3, 3]
        key_origin_a = np.asarray(scene_a['key']['to_world'].matrix, float)[:3, 3]
        key_origin_b = np.asarray(scene_b['key']['to_world'].matrix, float)[:3, 3]
        np.testing.assert_allclose(ground_center_a, ground_center_b, atol=1e-8)
        np.testing.assert_allclose(key_origin_a, key_origin_b, atol=1e-8)

    def test_set_camera_angles_init_orbit_uses_scene_center_distance(self):
        viewer = MitsubaViewer(resolution=(640, 480), spp=4, ground=False)
        viewer.add(skrobot.models.PR2())
        angles = [np.deg2rad(50), 0.0, np.deg2rad(30)]
        viewer.set_camera(angles=angles)
        eye = np.asarray(viewer._camera[0], float)
        scene_center = _scene_center_from_viewer(viewer)
        true_distance = float(np.linalg.norm(eye - scene_center))
        viewer._init_orbit()
        np.testing.assert_allclose(viewer._orbit_target, scene_center, atol=1e-8)
        self.assertAlmostEqual(viewer._orbit_dist, true_distance, places=7)

    def test_set_camera_angles_without_distance_respects_non_default_fov(self):
        viewer = MitsubaViewer(
            resolution=(640, 480), spp=4, ground=False, fov=20.0)
        viewer.add(self.robot)
        angles = [0, 0, np.pi / 2.0]
        viewer.set_camera(angles=angles)
        expected = _look_at_camera_from_viewer(viewer, angles)
        np.testing.assert_allclose(viewer._camera[0], expected[0], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[1], expected[1], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[2], expected[2], atol=1e-8)

        default_viewer = MitsubaViewer(
            resolution=(640, 480), spp=4, ground=False)
        default_viewer.add(self.robot)
        default_camera = _look_at_camera_from_viewer(default_viewer, angles)
        self.assertGreater(
            np.linalg.norm(viewer._camera[0] - default_camera[0]),
            0.1)

    def test_auto_camera_keeps_legacy_offset_formula(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=4)
        viewer.add(self.robot)
        points = viewer._collect_world_points()
        center = points.min(axis=0) + 0.5 * np.ptp(points, axis=0)
        radius = max(0.3, 0.5 * float(np.linalg.norm(np.ptp(points, axis=0))))
        eye, target, up = viewer._auto_camera()
        expected_eye = center + radius * np.array([2.6, -2.2, 1.9])
        np.testing.assert_allclose(eye, expected_eye, atol=1e-12)
        np.testing.assert_allclose(target, center, atol=1e-12)
        np.testing.assert_allclose(up, [0.0, 0.0, 1.0], atol=1e-12)

    def test_set_camera_eye_target_up_take_precedence(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=4)
        viewer.add(self.robot)
        eye = np.array([1.1, -0.9, 0.8])
        target = np.array([0.2, 0.1, 0.4])
        up = np.array([0.0, 1.0, 0.0])
        viewer.set_camera(
            angles=[np.deg2rad(30), 0, 0],
            distance=2.5,
            eye=eye,
            target=target,
            up=up)
        np.testing.assert_allclose(viewer._camera[0], eye, atol=1e-8)
        np.testing.assert_allclose(viewer._camera[1], target, atol=1e-8)
        np.testing.assert_allclose(viewer._camera[2], up, atol=1e-8)

    def test_set_camera_coords_or_transform_uses_camera_pose(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=4)
        viewer.add(self.robot)
        coords = Coordinates(pos=[0.8, -1.1, 0.9])
        viewer.set_camera(coords_or_transform=coords)
        self.assertIsNotNone(viewer._camera)
        np.testing.assert_allclose(viewer._camera[0], [0.8, -1.1, 0.9],
                                   atol=1e-8)

    def test_set_camera_resolution_changes_next_render_size(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=4)
        viewer.add(self.robot)
        viewer.render()
        viewer.set_camera(resolution=(96, 72))
        image = viewer.render()
        self.assertEqual(image.shape, (72, 96, 3))

    def test_set_camera_angles_distance_center_matches_trimesh_look_at(self):
        viewer = MitsubaViewer(resolution=(128, 96), spp=4)
        viewer.add(self.robot)
        angles = [np.deg2rad(30), np.deg2rad(-10), np.deg2rad(45)]
        distance = 2.5
        center = [0.1, -0.2, 0.3]
        fov = (60, 45)
        viewer.set_camera(angles=angles, distance=distance, center=center,
                          fov=fov)
        self.assertIsNotNone(viewer._camera)
        expected = _look_at_camera_from_viewer(
            viewer, angles, distance=distance, center=center)
        np.testing.assert_allclose(viewer._camera[0], expected[0], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[1], expected[1], atol=1e-8)
        np.testing.assert_allclose(viewer._camera[2], expected[2], atol=1e-8)

    def test_set_camera_fov_after_first_render_rebuilds_scene(self):
        viewer = MitsubaViewer(resolution=(96, 72), spp=64, fov=20)
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        first = viewer.render()      # first build
        same = viewer.render()
        baseline = np.abs(same.astype(int) - first.astype(int)).mean()
        viewer.set_camera(fov=60)  # must force scene rebuild for new sensor fov
        changed = viewer.render()
        changed_diff = np.abs(changed.astype(int) - same.astype(int)).mean()
        self.assertGreater(changed_diff, baseline + 1.0)

    def test_two_element_fov_uses_y_axis(self):
        viewer = MitsubaViewer(resolution=(160, 120), spp=4, fov=(60, 45))
        sensor = viewer._scene_dict()['sensor']
        self.assertAlmostEqual(sensor['fov'], 45.0, places=7)
        self.assertEqual(sensor['fov_axis'], 'y')

    def test_ground_height_auto_detects_lowest_geometry(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(
            resolution=(160, 120), spp=4, ground=True, ground_height=None)
        box = Box(extents=[0.2, 0.2, 0.2])
        box.translate([0.0, 0.0, -0.5])
        viewer.add(box)
        ground = viewer._scene_dict()['ground']
        ground_z = _transform_matrix_element(ground['to_world'], 2, 3)
        self.assertAlmostEqual(ground_z, -0.6, places=6)

    def test_ground_height_explicit_overrides_auto_detection(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(
            resolution=(160, 120), spp=4, ground=True, ground_height=0.3)
        box = Box(extents=[0.2, 0.2, 0.2])
        box.translate([0.0, 0.0, -0.5])
        viewer.add(box)
        ground = viewer._scene_dict()['ground']
        ground_z = _transform_matrix_element(ground['to_world'], 2, 3)
        self.assertAlmostEqual(ground_z, 0.3, places=7)

    def test_light_intensity_and_ambient_light_in_scene_dict(self):
        viewer = MitsubaViewer(
            resolution=(160, 120), spp=4, light_intensity=3.5,
            ambient_light=0.33)
        scene = viewer._scene_dict()
        self.assertAlmostEqual(
            scene['key']['emitter']['radiance']['value'], 3.5, places=7)
        self.assertAlmostEqual(
            scene['ambient']['radiance']['value'], 0.33, places=7)

    def test_key_light_keeps_legacy_size_for_single_robot_scenes(self):
        # Backward-compatibility guard: with the one-sided clamp, single-robot
        # scenes (Panda/PR2) must match legacy constants exactly.
        scenes = [
            (skrobot.models.Panda(), [1.45, -1.55, 1.05], [0.32, 0.05, 0.38]),
            (skrobot.models.PR2(), [2.5, -2.4, 1.8], [0.0, 0.0, 0.9]),
        ]
        for robot, eye, target in scenes:
            viewer = MitsubaViewer(resolution=(64, 48), spp=1)
            viewer.add(robot)
            viewer.set_camera(eye=eye, target=target)
            key = viewer._scene_dict()['key']['to_world']
            scale = _transform_uniform_scale(key)
            height = _transform_matrix_element(
                key, 2, 3) - float(np.asarray(viewer._camera[1])[2])
            self.assertAlmostEqual(scale, 1.5, places=6)
            self.assertAlmostEqual(height, 2.0, places=6)

    def test_key_light_radius_is_clamped_for_tiny_scenes(self):
        viewer = MitsubaViewer(resolution=(64, 48), spp=1)
        viewer.add_box([0.0, 0.0, 0.05], [0.1, 0.1, 0.1], name='tiny')
        key = viewer._scene_dict()['key']['to_world']
        self.assertAlmostEqual(_transform_uniform_scale(key), 1.5, places=6)

    def test_key_light_scales_with_scene_extent(self):
        small = MitsubaViewer(resolution=(160, 120), spp=8)
        large = MitsubaViewer(resolution=(160, 120), spp=8)
        for viewer in (small, large):
            viewer.set_camera(eye=[0.0, -9.0, 3.0], target=[0.0, 0.0, 0.2])
        _add_box_row(small, 1.5, prefix='small')
        _add_box_row(large, 8.0, prefix='large')

        key_small = small._scene_dict()['key']['to_world']
        key_large = large._scene_dict()['key']['to_world']
        scale_small = _transform_uniform_scale(key_small)
        scale_large = _transform_uniform_scale(key_large)
        height_small = _transform_matrix_element(key_small, 2, 3) - 0.2
        height_large = _transform_matrix_element(key_large, 2, 3) - 0.2

        self.assertGreater(scale_large / scale_small, 2.5)
        self.assertGreater(height_large / height_small, 2.5)

    def test_large_scene_is_not_underlit_by_fixed_key_light(self):
        # Measured on the same 5-box 8m scene at 320x240: fixed light gives
        # mean brightness ~110, scaled light ~174 (cuda_ad_rgb, 128 spp).
        viewer = MitsubaViewer(resolution=(320, 240), spp=32)
        viewer.set_camera(eye=[0.0, -11.0, 3.2], target=[0.0, 0.0, 0.2])
        _add_box_row(viewer, 8.0)
        image = viewer.render()
        self.assertGreater(float(image.mean()), 140.0)

    def test_default_scene_dict_uses_shared_fov_defaults(self):
        # Default moved from implicit Mitsuba x-axis FOV to the shared
        # (xfov, yfov) tuple convention so set_camera(angles=...) frames like
        # pyrender/trimesh by default.
        viewer = MitsubaViewer(resolution=(160, 120), spp=4)
        scene = viewer._scene_dict()
        self.assertAlmostEqual(scene['sensor']['fov'], 45.0, places=7)
        self.assertEqual(scene['sensor']['fov_axis'], 'y')
        self.assertAlmostEqual(scene['ambient']['radiance']['value'],
                               0.12, places=7)
        self.assertAlmostEqual(scene['key']['emitter']['radiance']['value'],
                               5.0, places=7)
        self.assertAlmostEqual(
            _transform_matrix_element(scene['ground']['to_world'], 2, 3),
            0.0, places=7)
        self.assertAlmostEqual(
            _transform_matrix_element(scene['ground']['to_world'], 0, 0),
            6.0, places=7)

    def test_configurable_scene_smoke_without_subclassing(self):
        viewer = MitsubaViewer(resolution=(640, 360), spp=512, ground=True,
                               fov=39.6, fov_axis='y', ground_height=-0.79,
                               light_intensity=3.5)
        viewer.resolution = (64, 36)  # keep this smoke test fast
        viewer.spp = 1
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        image = viewer.render()
        self.assertEqual(image.shape, (36, 64, 3))

    def test_auto_ground_height_is_sticky_for_same_geometry_set(self):
        viewer = MitsubaViewer(resolution=(64, 48), spp=1)
        viewer.add(self.robot)
        viewer.add_box([0.4, 0.0, 0.2], [0.05, 0.05, 0.05], name='cube')
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        viewer.render()
        z_before = _transform_matrix_element(
            viewer._scene_dict()['ground']['to_world'], 2, 3)
        self.robot.translate([0.0, 0.0, -0.4], 'world')
        # Replacing an existing marker by name must not move auto-ground.
        viewer.add_box([0.4, 0.0, -0.2], [0.05, 0.05, 0.05], name='cube')
        viewer.render()
        z_after = _transform_matrix_element(
            viewer._scene_dict()['ground']['to_world'], 2, 3)
        self.assertAlmostEqual(z_after, z_before, places=7)

    def test_auto_ground_height_recomputes_when_new_link_is_added(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(resolution=(64, 48), spp=1)
        viewer.add(self.robot)
        viewer.set_camera(eye=[1.5, -1.5, 1.0], target=[0.0, 0.0, 0.5])
        viewer.render()
        z_before = _transform_matrix_element(
            viewer._scene_dict()['ground']['to_world'], 2, 3)

        low = Box(extents=[0.2, 0.2, 0.2])
        low.translate([0.0, 0.0, -1.0], 'world')
        viewer.add(low)  # geometry set changed -> cached auto-ground must reset
        viewer.render()
        z_after = _transform_matrix_element(
            viewer._scene_dict()['ground']['to_world'], 2, 3)

        self.assertAlmostEqual(z_after, -1.1, places=7)
        self.assertLess(z_after, z_before - 0.5)

    def test_auto_light_size_is_sticky_for_same_geometry_set(self):
        viewer = MitsubaViewer(resolution=(96, 72), spp=2)
        viewer.set_camera(eye=[0.0, -9.0, 3.0], target=[0.0, 0.0, 0.2])
        _add_box_row(viewer, 2.0, prefix='box')
        viewer.render()
        before = _key_light_matrix(viewer)

        # Moving existing named geometry must not change cached auto light size.
        viewer.add_box([5.0, 0.0, 0.2], [0.35, 0.35, 0.35], name='box_0')
        viewer.render()
        after_move = _key_light_matrix(viewer)
        np.testing.assert_allclose(after_move, before, atol=1e-7)

        # Adding a new geometry name changes the set, so auto size may update.
        viewer.add_box([6.0, 0.0, 0.2], [0.35, 0.35, 0.35], name='box_new')
        viewer.render()
        after_add = _key_light_matrix(viewer)
        self.assertFalse(np.allclose(after_add, before, atol=1e-7))

    def test_explicit_light_size_overrides_scene_derived_size(self):
        small = MitsubaViewer(resolution=(160, 120), spp=8, light_size=0.9)
        large = MitsubaViewer(resolution=(160, 120), spp=8, light_size=0.9)
        for viewer in (small, large):
            viewer.set_camera(eye=[0.0, -9.0, 3.0], target=[0.0, 0.0, 0.2])
        _add_box_row(small, 1.5, prefix='small')
        _add_box_row(large, 8.0, prefix='large')
        key_small = small._scene_dict()['key']['to_world']
        key_large = large._scene_dict()['key']['to_world']

        self.assertAlmostEqual(_transform_uniform_scale(key_small),
                               0.9, places=7)
        self.assertAlmostEqual(_transform_uniform_scale(key_large),
                               0.9, places=7)
        np.testing.assert_allclose(
            np.asarray(key_small.matrix, float),
            np.asarray(key_large.matrix, float),
            atol=1e-7)

    def test_set_color_after_add_updates_render(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(resolution=(128, 96), spp=16)
        box = Box(extents=[0.2, 0.2, 0.2])
        viewer.add(box)
        viewer.set_camera(eye=[0.8, -0.8, 0.6], target=[0.0, 0.0, 0.0])
        before = viewer.render()
        before_red = _count_strong_red_pixels(before)
        box.set_color([255, 0, 0, 255])
        after = viewer.render()
        after_red = _count_strong_red_pixels(after)
        self.assertGreater(after_red, before_red + 200)

    def test_set_color_before_add_keeps_color_across_two_viewers(self):
        from skrobot.model.primitives import Box
        box = Box(extents=[0.2, 0.2, 0.2])
        box.set_color([255, 0, 0, 255])

        viewer_a = MitsubaViewer(resolution=(128, 96), spp=8, variant='llvm_ad_rgb')
        viewer_a.set_camera(eye=[0.8, -0.8, 0.6], target=[0.0, 0.0, 0.0])
        viewer_a.add(box)
        viewer_a.render()
        colors_a = [v[2] for v in viewer_a._links.values() if v[0] is box]
        self.assertGreater(len(colors_a), 0)
        for c in colors_a:
            np.testing.assert_allclose(c, [1.0, 0.0, 0.0, 1.0], atol=1e-6)

        viewer_b = MitsubaViewer(resolution=(128, 96), spp=8, variant='llvm_ad_rgb')
        viewer_b.set_camera(eye=[0.8, -0.8, 0.6], target=[0.0, 0.0, 0.0])
        viewer_b.add(box)
        colors_b = [v[2] for v in viewer_b._links.values() if v[0] is box]
        self.assertGreater(len(colors_b), 0)
        for c in colors_b:
            np.testing.assert_allclose(c, [1.0, 0.0, 0.0, 1.0], atol=1e-6)

    def test_set_color_clears_visual_mesh_changed_and_keeps_incremental(self):
        from skrobot.model.primitives import Box
        viewer = MitsubaViewer(resolution=(96, 72), spp=8)
        box = Box(extents=[0.2, 0.2, 0.2])
        viewer.add(box)
        viewer.set_camera(eye=[0.8, -0.8, 0.6], target=[0.0, 0.0, 0.0])
        build_calls = {'count': 0}
        build_scene = viewer._build_scene

        def wrapped_build_scene():
            build_calls['count'] += 1
            return build_scene()

        viewer._build_scene = wrapped_build_scene
        viewer.render()
        box.set_color([255, 0, 0, 255])
        self.assertTrue(box.visual_mesh_changed)
        viewer.render()
        self.assertFalse(box.visual_mesh_changed)
        viewer.render()
        self.assertEqual(build_calls['count'], 2)

    def test_pr2_render_emits_no_ply_attribute_warnings(self):
        import mitsuba as mi
        if 'llvm_ad_rgb' not in mi.variants():
            self.skipTest('llvm_ad_rgb is not available')

        warning_color = (
            'has integer fields: color attributes are expected to be in the '
            '[0, 1] range.')
        warning_stl = (
            'attributes without postfix are not handled for now: '
            'attribute "stl" ignored.')
        viewer = MitsubaViewer(
            resolution=(160, 120), spp=2, variant='llvm_ad_rgb')
        robot = skrobot.models.PR2()
        viewer.add(robot)
        viewer.set_camera(eye=[2.5, -2.4, 1.8], target=[0.0, 0.0, 0.9])

        mesh_links = [
            link for link in robot.link_list
            if getattr(link, 'concatenated_visual_mesh', None) is not None]
        self.assertGreater(len(mesh_links), 0)

        def _render_sequence():
            viewer.render()
            for i in range(3):
                link = mesh_links[i % len(mesh_links)]
                if i % 2 == 0:
                    color = [255, 0, 0, 255]
                else:
                    color = [30, 120, 255, 255]
                link.set_color(color)
                viewer.render()

        logs = _capture_mitsuba_logs(_render_sequence, log_level='Warn')
        joined = '\n'.join(logs)
        self.assertNotIn(warning_color, joined)
        self.assertNotIn(warning_stl, joined)

    def test_pr2_render_matches_legacy_ply_export(self):
        import mitsuba as mi
        if 'llvm_ad_rgb' not in mi.variants():
            self.skipTest('llvm_ad_rgb is not available')

        def _legacy_export(self, mesh, path):
            mesh.export(path)

        resolution = (320, 240)
        camera_eye = [2.5, -2.4, 1.8]
        camera_target = [0.0, 0.0, 0.9]

        legacy = MitsubaViewer(
            resolution=resolution, spp=16, variant='llvm_ad_rgb')
        legacy._export_ply_geometry_only = types.MethodType(
            _legacy_export, legacy)
        legacy.add(skrobot.models.PR2())
        legacy.set_camera(eye=camera_eye, target=camera_target)
        legacy_image = np.asarray(legacy.render(), dtype=np.int16)

        updated = MitsubaViewer(
            resolution=resolution, spp=16, variant='llvm_ad_rgb')
        updated.add(skrobot.models.PR2())
        updated.set_camera(eye=camera_eye, target=camera_target)
        updated_image = np.asarray(updated.render(), dtype=np.int16)

        diff = np.abs(updated_image - legacy_image)
        self.assertLessEqual(int(diff.max()), 1)

    def test_exported_ply_files_are_geometry_only(self):
        import mitsuba as mi
        if 'llvm_ad_rgb' not in mi.variants():
            self.skipTest('llvm_ad_rgb is not available')

        viewer = MitsubaViewer(
            resolution=(160, 120), spp=2, variant='llvm_ad_rgb')
        robot = skrobot.models.PR2()
        viewer.add(robot)
        viewer.set_camera(eye=[2.5, -2.4, 1.8], target=[0.0, 0.0, 0.9])
        viewer.add_sphere([0.4, 0.0, 1.0], 0.07, name='ball')
        viewer.add_box([0.4, -0.2, 0.8], [0.08, 0.06, 0.04], name='box')
        viewer.add_joint_axis(robot.rarm.joint_list[0], axis_length=0.08)

        # Trigger _refresh_changed_link_meshes(), which re-exports link meshes.
        link = next(
            l for l in robot.link_list
            if getattr(l, 'concatenated_visual_mesh', None) is not None)
        link.set_color([255, 0, 0, 255])
        viewer.render()

        ply_paths = {entry[1] for entry in viewer._links.values()}
        for marker in viewer._extra.values():
            if marker.get('type') == 'ply':
                ply_paths.add(marker['filename'])
        self.assertGreater(len(ply_paths), 0)

        expected_prefix = ['property float x',
                           'property float y',
                           'property float z']
        for path in sorted(ply_paths):
            props = _ply_header_property_lines(path)
            self.assertEqual(props[:3], expected_prefix)
            self.assertEqual(len(props), 4)
            self.assertRegex(
                props[3],
                r'^property list [A-Za-z0-9_]+ [A-Za-z0-9_]+ vertex_indices$')
            joined = ' '.join(props).lower()
            for token in ('red', 'green', 'blue', 'alpha', 'stl'):
                self.assertNotIn(token, joined)


if __name__ == '__main__':
    unittest.main()
