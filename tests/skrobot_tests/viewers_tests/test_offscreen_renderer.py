import os
import tempfile

import numpy as np
from PIL import Image
import pytest
import trimesh

from skrobot.coordinates import Coordinates
from skrobot.model import CascadedLink
from skrobot.model.primitives import Box
from skrobot.viewers import create_viewer
from skrobot.viewers import look_at_pose
from skrobot.viewers import merge_by_material
from skrobot.viewers import OffscreenRenderer
from skrobot.viewers import render_parallel


EYE = [1.5, -1.5, 1.0]
TARGET = [0.0, 0.0, 0.1]


def _boxes():
    base = Box(extents=(0.3, 0.3, 0.2), face_colors=[200, 40, 40, 255])
    top = Box(extents=(0.1, 0.1, 0.3), face_colors=[40, 40, 200, 255])
    base.assoc(top, relative_coords=Coordinates(pos=[0, 0, 0.25]))
    return CascadedLink(link_list=[base, top]), base, top


def _drawn(color):
    """Pixels that are not the white background."""
    return (color < 250).any(axis=2)


@pytest.fixture
def renderer():
    r = OffscreenRenderer((160, 120))
    yield r
    r.close()


def test_renders_robot_on_white(renderer):
    robot, _, _ = _boxes()
    renderer.add(robot)
    color = renderer.render(eye=EYE, target=TARGET)
    assert color.shape == (120, 160, 3)
    assert color.dtype == np.uint8
    # integer colours are 0-255 to pyrender: the default background must stay white
    assert (color[0, 0] == 255).all()
    assert _drawn(color).sum() > 100


def test_links_follow_their_robot(renderer):
    robot, base, _ = _boxes()
    renderer.add(robot)
    before = renderer.render(eye=EYE, target=TARGET)
    base.translate([0.3, 0.0, 0.0])
    after = renderer.render(eye=EYE, target=TARGET)
    assert not np.array_equal(before, after)


def test_link_added_twice_is_drawn_once(renderer):
    robot, base, _ = _boxes()
    renderer.add(robot)
    n = len(renderer._links)
    renderer.add(base)
    assert len(renderer._links) == n


def test_delete_removes_robot(renderer):
    robot, _, _ = _boxes()
    renderer.add(robot)
    renderer.delete(robot)
    color = renderer.render(eye=EYE, target=TARGET)
    assert not _drawn(color).any()


def test_mesh_can_be_moved(renderer):
    node = renderer.add(trimesh.creation.box(extents=(0.2, 0.2, 0.2)))
    first = renderer.render(eye=EYE, target=TARGET)
    pose = np.eye(4)
    pose[:3, 3] = [0.0, 0.4, 0.0]
    renderer.set_pose(node, pose)
    assert not np.array_equal(first, renderer.render(eye=EYE, target=TARGET))


def test_default_camera_sees_the_scene(renderer):
    robot, _, _ = _boxes()
    renderer.add(robot)
    drawn = _drawn(renderer.render())
    assert drawn.any()
    # PyrenderViewer's first view fits the whole scene: nothing touches the border
    assert not (drawn[0].any() or drawn[-1].any() or drawn[:, 0].any() or drawn[:, -1].any())


def test_set_camera_matches_look_at_pose(renderer):
    renderer.set_camera(coords_or_transform=look_at_pose(EYE, TARGET))
    np.testing.assert_allclose(renderer.camera_pose, look_at_pose(EYE, TARGET))
    renderer.set_camera(coords_or_transform=Coordinates(pos=[1, 2, 3]))
    np.testing.assert_allclose(renderer.camera_pose[:3, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        renderer.set_camera(coords_or_transform=np.eye(3))


def test_look_at_pose_points_minus_z_at_target():
    pose = look_at_pose(EYE, TARGET)
    forward = -pose[:3, 2]
    expected = np.subtract(TARGET, EYE)
    np.testing.assert_allclose(forward, expected / np.linalg.norm(expected))
    assert pose[2, 1] > 0   # +y is up in the image
    with pytest.raises(ValueError):
        look_at_pose([0, 0, 1], [0, 0, 0])


def test_merge_by_material_joins_alike_pieces():
    def box(rgba, x):
        m = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
        m.apply_translation([x, 0, 0])
        m.visual.face_colors = rgba
        return m
    merged = merge_by_material([box([255, 0, 0, 255], 0), box([255, 0, 0, 255], 1), box([0, 255, 0, 255], 2)])
    assert len(merged) == 1   # face-coloured pieces all go into one mesh
    assert len(merged[0].faces) == 36


def test_works_as_a_viewer():
    robot, _, _ = _boxes()
    viewer = create_viewer('offscreen', resolution=(160, 120))
    assert isinstance(viewer, OffscreenRenderer)
    viewer.add(robot)
    viewer.show()
    viewer.set_camera(coords_or_transform=look_at_pose(EYE, TARGET))
    viewer.redraw()
    viewer.pause(0.5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'frame.png')
        viewer.save_image(path)
        img = np.asarray(Image.open(path).convert('RGB'))
    assert img.shape == (120, 160, 3)
    assert _drawn(img).any()
    assert viewer.is_active
    viewer.close()
    assert not viewer.is_active


def _setup_box_renderer(size):
    robot, base, _ = _boxes()
    r = OffscreenRenderer(size)
    r.add(robot)

    def render_one(x):
        base.newcoords(Coordinates(pos=[x, 0, 0]))
        return r.render(eye=EYE, target=TARGET)
    return render_one


def test_render_parallel_keeps_job_order():
    jobs = [0.0, 0.2, -0.2, 0.1]
    serial = render_parallel(_setup_box_renderer, jobs, workers=1, setup_args=((80, 60),))
    split = render_parallel(_setup_box_renderer, jobs, workers=2, setup_args=((80, 60),), chunksize=1)
    assert len(split) == len(jobs)
    for a, b in zip(serial, split):
        # the same frame, give or take GL rounding at edges
        assert (np.abs(a.astype(int) - b.astype(int)).max(axis=2) > 8).mean() < 0.01
    assert not np.array_equal(serial[0], serial[1])
