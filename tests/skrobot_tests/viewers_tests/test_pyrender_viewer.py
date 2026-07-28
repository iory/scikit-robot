import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from skrobot.coordinates import Coordinates
from skrobot.viewers import PyrenderViewer


def _reset_pyrender_singleton():
    viewer = PyrenderViewer._instance
    if viewer is None:
        return
    thread = getattr(viewer, 'thread', None)
    if thread is not None and thread.is_alive():
        try:
            viewer.close_external()
        except Exception:
            pass
    PyrenderViewer._instance = None


@pytest.fixture(autouse=True)
def _cleanup_pyrender_singleton():
    _reset_pyrender_singleton()
    yield
    _reset_pyrender_singleton()


def _repo_root():
    return Path(__file__).resolve().parents[3]


def _display_command_prefix():
    if os.environ.get('DISPLAY'):
        return []
    if shutil.which('xvfb-run') is not None:
        return ['xvfb-run', '-a']
    pytest.skip('requires DISPLAY or xvfb-run')


def _run_display_script(script, timeout=60):
    env = os.environ.copy()
    root = str(_repo_root())
    current_pythonpath = env.get('PYTHONPATH')
    if current_pythonpath:
        env['PYTHONPATH'] = '{}{}{}'.format(
            root, os.pathsep, current_pythonpath)
    else:
        env['PYTHONPATH'] = root

    command = _display_command_prefix() + [sys.executable, '-c', script]
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        timeout=timeout)


def _extract_result(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith('RESULT:'):
            return json.loads(line.split('RESULT:', 1)[1])
    raise AssertionError(
        'subprocess did not emit RESULT payload. stdout:\n{}'.format(stdout))


def test_set_camera_accepts_transform_and_coordinates():
    viewer = PyrenderViewer(resolution=(320, 240))

    transform = np.eye(4)
    transform[:3, 3] = [0.1, 0.2, 0.3]
    viewer.set_camera(coords_or_transform=transform)
    np.testing.assert_allclose(viewer._camera_node.matrix, transform)

    coords = Coordinates(pos=[0.4, 0.5, 0.6], rot=np.eye(3))
    expected = coords.worldcoords().T()
    viewer.set_camera(coords_or_transform=coords)
    np.testing.assert_allclose(viewer._camera_node.matrix, expected)

    with pytest.raises((TypeError, ValueError), match='coords_or_transform'):
        viewer.set_camera(coords_or_transform='not-a-transform')


def test_linestring_draw_keeps_thread_alive_and_renders_nonblank_frame():
    thread_script = textwrap.dedent(
        """
        import json
        import time

        import numpy as np

        from skrobot.model.primitives import LineString
        from skrobot.viewers import PyrenderViewer

        viewer = PyrenderViewer(resolution=(320, 240))
        points = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0]],
            dtype=np.float64)
        viewer.add(LineString(points, color=[255, 0, 0, 255]))
        viewer.show()
        time.sleep(1.2)

        result = dict(
            thread_alive=bool(
                viewer.thread is not None and viewer.thread.is_alive()))
        print('RESULT:{}'.format(json.dumps(result, sort_keys=True)))
        """)
    completed = _run_display_script(thread_script)
    assert completed.returncode == 0, (
        'thread check subprocess failed\nstdout:\n{}\nstderr:\n{}'.format(
            completed.stdout, completed.stderr))
    thread_result = _extract_result(completed.stdout)
    assert thread_result['thread_alive'], (
        'pyrender viewer thread died after adding LineString.\n'
        'stdout:\n{}\nstderr:\n{}'.format(
            completed.stdout, completed.stderr))

    render_script = textwrap.dedent(
        """
        import json

        import numpy as np
        import pyrender

        from skrobot.model import Box
        from skrobot.model.primitives import LineString
        from skrobot.viewers import PyrenderViewer

        viewer = PyrenderViewer(resolution=(320, 240))
        points = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0]],
            dtype=np.float64)
        viewer.add(LineString(points, color=[255, 0, 0, 255]))

        # Add a bright mesh so a successful render is observably non-blank.
        box = Box(extents=(0.05, 0.05, 0.05), face_colors=[255, 255, 255, 255])
        box.translate((0.05, 0.0, 0.0))
        viewer.add(box)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=320, viewport_height=240)
        color, _ = renderer.render(viewer.scene)
        renderer.delete()

        result = dict(frame_std=float(np.std(color)))
        print('RESULT:{}'.format(json.dumps(result, sort_keys=True)))
        """)
    completed = _run_display_script(render_script)
    assert completed.returncode == 0, (
        'frame capture subprocess failed\nstdout:\n{}\nstderr:\n{}'.format(
            completed.stdout, completed.stderr))
    render_result = _extract_result(completed.stdout)
    assert render_result['frame_std'] > 0.0, (
        'captured frame is blank.\nstdout:\n{}\nstderr:\n{}'.format(
            completed.stdout, completed.stderr))


def test_pointcloud_with_mismatched_colors_does_not_kill_render_thread():
    script = textwrap.dedent(
        """
        import json
        import time

        import numpy as np

        from skrobot.model.primitives import PointCloudLink
        from skrobot.viewers import PyrenderViewer

        viewer = PyrenderViewer(resolution=(320, 240))
        points = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.2, 0.2, 0.0]],
            dtype=np.float64)
        colors = np.array([[255, 0, 0, 255]], dtype=np.uint8)
        viewer.add(PointCloudLink(points, colors=colors))
        viewer.show()
        time.sleep(1.2)

        result = dict(
            thread_alive=bool(
                viewer.thread is not None and viewer.thread.is_alive()))
        print('RESULT:{}'.format(json.dumps(result, sort_keys=True)))
        """)
    completed = _run_display_script(script)
    assert completed.returncode == 0, (
        'pointcloud subprocess failed\nstdout:\n{}\nstderr:\n{}'.format(
            completed.stdout, completed.stderr))
    result = _extract_result(completed.stdout)
    assert result['thread_alive'], (
        'pyrender viewer thread died for mismatched PointCloud colors.\n'
        'stdout:\n{}\nstderr:\n{}'.format(
            completed.stdout, completed.stderr))
