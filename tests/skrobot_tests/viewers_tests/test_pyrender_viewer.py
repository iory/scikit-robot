import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

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


def _pyrender_available():
    try:
        import pyrender  # noqa: F401
    except ImportError:
        return False
    return True


def _python_with_x_server_command(script):
    python_cmd = [sys.executable, '-c', script]
    if os.environ.get('DISPLAY'):
        return python_cmd
    xvfb_run = shutil.which('xvfb-run')
    if xvfb_run:
        return [xvfb_run, '-a'] + python_cmd
    return None


class TestPyrenderViewerSaveImage(unittest.TestCase):

    def _run_capture_script(self, script, timeout=180):
        cmd = _python_with_x_server_command(script)
        if cmd is None:
            self.skipTest('requires an X server or xvfb-run')
        env = os.environ.copy()
        existing = env.get('PYTHONPATH')
        if existing:
            env['PYTHONPATH'] = str(_repo_root()) + os.pathsep + existing
        else:
            env['PYTHONPATH'] = str(_repo_root())
        return subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=timeout,
        )

    def test_save_image_captures_scene_pixels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pyrender.png')
            script = textwrap.dedent(
                """
                import json
                import os
                import sys

                import numpy as np
                from PIL import Image

                sys.path.insert(0, {repo_root!r})
                import skrobot
                from skrobot.viewers import PyrenderViewer

                viewer = PyrenderViewer(resolution=(320, 240))
                viewer.add(skrobot.models.PR2())
                viewer.show()
                try:
                    viewer.set_camera(angles=[0.0, 0.0, np.pi / 2.0])
                    with open({path!r}, 'wb') as file_obj:
                        viewer.save_image(file_obj)
                finally:
                    viewer.close()

                image = np.array(Image.open({path!r}).convert('RGB'))
                background = image[0, 0, :].astype(np.int16)
                diff = np.abs(image.astype(np.int16) - background)
                fraction = float(np.any(diff > 0, axis=2).mean())
                stats = {{
                    'size': int(os.path.getsize({path!r})),
                    'fraction': fraction,
                }}
                print('PYRENDER_SAVE_IMAGE_STATS=' + json.dumps(stats))
                """.format(repo_root=str(_repo_root()), path=path))
            proc = self._run_capture_script(script)
            self.assertEqual(
                proc.returncode, 0,
                msg='stdout:\\n{}\\nstderr:\\n{}'.format(
                    proc.stdout, proc.stderr))
            lines = [
                line for line in proc.stdout.splitlines()
                if line.startswith('PYRENDER_SAVE_IMAGE_STATS=')
            ]
            self.assertTrue(
                lines, msg='stdout:\\n{}\\nstderr:\\n{}'.format(
                    proc.stdout, proc.stderr))
            stats = json.loads(lines[-1].split('=', 1)[1])
            self.assertGreater(stats['size'], 0)
            self.assertGreater(stats['fraction'], 0.02)

    def test_save_image_raises_when_viewer_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'unused.png')
            script = textwrap.dedent(
                """
                import json
                import sys

                sys.path.insert(0, {repo_root!r})
                from skrobot.viewers import PyrenderViewer

                viewer = PyrenderViewer(resolution=(160, 120))
                try:
                    viewer.save_image({path!r})
                except RuntimeError as exc:
                    payload = {{
                        'raised': True,
                        'message': str(exc),
                    }}
                else:
                    payload = {{
                        'raised': False,
                        'message': '',
                    }}
                finally:
                    try:
                        viewer.close()
                    except Exception:
                        pass
                print('PYRENDER_NOT_RUNNING=' + json.dumps(payload))
                """.format(repo_root=str(_repo_root()), path=path))
            proc = self._run_capture_script(script)
            self.assertEqual(
                proc.returncode, 0,
                msg='stdout:\\n{}\\nstderr:\\n{}'.format(
                    proc.stdout, proc.stderr))
            lines = [
                line for line in proc.stdout.splitlines()
                if line.startswith('PYRENDER_NOT_RUNNING=')
            ]
            self.assertTrue(
                lines, msg='stdout:\\n{}\\nstderr:\\n{}'.format(
                    proc.stdout, proc.stderr))
            result = json.loads(lines[-1].split('=', 1)[1])
            self.assertTrue(result['raised'])
            msg = result['message'].lower()
            self.assertTrue('not running' in msg or 'show()' in msg)


if __name__ == '__main__':
    unittest.main()
