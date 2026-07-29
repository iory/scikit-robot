import os
import tempfile
import unittest

import skrobot
from skrobot.utils.video import record_viewer
from skrobot.utils.video import VideoRecorder


def _deps_available():
    try:
        import imageio  # noqa: F401
        import mitsuba  # noqa: F401
    except ImportError:
        return False
    return True


class TestRecordViewerFactory(unittest.TestCase):

    def test_returns_none_without_path(self):
        # record_viewer is a no-op unless the user passes --save-video, so it
        # can be dropped straight into an example without branching.
        self.assertIsNone(record_viewer(object(), None))
        self.assertIsNone(record_viewer(object(), ''))


@unittest.skipUnless(_deps_available(), 'imageio or mitsuba is not installed')
class TestVideoRecorder(unittest.TestCase):

    def setUp(self):
        from skrobot.viewers import MitsubaViewer
        self.robot = skrobot.models.Panda()
        self.viewer = MitsubaViewer(resolution=(96, 72), spp=1)
        self.viewer.add(self.robot)

    def test_pause_and_redraw_are_captured(self):
        recorder = VideoRecorder(self.viewer, 'unused.mp4')
        self.assertEqual(recorder.n_frames, 0)
        self.viewer.pause(0.001)                      # wrapped -> captures
        self.viewer.redraw()                          # wrapped -> captures
        self.assertEqual(recorder.n_frames, 2)

    def test_save_writes_a_video_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'out.mp4')
            recorder = VideoRecorder(self.viewer, path, fps=5)
            for _ in range(3):
                self.viewer.pause(0.001)
            out = recorder.save()
            self.assertEqual(out, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            import imageio.v2 as imageio
            # Close the reader: on Windows an open handle keeps
            # TemporaryDirectory from removing the file it is reading.
            reader = imageio.get_reader(path)
            try:
                self.assertEqual(reader.count_frames(), 3)
            finally:
                reader.close()

    def test_save_without_frames_raises(self):
        recorder = VideoRecorder(self.viewer, 'unused.mp4')
        with self.assertRaises(RuntimeError):
            recorder.save()

    def test_close_stops_capturing(self):
        recorder = VideoRecorder(self.viewer, 'unused.mp4')
        self.viewer.pause(0.001)                      # wrapped -> captured
        self.assertEqual(recorder.n_frames, 1)
        recorder.close()                              # also drops the frames
        self.viewer.pause(0.001)                      # restored -> not captured
        self.assertEqual(recorder.n_frames, 0)

    def test_close_removes_the_temporary_frame_directory(self):
        # Every captured frame is a PNG in a private temp directory. A long
        # recording leaves hundreds of megabytes behind if close() does not
        # remove it.
        recorder = VideoRecorder(self.viewer, 'unused.mp4')
        frame_dir = recorder._dir
        self.viewer.pause(0.001)
        self.assertTrue(os.path.isdir(frame_dir))
        self.assertEqual(len(os.listdir(frame_dir)), 1)
        recorder.close()
        self.assertFalse(os.path.exists(frame_dir))
        recorder.close()                              # idempotent


if __name__ == '__main__':
    unittest.main()
