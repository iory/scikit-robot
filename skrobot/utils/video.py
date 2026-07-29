"""Record a viewer's frames to a video file.

This provides a single, backend-agnostic way to turn any example's animation
loop into a video. It relies only on a viewer's ``save_image`` method, which
is implemented by the built-in image-producing viewers
(``TrimeshSceneViewer``, ``PyrenderViewer`` and ``MitsubaViewer``), so
``--save-video`` can record the same loop on any of those backends. With the
headless ``mitsuba`` viewer it needs no display server at all (works over SSH
/ in CI).
"""

import os
import shutil
import tempfile


def _load_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError(
            'Saving a video needs imageio (and imageio-ffmpeg for .mp4). '
            "Install them with 'pip install imageio imageio-ffmpeg'.")
    return imageio


class VideoRecorder(object):
    """Capture the frames a viewer renders and encode them into a video.

    The recorder wraps a live viewer: each time the scene is refreshed -- via
    the viewer's ``redraw`` / ``pause`` (or an explicit :meth:`capture`) -- the
    currently displayed frame is written to a temporary image, and :meth:`save`
    encodes every collected frame into a single video file. Because it only
    uses ``save_image``, the same recorder drives the trimesh, pyrender and
    mitsuba backends unchanged.

    Parameters
    ----------
    viewer : object
        A viewer exposing ``save_image(path)`` (for example trimesh, pyrender
        or mitsuba viewers). The ``hook`` methods below are wrapped in place so
        existing animation loops record without any further changes.
    path : str
        Output video path. The extension selects the format (``.mp4`` needs
        imageio-ffmpeg; ``.gif`` does not).
    fps : float, optional
        Frames per second of the encoded video. Default ``30``.
    hook : tuple(str), optional
        Viewer methods after which a frame is captured. Default
        ``('redraw', 'pause')`` -- the two refresh calls the examples use in
        their animation loops.

    Examples
    --------
    >>> viewer = skrobot.viewers.create_viewer('mitsuba')
    >>> viewer.add(robot)
    >>> viewer.show()
    >>> recorder = VideoRecorder(viewer, 'out.mp4', fps=20)
    >>> for av in trajectory:          # loop calls viewer.pause()/redraw()
    ...     robot.angle_vector(av)
    ...     viewer.pause(0.1)          # each pause is captured automatically
    >>> recorder.save()
    """

    def __init__(self, viewer, path, fps=30, hook=('redraw', 'pause')):
        if not callable(getattr(viewer, 'save_image', None)):
            raise TypeError(
                'VideoRecorder needs viewer.save_image(file_obj); got {}.'
                .format(type(viewer).__name__))
        self.viewer = viewer
        self.path = path
        self.fps = float(fps)
        self._dir = tempfile.mkdtemp(prefix='skrobot_video_')
        self._frame_paths = []
        self._wrapped = {}
        for name in hook:
            method = getattr(viewer, name, None)
            if callable(method):
                # Remember the original method and whether it lived on the
                # instance, so close() can restore the exact prior state.
                self._wrapped[name] = (method, name in vars(viewer))
                setattr(viewer, name, self._make_wrapper(method))

    def _make_wrapper(self, method):
        def wrapper(*args, **kwargs):
            result = method(*args, **kwargs)
            self.capture()
            return result
        return wrapper

    def capture(self):
        """Save the currently displayed frame as one video frame."""
        path = os.path.join(
            self._dir, 'frame_{:06d}.png'.format(len(self._frame_paths)))
        self.viewer.save_image(path)
        self._frame_paths.append(path)

    @property
    def n_frames(self):
        """Number of frames captured so far."""
        return len(self._frame_paths)

    def save(self, path=None, fps=None):
        """Encode the captured frames into a video and return its path."""
        imageio = _load_imageio()
        out = path or self.path
        if not self._frame_paths:
            raise RuntimeError(
                'VideoRecorder captured no frames. Refresh the viewer '
                '(redraw/pause) or call capture() at least once before save().')
        fps_value = fps or self.fps
        frame_paths = self._frame_paths
        # A single captured frame (a static scene) is held for ~1 s so the file
        # is a valid, viewable clip rather than a one-frame flash.
        if len(frame_paths) == 1:
            frame_paths = frame_paths * max(1, int(round(fps_value)))
        # Stream frames straight to the encoder so a long recording never holds
        # every decoded frame in memory at once.
        with imageio.get_writer(
                out, fps=fps_value, macro_block_size=None) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))
        return out

    def close(self):
        """Restore the viewer methods this recorder wrapped and drop frames.

        The captured frames live in a temporary directory of their own; a
        720-frame recording leaves hundreds of megabytes behind if it is never
        removed. Calling this twice is harmless.
        """
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
            self._frame_paths = []
        for name, (method, was_instance_attr) in self._wrapped.items():
            if was_instance_attr:
                setattr(self.viewer, name, method)
            else:
                # The method came from the class; drop our instance-level
                # override so the class method shows through again.
                try:
                    delattr(self.viewer, name)
                except AttributeError:
                    setattr(self.viewer, name, method)
        self._wrapped = {}


def record_viewer(viewer, path, fps=30, **kwargs):
    """Return a :class:`VideoRecorder` for ``viewer``, or ``None`` if ``path``
    is falsy.

    This is the convenience entry point for the examples' ``--save-video``
    option: pass the argument straight through and only record when the user
    asked for it.

    >>> recorder = record_viewer(viewer, args.save_video)
    >>> ...                                   # run the animation loop
    >>> if recorder is not None:
    ...     recorder.save()
    """
    if not path:
        return None
    return VideoRecorder(viewer, path, fps=fps, **kwargs)
