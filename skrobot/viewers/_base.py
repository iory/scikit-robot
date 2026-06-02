import math
import time


class _InteractiveViewerMixin(object):
    """Mixin providing helpers shared by interactive viewers.

    The mixin only relies on the ``is_active`` property and the
    ``redraw`` / ``close`` methods, all of which are implemented by every
    interactive viewer backend (TrimeshSceneViewer, PyrenderViewer,
    ViserViewer). It is intentionally placed last in the base class list so
    that it never shadows the backend's own ``__init__`` / ``__new__``.
    """

    def wait_until_close(self, redraw=True, interval=0.1,
                         message='==> Press [q] to close window'):
        """Block until the viewer window is closed by the user.

        This replaces the ``while viewer.is_active: time.sleep(...);
        viewer.redraw()`` loop that is duplicated across the examples.

        Parameters
        ----------
        redraw : bool, optional
            If True, call :meth:`redraw` every ``interval`` seconds while
            waiting. Default is True.
        interval : float, optional
            Seconds to sleep between redraws. Default is 0.1.
        message : str or None, optional
            Message printed once before waiting starts. Pass None to
            suppress it. Default is '==> Press [q] to close window'.
        """
        if message:
            print(message)
        try:
            while self.is_active:
                time.sleep(interval)
                if redraw:
                    self.redraw()
        except KeyboardInterrupt:
            pass
        self.close()

    def pause(self, duration, fps=30.0):
        """Pause for ``duration`` seconds while keeping the viewer interactive.

        Use this in place of ``time.sleep(duration)`` inside animation
        loops. On macOS the trimesh and pyrender viewers run their GL event
        loop on the main thread, so a bare ``time.sleep`` freezes the window
        (the camera cannot be dragged) for the whole pause because no events
        are dispatched. This pumps :meth:`redraw` at ``fps`` for the entire
        duration so the view stays responsive. On backends that already
        render in a separate thread (trimesh/pyrender on Linux) or process
        (viser) the extra redraws are harmless.

        Parameters
        ----------
        duration : float, optional
            Seconds to pause. Non-positive (or non-finite) values just
            trigger a single redraw and return immediately.
        fps : float, optional
            Redraw frequency in Hz while pausing. Must be a positive finite
            number. Default is 30.
        """
        if not (math.isfinite(fps) and fps > 0):
            raise ValueError(
                'fps must be a positive finite number, got {}.'.format(fps))
        self.redraw()
        if not (math.isfinite(duration) and duration > 0):
            return
        interval = 1.0 / fps
        # monotonic() is used so a wall-clock adjustment cannot shorten or
        # extend the pause.
        end = time.monotonic() + duration
        while True:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(interval, remaining))
            self.redraw()
