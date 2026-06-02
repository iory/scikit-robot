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
