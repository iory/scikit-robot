"""In-memory robot interface, for tests and examples.

:class:`FakeRobotInterface` is the smallest complete backend of
:class:`skrobot.interfaces.base.RobotInterfaceBase`: it "executes" the
trajectories it is sent by cubic Hermite interpolation over time and
reports the interpolated positions as the measured state. There is no
physics, no tracking error and no joint limit, so what it reports is
exactly what was commanded -- and a state injected with
:meth:`FakeRobotInterface.set_joint_positions` is reported as it is, even
past the model's limits.

By default time is virtual: it stands still until :meth:`advance` or
``wait_interpolation`` moves it, so a test never sleeps. Passing
``clock=time.monotonic`` runs on wall-clock time instead.
"""

import numpy as np

from skrobot.interfaces.base import RobotInterfaceBase


def _hermite(t, t0, t1, p0, p1, v0, v1):
    """Cubic Hermite interpolation between ``(t0, p0, v0)``, ``(t1, p1, v1)``."""
    h = t1 - t0
    if h <= 0.0:
        return p1
    s = (t - t0) / h
    s2 = s * s
    s3 = s2 * s
    return ((2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * h * v0
            + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * h * v1)


class FakeRobotInterface(RobotInterfaceBase):
    """Robot interface that moves an in-memory robot.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Model of the robot (read only).
    controllers : dict[str, list[str]] or None
        See :class:`skrobot.interfaces.base.RobotInterfaceBase`.
    controller_type : str or None
        See :class:`skrobot.interfaces.base.RobotInterfaceBase`.
    initial_state : dict[str, float] or None
        Starting position of each joint. Joints not given start at the
        model's current angle.
    clock : callable or None
        Function returning the time [s]. ``None`` uses a virtual clock that
        only :meth:`advance` and ``wait_interpolation`` move.
    servo : bool
        Whether this fake accepts ``servo_on``/``servo_off``. With False it
        behaves like a backend without torque control.

    Attributes
    ----------
    sent : list[dict]
        Every trajectory received, as the arguments of ``_send_trajectory``
        plus the time it arrived, for tests to inspect.
    servo_states : dict[str, bool]
        Servo state per joint (only when ``servo`` is True).

    Examples
    --------
    ::

        ri = FakeRobotInterface(robot)
        ri.angle_vector({'joint1': 0.5}, 1.0)   # returns at once
        ri.advance(0.5)                          # virtual time
        ri.angle_vector()                        # measured, half way there
        ri.wait_interpolation()                  # jumps to the end
    """

    def __init__(self, robot, controllers=None, controller_type=None,
                 initial_state=None, clock=None, servo=True):
        super(FakeRobotInterface, self).__init__(
            robot, controllers=controllers, controller_type=controller_type)
        self._clock = clock
        self._virtual_time = 0.0
        self._servo = servo
        positions = {j.name: float(j.joint_angle())
                     for j in robot.joint_list if j.joint_dof == 1}
        if initial_state is not None:
            self._check_names(list(initial_state))
            positions.update(
                {k: float(v) for k, v in initial_state.items()})
        # joint name -> (hold position, [segments]); a segment is
        # (t0, t1, p0, p1, v0, v1).
        self._hold = positions
        self._segments = {name: [] for name in positions}
        self.sent = []
        self.servo_states = {name: True for name in positions}
        self.update_robot_state()

    # Clock --------------------------------------------------------------

    def now(self):
        """Current time [s] of this interface's clock."""
        if self._clock is None:
            return self._virtual_time
        return float(self._clock())

    def advance(self, dt):
        """Move the virtual clock forward.

        Parameters
        ----------
        dt : float
            Time [s] to advance.
        """
        if self._clock is not None:
            raise RuntimeError('advance() needs the virtual clock '
                               '(clock=None)')
        if dt < 0:
            raise ValueError('dt must be >= 0, but given: {}'.format(dt))
        self._virtual_time += dt

    # State --------------------------------------------------------------

    def _position(self, name, t):
        for t0, t1, p0, p1, v0, v1 in self._segments[name]:
            if t <= t1:
                if t < t0:
                    return p0
                return _hermite(t, t0, t1, p0, p1, v0, v1)
        return self._hold[name]

    def _end_time(self, name):
        segments = self._segments[name]
        if not segments:
            return -np.inf
        return segments[-1][1]

    def set_joint_positions(self, positions):
        """Put joints at ``positions`` at once, dropping their motion.

        Simulates the robot being somewhere it was not commanded to go
        (pushed by hand, calibration offset, past a URDF limit).

        Parameters
        ----------
        positions : dict[str, float]
            ``{joint_name: position}``.
        """
        self._check_names(list(positions))
        for name, value in positions.items():
            self._hold[name] = float(value)
            self._segments[name] = []

    # Backend hooks ------------------------------------------------------

    def _send_trajectory(self, controller_type, joint_names, positions,
                         velocities, times, start_time):
        now = self.now()
        self.sent.append(dict(
            controller_type=controller_type, joint_names=list(joint_names),
            positions=np.array(positions), velocities=np.array(velocities),
            times=np.array(times), start_time=start_time, received=now))
        t_start = now + start_time
        for j, name in enumerate(joint_names):
            p = self._position(name, now)
            v = 0.0
            t = t_start
            segments = []
            for i in range(len(times)):
                t_next = t_start + times[i]
                segments.append((t, t_next, p, positions[i, j],
                                 v, velocities[i, j]))
                t, p, v = t_next, positions[i, j], velocities[i, j]
            self._segments[name] = segments
            self._hold[name] = float(positions[-1, j])

    def _read_joint_state(self):
        t = self.now()
        return {name: float(self._position(name, t)) for name in self._hold}

    def _is_interpolating(self, controller_type):
        t = self.now()
        return any(t < self._end_time(name)
                   for name in self._controllers[controller_type])

    def _cancel(self, controller_type):
        t = self.now()
        for name in self._controllers[controller_type]:
            self._hold[name] = float(self._position(name, t))
            self._segments[name] = []

    def _wait_interpolation(self, controller_type, timeout):
        if self._clock is not None:
            return super(FakeRobotInterface, self)._wait_interpolation(
                controller_type, timeout)
        end = max(self._end_time(n)
                  for n in self._controllers[controller_type])
        if timeout is not None:
            end = min(end, self._virtual_time + timeout)
        if end > self._virtual_time:
            self._virtual_time = end

    def _set_servo(self, joint_names, on):
        if not self._servo:
            return super(FakeRobotInterface, self)._set_servo(joint_names, on)
        for name in joint_names:
            self.servo_states[name] = on
