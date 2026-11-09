"""In-memory robot interface, for tests and examples.

:class:`FakeRobotInterface` is the smallest complete backend of
:class:`skrobot.interfaces.base.RobotInterfaceBase`: it "executes" the
trajectories it is sent by cubic Hermite interpolation over time
(:func:`skrobot.interpolation.hermite_trajectory`) and
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
from skrobot.interpolation import hermite_trajectory


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
        # Where each joint rests, and for joints with a trajectory sent
        # ``name -> (trajectory, column, end time)``: one trajectory (see
        # hermite_trajectory) covers all joints of a send.
        self._hold = positions
        self._motions = {}
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
        if name not in self._motions:
            return self._hold[name]
        trajectory, column, _ = self._motions[name]
        return float(trajectory(t)[column])

    def _end_time(self, name):
        if name not in self._motions:
            return -np.inf
        return self._motions[name][2]

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
            self._motions.pop(name, None)

    # Backend hooks ------------------------------------------------------

    def _send_trajectory(self, controller_type, joint_names, positions,
                         velocities, times, start_time):
        now = self.now()
        self.sent.append(dict(
            controller_type=controller_type, joint_names=list(joint_names),
            positions=np.array(positions), velocities=np.array(velocities),
            times=np.array(times), start_time=start_time, received=now))
        t_start = now + start_time
        # The motion starts at rest from where the joints are now.
        start = [self._position(name, now) for name in joint_names]
        knot_times = np.concatenate([[t_start], t_start + np.asarray(times)])
        trajectory = hermite_trajectory(
            knot_times,
            np.vstack([start, positions]),
            np.vstack([np.zeros(len(joint_names)), velocities]))
        for j, name in enumerate(joint_names):
            self._motions[name] = (trajectory, j, knot_times[-1])
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
            self._motions.pop(name, None)

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
