"""What a simulated robot interface does the same in every simulator.

:class:`SimulatedRobotInterfaceBase` turns angle vectors into joint
trajectories, and base goals into base paths, the way the ROS interface
does, and leaves stepping the physics to the subclass. A subclass
(:class:`skrobot.interfaces._mujoco.MujocoRobotInterface`,
:class:`skrobot.interfaces._isaacsim.IsaacSimRobotInterface`) provides

- ``time``: the simulation time [s],
- ``_timestep``: one physics step [s],
- ``step(n)``: advance ``n`` physics steps, writing ``_target(time)`` (and
  ``_base_target(time)``) to the joints as the commanded positions,
- ``sync()``: read the simulated joints into ``self.robot``,
- ``_base_pose()``: the simulated base ``(x, y, yaw)``, when ``planar_base``.

Nothing advances on its own: :meth:`wait_interpolation` and ``step`` are
what run the physics, so the classes need no threads.
"""

import logging

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.model.joint import LinearJoint
from skrobot.model.joint import RotationalJoint


logger = logging.getLogger(__name__)

# Below these the ROS interface treats a joint as already at its goal; the
# same thresholds keep the interfaces choosing the same durations.
_ROTATIONAL_EPS = np.deg2rad(0.1)
_LINEAR_EPS = 0.01


def _planar_pose(coords):
    """``(x, y, yaw)`` of coordinates, dropping height, roll and pitch."""
    rot = coords.rotation
    x, y, _ = coords.translation
    return np.array([x, y, np.arctan2(rot[1, 0], rot[0, 0])])


def _planar_coords(pose):
    """Coordinates on the floor at ``(x, y, yaw)``."""
    return Coordinates(pos=(pose[0], pose[1], 0.0)).rotate(pose[2], 'z')


def _compose_planar(pose, delta):
    """``pose`` moved by ``delta``, given in ``pose``'s own frame."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.array([pose[0] + c * delta[0] - s * delta[1],
                     pose[1] + s * delta[0] + c * delta[1],
                     pose[2] + delta[2]])


def _hermite(t, t0, t1, p0, p1, v0, v1):
    """Cubic Hermite segment, the shape a joint trajectory controller follows.

    Parameters
    ----------
    t : float
        Query time, clipped to ``[t0, t1]``.
    t0, t1 : float
        Segment start and end time.
    p0, p1 : numpy.ndarray
        Positions at ``t0`` and ``t1``.
    v0, v1 : numpy.ndarray
        Velocities at ``t0`` and ``t1``.

    Returns
    -------
    numpy.ndarray
        Position at ``t``.
    """
    duration = t1 - t0
    if duration <= 0.0:
        return p1.copy()
    s = min(max((t - t0) / duration, 0.0), 1.0)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1
    h10 = s ** 3 - 2 * s ** 2 + s
    h01 = -2 * s ** 3 + 3 * s ** 2
    h11 = s ** 3 - s ** 2
    return (h00 * p0 + h10 * duration * v0
            + h01 * p1 + h11 * duration * v1)


def _camera_specs(cameras):
    """``cameras`` with the defaults filled in: 45 deg, 640x480, optical."""
    specs = {}
    for name, spec in (cameras or {}).items():
        spec = dict(spec)
        spec.setdefault('fovy', 45.0)
        spec.setdefault('width', 640)
        spec.setdefault('height', 480)
        spec.setdefault('frame', 'optical')
        specs[name] = spec
    return specs


class SimulatedRobotInterfaceBase(object):
    """Trajectories and base paths for a robot interface over a simulator.

    Subclasses set ``self.robot``, ``self.planar_base``,
    ``self.base_max_velocity`` and ``self.cameras``, call
    :meth:`_reset_commands` from their ``reset``, and implement the hooks
    listed in the module docstring.
    """

    # Angle-vector indices the interface turns itself (wheels following a
    # planar base); an angle vector does not move them.
    _wheel_index = np.zeros(0, dtype=np.int64)

    @property
    def time(self):
        """Simulation time [s]."""
        raise NotImplementedError

    @property
    def _timestep(self):
        raise NotImplementedError

    def step(self, n=1):
        """Advance the simulation ``n`` physics steps."""
        raise NotImplementedError

    def sync(self):
        """Copy the simulated joints into ``ri.robot``; return them."""
        raise NotImplementedError

    def _base_pose(self):
        """Simulated base ``(x, y, yaw)``."""
        raise NotImplementedError

    def _reset_commands(self, av):
        """Hold ``av`` (and the base where the robot model stands)."""
        self._segments = []
        self._hold = np.asarray(av, dtype=np.float64).copy()
        if self.planar_base:
            self._base_segments = []
            self._base_hold = _planar_pose(self.robot.copy_worldcoords())

    def _target(self, t):
        """Commanded angle vector at simulation time ``t``."""
        for t0, t1, p0, p1, v0, v1 in self._segments:
            if t <= t1:
                if t < t0:
                    return p0.copy()
                return _hermite(t, t0, t1, p0, p1, v0, v1)
        return self._hold.copy()

    def angle_vector_duration(self, start_av, end_av):
        """Time each joint needs at its velocity limit.

        Parameters
        ----------
        start_av, end_av : numpy.ndarray
            Angle vectors to move between.

        Returns
        -------
        list[tuple[float, str]]
            ``(time, joint_name)`` per joint, like the ROS interface's
            ``return_joint_names=True``.
        """
        diff = np.asarray(end_av) - np.asarray(start_av)
        # The base turns the wheels; an angle vector does not.
        diff[self._wheel_index] = 0.0
        times = []
        for d, joint in zip(diff, self.robot.joint_list):
            if ((isinstance(joint, RotationalJoint)
                 and abs(d) < _ROTATIONAL_EPS)
                    or (isinstance(joint, LinearJoint)
                        and abs(d) < _LINEAR_EPS)):
                times.append((0.0, joint.name))
            else:
                times.append((abs(d) / joint.max_joint_velocity, joint.name))
        return times

    @staticmethod
    def _check_time(time, fastest_time, time_scale):
        if time_scale < 1:
            raise ValueError(
                'time_scale must be >=1, but given: {}'.format(time_scale))
        fastest = max([t for t, _ in fastest_time] + [0.0])
        if time is None:
            return time_scale * fastest
        if time < fastest:
            # Under a millisecond short is float32 rounding in the angle
            # vector (1.0 s becoming 1.0000000149 s), not worth a warning.
            if fastest - time > 1e-3:
                slow = [name for t, name in fastest_time if t > time]
                logger.warning(
                    'Time has been changed from %s to %s due to joint '
                    'velocity limit of %s.', time, fastest, slow)
            return fastest
        return float(time)

    def angle_vector(self, av=None, time=None, start_time=0.0,
                     time_scale=5.0):
        """Send an angle vector, or read the simulated one.

        Returns immediately; call :meth:`wait_interpolation` to let the
        simulation carry the motion out. A new command replaces the
        trajectory still running, starting from where it is now.

        Parameters
        ----------
        av : numpy.ndarray or None
            Goal angle vector. ``None`` returns the simulated joints (and
            updates ``ri.robot`` with them).
        time : float or None
            Time to goal [s]. Raised to what the velocity limits allow;
            ``None`` uses ``time_scale`` times the fastest time.
        start_time : float
            Delay before the motion starts [s].
        time_scale : float
            See ``time``. Must be >= 1.

        Returns
        -------
        numpy.ndarray
            ``av``, or the simulated angle vector when ``av`` is None.
        """
        if av is None:
            return self.sync()
        # ri.robot gets ``av`` as given: robot.angle_vector() is float32,
        # and widening a value sitting on a limit to float64 lands it just
        # past that limit.
        self.robot.angle_vector(av)
        return self.angle_vector_sequence(
            [av], [time], start_time=start_time, time_scale=time_scale)[0]

    def angle_vector_sequence(self, avs, times=None, start_time=0.0,
                              time_scale=5.0):
        """Send a sequence of angle vectors.

        Velocities at the via points follow the ROS interface's rule: the
        mean of the neighbouring segment velocities where they agree in
        sign, zero where the joint turns back.

        Parameters
        ----------
        avs : list[numpy.ndarray]
            Angle vectors to pass through.
        times : list[float] or float or None
            Duration of each segment [s], or one value for all.
        start_time : float
            Delay before the motion starts [s].
        time_scale : float
            Used for segments whose time is ``None``. Must be >= 1.

        Returns
        -------
        list[numpy.ndarray]
            ``avs``.
        """
        avs = [np.asarray(av, dtype=np.float64) for av in avs]
        if not isinstance(times, (list, tuple)):
            times = [times] * len(avs)
        now = self.time
        prev = self._target(now)
        durations = []
        for av, tm in zip(avs, times):
            durations.append(self._check_time(
                tm, self.angle_vector_duration(prev, av), time_scale))
            prev = av

        points = [self._target(now)] + avs
        velocities = [np.zeros_like(points[0])]
        for i in range(1, len(points) - 1):
            v0 = points[i] - points[i - 1]
            v1 = points[i + 1] - points[i]
            vel = np.zeros_like(v0)
            if durations[i - 1] > 0.0 and durations[i] > 0.0:
                agree = v0 * v1 >= 0.0
                vel[agree] = 0.5 * (v0[agree] / durations[i - 1]
                                    + v1[agree] / durations[i])
            velocities.append(vel)
        velocities.append(np.zeros_like(points[0]))

        segments = []
        t = now + start_time
        for i, duration in enumerate(durations):
            segments.append((t, t + duration, points[i], points[i + 1],
                             velocities[i], velocities[i + 1]))
            t += duration
        self._segments = segments
        self._hold = avs[-1].copy()
        return avs

    def is_interpolating(self):
        """Whether a sent trajectory has time left to run."""
        return bool(self._segments) and self.time < self._segments[-1][1]

    def wait_interpolation(self, timeout=0):
        """Simulate until the trajectory sent last has run out.

        Parameters
        ----------
        timeout : float
            Longest simulation time to run [s]. 0 waits for the whole
            trajectory.

        Returns
        -------
        list[bool]
            ``[is_interpolating()]`` afterwards, as the ROS interface returns
            one entry per controller.
        """
        if self._segments:
            end = self._segments[-1][1]
            if timeout > 0:
                end = min(end, self.time + timeout)
            n = int(np.ceil((end - self.time) / self._timestep))
            self.step(max(n, 0))
            # Summing timestep n times can land a hair short of ``end``.
            while self.time < end:
                self.step(1)
        return [self.is_interpolating()]

    def cancel_angle_vector(self):
        """Stop where the commanded trajectory is now."""
        self._hold = self._target(self.time)
        self._segments = []

    def potentio_vector(self):
        """Simulated angle vector (see :meth:`sync`)."""
        return self.sync()

    def simulate(self, duration):
        """Let ``duration`` seconds of simulation pass.

        Parameters
        ----------
        duration : float
            Simulation time to run [s].
        """
        self.step(int(round(duration / self._timestep)))

    # --- base ------------------------------------------------------------

    def _require_base(self):
        if not self.planar_base:
            raise RuntimeError(
                'the base is welded to the world; construct the interface '
                'with planar_base=True to move it.')

    def _base_target(self, t):
        """Commanded base ``(x, y, yaw)`` at simulation time ``t``."""
        for t0, t1, path in self._base_segments:
            if t <= t1:
                return path(max(t, t0))
        return self._base_hold.copy()

    def _base_start(self):
        """Where a new base motion starts: the commanded pose while one is
        under way, so that motions chain smoothly, otherwise where the base
        is, so that a motion that was blocked is not carried into the next
        one (a real robot's move_base also starts from where it is)."""
        if self.is_moving_base():
            return self._base_target(self.time)
        return self._base_pose()

    def _send_base(self, duration, path, wait):
        t0 = self.time
        t1 = t0 + duration
        self._base_segments = [(t0, t1, path)]
        self._base_hold = path(t1)
        if wait:
            return self.wait_base()
        return True

    def _base_duration(self, start, goal):
        linear, angular = self.base_max_velocity
        distance = np.linalg.norm(goal[:2] - start[:2])
        turn = abs(goal[2] - start[2])
        # The Hermite profile peaks at 1.5x its mean speed; this keeps the
        # peak, not the mean, within base_max_velocity.
        return 1.5 * max(distance / linear, turn / angular)

    def move_to(self, coords, wait=True):
        """Drive the base to ``coords``, given in the world frame.

        The base moves and turns at the same time, along a straight line.

        Parameters
        ----------
        coords : skrobot.coordinates.Coordinates
            Goal. Only its x, y and yaw are used.
        wait : bool
            Simulate until the base arrives (see :meth:`wait_base`).

        Returns
        -------
        bool
            Whether the base arrived (see :meth:`wait_base`); True when
            not waiting.
        """
        self._require_base()
        start = self._base_start()
        goal = _planar_pose(coords)
        # Turn the short way round.
        goal[2] = start[2] + np.arctan2(np.sin(goal[2] - start[2]),
                                        np.cos(goal[2] - start[2]))
        duration = self._base_duration(start, goal)
        zero = np.zeros(3)
        t0 = self.time

        def path(t):
            return _hermite(t, t0, t0 + duration, start, goal, zero, zero)
        return self._send_base(duration, path, wait)

    def go_pos(self, x=0.0, y=0.0, yaw=0.0, wait=True):
        """Drive the base by ``(x, y, yaw)`` in its own frame.

        Parameters
        ----------
        x, y : float
            Distance forward and to the left [m].
        yaw : float
            Turn, counterclockwise [rad].
        wait : bool
            Simulate until the base arrives (see :meth:`wait_base`).

        Returns
        -------
        bool
            Whether the base arrived (see :meth:`wait_base`); True when
            not waiting.
        """
        self._require_base()
        start = self._base_start()
        goal = _compose_planar(start, (x, y, yaw))
        duration = self._base_duration(start, goal)
        zero = np.zeros(3)
        t0 = self.time

        def path(t):
            return _hermite(t, t0, t0 + duration, start, goal, zero, zero)
        return self._send_base(duration, path, wait)

    def go_velocity(self, x=0.0, y=0.0, yaw=0.0, sec=1.0, wait=False):
        """Drive the base at a velocity given in its own frame, then stop.

        Parameters
        ----------
        x, y : float
            Velocity forward and to the left [m/s].
        yaw : float
            Turning rate, counterclockwise [rad/s].
        sec : float
            How long to drive [s].
        wait : bool
            Simulate until the time is up (see :meth:`wait_base`).

        Returns
        -------
        bool
            Whether the base arrived (see :meth:`wait_base`); True when
            not waiting.
        """
        self._require_base()
        start = self._base_start()
        t0 = self.time

        def path(t):
            tau = t - t0
            turn = yaw * tau
            if abs(yaw) < 1e-9:
                dx, dy = x * tau, y * tau
            else:
                # Constant body velocity traces a circular arc.
                dx = (np.sin(turn) * x - (1.0 - np.cos(turn)) * y) / yaw
                dy = ((1.0 - np.cos(turn)) * x + np.sin(turn) * y) / yaw
            return _compose_planar(start, (dx, dy, turn))
        return self._send_base(sec, path, wait)

    def go_stop(self):
        """Stop the base where it is commanded to be now."""
        self._require_base()
        self._base_hold = self._base_target(self.time)
        self._base_segments = []

    def is_moving_base(self):
        """Whether a base motion has time left to run."""
        return (self.planar_base and bool(self._base_segments)
                and self.time < self._base_segments[-1][1])

    def wait_base(self):
        """Simulate until the base has arrived where it was sent last.

        Your robot model stays where it was. To plan toward world
        coordinates, such as an object's, from where the base now is, move
        it there: ``robot.newcoords(ri.odom)``.

        Returns
        -------
        bool
            Whether the base arrived. False when it is still off its goal
            2 s after the motion ended, as when something blocks it; the
            base then stops where it is, as move_base gives up, and the
            next motion starts from there.
        """
        self._require_base()
        if self._base_segments:
            end = self._base_segments[-1][1]
            while self.time < end:
                self.step(max(1, int(np.ceil(
                    (end - self.time) / self._timestep))))
        # The base trails its command; finish when it has arrived, as
        # move_base does, rather than when the command has.
        settle_end = self.time + 2.0
        while True:
            error = self._base_pose() - self._base_hold
            if (np.linalg.norm(error[:2]) < 1e-3
                    and abs(error[2]) < np.deg2rad(0.1)):
                return True
            if self.time >= settle_end:
                logger.warning(
                    'base did not reach its goal within 2 s; still %s off '
                    '(x [m], y [m], yaw [rad])', np.round(error, 4))
                self._base_hold = self._base_pose()
                self._base_segments = []
                return False
            self.step(10)

    @property
    def odom(self):
        """Simulated base pose in the world frame.

        Returns
        -------
        skrobot.coordinates.Coordinates
            Where the base is, which lags the command a little while moving.
        """
        self._require_base()
        return _planar_coords(self._base_pose())

    # --- cameras ---------------------------------------------------------

    def _camera(self, name):
        if name not in self.cameras:
            raise ValueError('no camera {}; cameras are {}'.format(
                name, sorted(self.cameras)))
        return self.cameras[name]

    def camera_info(self, name):
        """Image size and pinhole intrinsics of a camera.

        Parameters
        ----------
        name : str
            Camera name.

        Returns
        -------
        dict
            ``width``, ``height`` [px] and ``K``, the 3x3 matrix taking a
            point in the optical frame to homogeneous pixel coordinates.
        """
        spec = self._camera(name)
        width, height = spec['width'], spec['height']
        f = 0.5 * height / np.tan(0.5 * np.deg2rad(spec['fovy']))
        K = np.array([[f, 0.0, 0.5 * width],
                      [0.0, f, 0.5 * height],
                      [0.0, 0.0, 1.0]])
        return {'width': width, 'height': height, 'K': K}
