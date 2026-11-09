"""Backend-agnostic base class for robot interfaces.

A *robot interface* (``ri``) sends joint commands to a robot -- real
hardware, a simulator, a test double -- and reads its measured state back,
behind one call shape::

    ri.angle_vector(av, 3.0)      # move there in 3 s; returns at once
    ri.wait_interpolation()       # block until the motion is done
    ri.potentio_vector()          # last measured joint angles

:class:`RobotInterfaceBase` implements that call shape once: joint
bookkeeping, vector-or-dict goals, durations from velocity limits, via-point
velocities of a sequence, and the measured state. A backend only says how
to talk to its robot by implementing four hooks:

``_send_trajectory(controller_type, joint_names, positions, velocities,
times, start_time)``
    Start a timed trajectory for ``joint_names``. Must return without
    waiting for the motion.
``_read_joint_state()``
    Return the latest measured positions as ``{joint_name: position}``.
``_is_interpolating(controller_type)``
    Whether a trajectory sent to that controller is still running.
``_cancel(controller_type)``
    Stop that controller's motion where it is.

Optional hooks: ``_wait_interpolation`` (the default polls
``_is_interpolating``; a simulator overrides it to step its own clock) and
``_set_servo`` (torque on/off; the default raises ``NotImplementedError``).

The base class never writes into the robot model it is given. The model
clamps every angle into its URDF limits, so a measured angle that sits
past a limit would read back as the limit, and replaying it as a command
would make the robot jump. Measured angles are kept by joint name instead,
reported as they are, and :meth:`RobotInterfaceBase.outside_limits` lists
the joints that are past their limits.

Examples
--------
A backend for a robot driven through a vendor SDK could look like this
(``sdk`` stands for the vendor's library)::

    from skrobot.interfaces.base import RobotInterfaceBase

    class MyArmInterface(RobotInterfaceBase):

        def __init__(self, robot, port):
            self.dev = sdk.open(port)
            super(MyArmInterface, self).__init__(robot)

        def _send_trajectory(self, controller_type, joint_names,
                             positions, velocities, times, start_time):
            self.dev.play(joint_names, positions, times, delay=start_time)

        def _read_joint_state(self):
            return dict(zip(self.dev.joint_names(), self.dev.positions()))

        def _is_interpolating(self, controller_type):
            return self.dev.busy()

        def _cancel(self, controller_type):
            self.dev.stop()

        def _set_servo(self, joint_names, on):
            self.dev.torque(joint_names, on)

and is then used like any other interface::

    ri = MyArmInterface(robot, '/dev/ttyUSB0')
    ri.angle_vector(robot.angle_vector(), 3.0)       # whole controller group
    ri.angle_vector({'elbow_joint': 0.5})            # only that joint
    ri.wait_interpolation()
    print(ri.outside_limits())                       # measured, unclamped

:class:`skrobot.interfaces.fake.FakeRobotInterface` is a complete backend
that only interpolates in memory; it is the reference for writing one.
"""

import abc
import logging
import numbers
import time as _time

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from skrobot.model.joint import LinearJoint


logger = logging.getLogger(__name__)

# Below these a joint counts as already at its goal (the ROS interface uses
# the same thresholds, so both choose the same durations).
_ROTATIONAL_EPS = np.deg2rad(0.1)
_LINEAR_EPS = 0.01
# A requested time shorter than the velocity limit allows by less than this
# is float rounding, not worth a warning.
_TIME_ROUNDING = 1e-3

DEFAULT_CONTROLLER = 'default_controller'


def hermite_trajectory(times, positions, velocities):
    """Piecewise cubic Hermite trajectory through timed knots.

    The shape a joint trajectory controller follows between its points:
    each segment matches the positions and velocities at both of its ends
    (:class:`scipy.interpolate.CubicHermiteSpline`). A backend that
    executes trajectories itself (a simulator, a test double) can sample it
    at its own clock.

    Parameters
    ----------
    times : array_like
        ``(n_knots,)`` non-decreasing knot times [s]. Two knots with the
        same time are a zero-length segment: the trajectory jumps there,
        to the later knot's position.
    positions : array_like
        ``(n_knots, ...)`` positions at the knots, e.g.
        ``(n_knots, n_joints)``.
    velocities : array_like
        Velocities at the knots, shaped like ``positions``.

    Returns
    -------
    callable
        ``f(t)`` returning the position at time ``t`` as a new array.
        Before the first knot it holds the first position, after the last
        knot the last one (no extrapolation). At the time of a jump it
        returns the position reached before the jump, except at the first
        knot, where it returns the position after it.

    Raises
    ------
    ValueError
        If ``times`` decreases or the shapes do not match.
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError('times must be a non-empty 1-D array')
    if positions.shape[0] != len(times) \
            or velocities.shape != positions.shape:
        raise ValueError(
            'positions {} and velocities {} must have one row per time '
            '({})'.format(positions.shape, velocities.shape, len(times)))
    if np.any(np.diff(times) < 0):
        raise ValueError('times must not decrease: {}'.format(times))

    # CubicHermiteSpline needs strictly increasing times, so the knots are
    # split at zero-length segments into pieces, each one spline:
    # (end time, position at the end, spline or None for a jump).
    pieces = []
    n = len(times)
    k = 0
    while k < n - 1:
        if times[k + 1] == times[k]:
            pieces.append((times[k], positions[k + 1], None))
            k += 1
            continue
        j = k + 1
        while j + 1 < n and times[j + 1] > times[j]:
            j += 1
        pieces.append((times[j], positions[j], CubicHermiteSpline(
            times[k:j + 1], positions[k:j + 1], velocities[k:j + 1],
            axis=0)))
        k = j
    first_time = times[0]
    first = positions[0].copy()
    last = positions[-1].copy()

    def evaluate(t):
        if t < first_time:
            return first.copy()
        for piece_end, end_position, spline in pieces:
            if t <= piece_end:
                if spline is None or t == piece_end:
                    return end_position.copy()
                return spline(t)
        return last.copy()
    return evaluate


class RobotInterfaceBase(object, metaclass=abc.ABCMeta):
    """Base class of robot interfaces, independent of any middleware.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Model of the robot. It is read (joint names, limits, velocity
        limits) but never written to.
    controllers : dict[str, list[str]] or None
        Joint groups that can be commanded independently, as
        ``{controller_type: [joint names]}``. ``None`` makes one group,
        ``'default_controller'``, of every joint in ``robot.joint_list``.
    controller_type : str or None
        Group used when a call does not name one. ``None`` takes the first
        key of ``controllers``.
    poll_period : float
        Period [s] at which the default :meth:`_wait_interpolation` checks
        whether the motion is done.

    Attributes
    ----------
    robot : skrobot.model.RobotModel
        The model given.
    controller_type : str
        Default controller group.
    """

    def __init__(self, robot, controllers=None, controller_type=None,
                 poll_period=0.01):
        self.robot = robot
        self.poll_period = poll_period
        # Measured positions by joint name, exactly as the backend reported
        # them (never passed through the clamping model).
        self._measured = {}
        self._init_controllers(controllers, controller_type)

    def _init_controllers(self, controllers, controller_type):
        """Validate and store the controller groups.

        A backend that keeps its controller groups elsewhere (the ROS 2
        interface builds them from its action servers) overrides this and
        the :attr:`_controllers` property.
        """
        if controllers is None:
            controllers = {DEFAULT_CONTROLLER: list(self._model_joint_names)}
        if len(controllers) == 0:
            raise ValueError('controllers must name at least one group')
        joints = self._joints
        groups = {}
        for name, joint_names in controllers.items():
            joint_names = list(joint_names)
            self._check_names(joint_names)
            if len(set(joint_names)) != len(joint_names):
                raise ValueError(
                    'controller {} lists a joint twice: {}'.format(
                        name, joint_names))
            multi_dof = [n for n in joint_names
                         if joints[n].joint_dof != 1]
            if multi_dof:
                raise ValueError(
                    'only 1-DOF joints can be commanded, but controller {} '
                    'has {}'.format(name, multi_dof))
            groups[name] = joint_names
        self._controller_groups = groups
        if controller_type is None:
            controller_type = next(iter(groups))
        self._check_controller(controller_type)
        self.controller_type = controller_type

    @property
    def _controllers(self):
        """``{controller_type: [joint names]}`` of the controller groups."""
        return self._controller_groups

    @property
    def _joints(self):
        """``{joint_name: joint}`` of the current ``self.robot``."""
        return {j.name: j for j in self.robot.joint_list}

    @property
    def _model_joint_names(self):
        """Joint names of the current ``self.robot``, in model order."""
        return [j.name for j in self.robot.joint_list]

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _send_trajectory(self, controller_type, joint_names, positions,
                         velocities, times, start_time):
        """Start a timed joint trajectory. Must not wait for it.

        Parameters
        ----------
        controller_type : str
            Group the joints belong to.
        joint_names : list[str]
            Joints to move: all of the group's joints, or a subset of them
            when the goal was given as a dict. Joints of the group not
            listed must keep doing what they do.
        positions : numpy.ndarray
            ``(n_points, len(joint_names))`` positions to pass through.
            The trajectory starts from where the joints are now; that
            point is not included.
        velocities : numpy.ndarray
            ``(n_points, len(joint_names))`` velocities at those points.
        times : numpy.ndarray
            ``(n_points,)`` time [s] of each point, counted from the start
            of the motion (``time_from_start`` in ROS terms); increasing.
        start_time : float
            Delay [s] from now before the motion starts.
        """

    @abc.abstractmethod
    def _read_joint_state(self):
        """Return the latest measured joint positions.

        Returns
        -------
        dict[str, float]
            ``{joint_name: position}``. It may cover only some joints (the
            others keep their last value) and may contain names the model
            does not have (they are stored but not used).
        """

    @abc.abstractmethod
    def _is_interpolating(self, controller_type):
        """Whether a trajectory sent to ``controller_type`` is running.

        Parameters
        ----------
        controller_type : str
            Group to ask about.

        Returns
        -------
        bool
            True while a motion is in progress.
        """

    @abc.abstractmethod
    def _cancel(self, controller_type):
        """Stop the motion of ``controller_type`` where it is.

        Parameters
        ----------
        controller_type : str
            Group to stop.
        """

    def _wait_interpolation(self, controller_type, timeout):
        """Block until ``controller_type`` stops or ``timeout`` runs out.

        The default polls :meth:`_is_interpolating` every
        ``poll_period`` seconds of wall-clock time. A backend with its own
        clock (a simulator) or a blocking wait of its own overrides this.

        Parameters
        ----------
        controller_type : str
            Group to wait for.
        timeout : float or None
            Longest wait [s]; ``None`` waits until the motion is done.
        """
        start = _time.monotonic()
        while self._is_interpolating(controller_type):
            if timeout is not None and _time.monotonic() - start >= timeout:
                return
            _time.sleep(self.poll_period)

    def _set_servo(self, joint_names, on):
        """Switch joint torque on or off.

        Parameters
        ----------
        joint_names : list[str]
            Joints to switch.
        on : bool
            True for servo on.
        """
        raise NotImplementedError(
            '{} does not support switching servos on and off; a backend '
            'that can implements _set_servo(joint_names, on)'.format(
                type(self).__name__))

    # ------------------------------------------------------------------
    # Joints and controllers
    # ------------------------------------------------------------------

    @property
    def controllers(self):
        """Controller groups, as ``{controller_type: [joint names]}``."""
        return {k: list(v) for k, v in self._controllers.items()}

    def joint_names(self, controller_type=None):
        """Names of the joints a controller group moves.

        Parameters
        ----------
        controller_type : str or None
            Group; ``None`` is :attr:`controller_type`.

        Returns
        -------
        list[str]
            Joint names in the group's order.
        """
        return list(self._controllers[self._controller(controller_type)])

    def _controller(self, controller_type):
        if controller_type is None:
            return self.controller_type
        self._check_controller(controller_type)
        return controller_type

    def _check_controller(self, controller_type):
        if controller_type not in self._controllers:
            raise KeyError('unknown controller_type {}; known: {}'.format(
                controller_type, sorted(self._controllers)))

    def _controllers_to_use(self, controller_type):
        if controller_type is None:
            return list(self._controllers)
        self._check_controller(controller_type)
        return [controller_type]

    def _check_names(self, names):
        unknown = [n for n in names if n not in self._joints]
        if unknown:
            raise KeyError('not joints of {}: {}'.format(
                self.robot.name, ', '.join(map(str, unknown))))

    def _resolve(self, av, controller_type):
        """Turn a goal into ``(joint_names, positions)`` of one group.

        A vector is in ``robot.joint_list`` order and covers the whole
        group; a dict names the joints to move, which must all belong to
        the group.
        """
        group = self._controllers[controller_type]
        if isinstance(av, dict):
            self._check_names(list(av))
            outside = [n for n in av if n not in group]
            if outside:
                raise ValueError(
                    'not joints of controller {}: {}'.format(
                        controller_type, ', '.join(outside)))
            names = [n for n in group if n in av]
            values = np.array([av[n] for n in names], dtype=np.float64)
        else:
            vector = np.asarray(av, dtype=np.float64)
            model_names = self._model_joint_names
            if vector.shape != (len(model_names),):
                raise ValueError(
                    'angle vector must have shape ({},) in robot.joint_list '
                    'order, but got {}'.format(
                        len(model_names), vector.shape))
            index = {n: i for i, n in enumerate(model_names)}
            names = list(group)
            values = np.array([vector[index[n]] for n in names],
                              dtype=np.float64)
        if not np.all(np.isfinite(values)):
            bad = [n for n, v in zip(names, values) if not np.isfinite(v)]
            raise ValueError('goal is not finite for {}'.format(bad))
        return names, values

    def _diff(self, names, goal, start):
        """``goal - start``, the short way round for continuous joints."""
        diff = np.asarray(goal, dtype=np.float64) - start
        joints = self._joints
        for i, name in enumerate(names):
            joint = joints[name]
            if np.isinf(joint.min_angle) and np.isinf(joint.max_angle):
                diff[i] = (diff[i] + np.pi) % (2 * np.pi) - np.pi
        return diff

    # ------------------------------------------------------------------
    # Measured state
    # ------------------------------------------------------------------

    def update_robot_state(self):
        """Read the backend's measured joint state into the interface.

        Returns
        -------
        dict[str, float]
            All measured positions known so far, by joint name.
        """
        state = self._read_joint_state()
        for name, position in state.items():
            self._measured[name] = float(position)
        return dict(self._measured)

    def measured_joint_angles(self, joint_names=None):
        """Last measured positions by name, as measured (not clamped).

        Parameters
        ----------
        joint_names : list[str] or None
            Joints to return; ``None`` returns all measured joints.

        Returns
        -------
        dict[str, float]
            ``{joint_name: position}``.

        Raises
        ------
        KeyError
            If a name is not a joint of the model.
        ValueError
            If a requested joint has never been measured.
        """
        if joint_names is None:
            return {n: self._measured[n] for n in self._model_joint_names
                    if n in self._measured}
        joint_names = list(joint_names)
        self._check_names(joint_names)
        missing = [n for n in joint_names if n not in self._measured]
        if missing:
            raise ValueError(
                'never measured: {}. Call update_robot_state() or check that '
                'the backend reports these joints.'.format(
                    ', '.join(missing)))
        return {n: self._measured[n] for n in joint_names}

    def potentio_vector(self):
        """Last measured joint angles, in ``robot.joint_list`` order.

        Same as :meth:`measured_angle_vector` here. A backend may override
        it for compatibility (the ROS 2 interface returns its model's,
        clamped, angles); :meth:`measured_angle_vector` stays unclamped.

        Returns
        -------
        numpy.ndarray
            ``(len(robot.joint_list),)`` float64 vector.
        """
        return self.measured_angle_vector()

    def measured_angle_vector(self):
        """Last measured joint angles, unclamped, in ``robot.joint_list`` order.

        Built from the measured values by joint name, without going through
        the robot model, so an angle past its URDF limit is returned as it
        is. Joints that have never been measured are ``nan``. Call
        :meth:`update_robot_state` (or ``angle_vector()``) to refresh.

        Returns
        -------
        numpy.ndarray
            ``(len(robot.joint_list),)`` float64 vector.
        """
        return np.array([self._measured.get(n, np.nan)
                         for n in self._model_joint_names],
                        dtype=np.float64)

    def outside_limits(self, av=None, margin=0.0):
        """Joints whose angle is outside the model's joint limits.

        Parameters
        ----------
        av : numpy.ndarray, dict or None
            Angles to check: a vector in ``robot.joint_list`` order or a
            ``{joint_name: angle}`` dict. ``None`` checks the last measured
            state.
        margin : float
            Tolerance; an angle counts as outside only when it is more than
            ``margin`` past a limit.

        Returns
        -------
        dict[str, tuple[float, float, float]]
            ``{joint_name: (angle, min_angle, max_angle)}``; empty when all
            joints are within their limits.
        """
        if av is None:
            angles = self.measured_joint_angles()
        elif isinstance(av, dict):
            self._check_names(list(av))
            angles = av
        else:
            vector = np.asarray(av, dtype=np.float64)
            model_names = self._model_joint_names
            if vector.shape != (len(model_names),):
                raise ValueError(
                    'angle vector must have shape ({},), but got {}'.format(
                        len(model_names), vector.shape))
            angles = dict(zip(model_names, vector))
        result = {}
        joints = self._joints
        for name, angle in angles.items():
            joint = joints[name]
            if joint.joint_dof != 1 or np.isnan(angle):
                continue
            lower = float(joint.min_angle)
            upper = float(joint.max_angle)
            if angle < lower - margin or angle > upper + margin:
                result[name] = (float(angle), lower, upper)
        return result

    # ------------------------------------------------------------------
    # Durations
    # ------------------------------------------------------------------

    def _joint_durations(self, names, start, goal):
        """``[(time, name)]`` each joint needs at its velocity limit.

        ``time`` is ``nan`` for a joint that moves but has no usable
        velocity limit.
        """
        diff = self._diff(names, goal, start)
        times = []
        joints = self._joints
        for name, d in zip(names, diff):
            joint = joints[name]
            eps = _LINEAR_EPS if isinstance(joint, LinearJoint) \
                else _ROTATIONAL_EPS
            if abs(d) < eps:
                times.append((0.0, name))
                continue
            velocity = float(joint.max_joint_velocity)
            if not velocity > 0:
                times.append((np.nan, name))
            else:
                times.append((abs(d) / velocity, name))
        return times

    def angle_vector_duration(self, start_av, end_av, controller_type=None,
                              return_joint_names=False):
        """Time to move between two postures at the joint velocity limits.

        Parameters
        ----------
        start_av, end_av : numpy.ndarray or dict
            Postures, as vectors in ``robot.joint_list`` order or as
            ``{joint_name: angle}`` dicts naming the same joints.
        controller_type : str or None
            Only this group's joints count; ``None`` is
            :attr:`controller_type`.
        return_joint_names : bool
            Return per-joint times instead of their maximum.

        Returns
        -------
        float or list[tuple[float, str]]
            The longest joint time, or ``[(time, joint_name), ...]``.
            A joint without a positive velocity limit has time ``nan``.
        """
        controller_type = self._controller(controller_type)
        names, start = self._resolve(start_av, controller_type)
        end_names, end = self._resolve(end_av, controller_type)
        if names != end_names:
            raise ValueError('start_av and end_av name different joints')
        times = self._joint_durations(names, start, end)
        if return_joint_names:
            return times
        return max([t for t, _ in times] + [0.0])

    def _check_time(self, time, joint_times, time_scale):
        """Duration to use for a segment.

        ``None`` is ``time_scale`` times the slowest joint at its velocity
        limit; a time shorter than that is raised to it, with a warning.
        """
        if time_scale < 1:
            raise ValueError(
                'time_scale must be >=1, but given: {}'.format(time_scale))
        no_limit = [n for t, n in joint_times if np.isnan(t)]
        fastest = max([t for t, _ in joint_times if not np.isnan(t)] + [0.0])
        if time is None:
            if no_limit:
                raise ValueError(
                    'cannot compute a duration: no positive velocity limit '
                    'for {}. Pass time explicitly.'.format(no_limit))
            return time_scale * fastest
        if not isinstance(time, numbers.Real) or not np.isfinite(time) \
                or time < 0:
            raise ValueError(
                'time must be a non-negative number, but given: {}'.format(
                    time))
        if time < fastest:
            if fastest - time > _TIME_ROUNDING:
                slow = [n for t, n in joint_times if t > time]
                logger.warning(
                    'Time has been changed from %s to %s due to joint '
                    'velocity limit of %s.', time, fastest, slow)
            return float(fastest)
        return float(time)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _start_positions(self, names):
        self.update_robot_state()
        measured = self.measured_joint_angles(names)
        return np.array([measured[n] for n in names], dtype=np.float64)

    def angle_vector(self, av=None, time=None, controller_type=None,
                     start_time=0.0, time_scale=5.0, velocities=None):
        """Send a goal posture, or read the measured one.

        Returns at once; use :meth:`wait_interpolation` to block until the
        motion is done.

        Parameters
        ----------
        av : numpy.ndarray, list, dict or None
            Goal. A vector in ``robot.joint_list`` order moves every joint
            of the controller group (other entries are ignored). A
            ``{joint_name: angle}`` dict moves only the joints it names;
            the group's other joints are left alone. ``None`` refreshes and
            returns the measured state (see :meth:`potentio_vector`).
        time : float or None
            Time to the goal [s]. A time shorter than the joint velocity
            limits allow is raised to that. ``None`` uses ``time_scale``
            times the shortest possible time.
        controller_type : str or None
            Controller group; ``None`` is :attr:`controller_type`.
        start_time : float
            Delay [s] before the motion starts.
        time_scale : float
            Used when ``time`` is None. Must be >= 1.
        velocities : numpy.ndarray, dict or None
            Joint velocities at the goal, in the same form as ``av``.
            ``None`` stops there.

        Returns
        -------
        numpy.ndarray or dict
            The goal sent (a float64 vector, or the dict given), or the
            measured angle vector when ``av`` is None.

        Raises
        ------
        KeyError
            If ``av`` names a joint the model does not have.
        ValueError
            If ``av`` is malformed, names a joint outside the controller
            group, or a joint to move has never been measured.
        """
        if av is None:
            self.update_robot_state()
            return self.measured_angle_vector()
        controller_type = self._controller(controller_type)
        names, goal = self._resolve(av, controller_type)
        start = self._start_positions(names)
        duration = self._check_time(
            time, self._joint_durations(names, start, goal), time_scale)
        if velocities is None:
            goal_velocity = np.zeros_like(goal)
        else:
            velocity_names, goal_velocity = self._resolve(
                velocities, controller_type)
            if velocity_names != names:
                raise ValueError('velocities must name the same joints as av')
        self._send_trajectory(
            controller_type, names, goal[None, :], goal_velocity[None, :],
            np.array([duration]), float(start_time))
        if isinstance(av, dict):
            return av
        return np.asarray(av, dtype=np.float64)

    def angle_vector_sequence(self, avs, times=None, controller_type=None,
                              start_time=0.0, time_scale=5.0):
        """Send a sequence of postures to pass through.

        Velocities at the via points are the mean of the neighbouring
        segment velocities where those agree in sign, and zero where the
        joint turns back. The motion stops at the last posture.

        Parameters
        ----------
        avs : list[numpy.ndarray] or list[dict]
            Postures, each as :meth:`angle_vector` takes them. All must
            name the same joints.
        times : list[float or None], float or None
            Duration [s] of each segment from the previous posture, or one
            value for all. Each is treated as ``time`` of
            :meth:`angle_vector`.
        controller_type : str or None
            Controller group; ``None`` is :attr:`controller_type`.
        start_time : float
            Delay [s] before the motion starts.
        time_scale : float
            Used for segments whose time is None. Must be >= 1.

        Returns
        -------
        list
            ``avs``.
        """
        if len(avs) == 0:
            raise ValueError('avs is empty')
        controller_type = self._controller(controller_type)
        if not isinstance(times, (list, tuple, np.ndarray)):
            times = [times] * len(avs)
        if len(times) != len(avs):
            raise ValueError('times has {} entries for {} postures'.format(
                len(times), len(avs)))
        names = None
        goals = []
        for av in avs:
            av_names, goal = self._resolve(av, controller_type)
            if names is not None and av_names != names:
                raise ValueError('all postures must name the same joints')
            names = av_names
            goals.append(goal)
        start = self._start_positions(names)

        points = [start] + goals
        durations = []
        for i in range(len(goals)):
            durations.append(self._check_time(
                times[i],
                self._joint_durations(names, points[i], points[i + 1]),
                time_scale))

        velocities = []
        for i in range(1, len(points)):
            vel = np.zeros(len(names))
            if i < len(points) - 1 and durations[i - 1] > 0 \
                    and durations[i] > 0:
                v0 = self._diff(names, points[i], points[i - 1])
                v1 = self._diff(names, points[i + 1], points[i])
                agree = v0 * v1 >= 0.0
                vel[agree] = 0.5 * (v0[agree] / durations[i - 1]
                                    + v1[agree] / durations[i])
            velocities.append(vel)

        self._send_trajectory(
            controller_type, names, np.array(goals), np.array(velocities),
            np.cumsum(durations), float(start_time))
        return avs

    def cancel_angle_vector(self, controller_type=None):
        """Stop the motion where it is.

        Parameters
        ----------
        controller_type : str or None
            Group to stop; ``None`` stops all groups.
        """
        for name in self._controllers_to_use(controller_type):
            self._cancel(name)

    def is_interpolating(self, controller_type=None):
        """Whether a sent motion is still running.

        Parameters
        ----------
        controller_type : str or None
            Group to ask about; ``None`` asks about all groups.

        Returns
        -------
        bool
            True if any of them is moving.
        """
        return any(self._is_interpolating(name)
                   for name in self._controllers_to_use(controller_type))

    def wait_interpolation(self, controller_type=None, timeout=0):
        """Block until the sent motion is done.

        Refreshes the measured state before returning.

        Parameters
        ----------
        controller_type : str or None
            Group to wait for; ``None`` waits for all groups.
        timeout : float
            Longest wait [s] in total; 0 (or None) waits for as long as the
            motion takes.

        Returns
        -------
        list[bool]
            :meth:`is_interpolating` of each group waited for, afterwards:
            all False unless the timeout ran out.
        """
        names = self._controllers_to_use(controller_type)
        deadline = None
        if timeout:
            deadline = _time.monotonic() + timeout
        for name in names:
            remaining = None
            if deadline is not None:
                remaining = max(deadline - _time.monotonic(), 0.0)
            self._wait_interpolation(name, remaining)
        self.update_robot_state()
        return [self._is_interpolating(name) for name in names]

    def servo_on(self, joint_names=None):
        """Switch joint torque on.

        Parameters
        ----------
        joint_names : list[str] or None
            Joints to switch; ``None`` is every joint of every group.

        Raises
        ------
        NotImplementedError
            If the backend cannot switch servos.
        """
        self._set_servo(self._servo_names(joint_names), True)

    def servo_off(self, joint_names=None):
        """Switch joint torque off. The robot may fall.

        Parameters
        ----------
        joint_names : list[str] or None
            Joints to switch; ``None`` is every joint of every group.

        Raises
        ------
        NotImplementedError
            If the backend cannot switch servos.
        """
        self._set_servo(self._servo_names(joint_names), False)

    def _servo_names(self, joint_names):
        if joint_names is None:
            names = []
            for group in self._controllers.values():
                names.extend(n for n in group if n not in names)
            return names
        joint_names = list(joint_names)
        self._check_names(joint_names)
        return joint_names
