"""Robot interface backed by a MuJoCo simulation.

``MujocoRobotInterface`` plays the role ``*ri*`` plays in EusLisp: the robot
model holds the posture you plan with, the interface holds the robot that
actually moves. Like the ROS interface it keeps its own copy of the model
(``ri.robot``) and never writes into yours; reading the simulated robot back
is ``robot.angle_vector(ri.angle_vector())``. It follows the call shape of
:class:`skrobot.interfaces.ros.base.ROSRobotInterfaceBase` --
``angle_vector(av, time)`` returns at once, ``wait_interpolation()`` blocks
until the motion is done -- so code written against a real robot runs here
unchanged.

Time is simulation time. Nothing advances on its own: ``wait_interpolation``
and :meth:`MujocoRobotInterface.step` are what call ``mj_step``. That keeps
the class free of threads, which is what lets it run inside Pyodide as well,
where the ``mujoco`` module is a thin layer over MuJoCo's WebAssembly build
(see https://github.com/iory/scikit-robot-web).
"""

import copy
import logging
import os
import tempfile
import time as _time
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import quaternion2matrix
from skrobot.interfaces import _mujoco_scene
from skrobot.model.joint import LinearJoint
from skrobot.model.joint import RotationalJoint


logger = logging.getLogger(__name__)

# Below these the ROS interface treats a joint as already at its goal; the
# same thresholds keep the two interfaces choosing the same durations.
_ROTATIONAL_EPS = np.deg2rad(0.1)
_LINEAR_EPS = 0.01

# URDFs already converted in this process -> their MJCF. Converting a robot
# with detailed meshes takes seconds (over ten for Jedy in a browser), and a
# program run again, or a second interface on the same robot, needs the
# same file.
_converted = {}


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


class MujocoRobotInterface(object):
    """Send angle vectors to a robot simulated in MuJoCo.

    The URDF the robot model was loaded from is converted with
    :func:`skrobot.urdf.urdf_to_mjcf`, which gives every actuated joint a
    position actuator named ``<joint>_act``. A trajectory sent with
    :meth:`angle_vector` or :meth:`angle_vector_sequence` is fed to those
    actuators one physics step at a time.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Robot model. Its ``joint_list`` decides the order of angle vectors.
        The interface works on a copy, ``ri.robot``, as the ROS interface
        does; the model passed in is left alone.
    urdf_path : str or None
        URDF to simulate. Defaults to ``robot.urdf_path``.
    mjcf_path : str or None
        An MJCF to load as is, instead of converting the URDF. It must name
        its joints like the URDF and its position actuators ``<joint>_act``.
    floating_base : bool
        Give the base link a free joint instead of welding it to the world.
        :meth:`sync` then also moves the root of ``ri.robot``.
    planar_base : bool
        Let the base slide and turn on the floor, driven by :meth:`go_pos`,
        :meth:`go_velocity` and :meth:`move_to`. The base moves like an
        omnidirectional one, wheels or not, and does not touch the ground
        (it still collides with ``objects``). Cannot be combined with
        ``floating_base``.
    base_max_velocity : tuple[float, float]
        Translation [m/s] and rotation [rad/s] speed :meth:`go_pos` and
        :meth:`move_to` plan with.
    wheels : dict[str, tuple[float, float]] or None
        Wheel joints to turn with the planar base, as ``{joint_name:
        (radius, roller_angle)}``: radius [m], and the angle [rad] of the
        wheel's rollers to its axle's normal -- 0 for a plain wheel,
        about +-pi/4 for a mecanum wheel. Each wheel turns as it would
        roll without slipping under the base's motion, so the wheels
        spin while the base drives and :meth:`angle_vector` reads their
        angles. Only for the look: the base does not touch the ground,
        so the wheels do not drive it, and what an angle vector sent to
        the interface asks of them is ignored.
    objects : list[skrobot.model.Link]
        Box, Cylinder and Sphere primitives fixed in the world where they
        are now, such as a table. Each needs a unique ``name``.
    movable_objects : list[skrobot.model.Link]
        Primitives free to be pushed and picked up, such as a cube on that
        table. Their mass follows from their volume at the density of water.
    cameras : dict[str, dict] or None
        Camera name -> ``{'link': <link>, 'fovy': <deg>, 'width': <px>,
        'height': <px>, 'frame': 'optical' or 'x_forward'}``. The camera
        looks along the link's +z when ``frame`` is ``'optical'`` (a ROS
        optical frame, the default) and along its +x, +z up, when it is
        ``'x_forward'``. ``fovy`` defaults to 45, the size to 640x480. Read
        with :meth:`get_image` and :meth:`get_depth`.
    renderer : object or None
        What draws camera images: an object with ``rgb(ri, name)`` and
        ``depth(ri, name)``. ``None`` uses ``mujoco.Renderer``, which needs
        an OpenGL context (set ``MUJOCO_GL=egl`` on a machine without a
        display). scikit-robot-web passes one that draws with three.js.
    actuator_kp : float or None
        Proportional gain of every joint's position actuator. ``None``
        (the default) gives each joint its own, from the inertia it moves
        in the starting pose, so that it responds at ``actuator_bandwidth``:
        one gain for all suits a 100 kg robot or a 2 kg one, not both (on
        a 2 kg arm, a gain a heavy robot needs makes the simulation blow
        up).
    actuator_bandwidth : float
        Response of the joints [rad/s] when ``actuator_kp`` is None. The
        actuators are critically damped. Stiff, as real servos are, so a
        held object or a squeezing gripper barely deflects the arm (at 50
        Jedy's elbow gave 0.045 rad under a 27 g cube); much above this the
        smaller joints hit their torque limits and chatter. It must stay
        well under ``1 / timestep`` for the simulation to stay stable.
    gravity_compensation : bool
        Cancel gravity on every body, as a force outside the actuators:
        it does not count against their torque limits. A URDF's effort
        limits often cannot hold its own masses (Fetch's torso needs about
        950 N against an effort of 450 N), so counting it would leave such
        a robot unable to lift itself. Without compensation the arm sags
        under a position gain this soft.
    timestep : float or None
        Physics step [s]. ``None`` keeps the MJCF's value.
    realtime : bool
        Make :meth:`wait_interpolation` take as long in wall-clock time as it
        simulates.
    viewer : bool
        Open MuJoCo's passive viewer (desktop ``mujoco`` only). Implies
        ``realtime``.
    ghost : bool
        In the viewer, draw the robot model you passed in -- the posture
        you plan with -- as a translucent ghost over the simulated robot,
        so the two can be told apart: ``robot.angle_vector(...)`` moves the
        ghost, only ``ri`` moves the robot. It is redrawn as simulation
        time passes. Shown by viewers with ``sync_ghost``, such as
        scikit-robot-web's; MuJoCo's desktop viewer does not draw it yet.
    mujoco : module or None
        The MuJoCo bindings to use. ``None`` imports the ``mujoco`` package;
        a stand-in with the same API can be passed instead.
    converter_kwargs : dict or None
        Extra keyword arguments for :func:`skrobot.urdf.urdf_to_mjcf`.

    Examples
    --------
    >>> import skrobot
    >>> from skrobot.interfaces import MujocoRobotInterface
    >>> robot = skrobot.models.Fetch()
    >>> ri = MujocoRobotInterface(robot)
    >>> robot.reset_pose()
    >>> ri.angle_vector(robot.angle_vector(), 3.0)
    >>> ri.wait_interpolation()
    >>> robot.angle_vector(ri.angle_vector())
    """

    def __init__(self, robot, urdf_path=None, mjcf_path=None,
                 floating_base=False, planar_base=False,
                 base_max_velocity=(0.5, 1.0), wheels=None,
                 objects=(), movable_objects=(), cameras=None,
                 actuator_kp=None, actuator_bandwidth=150.0,
                 gravity_compensation=True,
                 timestep=None, realtime=False, viewer=False,
                 ghost=True, renderer=None, mujoco=None,
                 converter_kwargs=None):
        if floating_base and planar_base:
            raise ValueError(
                'floating_base and planar_base cannot both be set.')
        if wheels and not planar_base:
            raise ValueError('wheels need planar_base=True.')
        if mujoco is None:
            try:
                import mujoco
            except ImportError:
                raise ImportError(
                    'MujocoRobotInterface needs the mujoco package.\n\n'
                    '  $ pip install scikit-robot[mujoco]\n')
        self.mujoco = mujoco
        # Its own copy, as ROSRobotInterfaceBase keeps: what the simulation
        # does must not overwrite the posture the caller is planning with.
        self.robot = copy.deepcopy(robot)
        # The caller's model, only ever read: the ghost draws it.
        self._planning_robot = robot
        self.floating_base = floating_base
        self.planar_base = planar_base
        self.base_max_velocity = base_max_velocity
        self.cameras = {}
        for name, spec in (cameras or {}).items():
            spec = dict(spec)
            spec.setdefault('fovy', 45.0)
            spec.setdefault('width', 640)
            spec.setdefault('height', 480)
            spec.setdefault('frame', 'optical')
            self.cameras[name] = spec

        if mjcf_path is None:
            if urdf_path is None:
                urdf_path = robot.urdf_path
            if urdf_path is None:
                raise ValueError(
                    'robot.urdf_path is None; pass urdf_path or mjcf_path.')
            # With actuator_kp None the gains are set after loading, from
            # the model's inertia; the converter's are placeholders.
            mjcf_path = self._convert(
                urdf_path, floating_base,
                1.0 if actuator_kp is None else actuator_kp,
                converter_kwargs or {})
        if (gravity_compensation or planar_base or objects
                or movable_objects or self.cameras):
            mjcf_path = self._build_scene(
                mjcf_path, objects, movable_objects, gravity_compensation)
        self.mjcf_path = mjcf_path
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        if timestep is not None:
            self.model.opt.timestep = timestep
        # Integrate the actuators' damping implicitly. Gains as stiff as
        # actuator_bandwidth asks for make coupled joints oscillate under
        # the default (explicit) Euler integrator -- MuJoCo recommends this
        # one for position servos.
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        self._map_joints()
        self._map_wheels(wheels or {})
        self.renderer = renderer

        self.frame_callbacks = []
        self.frame_period = 1.0 / 60.0
        self._next_frame_time = 0.0
        self.realtime = realtime or viewer
        self.viewer = None
        self.reset(robot.angle_vector())
        # After reset() has posed everything, so the viewer's first frame
        # shows the robot where it stands.
        if viewer:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.frame_callbacks.append(lambda _ri: self.viewer.sync())
            if ghost and hasattr(self.viewer, 'sync_ghost'):
                self._ghost_data = mujoco.MjData(self.model)
                self._ghost_geoms = self._robot_geoms()
                self.frame_callbacks.append(lambda _ri: self._sync_ghost())
                self.viewer.on_finish.append(self._sync_ghost)

        # In the starting pose, since the inertia a joint moves depends on
        # the pose of the joints beyond it.
        if actuator_kp is None:
            self._tune_gains(
                [j.name for j, commanded in zip(robot.joint_list,
                                                self._commanded)
                 if commanded], actuator_bandwidth)
        if planar_base:
            self._tune_gains(_mujoco_scene.PLANAR_BASE_JOINTS, 20.0)

    @staticmethod
    def _convert(urdf_path, floating_base, actuator_kp, converter_kwargs):
        from skrobot.urdf import urdf_to_mjcf

        key = (os.path.abspath(urdf_path), os.path.getmtime(urdf_path),
               floating_base, actuator_kp,
               repr(sorted(converter_kwargs.items())))
        out_path = _converted.get(key)
        if out_path is not None and os.path.exists(out_path):
            return out_path
        out_dir = tempfile.mkdtemp(prefix='skrobot_mujoco_')
        out_path = os.path.join(out_dir, 'robot.xml')
        urdf_to_mjcf(urdf_path, out_path, floating_base=floating_base,
                     actuator_kp=actuator_kp, **converter_kwargs)
        _converted[key] = out_path
        return out_path

    def _build_scene(self, mjcf_path, objects, movable_objects,
                     gravity_compensation):
        """Write the scene MJCF beside ``mjcf_path`` and return its path.

        Beside it, not in a temporary directory, so the relative
        ``meshdir`` still finds the meshes.
        """
        tree = ET.parse(mjcf_path)
        root = tree.getroot()
        # Before the objects go in: those must fall.
        if gravity_compensation:
            _mujoco_scene.compensate_gravity(root)
        if self.planar_base:
            _mujoco_scene.add_planar_base(root, self.robot.root_link.name)
        _mujoco_scene.add_objects(root, objects, movable=False)
        _mujoco_scene.add_objects(root, movable_objects, movable=True)
        _mujoco_scene.add_cameras(root, self.cameras)
        stem, ext = os.path.splitext(mjcf_path)
        out_path = stem + '_scene' + ext
        tree.write(out_path)
        return out_path

    def _tune_gains(self, joint_names, bandwidth):
        """Critically damped position actuators at ``bandwidth`` [rad/s].

        Each gain scales with the inertia its joint moves, read from the
        diagonal of the mass matrix in the current pose, so a 100 kg
        mobile manipulator and a 2 kg one track equally well.

        Parameters
        ----------
        joint_names : list[str]
            Joints whose ``<joint>_act`` actuator to set.
        bandwidth : float
            Natural frequency [rad/s].
        """
        mj = self.mujoco
        mj.mj_forward(self.model, self.data)
        for name in joint_names:
            jid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name)
            aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR,
                                name + '_act')
            dof = self.model.jnt_dofadr[jid]
            # e_dof^T M e_dof through mj_mulM, which every MuJoCo version
            # has, rather than indexing the mass matrix, whose storage
            # changed (MjData.qM is gone in 3.17).
            unit = np.zeros(self.model.nv)
            unit[dof] = 1.0
            column = np.zeros(self.model.nv)
            mj.mj_mulM(self.model, self.data, column, unit)
            inertia = column[dof]
            kp = inertia * bandwidth ** 2
            kv = 2.0 * inertia * bandwidth
            # A position actuator is gain*ctrl + bias0 + bias1*q + bias2*qdot.
            self.model.actuator_gainprm[aid, 0] = kp
            self.model.actuator_biasprm[aid, 1] = -kp
            self.model.actuator_biasprm[aid, 2] = -kv

    def _map_joints(self):
        mj = self.mujoco
        joint_type = mj.mjtObj.mjOBJ_JOINT
        actuator_type = mj.mjtObj.mjOBJ_ACTUATOR
        qpos_index = []
        qvel_index = []
        actuator_index = []
        for joint in self.robot.joint_list:
            jid = mj.mj_name2id(self.model, joint_type, joint.name)
            if jid < 0:
                raise ValueError(
                    'joint {} of the robot model is not in the MJCF {}'.format(
                        joint.name, self.mjcf_path))
            qpos_index.append(self.model.jnt_qposadr[jid])
            qvel_index.append(self.model.jnt_dofadr[jid])
            # A mimic joint has no actuator: an equality constraint drives it.
            actuator_index.append(
                mj.mj_name2id(self.model, actuator_type, joint.name + '_act'))
        self._qpos_index = np.array(qpos_index, dtype=np.int64)
        self._qvel_index = np.array(qvel_index, dtype=np.int64)
        actuator_index = np.array(actuator_index, dtype=np.int64)
        self._commanded = actuator_index >= 0
        self._actuator_index = actuator_index[self._commanded]

        self._free_qpos = None
        if self.floating_base:
            free = [i for i in range(self.model.njnt)
                    if self.model.jnt_type[i] == mj.mjtJoint.mjJNT_FREE]
            if free:
                self._free_qpos = self.model.jnt_qposadr[free[0]]

        self._base_qpos = None
        if self.planar_base:
            jids = [mj.mj_name2id(self.model, joint_type, name)
                    for name in _mujoco_scene.PLANAR_BASE_JOINTS]
            self._base_qpos = np.array(
                [self.model.jnt_qposadr[j] for j in jids], dtype=np.int64)
            self._base_actuator = np.array(
                [mj.mj_name2id(self.model, actuator_type, name + '_act')
                 for name in _mujoco_scene.PLANAR_BASE_JOINTS],
                dtype=np.int64)

    def _map_wheels(self, wheels):
        """Where each wheel sits on the base, for :meth:`_turn_wheels`."""
        names = [j.name for j in self.robot.joint_list]
        base = self.robot.copy_worldcoords()
        self._wheels = []
        for name, (radius, roller_angle) in wheels.items():
            if name not in names:
                raise ValueError('wheel {} is not a joint of the robot '
                                 'model'.format(name))
            k = names.index(name)
            joint = self.robot.joint_list[k]
            child = joint.child_link
            # In the base frame, on the floor plane.
            position = base.inverse_transform_vector(child.worldpos())[:2]
            axle = base.worldrot().T.dot(
                child.worldrot().dot(np.asarray(joint.axis, np.float64)))
            axle = axle[:2] / np.linalg.norm(axle[:2])
            # The way the wheel's centre goes when it turns about +axle.
            forward = np.array([axle[1], -axle[0]])
            self._wheels.append(
                (k, position, axle, forward, radius, np.tan(roller_angle)))
        # Angle-vector index of each wheel, in self._wheels' order.
        self._wheel_index = np.array([w[0] for w in self._wheels],
                                     dtype=np.int64)
        if self._wheels:
            joint_type = self.mujoco.mjtObj.mjOBJ_JOINT
            jids = [self.mujoco.mj_name2id(self.model, joint_type, name)
                    for name in _mujoco_scene.PLANAR_BASE_JOINTS]
            self._base_qvel = np.array(
                [self.model.jnt_dofadr[j] for j in jids], dtype=np.int64)
        self._wheel_angles = np.zeros(len(self._wheels))

    def _turn_wheels(self, dt):
        """Advance the wheels by ``dt`` of the base's current motion."""
        vx, vy, wz = self.data.qvel[self._base_qvel]
        yaw = self.data.qpos[self._base_qpos[2]]
        c, s = np.cos(yaw), np.sin(yaw)
        velocity = np.array([c * vx + s * vy, -s * vx + c * vy])
        for i, (_, position, axle, forward, radius, tan_roller) \
                in enumerate(self._wheels):
            contact = velocity + wz * np.array([-position[1], position[0]])
            self._wheel_angles[i] += dt * (
                contact.dot(forward) + tan_roller * contact.dot(axle)) \
                / radius

    def reset(self, av=None):
        """Put the simulated robot at ``av`` at rest and drop any trajectory.

        Movable objects go back to where they were placed, and a planar
        base to where the robot model stands.

        Parameters
        ----------
        av : numpy.ndarray or None
            Angle vector to start from. ``None`` uses the robot model's.
        """
        if av is None:
            av = self.robot.angle_vector()
        av = np.asarray(av, dtype=np.float64)
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self._qpos_index] = av
        self._segments = []
        self._hold = av.copy()
        self._wheel_angles = av[self._wheel_index].copy()
        self._write_ctrl(av)
        if self._base_qpos is not None:
            pose = _planar_pose(self.robot.copy_worldcoords())
            self.data.qpos[self._base_qpos] = pose
            self._base_segments = []
            self._base_hold = pose
            self.data.ctrl[self._base_actuator] = pose
        self.mujoco.mj_forward(self.model, self.data)
        self._next_frame_time = self.data.time

    @property
    def time(self):
        """Simulation time [s]."""
        return self.data.time

    def _write_ctrl(self, av):
        if self._wheels:
            av = av.copy()
            av[self._wheel_index] = self._wheel_angles
        self.data.ctrl[self._actuator_index] = av[self._commanded]

    def _target(self, t):
        """Commanded angle vector at simulation time ``t``."""
        for t0, t1, p0, p1, v0, v1 in self._segments:
            if t <= t1:
                if t < t0:
                    return p0.copy()
                return _hermite(t, t0, t1, p0, p1, v0, v1)
        return self._hold.copy()

    def step(self, n=1):
        """Advance the simulation ``n`` physics steps.

        Parameters
        ----------
        n : int
            Number of ``mj_step`` calls.
        """
        mj = self.mujoco
        start_wall = _time.time()
        start_sim = self.data.time
        for _ in range(n):
            if self._wheels:
                self._turn_wheels(self.model.opt.timestep)
            self._write_ctrl(self._target(self.data.time))
            if self._base_qpos is not None:
                self.data.ctrl[self._base_actuator] = self._base_target(
                    self.data.time)
            mj.mj_step(self.model, self.data)
            if self.data.time >= self._next_frame_time:
                self._next_frame_time = self.data.time + self.frame_period
                for callback in self.frame_callbacks:
                    callback(self)
                if self.realtime:
                    lag = (self.data.time - start_sim) \
                        - (_time.time() - start_wall)
                    if lag > 0:
                        _time.sleep(lag)

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
        now = self.data.time
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
        return bool(self._segments) and self.data.time < self._segments[-1][1]

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
                end = min(end, self.data.time + timeout)
            n = int(np.ceil((end - self.data.time) / self.model.opt.timestep))
            self.step(max(n, 0))
            # Summing timestep n times can land a hair short of ``end``.
            while self.data.time < end:
                self.step(1)
        return [self.is_interpolating()]

    def cancel_angle_vector(self):
        """Stop where the commanded trajectory is now."""
        self._hold = self._target(self.data.time)
        self._segments = []

    def sync(self):
        """Copy the simulated joints (and base) into ``ri.robot``.

        Returns
        -------
        numpy.ndarray
            The simulated angle vector as a robot model can hold it: a joint
            the simulation pressed a little past its limit reads as at the
            limit, so ``robot.angle_vector(ri.angle_vector())`` goes through
            without joint-limit warnings.
        """
        av = self.data.qpos[self._qpos_index].copy()
        lower = np.array([j.min_angle for j in self.robot.joint_list])
        upper = np.array([j.max_angle for j in self.robot.joint_list])
        self.robot.angle_vector(np.clip(av, lower, upper))
        av = self.robot.angle_vector()
        if self._free_qpos is not None:
            q = self.data.qpos[self._free_qpos:self._free_qpos + 7]
            self.robot.newcoords(Coordinates(
                pos=q[:3], rot=quaternion2matrix(q[3:7])))
        if self._base_qpos is not None:
            self.robot.newcoords(
                _planar_coords(self.data.qpos[self._base_qpos]))
        return av

    def potentio_vector(self):
        """Simulated angle vector (see :meth:`sync`)."""
        return self.sync()

    def joint_velocity_vector(self):
        """Simulated joint velocities, in ``robot.joint_list`` order."""
        return self.data.qvel[self._qvel_index].copy()

    def joint_effort_vector(self):
        """Actuator torque [N m] or force [N] on each joint, in
        ``robot.joint_list`` order, gravity compensation included."""
        return self.data.qfrc_actuator[self._qvel_index].copy()

    def simulate(self, duration):
        """Let ``duration`` seconds of simulation pass.

        Parameters
        ----------
        duration : float
            Simulation time to run [s].
        """
        self.step(int(round(duration / self.model.opt.timestep)))

    # --- base ------------------------------------------------------------

    def _require_base(self):
        if self._base_qpos is None:
            raise RuntimeError(
                'the base is welded to the world; construct the interface '
                'with planar_base=True to move it.')

    def _base_target(self, t):
        """Commanded base ``(x, y, yaw)`` at simulation time ``t``."""
        for t0, t1, path in self._base_segments:
            if t <= t1:
                return path(max(t, t0))
        return self._base_hold.copy()

    def _send_base(self, duration, path, wait):
        t0 = self.data.time
        t1 = t0 + duration
        self._base_segments = [(t0, t1, path)]
        self._base_hold = path(t1)
        if wait:
            self.wait_base()
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
            True.
        """
        self._require_base()
        start = self._base_target(self.data.time)
        goal = _planar_pose(coords)
        # Turn the short way round.
        goal[2] = start[2] + np.arctan2(np.sin(goal[2] - start[2]),
                                        np.cos(goal[2] - start[2]))
        duration = self._base_duration(start, goal)
        zero = np.zeros(3)
        t0 = self.data.time

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
            True.
        """
        self._require_base()
        start = self._base_target(self.data.time)
        goal = _compose_planar(start, (x, y, yaw))
        duration = self._base_duration(start, goal)
        zero = np.zeros(3)
        t0 = self.data.time

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
            True.
        """
        self._require_base()
        start = self._base_target(self.data.time)
        t0 = self.data.time

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
        self._base_hold = self._base_target(self.data.time)
        self._base_segments = []

    def is_moving_base(self):
        """Whether a base motion has time left to run."""
        return (self._base_qpos is not None and bool(self._base_segments)
                and self.data.time < self._base_segments[-1][1])

    def wait_base(self):
        """Simulate until the base has arrived where it was sent last.

        Your robot model stays where it was. To plan toward world
        coordinates, such as an object's, from where the base now is, move
        it there: ``robot.newcoords(ri.odom)``.
        """
        self._require_base()
        if self._base_segments:
            end = self._base_segments[-1][1]
            while self.data.time < end:
                self.step(max(1, int(np.ceil(
                    (end - self.data.time) / self.model.opt.timestep))))
        # The base trails its command; finish when it has arrived, as
        # move_base does, rather than when the command has.
        settle_end = self.data.time + 2.0
        while True:
            error = self.data.qpos[self._base_qpos] - self._base_hold
            if (np.linalg.norm(error[:2]) < 1e-3
                    and abs(error[2]) < np.deg2rad(0.1)):
                break
            if self.data.time >= settle_end:
                logger.warning(
                    'base did not reach its goal within 2 s; still %s off '
                    '(x [m], y [m], yaw [rad])', np.round(error, 4))
                break
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
        return _planar_coords(self.data.qpos[self._base_qpos])

    # --- sensors ---------------------------------------------------------

    def _body_id(self, name):
        bid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError('no link or object named {}'.format(name))
        return bid

    def _body_coords(self, bid):
        return Coordinates(pos=self.data.xpos[bid].copy(),
                           rot=self.data.xmat[bid].reshape(3, 3).copy())

    def object_coords(self, name):
        """Where an object (or a robot link) is, in the world frame.

        Parameters
        ----------
        name : str
            Name of a primitive passed as ``objects``/``movable_objects``,
            or of a link.

        Returns
        -------
        skrobot.coordinates.Coordinates
        """
        return self._body_coords(self._body_id(name))

    def contacts(self, name):
        """Names of the links and objects touching ``name`` now.

        Parameters
        ----------
        name : str
            Link or object name.

        Returns
        -------
        list[str]
        """
        bid = self._body_id(name)
        mj = self.mujoco
        touching = set()
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            b1 = self.model.geom_bodyid[contact.geom1]
            b2 = self.model.geom_bodyid[contact.geom2]
            if b1 == bid:
                other = b2
            elif b2 == bid:
                other = b1
            else:
                continue
            touching.add(mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY,
                                       other) or 'world')
        return sorted(touching)

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

    def camera_coords(self, name):
        """Optical frame of a camera in the world frame (+z forward, +x
        right, +y down), whichever way it was mounted."""
        self._camera(name)
        cid = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, name)
        # MuJoCo's camera frame (-z forward, +y up) turned half a turn
        # about its x axis.
        rot = self.data.cam_xmat[cid].reshape(3, 3) * np.array([1, -1, -1])
        return Coordinates(pos=self.data.cam_xpos[cid].copy(), rot=rot)

    def _renderer(self):
        if self.renderer is None:
            self.renderer = _MujocoRenderer()
        return self.renderer

    def get_image(self, name):
        """RGB image from a camera.

        Parameters
        ----------
        name : str
            Camera name.

        Returns
        -------
        numpy.ndarray
            ``(height, width, 3)`` uint8, first row at the top.
        """
        self._camera(name)
        return self._renderer().rgb(self, name)

    def get_depth(self, name):
        """Depth image from a camera.

        Parameters
        ----------
        name : str
            Camera name.

        Returns
        -------
        numpy.ndarray
            ``(height, width)`` float32 distance along the optical axis [m].
        """
        self._camera(name)
        return self._renderer().depth(self, name)

    def _robot_geoms(self):
        """Ids of the robot's geoms (not the floor's or the objects')."""
        mj = self.mujoco
        bodies = set()
        for link in self.robot.link_list:
            bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, link.name)
            if bid >= 0:
                bodies.add(bid)
        body_of = self.model.geom_bodyid
        return [int(g) for g in range(self.model.ngeom)
                if body_of[g] in bodies]

    def _sync_ghost(self):
        """Pose the robot model passed in, by kinematics alone, and draw it."""
        ghost = self._ghost_data
        robot = self._planning_robot
        ghost.qpos[self._qpos_index] = robot.angle_vector()
        coords = robot.copy_worldcoords()
        if self._base_qpos is not None:
            ghost.qpos[self._base_qpos] = _planar_pose(coords)
        if self._free_qpos is not None:
            q = np.concatenate([coords.translation,
                                matrix2quaternion(coords.rotation)])
            ghost.qpos[self._free_qpos:self._free_qpos + 7] = q
        self.mujoco.mj_kinematics(self.model, ghost)
        self.viewer.sync_ghost(ghost, self._ghost_geoms)

    def close(self):
        """Close the viewer, if one is open."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


class _MujocoRenderer(object):
    """Camera images drawn by ``mujoco.Renderer`` (desktop ``mujoco``)."""

    def __init__(self):
        self._renderers = {}

    def _get(self, ri, name):
        spec = ri.cameras[name]
        key = (spec['width'], spec['height'])
        if key not in self._renderers:
            self._renderers[key] = ri.mujoco.Renderer(
                ri.model, height=spec['height'], width=spec['width'])
        renderer = self._renderers[key]
        renderer.update_scene(ri.data, camera=name)
        return renderer

    def rgb(self, ri, name):
        renderer = self._get(ri, name)
        renderer.disable_depth_rendering()
        return renderer.render().copy()

    def depth(self, ri, name):
        renderer = self._get(ri, name)
        renderer.enable_depth_rendering()
        try:
            return renderer.render().astype(np.float32)
        finally:
            renderer.disable_depth_rendering()
