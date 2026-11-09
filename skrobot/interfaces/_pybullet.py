import importlib
import time as _time

import numpy as np

from skrobot.coordinates import Coordinates
from skrobot.coordinates import matrix2quaternion
from skrobot.coordinates import quaternion2rpy
from skrobot.coordinates.math import wxyz2xyzw
from skrobot.coordinates.math import xyzw2wxyz
from skrobot.interfaces.base import DEFAULT_CONTROLLER
from skrobot.interfaces.base import RobotInterfaceBase
from skrobot.interpolation import hermite_trajectory


_available = False
_import_checked = False
p = None


def _check_available(silent=False):
    global _available
    global _import_checked
    global p
    if not _import_checked:
        try:
            p = importlib.import_module('pybullet')
        except ImportError as e:
            if not silent:
                if 'numpy.core.multiarray' in str(e):
                    print('Failed to import pybullet due to a NumPy compatibility issue '  # NOQA
                          + '(numpy.core.multiarray failed to import). '
                          + 'This may be caused by an incompatible NumPy version.\n'  # NOQA
                          + 'Please try installing a different NumPy version, e.g., 1.20.3:\n\n'  # NOQA
                          + '  $ pip install numpy==1.20.3\n'
                          + 'Then reinstall pybullet:\n'
                          + '  $ pip install pybullet\n')
                else:
                    _available = False
                    print('pybullet is not installed on your environment, '
                          'so nothing will be drawn at this time. '
                          'Please install pybullet.\n\n'
                          '  $ pip install pybullet\n')
            else:
                _available = False
        except TypeError:
            _available = False
            if not silent:
                print('Unexpected TypeError occurred while importing pybullet.')
        finally:
            _import_checked = True
            if p is not None:
                _available = True
    if not _available and not silent:
        print('pybullet is unavailable. '
              'No drawing will occur until the issue is resolved.')
    return _available


class PybulletRobotInterface(Coordinates, RobotInterfaceBase):

    """Pybullet Interface Class

    It is a :class:`skrobot.interfaces.base.RobotInterfaceBase` as well as
    the robot's base pose (:class:`skrobot.coordinates.Coordinates`). The
    calls it had before behave as before:

    - ``angle_vector(av)`` / ``angle_vector(av, realtime_simulation=...)``
      sets the joint motor targets at once (and, unless realtime
      simulation is on, puts the joints there with ``resetJointState``);
      ``angle_vector()`` sends the robot model's angle vector.
    - ``wait_interpolation(thresh, timeout)`` steps the simulation until
      every joint is slower than ``thresh`` and returns True, or False after
      ``timeout`` seconds of wall-clock time.
    - ``sync()`` copies the simulated joints into the robot model.

    Timed motions are what the base class adds: ``angle_vector(av, 2.0)``
    (any ``time`` that is a number, not a bool), a ``{joint_name: angle}``
    goal, or any of ``controller_type``, ``start_time``, ``time_scale``,
    ``velocities`` sends a trajectory that :meth:`step` and
    :meth:`wait_interpolation` carry out in simulation time. The
    simulation only advances in :meth:`step`, which writes the trajectory's
    motor targets and calls ``pybullet.stepSimulation``; a program that
    calls ``stepSimulation`` itself does not move a timed trajectory.
    :meth:`measured_angle_vector`, :meth:`measured_joint_angles` and
    :meth:`outside_limits` report the simulated joints without the robot
    model's clamping.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        robot model
    urdf_path : None or str
        urdf path. If this value is `None`,
        get `urdf_path` from `robot.urdf_path`.
    use_fixed_base : bool
        If this value is `True`, robot in pybullet simulator will be fixed.
    connect : int
        pybullet's connection mode. If you have already connected
        to pybullet physics server, specify the server id.
        The default value is 1 (pybullet.GUI).

    Examples
    --------
    >>> from skrobot.models import PR2
    >>> from skrobot.interfaces import PybulletRobotInterface
    >>> robot_model = PR2()
    >>> interface = PybulletRobotInterface(robot_model)

    If you have already connected to pybullet physics server

    >>> import pybullet
    >>> client_id = pybullet.connect(pybullet.GUI)
    >>> robot_model = PR2()
    >>> interface = PybulletRobotInterface(robot_model, connect=client_id)

    A timed motion, carried out in simulation time

    >>> interface.angle_vector({'r_shoulder_pan_joint': 0.5}, 2.0)
    >>> interface.wait_interpolation()
    """

    def __init__(self, robot, urdf_path=None, use_fixed_base=False,
                 connect=1, *args, **kwargs):
        if _check_available() is False:
            raise ImportError('pybullet is not available.')
        Coordinates.__init__(self, *args, **kwargs)
        if urdf_path is None:
            if robot.urdf_path is not None:
                urdf_path = robot.urdf_path
            else:
                raise ValueError('urdf_path should be given.')
        self.robot = robot
        if connect == 2:
            p.connect(connect)
        elif connect == 1:
            try:
                p.connect(connect)
            except Exception as e:
                print(e)
        self.robot_id = p.loadURDF(urdf_path, self.translation,
                                   self.quaternion_xyzw,
                                   useFixedBase=use_fixed_base)

        self.load_bullet()
        self.realtime_simulation = False
        # Timed trajectories: simulation time [s] advanced by step(), and
        # for each joint still moving ``name -> (trajectory, column, end
        # time)``, one hermite_trajectory covering all joints of a send.
        self.sim_time = 0.0
        self._motions = {}
        RobotInterfaceBase.__init__(self, robot, controllers={
            DEFAULT_CONTROLLER: [
                j.name for j in robot.joint_list
                if j.joint_dof == 1
                and j.name in self.joint_name_to_joint_id]})

    @staticmethod
    def available():
        """Check Pybullet is available.

        Returns
        -------
        _available : bool
            If `False`, pybullet is not available.
        """
        _check_available()
        return _available

    @property
    def pose(self):
        """Getter of Pose in pybullet physics simulator.

        Wrapper of pybullet.getBasePositionAndOrientation.

        Returns
        -------
        pose : skrobot.coordinates.Coordinates
            pose of this robot in the physics simulator.
        """
        pos, q_xyzw = p.getBasePositionAndOrientation(
            self.robot_id)
        q_wxyz = xyzw2wxyz(q_xyzw)
        return Coordinates(pos=pos, rot=q_wxyz)

    def _reset_position_and_orientation(self):
        """Reset base position and orientation.

        This function is wrapper of pybullet.resetBasePositionAndOrientation.
        """
        p.resetBasePositionAndOrientation(self.robot_id, self.translation,
                                          self.quaternion_xyzw)

    def translate(self, vec, wrt='local'):
        """Translate robot in simulator.

        For more detail,
        please see docs of skrobot.coordinates.Coordinates.translate.
        The difference between the translate, this function internally
        call pybullet.resetBasePositionAndOrientation.

        Parameters
        ----------
        vec : list or np.ndarray
            shape of (3,) translation vector. unit is [m] order.
        wrt : string or Coordinates (optional)
            translate with respect to wrt.
        """
        super(PybulletRobotInterface, self).translate(vec, wrt)
        self._reset_position_and_orientation()
        return self

    def rotate(self, theta, axis=None, wrt='local'):
        """Rotate this robot by given theta and axis.

        For more detail,
        please see docs of skrobot.coordinates.Coordinates.rotate.
        The difference between the rotate, this function internally
        call pybullet.resetBasePositionAndOrientation.

        Parameters
        ----------
        theta : float
            radian
        wrt : string or skrobot.coordinates.Coordinates
        """
        super(PybulletRobotInterface, self).rotate(theta, axis, wrt)
        self._reset_position_and_orientation()
        return self

    def transform(self, c, wrt='local'):
        """Transform this coordinate by coords based on wrt

        For more detail,
        please see docs of skrobot.coordinates.Coordinates.transform.
        The difference between the transform, this function internally
        call pybullet.resetBasePositionAndOrientation.

        Parameters
        ----------
        c : skrobot.coordinates.Coordinates
            coordinate
        wrt : string or skrobot.coordinates.Coordinates
            If wrt is 'local' or self, multiply c from the right.
            If wrt is 'world' or 'parent' or self.parent,
            transform c with respect to worldcoord.
            If wrt is Coordinates, transform c with respect to c.
        """
        super(PybulletRobotInterface, self).transform(c, wrt)
        self._reset_position_and_orientation()
        return self

    def newcoords(self, c, pos=None, **kwargs):
        """Update of position and orientation.

        Keyword arguments (``check_validity``, ``relative_coords``) are
        passed to :meth:`skrobot.coordinates.Coordinates.newcoords`, which
        ``translate``, ``rotate`` and ``transform`` call with them.
        """
        super(PybulletRobotInterface, self).newcoords(c, pos, **kwargs)
        self._reset_position_and_orientation()
        return self

    def load_bullet(self):
        """Load bullet configurations.

        This function internally called.
        """
        joint_num = p.getNumJoints(self.robot_id)
        joint_ids = [None] * joint_num
        joint_name_to_joint_id = {}
        for i in range(len(joint_ids)):
            joint_name = p.getJointInfo(self.robot_id, i)[1]
            try:
                idx = self.robot.joint_names.index(joint_name.decode('utf-8'))
            except ValueError:
                continue
            if idx != -1:
                joint_ids[idx] = i
                joint_name_to_joint_id[joint_name.decode('utf-8')] = i
            else:
                joint_name_to_joint_id[joint_name.decode('utf-8')] = idx
        self.joint_ids = joint_ids
        self.joint_name_to_joint_id = joint_name_to_joint_id

        self.max_velocity = 1.0
        self.position_gain = 0.1
        self.target_velocity = 0.0
        self.velocity_gain = 0.1

    def angle_vector(self, av=None, time=None, controller_type=None,
                     start_time=None, time_scale=None, velocities=None,
                     realtime_simulation=None, angle_vector=None):
        """Send an angle vector to pybullet's physics engine.

        Called as before -- ``angle_vector(av)``,
        ``angle_vector(av, realtime_simulation=True)`` or the old positional
        ``angle_vector(av, True)`` -- it sets the motor targets at once.
        Given a ``time`` (a number), a dict, or any of the timed-motion
        arguments, it sends a timed trajectory instead, as
        :meth:`skrobot.interfaces.base.RobotInterfaceBase.angle_vector`
        does; :meth:`step` and :meth:`wait_interpolation` carry it out.

        Parameters
        ----------
        av : None, numpy.ndarray or dict
            angle vector, or ``{joint_name: angle}`` for a timed motion of
            those joints. If `None`, send self.robot.angle_vector().
        time : None, float or bool
            Time to the goal [s] of a timed motion. A bool is taken as
            ``realtime_simulation``, the old second positional argument.
        controller_type, start_time, time_scale, velocities
            Timed motion only; see
            :meth:`skrobot.interfaces.base.RobotInterfaceBase.angle_vector`.
            ``start_time`` defaults to 0 and ``time_scale`` to 5.
        realtime_simulation : None or bool
            If this value is `True`, send angle_vector
            by pybullet.setJointMotorControl2 only, without resetting the
            joints. The setting is kept for later calls.
        angle_vector : None or numpy.ndarray
            Old name of ``av``.

        Returns
        -------
        angle_vector : numpy.ndarray or dict
            return sent angle vector.
        """
        if angle_vector is not None:
            if av is not None:
                raise TypeError('give av or angle_vector, not both')
            av = angle_vector
        if isinstance(time, bool):
            if realtime_simulation is not None:
                raise TypeError('realtime_simulation given twice')
            realtime_simulation = time
            time = None
        if realtime_simulation is not None and isinstance(
                realtime_simulation, bool):
            self.realtime_simulation = realtime_simulation

        timed = (time is not None or isinstance(av, dict)
                 or controller_type is not None or start_time is not None
                 or time_scale is not None or velocities is not None)
        if timed:
            if self.robot_id is None:
                raise RuntimeError('no robot is loaded in pybullet')
            if av is None:
                av = self.robot.angle_vector()
            return RobotInterfaceBase.angle_vector(
                self, av, time, controller_type=controller_type,
                start_time=0.0 if start_time is None else start_time,
                time_scale=5.0 if time_scale is None else time_scale,
                velocities=velocities)

        if self.robot_id is None:
            return self.robot.angle_vector()
        if av is None:
            av = self.robot.angle_vector()

        for i, (joint, angle) in enumerate(
                zip(self.robot.joint_list, av)):
            idx = self.joint_name_to_joint_id[joint.name]

            joint = self.robot.joint_list[i]
            # A direct command replaces a timed motion of this joint.
            self._motions.pop(joint.name, None)

            if self.realtime_simulation is False:
                p.resetJointState(self.robot_id, idx, angle)

            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                    jointIndex=idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=angle,
                                    targetVelocity=self.target_velocity,
                                    force=joint.max_joint_torque,
                                    positionGain=self.position_gain,
                                    velocityGain=self.velocity_gain,
                                    maxVelocity=self.max_velocity)

        return av

    def wait_interpolation(self, thresh=0.05, timeout=60.0):
        """Wait robot movement.

        This function usually called after self.angle_vector.
        Wait while a timed motion is running and while the robot joints
        are moving, or until time of timeout.
        This function steps the simulation (see :meth:`step`).

        Parameters
        ----------
        thresh : float
            velocity threshold for detecting movement stop.
        timeout : float
            maximum time of timeout (wall-clock seconds).

        Returns
        -------
        bool
            True when the robot stopped, False on timeout.
        """
        start = _time.time()
        while True:
            self.step()
            wait = bool(self._motions)
            if not wait:
                for idx in self.joint_ids:
                    if idx is None:
                        continue
                    _, velocity, _, _ = p.getJointState(self.robot_id,
                                                        idx)
                    if abs(velocity) > thresh:
                        wait = True
            if wait is False:
                break
            if _time.time() - start > timeout:
                return False
        self.update_robot_state()
        return True

    def sync(self):
        """Synchronize pybullet pose to robot_model.

        """
        if self.robot_id is None:
            return self.angle_vector()

        for idx, joint in zip(self.joint_ids, self.robot.joint_list):
            if idx is None:
                continue
            joint_state = p.getJointState(self.robot_id,
                                          idx)
            joint.joint_angle(joint_state[0])
        pos, orientation_xyzw = p.getBasePositionAndOrientation(self.robot_id)
        rpy, _ = quaternion2rpy(xyzw2wxyz(orientation_xyzw))
        self.robot.root_link.newcoords(np.array([rpy[0], rpy[1], rpy[2]]),
                                       pos=pos)
        return self.angle_vector()

    # ------------------------------------------------------------------
    # Timed motions (RobotInterfaceBase backend)
    # ------------------------------------------------------------------

    def _timestep(self):
        return p.getPhysicsEngineParameters()['fixedTimeStep']

    def _motor_target(self, name, position):
        joint = getattr(self.robot, name)
        kwargs = {}
        # The timed trajectory sets the pace; max_velocity (the cap of a
        # direct command) is only raised to the joint's own limit.
        velocity_limit = float(joint.max_joint_velocity)
        if np.isfinite(velocity_limit):
            kwargs['maxVelocity'] = max(self.max_velocity, velocity_limit)
        p.setJointMotorControl2(bodyIndex=self.robot_id,
                                jointIndex=self.joint_name_to_joint_id[name],
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=position,
                                targetVelocity=self.target_velocity,
                                force=joint.max_joint_torque,
                                positionGain=self.position_gain,
                                velocityGain=self.velocity_gain,
                                **kwargs)

    def _trajectory_target(self, name, t):
        trajectory, column, _ = self._motions[name]
        return float(trajectory(t)[column])

    def step(self, n=1):
        """Advance the simulation ``n`` steps, following timed motions.

        Before each ``pybullet.stepSimulation`` the motor target of every
        joint with a timed motion is set to where the trajectory is at
        :attr:`sim_time`.

        Parameters
        ----------
        n : int
            Number of simulation steps.
        """
        dt = self._timestep()
        for _ in range(n):
            for name in list(self._motions):
                self._motor_target(
                    name, self._trajectory_target(name, self.sim_time))
                if self.sim_time >= self._motions[name][2]:
                    del self._motions[name]
            p.stepSimulation()
            self.sim_time += dt

    def _send_trajectory(self, controller_type, joint_names, positions,
                         velocities, times, start_time):
        state = self._read_joint_state()
        t_start = self.sim_time + start_time
        # The motion starts at rest from where the joints are now.
        knot_times = np.concatenate([[t_start], t_start + np.asarray(times)])
        trajectory = hermite_trajectory(
            knot_times,
            np.vstack([[state[name] for name in joint_names], positions]),
            np.vstack([np.zeros(len(joint_names)), velocities]))
        for j, name in enumerate(joint_names):
            self._motions[name] = (trajectory, j, knot_times[-1])

    def _read_joint_state(self):
        state = {}
        for name in self._controllers[DEFAULT_CONTROLLER]:
            idx = self.joint_name_to_joint_id[name]
            state[name] = float(p.getJointState(self.robot_id, idx)[0])
        return state

    def _is_interpolating(self, controller_type):
        return any(name in self._motions
                   for name in self._controllers[controller_type])

    def _cancel(self, controller_type):
        state = self._read_joint_state()
        for name in self._controllers[controller_type]:
            if self._motions.pop(name, None) is not None:
                self._motor_target(name, state[name])

    def _wait_interpolation(self, controller_type, timeout):
        dt = self._timestep()
        end = self.sim_time + timeout if timeout is not None else None
        while self._is_interpolating(controller_type):
            if end is not None and self.sim_time + 0.5 * dt >= end:
                return
            self.step()


remove_user_item_indices = []
remove_body_indices = []


def draw(c,
         line_width=4,
         line_length=0.3,
         parent_link_index=0,
         color=[1, 1, 1, 1],
         radius=0.03,
         text=''):
    global remove_user_item_indices
    global remove_body_indices
    _check_available()

    if isinstance(c, np.ndarray):
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE,
                                              rgbaColor=color,
                                              radius=radius)
        idx = p.createMultiBody(baseVisualShapeIndex=visual_shape_id,
                                basePosition=c,
                                useMaximalCoordinates=True)
        remove_body_indices.append(idx)
        return
    coord = c.copy_worldcoords()
    orientation_xyzw = wxyz2xyzw(matrix2quaternion(coord.worldrot()))
    create_pose_marker(c.worldpos(),
                       orientation_xyzw,
                       text=text,
                       lineWidth=line_width,
                       lineLength=line_length,
                       parentLinkIndex=parent_link_index)


def flush():
    global remove_user_item_indices
    global remove_body_indices
    _check_available()
    for idx in remove_user_item_indices:
        p.removeUserDebugItem(idx)
    for idx in remove_body_indices:
        p.removeBody(idx)
    remove_user_item_indices = []


def create_pose_marker(position=np.array([0, 0, 0]),
                       orientation=np.array([0, 0, 0, 1]),
                       text='',
                       xColor=np.array([1, 0, 0]),
                       yColor=np.array([0, 1, 0]),
                       zColor=np.array([0, 0, 1]),
                       textColor=np.array([0, 0, 0]),
                       lineLength=0.1,
                       lineWidth=1,
                       textSize=1,
                       textPosition=np.array([0, 0, 0.1]),
                       textOrientation=None,
                       lifeTime=0,
                       parentObjectUniqueId=-1,
                       parentLinkIndex=-1):
    """Create a pose marker

    Create a pose marker that identifies a position and orientation in space
    with 3 colored lines.

    """
    global remove_user_item_indices
    _check_available()
    pts = np.array([[0, 0, 0], [lineLength, 0, 0], [
                   0, lineLength, 0], [0, 0, lineLength]])
    rotIdentity = np.array([0, 0, 0, 1])
    po, _ = p.multiplyTransforms(position, orientation, pts[0, :], rotIdentity)
    px, _ = p.multiplyTransforms(position, orientation, pts[1, :], rotIdentity)
    py, _ = p.multiplyTransforms(position, orientation, pts[2, :], rotIdentity)
    pz, _ = p.multiplyTransforms(position, orientation, pts[3, :], rotIdentity)
    idx = p.addUserDebugLine(po, px, xColor, lineWidth, lifeTime,
                             parentObjectUniqueId, parentLinkIndex)
    remove_user_item_indices.append(idx)
    idx = p.addUserDebugLine(po, py, yColor, lineWidth, lifeTime,
                             parentObjectUniqueId, parentLinkIndex)
    remove_user_item_indices.append(idx)
    idx = p.addUserDebugLine(po, pz, zColor, lineWidth, lifeTime,
                             parentObjectUniqueId, parentLinkIndex)
    remove_user_item_indices.append(idx)
    if textOrientation is None:
        textOrientation = orientation
    idx = p.addUserDebugText(text, [0, 0, 0.1], textColorRGB=textColor,
                             textSize=textSize,
                             parentObjectUniqueId=parentObjectUniqueId,
                             parentLinkIndex=parentLinkIndex)
    remove_user_item_indices.append(idx)
