"""Robot interface backed by an Isaac Sim (PhysX) simulation.

``IsaacSimRobotInterface`` takes the same calls as
:class:`skrobot.interfaces._mujoco.MujocoRobotInterface` -- and so as a real
robot's ROS interface: ``angle_vector(av, time)`` returns at once,
``wait_interpolation()`` runs the simulation until the motion is done -- but
the physics is PhysX and the cameras are drawn by Isaac Sim's RTX renderer.

The URDF is converted with :func:`skrobot.export.usd.urdf_to_usd`, whose
articulation gives every joint a position drive; the trajectory is written
to those drives one physics step at a time. As in the MuJoCo interface
nothing runs on its own and there are no threads: :meth:`step` and
:meth:`wait_interpolation` advance the simulation.

Isaac Sim must be started (``SimulationApp``) before any of its modules is
imported. The interface starts one, headless unless asked otherwise, when
none is running yet; start your own first to choose its settings. That
includes ``pxr``: imported before the app starts, it loads a USD that does
not match Isaac Sim's, and the app's extensions then fail to load.
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
from skrobot.interfaces._mujoco_scene import _CAMERA_IN_LINK
from skrobot.interfaces._sim_base import _camera_specs
from skrobot.interfaces._sim_base import _planar_coords
from skrobot.interfaces._sim_base import SimulatedRobotInterfaceBase


logger = logging.getLogger(__name__)

# The planar base joints skrobot.export.usd adds for mobile_base=True, in
# (x, y, yaw) order.
_PLANAR_BASE_JOINTS = ('base_x_joint', 'base_y_joint', 'base_yaw_joint')

_ROBOT_PRIM = '/World/robot'
_SCENE_PRIM = '/World/scene'
_PROPS_PRIM = '/World/props'

# Environments in Isaac Sim's asset library, by short name. They are
# fetched from NVIDIA's asset server when the stage loads.
SCENES = {
    'warehouse': '/Isaac/Environments/Simple_Warehouse/warehouse.usd',
    'full_warehouse':
        '/Isaac/Environments/Simple_Warehouse/full_warehouse.usd',
    'warehouse_with_forklifts':
        '/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd',
    'warehouse_multiple_shelves':
        '/Isaac/Environments/Simple_Warehouse/'
        'warehouse_multiple_shelves.usd',
    'hospital': '/Isaac/Environments/Hospital/hospital.usd',
    'office': '/Isaac/Environments/Office/office.usd',
}

# Horizontal aperture of a USD camera [tenths of a scene unit], Isaac Sim's
# default; the focal length is chosen against it to give the asked fovy.
_HORIZONTAL_APERTURE = 20.955

# Scene lighting: a dome light and a distant light tilted from the zenith
# (rotation about x, y, z [deg]).
_SKY_INTENSITY = 1000.0
_SUN_INTENSITY = 3000.0
_SUN_ROTATION_DEG = (0.0, 40.0, 30.0)

# Renders get_image waits through for the first image (shader compilation
# on a first run took 300 s on a T4), and renders it adds after, for the
# annotator to catch up with the current state.
_MAX_WARMUP_FRAMES = 10000
_CATCH_UP_FRAMES = 3

# URDFs already converted in this process -> their USD.
_converted = {}

# The SimulationApp this module started, if it started one.
_app = None


def _simulation_app(headless):
    """The SimulationApp this module started, a new one if Isaac Sim is not
    up, or None when the caller started their own.

    ``omni.kit.app.get_app()`` hands back the app only while one runs:
    before that ``omni`` is not importable, or (once ``isaacsim`` has been
    imported) acquiring the app interface raises RuntimeError.
    """
    global _app
    if _app is not None:
        return _app
    try:
        import omni.kit.app
        omni.kit.app.get_app()
        return None
    except (ImportError, RuntimeError):
        pass
    try:
        from isaacsim import SimulationApp
    except ImportError:
        raise ImportError(
            'IsaacSimRobotInterface needs Isaac Sim (Python 3.12):\n\n'
            '  $ pip install "isaacsim[all,extscache]" '
            '--extra-index-url https://pypi.nvidia.com\n')
    _app = SimulationApp({'headless': headless})
    return _app


def _scene_url(scene):
    """A name in :data:`SCENES`, a path in Isaac Sim's asset library
    (``/Isaac/...``) or any USD path or URL -> what to reference."""
    scene = SCENES.get(scene, scene)
    if scene.startswith('/Isaac/'):
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
        if root is None:
            raise RuntimeError(
                'Isaac Sim asset server not reachable; cannot load '
                + scene)
        scene = root + scene
    return scene


def _prim_names(urdf_path, planar_base):
    """URDF link and joint names -> their prim names in the exported USD.

    The same mappings :func:`skrobot.export.usd.urdf_to_usd` makes, over
    the links and joints in document order.

    Returns
    -------
    tuple[dict[str, str], dict[str, str]]
        Link and joint mappings.
    """
    from skrobot.export.usd import _BASE_JOINTS
    from skrobot.export.usd import _LOOKS_PATH
    from skrobot.export.usd import _PLANAR_BASE_LINKS
    from skrobot.urdf.sanitize import unique_sanitized_names

    root = ET.parse(urdf_path).getroot()
    links = unique_sanitized_names(
        [e.get('name') for e in root.findall('link')],
        reserved=('joints', _LOOKS_PATH.rsplit('/', 1)[-1])
        + (_PLANAR_BASE_LINKS if planar_base else ()))
    joints = unique_sanitized_names(
        [e.get('name') for e in root.findall('joint')],
        reserved=_BASE_JOINTS)
    return links, joints


class IsaacSimRobotInterface(SimulatedRobotInterfaceBase):
    """Send angle vectors to a robot simulated in Isaac Sim.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        Robot model. Its ``joint_list`` decides the order of angle vectors.
        The interface works on a copy, ``ri.robot``; the model passed in is
        left alone.
    urdf_path : str or None
        URDF to simulate. Defaults to ``robot.urdf_path``.
    usd_path : str or None
        A USD written by :func:`skrobot.export.usd.urdf_to_usd` to load as
        is, instead of converting the URDF (``urdf_path`` is still read for
        the joint names).
    floating_base : bool
        Leave the base free instead of welding it to the world.
    planar_base : bool
        Let the base slide and turn on the floor, driven by :meth:`go_pos`,
        :meth:`go_velocity` and :meth:`move_to`. Cannot be combined with
        ``floating_base``.
    base_max_velocity : tuple[float, float]
        Translation [m/s] and rotation [rad/s] speed :meth:`go_pos` and
        :meth:`move_to` plan with.
    cameras : dict[str, dict] or None
        Camera name -> ``{'link': <link>, 'fovy': <deg>, 'width': <px>,
        'height': <px>, 'frame': 'optical' or 'x_forward'}``, as in
        :class:`~skrobot.interfaces._mujoco.MujocoRobotInterface`. Read with
        :meth:`get_image` and :meth:`get_depth`.
    gravity_compensation : bool
        Turn gravity off for the robot's links, so the drives only have to
        follow the trajectory. The floor and other bodies still fall.
    timestep : float
        Physics step [s].
    headless : bool
        Start Isaac Sim without a window, when the interface starts it.
    realtime : bool
        Make :meth:`step` (and so :meth:`wait_interpolation`) take as long
        in wall-clock time as it simulates.
    viewer : bool
        Serve a page at ``http://localhost:<viewer_port>/`` showing the
        simulation through an RTX camera you orbit with the mouse (see
        :class:`~skrobot.interfaces._isaacsim_viewer.RtxWebViewer`; needs
        the ``websockets`` package). Frames are drawn while the simulation
        steps, so it implies ``realtime``; :meth:`spin` keeps it going when
        your program has nothing to send. In Colab, show it in the notebook
        with ``google.colab.output.serve_kernel_port_as_iframe(viewer_port)``.
    viewer_port : int
        Port of the viewer page.
    converter_kwargs : dict or None
        Extra keyword arguments for :func:`skrobot.export.usd.urdf_to_usd`.
    scene : str or None
        Environment to put the robot in, instead of the bare floor: a name
        in :data:`SCENES` (``'warehouse'``, ``'hospital'``, ...), a path in
        Isaac Sim's asset library (``'/Isaac/Environments/...'``) or a USD
        file or URL. It brings its own floor and lights. The robot stands
        where the robot model stands in the scene's frame.
    props : list[tuple[str, skrobot.coordinates.Coordinates]] or None
        Objects to place: ``(usd, coords)`` pairs, ``usd`` given as for
        ``scene`` (for example
        ``'/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd'``)
        and ``coords`` its pose in the world. A few props on the bare floor
        make a much lighter stage than a whole environment.

    Examples
    --------
    >>> import skrobot
    >>> from skrobot.interfaces import IsaacSimRobotInterface
    >>> robot = skrobot.models.Panda()
    >>> ri = IsaacSimRobotInterface(robot)
    >>> robot.reset_pose()
    >>> ri.angle_vector(robot.angle_vector(), 3.0)
    >>> ri.wait_interpolation()
    >>> robot.angle_vector(ri.angle_vector())
    """

    def __init__(self, robot, urdf_path=None, usd_path=None,
                 floating_base=False, planar_base=False,
                 base_max_velocity=(0.5, 1.0), cameras=None,
                 gravity_compensation=True, timestep=1.0 / 200.0,
                 headless=True, realtime=False, viewer=False,
                 viewer_port=8765, converter_kwargs=None, scene=None,
                 props=None):
        if floating_base and planar_base:
            raise ValueError(
                'floating_base and planar_base cannot both be set.')
        self.app = _simulation_app(headless)
        self.robot = copy.deepcopy(robot)
        self.floating_base = floating_base
        self.planar_base = planar_base
        self.base_max_velocity = base_max_velocity
        self.cameras = _camera_specs(cameras)

        if urdf_path is None:
            urdf_path = robot.urdf_path
        if urdf_path is None:
            raise ValueError('robot.urdf_path is None; pass urdf_path.')
        self.urdf_path = urdf_path
        if usd_path is None:
            usd_path = self._convert(urdf_path, floating_base, planar_base,
                                     converter_kwargs or {})
        self.usd_path = usd_path
        self._link_prim, self._joint_prim = _prim_names(urdf_path,
                                                        planar_base)

        self.scene = scene
        self.props = list(props or [])
        self._build_stage(timestep, gravity_compensation)
        self._map_joints()
        self._add_cameras()
        self.realtime = realtime or viewer
        self.viewer = None
        if viewer:
            from skrobot.interfaces._isaacsim_viewer import RtxWebViewer
            self.viewer = RtxWebViewer(self, port=viewer_port)
        self.reset(robot.angle_vector())

    @staticmethod
    def _convert(urdf_path, floating_base, planar_base, converter_kwargs):
        from skrobot.export.usd import urdf_to_usd

        key = (os.path.abspath(urdf_path), os.path.getmtime(urdf_path),
               floating_base, planar_base,
               repr(sorted(converter_kwargs.items())))
        out_path = _converted.get(key)
        if out_path is not None and os.path.exists(out_path):
            return out_path
        out_dir = tempfile.mkdtemp(prefix='skrobot_isaacsim_')
        out_path = os.path.join(out_dir, 'robot.usd')
        urdf_to_usd(urdf_path, out_path, floating_base=floating_base,
                    mobile_base=planar_base, **converter_kwargs)
        _converted[key] = out_path
        return out_path

    def _build_stage(self, timestep, gravity_compensation):
        """A fresh stage: physics scene, floor (or environment) and the
        robot."""
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.core.utils.stage import create_new_stage
        from pxr import Gf
        from pxr import PhysxSchema
        from pxr import Sdf
        from pxr import UsdGeom
        from pxr import UsdLux
        from pxr import UsdPhysics

        # A new interface starts over, as a notebook cell run again does.
        World.clear_instance()
        create_new_stage()
        self.world = World(physics_dt=timestep, rendering_dt=timestep,
                           stage_units_in_meters=1.0)
        if self.scene is None:
            self.world.scene.add_default_ground_plane()
        else:
            add_reference_to_stage(_scene_url(self.scene), _SCENE_PRIM)
        prim = add_reference_to_stage(self.usd_path, _ROBOT_PRIM)
        stage = prim.GetStage()
        for i, (usd, coords) in enumerate(self.props):
            # The pose goes on a parent of the reference, so the asset's own
            # transform (its scale, say) is kept.
            holder = UsdGeom.Xform.Define(
                stage, Sdf.Path('{}/prop_{}'.format(_PROPS_PRIM, i)))
            holder.AddTranslateOp().Set(
                Gf.Vec3d(*[float(v) for v in coords.translation]))
            # matrix2quaternion is (w, x, y, z), as Gf.Quatd takes it.
            holder.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(
                *[float(v) for v in matrix2quaternion(coords.rotation)]))
            add_reference_to_stage(
                _scene_url(usd), str(holder.GetPath().AppendChild('asset')))
        if self.scene is None:
            # A new stage has no lights, and RTX draws an unlit one black:
            # sky all round, and a sun high and to one side for shadows.
            UsdLux.DomeLight.Define(stage, Sdf.Path('/World/sky')) \
                .CreateIntensityAttr(_SKY_INTENSITY)
            sun = UsdLux.DistantLight.Define(stage, Sdf.Path('/World/sun'))
            sun.CreateIntensityAttr(_SUN_INTENSITY)
            sun.AddRotateXYZOp().Set(Gf.Vec3f(*_SUN_ROTATION_DEG))
        if not self.floating_base and not self.planar_base:
            # The exported weld holds the base at the world origin; move it
            # to where the robot model stands.
            coords = self.robot.copy_worldcoords()
            weld = UsdPhysics.FixedJoint(
                stage.GetPrimAtPath(_ROBOT_PRIM + '/joints/base_fixed'))
            weld.CreateLocalPos0Attr(
                Gf.Vec3f(*[float(v) for v in coords.translation]))
            # matrix2quaternion is (w, x, y, z), as Gf.Quatf takes it.
            weld.CreateLocalRot0Attr(Gf.Quatf(
                *[float(v) for v in matrix2quaternion(coords.rotation)]))
        if gravity_compensation:
            for p in stage.Traverse():
                if (p.GetPath().HasPrefix(prim.GetPath())
                        and p.HasAPI(UsdPhysics.RigidBodyAPI)):
                    PhysxSchema.PhysxRigidBodyAPI.Apply(p) \
                        .CreateDisableGravityAttr(True)
        self._stage = stage

    def _map_joints(self):
        # The batched view, not SingleArticulation: only it sets drive
        # targets. Its arrays carry a leading axis of one robot.
        from isaacsim.core.prims import Articulation

        self.articulation = Articulation(_ROBOT_PRIM)
        self.world.reset()
        self.articulation.initialize()
        dof_names = list(self.articulation.dof_names)
        index = []
        for joint in self.robot.joint_list:
            name = self._joint_prim.get(joint.name)
            if name not in dof_names:
                raise ValueError(
                    'joint {} of the robot model is not a degree of freedom '
                    'of {}'.format(joint.name, self.usd_path))
            index.append(dof_names.index(name))
        self._dof_index = np.array(index, dtype=np.int64)
        self._base_index = None
        if self.planar_base:
            self._base_index = np.array(
                [dof_names.index(n) for n in _PLANAR_BASE_JOINTS],
                dtype=np.int64)

    def _add_cameras(self):
        """A USD camera on each named link, and an RTX render product."""
        if not self.cameras:
            self._annotators = {}
            return
        import omni.replicator.core as rep
        from pxr import Gf
        from pxr import UsdGeom

        link_prim = self._link_prim
        self._annotators = {}
        for name, spec in self.cameras.items():
            link = spec['link']
            link_name = link if isinstance(link, str) else link.name
            if link_name not in link_prim:
                raise ValueError('camera {}: link {} is not in the URDF'
                                 .format(name, link_name))
            frame = spec['frame']
            if frame not in _CAMERA_IN_LINK:
                raise ValueError(
                    'camera {}: frame must be one of {}, not {!r}'.format(
                        name, sorted(_CAMERA_IN_LINK), frame))
            path = '{}/{}/{}'.format(_ROBOT_PRIM, link_prim[link_name], name)
            camera = UsdGeom.Camera.Define(self._stage, path)
            # A USD camera looks along -z with +y up, as MuJoCo's does.
            camera.AddOrientOp().Set(Gf.Quatf(
                *matrix2quaternion(_CAMERA_IN_LINK[frame])))
            aspect = spec['height'] / float(spec['width'])
            vertical = _HORIZONTAL_APERTURE * aspect
            camera.CreateHorizontalApertureAttr(_HORIZONTAL_APERTURE)
            camera.CreateVerticalApertureAttr(vertical)
            camera.CreateFocalLengthAttr(
                0.5 * vertical / np.tan(0.5 * np.deg2rad(spec['fovy'])))
            camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
            product = rep.create.render_product(
                path, (spec['width'], spec['height']))
            rgb = rep.AnnotatorRegistry.get_annotator('rgb')
            rgb.attach([product])
            depth = rep.AnnotatorRegistry.get_annotator(
                'distance_to_image_plane')
            depth.attach([product])
            self._annotators[name] = (rgb, depth)

    def reset(self, av=None):
        """Put the simulated robot at ``av`` at rest and drop any trajectory.

        A planar base goes back to where the robot model stands.

        Parameters
        ----------
        av : numpy.ndarray or None
            Angle vector to start from. ``None`` uses the robot model's.
        """
        if av is None:
            av = self.robot.angle_vector()
        av = np.asarray(av, dtype=np.float64)
        self._reset_commands(av)
        positions = np.zeros(self.articulation.num_dof)
        positions[self._dof_index] = av
        if self._base_index is not None:
            positions[self._base_index] = self._base_hold
        self.articulation.set_joint_positions(positions[None])
        self.articulation.set_joint_velocities(np.zeros((1, positions.size)))
        self._write_targets(self.time)

    def _read(self, getter, indices):
        """``getter`` (an Articulation ``get_*`` method) at ``indices``,
        as a flat float64 array."""
        return np.asarray(getter(joint_indices=indices),
                          dtype=np.float64)[0]

    @property
    def time(self):
        """Simulation time [s]."""
        return self.world.current_time

    @property
    def _timestep(self):
        return self.world.get_physics_dt()

    def _write_targets(self, t):
        """Drive the joints (and base) toward the command at time ``t``.

        The commanded velocity goes to the drives too, so their damping
        pulls along the trajectory instead of against it.
        """
        dt = self._timestep
        target = self._target(t)
        velocity = (self._target(t + dt) - target) / dt
        indices = self._dof_index
        if self._base_index is not None:
            base = self._base_target(t)
            target = np.concatenate([target, base])
            velocity = np.concatenate(
                [velocity, (self._base_target(t + dt) - base) / dt])
            indices = np.concatenate([indices, self._base_index])
        self.articulation.set_joint_position_targets(
            target[None], joint_indices=indices)
        self.articulation.set_joint_velocity_targets(
            velocity[None], joint_indices=indices)

    def step(self, n=1):
        """Advance the simulation ``n`` physics steps.

        Parameters
        ----------
        n : int
            Number of physics steps.
        """
        start_wall = _time.time()
        start_sim = self.time
        for _ in range(n):
            self._write_targets(self.time)
            self.world.step(render=False)
            if self.viewer is not None:
                self.viewer.service()
            if self.realtime:
                lag = (self.time - start_sim) - (_time.time() - start_wall)
                if lag > 0:
                    _time.sleep(lag)

    def spin(self, duration=None):
        """Keep the simulation running in real time, holding the last
        command, for ``duration`` seconds or until interrupted (Ctrl-C).

        For a program that has sent its last motion but should keep the
        viewer live.

        Parameters
        ----------
        duration : float or None
            Seconds to run; None runs until KeyboardInterrupt.
        """
        realtime = self.realtime
        self.realtime = True
        end = None if duration is None else self.time + duration
        try:
            while end is None or self.time < end:
                self.step(10)
        except KeyboardInterrupt:
            pass
        finally:
            self.realtime = realtime

    def _base_pose(self):
        return self._read(self.articulation.get_joint_positions,
                          self._base_index)

    def sync(self):
        """Copy the simulated joints (and base) into ``ri.robot``.

        Returns
        -------
        numpy.ndarray
            The simulated angle vector as a robot model can hold it: a joint
            pressed a little past its limit reads as at the limit.
        """
        av = self._read(self.articulation.get_joint_positions,
                        self._dof_index)
        lower = np.array([j.min_angle for j in self.robot.joint_list])
        upper = np.array([j.max_angle for j in self.robot.joint_list])
        self.robot.angle_vector(np.clip(av, lower, upper))
        av = self.robot.angle_vector()
        if self.floating_base:
            # (w, x, y, z), as quaternion2matrix takes it.
            pos, quat = self.articulation.get_world_poses(usd=False)
            self.robot.newcoords(Coordinates(
                pos=np.asarray(pos)[0],
                rot=quaternion2matrix(np.asarray(quat)[0])))
        if self.planar_base:
            self.robot.newcoords(_planar_coords(self._base_pose()))
        return av

    def joint_velocity_vector(self):
        """Simulated joint velocities, in ``robot.joint_list`` order."""
        return self._read(self.articulation.get_joint_velocities,
                          self._dof_index)

    def joint_effort_vector(self):
        """Torque [N m] or force [N] the drive of each joint applies, in
        ``robot.joint_list`` order."""
        return self._read(self.articulation.get_measured_joint_efforts,
                          self._dof_index)

    # --- cameras ---------------------------------------------------------

    def camera_coords(self, name):
        """Optical frame of a camera in the world frame (+z forward, +x
        right, +y down), whichever way it was mounted."""
        spec = self._camera(name)
        self.sync()
        link = spec['link']
        if isinstance(link, str):
            link = getattr(self.robot, link)
        else:
            link = getattr(self.robot, link.name)
        # The USD camera (-z forward, +y up) turned half a turn about x.
        rot = link.worldrot().dot(_CAMERA_IN_LINK[spec['frame']]) \
            * np.array([1, -1, -1])
        return Coordinates(pos=link.worldpos(), rot=rot)

    def _render(self, annotator):
        """Draw the cameras at the current state, without stepping physics,
        and return what ``annotator`` holds.

        The RTX pipeline hands an annotator its frame a few updates late,
        and gives nothing at all until its shaders are ready (minutes on a
        first run), so it renders until data arrives, then a few frames
        more for it to catch up with the present state.
        """
        for _ in range(_MAX_WARMUP_FRAMES):
            self.world.render()
            if np.asarray(annotator.get_data()).size > 0:
                break
        else:
            raise RuntimeError(
                'the renderer gave no image after {} frames'.format(
                    _MAX_WARMUP_FRAMES))
        for _ in range(_CATCH_UP_FRAMES):
            self.world.render()
        return np.asarray(annotator.get_data())

    def get_image(self, name):
        """RGB image from a camera, drawn by the RTX renderer.

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
        return self._render(self._annotators[name][0])[..., :3].copy()

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
        return self._render(self._annotators[name][1]).astype(np.float32)

    def close(self):
        """Stop the simulation. Isaac Sim itself keeps running; close it
        with ``ri.app.close()`` when the interface started it."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.world.stop()
