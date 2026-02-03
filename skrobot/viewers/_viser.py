from contextlib import contextmanager
import threading
import time
from typing import Dict
from typing import Optional
from typing import Union
import webbrowser

import numpy as np
import trimesh
import viser
import viser.transforms as vtf

from skrobot.coordinates import CascadedCoords
from skrobot.coordinates import Coordinates
from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import matrix2rpy
from skrobot.coordinates.math import rotation_matrix_from_rpy
from skrobot.model.joint import _MimicJointHook
from skrobot.model.joint import FixedJoint
from skrobot.model.link import Link
from skrobot.model.primitives import Axis
from skrobot.model.primitives import LineString
from skrobot.model.primitives import PointCloudLink
from skrobot.model.primitives import Sphere
from skrobot.model.robot_model import CascadedLink
from skrobot.model.robot_model import RobotModel


class ViserViewer:
    """Viser-based 3D viewer for scikit-robot.

    Parameters
    ----------
    draw_grid : bool
        Whether to draw the ground grid. Default is True.
    enable_ik : bool
        Whether to enable interactive IK controls. When enabled,
        transform controls are added for each detected end-effector
        and dragging them will solve IK in real-time. Default is False.
    enable_motion_planning : bool
        Whether to enable motion planning controls. When enabled,
        users can save waypoints, plan trajectories between them,
        and animate the results. Implicitly enables IK. Default is False.
    """

    _LIMB_ATTR_MAP = {
        'arm': ['arm', 'rarm'],
        'right_arm': ['rarm', 'right_arm'],
        'left_arm': ['larm', 'left_arm'],
        'right_leg': ['rleg', 'right_leg'],
        'left_leg': ['lleg', 'left_leg'],
        'head': ['head'],
        'torso': ['torso'],
    }

    _END_COORDS_ATTR_MAP = {
        'arm': ['arm_end_coords', 'rarm_end_coords', 'end_coords'],
        'right_arm': ['rarm_end_coords', 'right_arm_end_coords'],
        'left_arm': ['larm_end_coords', 'left_arm_end_coords'],
        'right_leg': ['rleg_end_coords', 'right_leg_end_coords'],
        'left_leg': ['lleg_end_coords', 'left_leg_end_coords'],
        'head': ['head_end_coords'],
        'torso': ['torso_end_coords'],
    }

    def __init__(
        self,
        draw_grid: bool = True,
        enable_ik: bool = False,
        enable_motion_planning: bool = False,
    ):
        self._server = viser.ViserServer()
        self._linkid_to_handle = dict()
        self._linkid_to_link = dict()
        self._is_active = True
        self._joint_sliders = dict()
        self._joint_folders = dict()

        # Motion planning implies IK
        self._enable_motion_planning = enable_motion_planning
        if enable_motion_planning:
            enable_ik = True

        # IK state
        self._enable_ik = enable_ik
        # robot_id -> {group_name -> target_info}
        self._ik_targets: Dict[int, Dict[str, dict]] = {}
        # robot_id -> RobotModel
        self._robot_models: Dict[int, RobotModel] = {}
        self._updating_from_ik = False

        # Batch IK state
        # robot_id -> {group_name -> solver}
        self._batch_ik_solvers: Dict[int, Dict[str, object]] = {}
        self._batch_ik_samples = None

        # JAX availability check (done once at init)
        self._jax_available = False
        try:
            from skrobot.backend import list_backends
            self._jax_available = 'jax' in list_backends()
        except Exception:
            pass

        # Throttling for IK callbacks
        self._last_ik_time: Dict[tuple, float] = {}  # (robot_id, group_name) -> timestamp
        self._ik_throttle_interval = 0.05  # 50ms throttle

        # Thread safety for IK solver creation
        self._ik_lock = threading.Lock()
        self._solver_creating_keys: set = set()

        # Motion planning state
        self._waypoints = []  # list of dicts: {'angle_vector': ndarray, 'group_angles': dict}
        self._selected_planning_group = None
        self._planned_trajectory = None  # ndarray (n_steps, n_joints)
        self._trajectory_ghost_handles = []  # viser handles for trajectory ghosts
        self._waypoint_ghost_handles = []  # viser handles for waypoint ghosts
        self._is_planning = False
        self._is_animating = False
        self._animation_thread = None
        self._planning_thread = None
        self._mp_lock = threading.Lock()
        self._cached_mp_solver = None
        self._cached_mp_solver_type = None

    @property
    def is_active(self) -> bool:
        return self._is_active

    def close(self):
        self._is_active = False
        self._server.stop()

    def _get_limb_attribute(self, robot_model, group_name, attr_name):
        """Get an attribute from a robot model's limb for a given group.

        Parameters
        ----------
        robot_model : RobotModel
            The robot model to search.
        group_name : str
            The group name (e.g., 'arm', 'right_arm').
        attr_name : str
            The attribute to retrieve from the limb (e.g., 'end_coords',
            'link_list').

        Returns
        -------
        object or None
            The attribute value if found, None otherwise.
        """
        limb_attrs = self._LIMB_ATTR_MAP.get(group_name, [group_name])
        for limb_attr in limb_attrs:
            try:
                if hasattr(robot_model, limb_attr):
                    limb = getattr(robot_model, limb_attr)
                    if hasattr(limb, attr_name):
                        return getattr(limb, attr_name)
            except NotImplementedError:
                continue
        return None

    @staticmethod
    def _update_handle_pose(handle, pos, rot):
        """Update a viser handle's position and orientation.

        Parameters
        ----------
        handle : object
            A viser scene handle with ``position`` and ``wxyz`` attributes.
        pos : numpy.ndarray
            Position (3,).
        rot : numpy.ndarray
            Rotation matrix (3, 3).
        """
        handle.position = pos
        handle.wxyz = matrix2quaternion(rot)

    @contextmanager
    def _ik_update_guard(self):
        """Context manager that sets ``_updating_from_ik`` while active."""
        self._updating_from_ik = True
        try:
            yield
        finally:
            self._updating_from_ik = False

    @staticmethod
    @contextmanager
    def _preserved_angle_vector(robot_model, update_links=True):
        """Context manager that saves and restores a robot's angle vector.

        Parameters
        ----------
        robot_model : RobotModel
            The robot model whose angle vector is preserved.
        update_links : bool
            Whether to force-update all links after restoring.
            Default is True.
        """
        original_av = robot_model.angle_vector().copy()
        try:
            yield
        finally:
            robot_model.angle_vector(original_av)
            if update_links:
                for link in robot_model.link_list:
                    link.update(force=True)

    def _check_ik_guard(self, robot_id, group_name):
        """Check IK preconditions and return the target if valid.

        Returns None if the call should be skipped (updating from IK,
        throttled, or target not found).

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.

        Returns
        -------
        dict or None
            Target info dict if the IK call should proceed, None otherwise.
        """
        if self._updating_from_ik:
            return None
        if self._is_throttled(robot_id, group_name):
            return None
        return self._get_ik_target(robot_id, group_name)

    def _find_existing_end_coords(self, robot_model: RobotModel, group_name: str):
        """Find existing end_coords in robot model for a given group.

        Parameters
        ----------
        robot_model : RobotModel
            The robot model to search.
        group_name : str
            The group name (e.g., 'arm', 'right_arm', 'left_arm', 'head').

        Returns
        -------
        CascadedCoords or None
            The existing end_coords if found, None otherwise.
        """
        attr_candidates = self._END_COORDS_ATTR_MAP.get(
            group_name, [f'{group_name}_end_coords'])
        for attr_name in attr_candidates:
            if hasattr(robot_model, attr_name):
                end_coords = getattr(robot_model, attr_name)
                if isinstance(end_coords, CascadedCoords):
                    return end_coords

        end_coords = self._get_limb_attribute(
            robot_model, group_name, 'end_coords')
        if isinstance(end_coords, CascadedCoords):
            return end_coords
        return None

    def _find_existing_link_list(self, robot_model: RobotModel, group_name: str):
        """Find existing link_list in robot model for a given group.

        Parameters
        ----------
        robot_model : RobotModel
            The robot model to search.
        group_name : str
            The group name (e.g., 'arm', 'right_arm', 'left_arm', 'head').

        Returns
        -------
        list or None
            The existing link_list if found, None otherwise.
        """
        link_list = self._get_limb_attribute(
            robot_model, group_name, 'link_list')
        if link_list:
            return list(link_list)
        return None

    def _setup_ik_controls(self, robot_model: RobotModel):
        """Set up interactive IK controls for detected end-effectors."""
        from skrobot.urdf.robot_class_generator import generate_groups_from_geometry

        robot_id = id(robot_model)
        self._robot_models[robot_id] = robot_model
        self._ik_targets[robot_id] = {}
        self._batch_ik_solvers[robot_id] = {}

        # Detect groups from geometry
        groups, end_effectors, end_coords_info, _ = generate_groups_from_geometry(
            robot_model
        )

        # Filter groups suitable for IK
        ik_groups = {}
        for group_name, group_data in groups.items():
            if group_data is None:
                continue
            # Skip torso-only and gripper groups
            if 'torso' in group_name and 'arm' not in group_name:
                continue
            if 'gripper' in group_name:
                continue
            # Need at least 3 joints
            links = group_data.get('links', [])
            if len(links) < 3:
                continue
            ik_groups[group_name] = (group_data, end_coords_info.get(group_name, {}))

        if not ik_groups:
            return

        # Get robot display name
        robot_name = getattr(robot_model, 'name', None) or f"robot_{robot_id}"

        # Create transform control for each group inside IK Controls folder
        with self._ik_controls_folder:
            for group_name, (group_data, ec_info) in ik_groups.items():
                # Try to use existing link_list from robot model (e.g., rarm.link_list)
                link_list = self._find_existing_link_list(robot_model, group_name)

                if link_list is None:
                    # Fall back to generating from group data
                    link_names = group_data.get('links', [])
                    link_list = []
                    for name in link_names:
                        for link in robot_model.link_list:
                            if link.name == name:
                                link_list.append(link)
                                break

                if not link_list:
                    continue

                # Try to use existing end_coords from robot model
                end_coords = self._find_existing_end_coords(robot_model, group_name)

                if end_coords is None:
                    # Create end_coords from detected info
                    link_names = group_data.get('links', [])
                    default_parent = link_names[-1] if link_names else link_list[-1].name
                    parent_link_name = ec_info.get('parent_link', default_parent)
                    parent_link = None
                    for link in robot_model.link_list:
                        if link.name == parent_link_name:
                            parent_link = link
                            break
                    if parent_link is None:
                        parent_link = link_list[-1]

                    pos = ec_info.get('pos', [0.0, 0.0, 0.0])
                    rot = ec_info.get('rot')

                    end_coords = CascadedCoords(
                        parent=parent_link,
                        pos=pos,
                        rot=rot,
                        name=f"{group_name}_end_coords",
                    )

                # Add transform control at end-effector position
                ee_pos = end_coords.worldpos()
                ee_rot = end_coords.worldrot()

                control = self._server.scene.add_transform_controls(
                    f"ik_target/{robot_name}/{group_name}",
                    scale=0.1,
                    position=ee_pos,
                    wxyz=matrix2quaternion(ee_rot),
                )

                # Add visibility checkbox for this target
                visibility_checkbox = self._server.gui.add_checkbox(
                    f"Show {robot_name}/{group_name}", initial_value=True
                )

                def make_visibility_callback(ctrl, checkbox):
                    def callback(_):
                        ctrl.visible = checkbox.value
                    return callback

                visibility_checkbox.on_update(
                    make_visibility_callback(control, visibility_checkbox)
                )

                # Add numeric input fields for position and rotation
                with self._server.gui.add_folder(
                    f"{robot_name}/{group_name} Target",
                    expand_by_default=False,
                ):
                    # Position inputs (in meters)
                    pos_x = self._server.gui.add_number(
                        "X [m]", initial_value=float(ee_pos[0]), step=0.01
                    )
                    pos_y = self._server.gui.add_number(
                        "Y [m]", initial_value=float(ee_pos[1]), step=0.01
                    )
                    pos_z = self._server.gui.add_number(
                        "Z [m]", initial_value=float(ee_pos[2]), step=0.01
                    )

                    # Get initial RPY from rotation matrix
                    # matrix2rpy returns [roll, pitch, yaw]
                    roll_init, pitch_init, yaw_init = matrix2rpy(ee_rot)

                    # Rotation inputs (in degrees for user convenience)
                    roll_input = self._server.gui.add_number(
                        "Roll [deg]",
                        initial_value=float(np.rad2deg(roll_init)),
                        step=1.0,
                    )
                    pitch_input = self._server.gui.add_number(
                        "Pitch [deg]",
                        initial_value=float(np.rad2deg(pitch_init)),
                        step=1.0,
                    )
                    yaw_input = self._server.gui.add_number(
                        "Yaw [deg]",
                        initial_value=float(np.rad2deg(yaw_init)),
                        step=1.0,
                    )

                # Store target info
                self._ik_targets[robot_id][group_name] = {
                    'link_list': link_list,
                    'end_coords': end_coords,
                    'control': control,
                    'visibility_checkbox': visibility_checkbox,
                    'pos_inputs': (pos_x, pos_y, pos_z),
                    'rot_inputs': (roll_input, pitch_input, yaw_input),
                    'robot_model': robot_model,
                }

                # Callback for when control is moved (updates numeric inputs)
                def make_ik_callback(rid, gname):
                    def callback(_):
                        self._solve_ik(rid, gname)
                    return callback

                control.on_update(make_ik_callback(robot_id, group_name))

                # Callback for numeric input changes
                def make_numeric_ik_callback(rid, gname):
                    def callback(_):
                        self._solve_ik_from_numeric(rid, gname)
                    return callback

                numeric_callback = make_numeric_ik_callback(robot_id, group_name)
                pos_x.on_update(numeric_callback)
                pos_y.on_update(numeric_callback)
                pos_z.on_update(numeric_callback)
                roll_input.on_update(numeric_callback)
                pitch_input.on_update(numeric_callback)
                yaw_input.on_update(numeric_callback)

    def _get_ik_target(self, robot_id, group_name):
        """Get IK target info dict, or None if not found.

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.

        Returns
        -------
        dict or None
            Target info dict if found, None otherwise.
        """
        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return None
        return robot_targets.get(group_name)

    def _is_throttled(self, robot_id, group_name):
        """Return True if this IK call should be skipped (throttled).

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.

        Returns
        -------
        bool
            True if the call should be throttled.
        """
        now = time.time()
        key = (robot_id, group_name)
        if now - self._last_ik_time.get(key, 0) < self._ik_throttle_interval:
            return True
        self._last_ik_time[key] = now
        return False

    def _update_pose_inputs(self, target, pos, rot):
        """Update numeric pos/rot input fields for a target.

        Parameters
        ----------
        target : dict
            IK target info dict.
        pos : numpy.ndarray
            Position (3,).
        rot : numpy.ndarray
            Rotation matrix (3, 3).
        """
        pos_inputs = target['pos_inputs']
        for i in range(3):
            pos_inputs[i].value = float(pos[i])
        roll, pitch, yaw = matrix2rpy(rot)
        rot_inputs = target['rot_inputs']
        rot_inputs[0].value = float(np.rad2deg(roll))
        rot_inputs[1].value = float(np.rad2deg(pitch))
        rot_inputs[2].value = float(np.rad2deg(yaw))

    def _run_ik(self, robot_id, group_name, target_pos, target_rot):
        """Core IK solving: regular IK with batch IK fallback.

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.
        target_pos : numpy.ndarray
            Target position (3,).
        target_rot : numpy.ndarray
            Target rotation matrix (3, 3).

        Returns
        -------
        numpy.ndarray, bool, or None
            Joint angles if successful, False or None otherwise.
        """
        target = self._get_ik_target(robot_id, group_name)
        if target is None:
            return None
        robot_model = target['robot_model']
        target_coords = Coordinates(pos=target_pos, rot=target_rot)
        constrain_rot = self._ik_constrain_rotation.value

        result = robot_model.inverse_kinematics(
            target_coords,
            link_list=target['link_list'],
            move_target=target['end_coords'],
            rotation_mask=constrain_rot,
            stop=30,
            revert_if_fail=True,
        )
        if (result is False or result is None) and self._jax_available:
            result = self._solve_ik_batch(
                robot_id, group_name, target_pos, target_rot)
        return result

    def _solve_ik(self, robot_id: int, group_name: str):
        """Solve IK for a group when its transform control is moved."""
        target = self._check_ik_guard(robot_id, group_name)
        if target is None:
            return

        control = target['control']
        target_pos = np.array(control.position)
        target_rot = vtf.SO3(control.wxyz).as_matrix()

        result = self._run_ik(robot_id, group_name, target_pos, target_rot)

        if result is not False and result is not None:
            self.redraw()
            self._sync_joint_sliders(robot_id)
            self._sync_ik_targets(robot_id, exclude=group_name)
            self._sync_numeric_inputs(robot_id, group_name)
        else:
            # IK failed: refresh cached worldcoords after revert_if_fail,
            # and update mesh handles only.  Do NOT call redraw() here
            # because it would _sync_ik_targets without exclude, which
            # snaps the user's dragged control back to the current EE pose.
            for link in target['robot_model'].link_list:
                link.update(force=True)
            for link_id, handle in self._linkid_to_handle.items():
                link = self._linkid_to_link[link_id]
                self._update_handle_pose(
                    handle, link.worldpos(), link.worldrot())

    def _solve_ik_from_numeric(self, robot_id: int, group_name: str):
        """Solve IK when numeric input fields are changed."""
        target = self._check_ik_guard(robot_id, group_name)
        if target is None:
            return

        # Build target from numeric inputs
        pos_inputs = target['pos_inputs']
        target_pos = np.array([
            pos_inputs[0].value,
            pos_inputs[1].value,
            pos_inputs[2].value,
        ])
        rot_inputs = target['rot_inputs']
        roll = np.deg2rad(rot_inputs[0].value)
        pitch = np.deg2rad(rot_inputs[1].value)
        yaw = np.deg2rad(rot_inputs[2].value)
        target_rot = rotation_matrix_from_rpy([yaw, pitch, roll])

        result = self._run_ik(robot_id, group_name, target_pos, target_rot)

        if result is not False and result is not None:
            self.redraw()
            self._sync_joint_sliders(robot_id)
            # Update the transform control to match numeric inputs
            with self._ik_update_guard():
                self._update_handle_pose(
                    target['control'], target_pos, target_rot)
            self._sync_ik_targets(robot_id, exclude=group_name)
        else:
            self._revert_ik_target(robot_id, group_name)

    def _revert_ik_target(self, robot_id: int, group_name: str):
        """Revert IK target to current end-effector pose when IK fails."""
        target = self._get_ik_target(robot_id, group_name)
        if target is None:
            return

        pos = target['end_coords'].worldpos()
        rot = target['end_coords'].worldrot()

        with self._ik_update_guard():
            self._update_handle_pose(target['control'], pos, rot)
            self._update_pose_inputs(target, pos, rot)

    def _sync_numeric_inputs(self, robot_id: int, group_name: str):
        """Sync numeric input fields with the transform control position."""
        target = self._get_ik_target(robot_id, group_name)
        if target is None:
            return

        control = target['control']
        pos = np.array(control.position)
        rot = vtf.SO3(control.wxyz).as_matrix()

        with self._ik_update_guard():
            self._update_pose_inputs(target, pos, rot)

    def _get_or_create_solver(self, robot_id, group_name,
                              target_pos, target_rot, current_angles):
        """Get an existing batch IK solver, or start async creation.

        If the solver does not exist yet, a background thread is spawned to
        create and JIT-warm it.  The caller receives ``None`` immediately so
        that the Viser callback thread is never blocked for seconds (which
        would cause race conditions with concurrent ``inverse_kinematics``
        calls that use ``revert_if_fail``).

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.
        target_pos : numpy.ndarray
            Target position (3,) used for JIT warmup.
        target_rot : numpy.ndarray
            Target rotation matrix (3, 3) used for JIT warmup.
        current_angles : numpy.ndarray
            Current joint angles used for JIT warmup.

        Returns
        -------
        callable or None
            The batch IK solver if already available, None otherwise.
        """
        # Fast path: solver already exists (no lock needed)
        robot_solvers = self._batch_ik_solvers.get(robot_id, {})
        if group_name in robot_solvers:
            return robot_solvers[group_name]

        solver_key = (robot_id, group_name)

        with self._ik_lock:
            # Double-check after acquiring lock
            robot_solvers = self._batch_ik_solvers.get(robot_id, {})
            if group_name in robot_solvers:
                return robot_solvers[group_name]
            if solver_key in self._solver_creating_keys:
                # Background thread is already creating this solver
                return None
            self._solver_creating_keys.add(solver_key)

        # Capture immutable data for the background thread
        target = self._get_ik_target(robot_id, group_name)
        if target is None:
            with self._ik_lock:
                self._solver_creating_keys.discard(solver_key)
            return None

        warmup_pos = target_pos.copy()
        warmup_rot = target_rot.copy()
        warmup_angles = current_angles.copy()
        robot_model = target['robot_model']
        link_list = target['link_list']
        end_coords = target['end_coords']

        def _create_solver_background():
            try:
                from skrobot.kinematics.differentiable import create_batch_ik_solver
                print(f"[Batch IK] Creating JAX solver for {group_name}...")
                t0 = time.time()
                solver = create_batch_ik_solver(
                    robot_model, link_list, end_coords,
                    backend_name='jax',
                )
                t1 = time.time()
                print(f"[Batch IK] Solver created "
                      f"({(t1 - t0) * 1000:.0f}ms), warming up JIT...")

                dummy_pos = warmup_pos.reshape(1, 3)
                dummy_rot = warmup_rot.reshape(1, 3, 3)
                dummy_angles = warmup_angles.reshape(1, -1)
                warmup_attempts = 50
                warmup_kwargs = dict(
                    max_iterations=20, damping=0.01,
                    pos_threshold=0.01,
                    attempts_per_pose=warmup_attempts,
                )

                t2 = time.time()
                solver(dummy_pos, dummy_rot,
                       initial_angles=dummy_angles,
                       rotation_mask=True, **warmup_kwargs)
                t3 = time.time()
                print(f"[Batch IK] JIT warmup 1/2 "
                      f"({(t3 - t2) * 1000:.0f}ms)")

                solver(dummy_pos, dummy_rot,
                       initial_angles=dummy_angles,
                       rotation_mask=False, **warmup_kwargs)
                t4 = time.time()
                print(f"[Batch IK] JIT warmup 2/2 "
                      f"({(t4 - t3) * 1000:.0f}ms)")
                print(f"[Batch IK] Solver ready for {group_name} "
                      f"(total: {(t4 - t0) * 1000:.0f}ms)")

                with self._ik_lock:
                    if robot_id not in self._batch_ik_solvers:
                        self._batch_ik_solvers[robot_id] = {}
                    self._batch_ik_solvers[robot_id][group_name] = solver
                    self._solver_creating_keys.discard(solver_key)
            except Exception as e:
                print(f"[Batch IK] Failed to create solver: {e}")
                with self._ik_lock:
                    self._solver_creating_keys.discard(solver_key)

        thread = threading.Thread(
            target=_create_solver_background, daemon=True)
        thread.start()
        return None

    def _solve_ik_batch(self, robot_id: int, group_name: str,
                        target_pos: np.ndarray, target_rot: np.ndarray):
        """Solve IK using JAX batch solver with random initial configurations.

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.
        target_pos : numpy.ndarray
            Target position (3,).
        target_rot : numpy.ndarray
            Target rotation matrix (3, 3).

        Returns
        -------
        numpy.ndarray or False
            Joint angles if successful, False otherwise.
        """
        if self._batch_ik_samples is None:
            return False

        target = self._get_ik_target(robot_id, group_name)
        if target is None:
            return False

        link_list = target['link_list']
        current_angles = np.array(
            [link.joint.joint_angle() for link in link_list])

        solver = self._get_or_create_solver(
            robot_id, group_name, target_pos, target_rot, current_angles)
        if solver is None:
            return False

        attempts = int(self._batch_ik_samples.value)
        rotation_mask = bool(self._ik_constrain_rotation.value)

        try:
            t0 = time.time()
            solutions, success_flags, errors = solver(
                target_pos.reshape(1, 3),
                target_rot.reshape(1, 3, 3),
                initial_angles=current_angles.reshape(1, -1),
                max_iterations=20,
                damping=0.01,
                rotation_mask=rotation_mask,
                pos_threshold=0.01,
                attempts_per_pose=attempts,
                use_current_angles=True,
                select_closest_to_initial=True,
            )
            t1 = time.time()
            solve_ms = (t1 - t0) * 1000
            if solve_ms > 5:
                print(
                    f"[Batch IK] Solve: {solve_ms:.1f}ms, "
                    f"attempts={attempts}, rot_mask={rotation_mask}, "
                    f"err={float(np.asarray(errors)[0]):.4f}")

            best_solution = np.asarray(solutions)[0]
            success = bool(np.asarray(success_flags)[0])
            error = float(np.asarray(errors)[0])

            if success or error < 0.05:
                for i, link in enumerate(link_list):
                    link.joint.joint_angle(float(best_solution[i]))
                return best_solution

        except Exception as e:
            print(f"[Batch IK] Exception: {e}")

        return False

    def _sync_joint_sliders(self, robot_id: int):
        """Sync joint sliders with current robot state."""
        robot_model = self._robot_models.get(robot_id)
        if robot_model is None:
            return

        with self._ik_update_guard():
            for joint_name, slider in self._joint_sliders.items():
                for joint in robot_model.joint_list:
                    if joint.name == joint_name:
                        angle = joint.joint_angle()
                        slider.value = float(np.clip(angle, slider.min, slider.max))
                        break

    def _sync_ik_targets(self, robot_id: int, exclude: Optional[str] = None):
        """Sync IK target positions with current end-effector poses."""
        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return

        with self._ik_update_guard():
            for name, target in robot_targets.items():
                if name == exclude:
                    continue
                pos = target['end_coords'].worldpos()
                rot = target['end_coords'].worldrot()
                self._update_handle_pose(target['control'], pos, rot)
                self._update_pose_inputs(target, pos, rot)

    def _render_ghost_robot(self, robot_model, joint_angles, ghost_id,
                             opacity=0.3, color=(100, 180, 255)):
        """Render a transparent ghost of the robot at given joint angles.

        Parameters
        ----------
        robot_model : RobotModel
            The robot model to ghost-render.
        joint_angles : numpy.ndarray
            Joint angle vector to pose the ghost.
        ghost_id : str
            Unique identifier for this ghost (used in scene names).
        opacity : float
            Transparency of the ghost (0.0 to 1.0).
        color : tuple
            RGB color tuple for the ghost mesh.

        Returns
        -------
        list
            List of viser handles for the ghost meshes.
        """
        handles = []
        with self._preserved_angle_vector(robot_model):
            robot_model.angle_vector(joint_angles)
            for link in robot_model.link_list:
                link.update(force=True)

            for link in robot_model.link_list:
                if isinstance(link, (Axis, PointCloudLink, LineString)):
                    continue
                if isinstance(link, Sphere):
                    link_color = link.visual_mesh.visual.face_colors[0, :3]
                    blended = (
                        np.array(link_color, dtype=float) * 0.3
                        + np.array(color, dtype=float) * 0.7
                    ).astype(np.uint8)
                    handle = self._server.scene.add_icosphere(
                        f"ghost/{ghost_id}/{link.name}",
                        radius=link.radius,
                        position=link.worldpos(),
                        color=tuple(blended),
                        opacity=opacity,
                    )
                    handles.append(handle)
                else:
                    mesh = link.concatenated_visual_mesh
                    if mesh is not None:
                        handle = self._server.scene.add_mesh_simple(
                            f"ghost/{ghost_id}/{link.name}",
                            vertices=np.array(mesh.vertices, dtype=np.float32),
                            faces=np.array(mesh.faces, dtype=np.uint32),
                            color=color,
                            opacity=opacity,
                            wxyz=matrix2quaternion(link.worldrot()),
                            position=link.worldpos(),
                            flat_shading=False,
                        )
                        handles.append(handle)
        return handles

    def _remove_ghost_handles(self, handles):
        """Remove ghost mesh handles from the scene.

        Parameters
        ----------
        handles : list
            List of viser handles to remove.
        """
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass

    def _add_waypoint(self):
        """Save current robot configuration as a waypoint."""
        if not self._robot_models:
            return
        robot_id = next(iter(self._robot_models))
        robot_model = self._robot_models[robot_id]

        angle_vector = robot_model.angle_vector().copy()

        # Store group-specific joint angles if a planning group is selected
        group_angles = {}
        if self._selected_planning_group and robot_id in self._ik_targets:
            group_name = self._selected_planning_group
            target = self._ik_targets[robot_id].get(group_name)
            if target is not None:
                link_list = target['link_list']
                group_angles[group_name] = np.array(
                    [link.joint.joint_angle() for link in link_list]
                )

        waypoint_idx = len(self._waypoints)
        waypoint = {
            'angle_vector': angle_vector,
            'group_angles': group_angles,
        }
        self._waypoints.append(waypoint)

        # Render ghost for this waypoint
        ghost_handles = self._render_ghost_robot(
            robot_model, angle_vector,
            ghost_id=f"waypoint_{waypoint_idx}",
            opacity=0.25,
            color=(100, 180, 255),
        )
        self._waypoint_ghost_handles.append(ghost_handles)

        # Clear stale planned trajectory since waypoints changed
        self._clear_trajectory_ghosts()
        self._planned_trajectory = None

        # Update GUI status
        self._update_mp_status()

    def _remove_last_waypoint(self):
        """Remove the last saved waypoint."""
        if not self._waypoints:
            return
        self._waypoints.pop()

        if self._waypoint_ghost_handles:
            handles = self._waypoint_ghost_handles.pop()
            self._remove_ghost_handles(handles)

        # Clear stale planned trajectory since waypoints changed
        self._clear_trajectory_ghosts()
        self._planned_trajectory = None

        self._update_mp_status()

    def _clear_waypoints(self):
        """Remove all saved waypoints."""
        self._waypoints.clear()

        for handles in self._waypoint_ghost_handles:
            self._remove_ghost_handles(handles)
        self._waypoint_ghost_handles.clear()

        # Also clear any planned trajectory visualization
        self._clear_trajectory_ghosts()
        self._planned_trajectory = None

        self._update_mp_status()

    def _clear_trajectory_ghosts(self):
        """Remove trajectory ghost visualizations."""
        for handles in self._trajectory_ghost_handles:
            self._remove_ghost_handles(handles)
        self._trajectory_ghost_handles.clear()

    def _update_mp_status(self):
        """Update the motion planning status display."""
        if not hasattr(self, '_mp_status_text'):
            return
        n = len(self._waypoints)
        if self._is_planning:
            self._mp_status_text.content = "*Planning...*"
        elif self._is_animating:
            self._mp_status_text.content = "*Animating...*"
        elif self._planned_trajectory is not None:
            n_steps = len(self._planned_trajectory)
            self._mp_status_text.content = (
                f"**{n} waypoints** | Trajectory: {n_steps} steps"
            )
        else:
            self._mp_status_text.content = f"**{n} waypoints**"

    def _setup_motion_planning_gui(self):
        """Set up GUI controls for motion planning."""
        self._mp_folder = self._server.gui.add_folder(
            "Motion Planning", expand_by_default=True
        )

        with self._mp_folder:
            # Planning Group dropdown (populated later when robot is added)
            self._mp_group_dropdown = self._server.gui.add_dropdown(
                "Planning Group",
                options=["(none)"],
                initial_value="(none)",
            )

            def on_group_change(_):
                val = self._mp_group_dropdown.value
                self._selected_planning_group = None if val == "(none)" else val

            self._mp_group_dropdown.on_update(on_group_change)

            # Waypoints subfolder
            with self._server.gui.add_folder(
                "Waypoints", expand_by_default=True
            ):
                add_wp_btn = self._server.gui.add_button("Add Waypoint")
                add_wp_btn.on_click(lambda _: self._add_waypoint())

                remove_wp_btn = self._server.gui.add_button("Remove Last Waypoint")
                remove_wp_btn.on_click(lambda _: self._remove_last_waypoint())

                clear_wp_btn = self._server.gui.add_button("Clear All Waypoints")
                clear_wp_btn.on_click(lambda _: self._clear_waypoints())

            # Planning Parameters subfolder (collapsed)
            with self._server.gui.add_folder(
                "Planning Parameters", expand_by_default=False
            ):
                self._mp_n_points = self._server.gui.add_slider(
                    "Interpolation Points",
                    min=5, max=50, step=1, initial_value=15,
                )
                self._mp_smoothness_weight = self._server.gui.add_number(
                    "Smoothness Weight",
                    initial_value=1.0, step=0.1,
                )
                self._mp_collision_weight = self._server.gui.add_number(
                    "Collision Weight",
                    initial_value=100.0, step=10.0,
                )
                self._mp_max_iterations = self._server.gui.add_slider(
                    "Max Iterations",
                    min=10, max=500, step=10, initial_value=100,
                )
                self._mp_solver_dropdown = self._server.gui.add_dropdown(
                    "Solver",
                    options=["jaxls", "scipy", "gradient_descent"],
                    initial_value="jaxls",
                )
                self._mp_self_collision = self._server.gui.add_checkbox(
                    "Self-Collision", initial_value=True,
                )
                self._mp_cartesian_interp = self._server.gui.add_checkbox(
                    "Cartesian Interpolation", initial_value=False,
                )
                self._mp_cartesian_weight = self._server.gui.add_number(
                    "Cartesian Path Weight",
                    initial_value=1000.0, step=100.0,
                )

            # Plan button
            plan_btn = self._server.gui.add_button("Plan Trajectory")
            plan_btn.on_click(lambda _: self._start_planning())

            # Status display
            self._mp_status_text = self._server.gui.add_markdown(
                "**0 waypoints**"
            )

            # Animation subfolder
            with self._server.gui.add_folder(
                "Animation", expand_by_default=True
            ):
                self._mp_play_btn = self._server.gui.add_button("Play")
                self._mp_play_btn.on_click(lambda _: self._toggle_animation())

                self._mp_speed = self._server.gui.add_slider(
                    "Speed", min=0.1, max=5.0, step=0.1, initial_value=1.0,
                )

                self._mp_progress = self._server.gui.add_slider(
                    "Progress", min=0.0, max=1.0, step=0.001, initial_value=0.0,
                )
                self._mp_progress.on_update(lambda _: self._scrub_trajectory())

                self._mp_show_trajectory = self._server.gui.add_checkbox(
                    "Show Trajectory", initial_value=True,
                )
                self._mp_show_trajectory.on_update(
                    lambda _: self._toggle_trajectory_visibility()
                )

    def _add_joint_sliders(self, robot_model: RobotModel):
        """Add GUI sliders for each joint in the robot model."""
        robot_id = id(robot_model)
        robot_name = getattr(robot_model, 'name', None) or f"robot_{robot_id}"

        # Collect mimic joints (joints that follow other joints)
        mimic_joints = set()
        for joint in robot_model.joint_list:
            for hook in joint._hooks:
                if isinstance(hook, _MimicJointHook):
                    mimic_joints.add(hook.other_joint)

        # Group joints by prefix (e.g., "little_module1" from "little_module1_joint1")
        joint_groups = {}
        for joint in robot_model.joint_list:
            if isinstance(joint, FixedJoint):
                continue
            if joint in mimic_joints:
                continue

            # Extract group name (everything before last underscore + identifier)
            name_parts = joint.name.rsplit('_', 1)
            if len(name_parts) == 2:
                group_name = name_parts[0]
                short_name = name_parts[1]
            else:
                group_name = "joints"
                short_name = joint.name

            if group_name not in joint_groups:
                joint_groups[group_name] = []
            joint_groups[group_name].append((joint, short_name))

        # Create folders for each group inside Joint Angles folder
        with self._joint_angles_folder:
            for group_name, joints in joint_groups.items():
                folder_name = f"{robot_name}/{group_name}"
                with self._server.gui.add_folder(
                    folder_name,
                    expand_by_default=True,
                ) as folder:
                    self._joint_folders[folder_name] = folder

                    for joint, short_name in joints:
                        min_angle = joint.min_angle
                        max_angle = joint.max_angle

                        current_angle = joint.joint_angle()

                        # Handle infinite limits (continuous joints)
                        if np.isinf(min_angle) or np.isinf(max_angle):
                            # For continuous joints, set range centered on current angle
                            min_angle = current_angle - 2 * np.pi
                            max_angle = current_angle + 2 * np.pi
                        else:
                            # Clamp current angle to valid range
                            current_angle = np.clip(current_angle, min_angle, max_angle)

                        slider = self._server.gui.add_slider(
                            short_name,
                            min=float(min_angle),
                            max=float(max_angle),
                            step=0.01,
                            initial_value=float(current_angle),
                        )

                        def make_callback(j, rid):
                            def callback(_):
                                if self._updating_from_ik:
                                    return
                                j.joint_angle(self._joint_sliders[j.name].value)
                                self.redraw()
                                self._sync_ik_targets(rid)
                            return callback

                        slider.on_update(make_callback(joint, robot_id))
                        self._joint_sliders[joint.name] = slider

        # Add joint angle export feature
        self._add_joint_angle_export(robot_model)

    def _add_joint_angle_export(self, robot_model: RobotModel):
        """Add GUI for exporting joint angles as Python code."""
        # Only initialize export GUI once
        if hasattr(self, '_export_initialized'):
            return
        self._export_initialized = True

        with self._export_folder:
            # Prefix input field
            self._export_prefix = self._server.gui.add_text(
                "Variable prefix",
                initial_value="robot_model.",
            )

            # Generate code button
            generate_button = self._server.gui.add_button("Generate Code")

            # Text area for generated code (initially empty)
            self._export_code_text = self._server.gui.add_text(
                "Code",
                initial_value="",
                multiline=True,
            )

        def generate_code_callback(_):
            prefix = self._export_prefix.value
            lines = []
            # Generate code for all robot models
            for robot_model in self._robot_models.values():
                for joint in robot_model.joint_list:
                    if isinstance(joint, FixedJoint):
                        continue
                    if joint.name not in self._joint_sliders:
                        continue
                    angle = joint.joint_angle()
                    # Format angle nicely
                    angle_deg = np.rad2deg(angle)
                    # Use np.deg2rad for cleaner code if angle is a nice degree value
                    if abs(angle_deg - round(angle_deg)) < 0.01:
                        angle_deg_int = int(round(angle_deg))
                        if angle_deg_int == 0:
                            angle_str = "0"
                        else:
                            angle_str = f"np.deg2rad({angle_deg_int})"
                    else:
                        angle_str = f"{angle:.6f}"
                    lines.append(f"{prefix}{joint.name}.joint_angle({angle_str})")
            self._export_code_text.value = "\n".join(lines)

        generate_button.on_click(generate_code_callback)

    def _detect_world_obstacles(self):
        """Detect world obstacles from objects added to the viewer.

        Returns
        -------
        list
            List of obstacle dicts with 'type', 'center', 'radius'.
        """
        from skrobot.model.primitives import Box
        from skrobot.model.primitives import Cylinder

        obstacles = []
        # Track robot model link IDs so we skip them
        robot_link_ids = set()
        for robot_model in self._robot_models.values():
            for link in robot_model.link_list:
                robot_link_ids.add(str(id(link)))

        for link_id, link in self._linkid_to_link.items():
            if link_id in robot_link_ids:
                continue
            if isinstance(link, Sphere):
                obstacles.append({
                    'type': 'sphere',
                    'center': link.worldpos().tolist(),
                    'radius': float(link.radius),
                })
            elif isinstance(link, Box):
                half_extents = np.array(link.extents) / 2
                radius = float(np.linalg.norm(half_extents))
                obstacles.append({
                    'type': 'sphere',
                    'center': link.worldpos().tolist(),
                    'radius': radius,
                })
            elif isinstance(link, Cylinder):
                half_h = link.height / 2
                r = link.radius
                radius = float(np.sqrt(r ** 2 + half_h ** 2))
                obstacles.append({
                    'type': 'sphere',
                    'center': link.worldpos().tolist(),
                    'radius': radius,
                })
        return obstacles

    def _build_cartesian_initial_trajectory(
        self, robot_model, link_list, move_target,
        waypoint_angles, n_segments, points_per_seg,
    ):
        """Build initial trajectory via Cartesian interpolation + batch IK.

        Interpolates end-effector poses linearly in Cartesian space (lerp for
        position, slerp for orientation), then solves IK for each interpolated
        pose using batch_inverse_kinematics.

        Parameters
        ----------
        robot_model : RobotModel
            Robot model.
        link_list : list
            Links in the kinematic chain.
        move_target : CascadedCoords
            End-effector coordinates.
        waypoint_angles : list of ndarray
            Group joint angles at each waypoint.
        n_segments : int
            Number of segments between waypoints.
        points_per_seg : int
            Number of interpolation points per segment.

        Returns
        -------
        initial_traj : ndarray
            Initial trajectory (total_points, n_joints).
        target_ee_positions : ndarray
            Target EE positions (total_points, 3) for Cartesian path cost.
        target_ee_rotations : ndarray
            Target EE rotation matrices (total_points, 3, 3) for Cartesian
            path cost.
        """
        from skrobot.coordinates import Coordinates
        from skrobot.coordinates.base import slerp_coordinates

        with self._preserved_angle_vector(robot_model, update_links=False):
            # Compute end-effector pose at each waypoint
            wp_coords = []
            for i, angles in enumerate(waypoint_angles):
                for link, angle in zip(link_list, angles):
                    link.joint.joint_angle(angle)
                pos = move_target.worldpos().copy()
                rot = move_target.worldrot().copy()
                wp_coords.append(Coordinates(pos=pos, rot=rot))

            # Build interpolated Cartesian poses for all segments
            all_target_coords = []
            all_target_positions = []
            all_target_rotations = []
            all_initial_group_angles = []
            for seg_idx in range(n_segments):
                c_start = wp_coords[seg_idx]
                c_end = wp_coords[seg_idx + 1]
                ga_start = waypoint_angles[seg_idx]
                ga_end = waypoint_angles[seg_idx + 1]
                for j in range(points_per_seg):
                    if seg_idx > 0 and j == 0:
                        continue
                    t = j / max(points_per_seg - 1, 1)
                    interp_c = slerp_coordinates(c_start, c_end, t)
                    all_target_coords.append(interp_c)
                    all_target_positions.append(interp_c.worldpos().copy())
                    all_target_rotations.append(interp_c.worldrot().copy())
                    all_initial_group_angles.append(
                        ga_start + t * (ga_end - ga_start)
                    )

            target_ee_positions = np.array(all_target_positions)
            target_ee_rotations = np.array(all_target_rotations)

            # Solve IK for all interpolated poses at once
            initial_angles_array = np.array(all_initial_group_angles)
            solutions, success_flags, _ = robot_model.batch_inverse_kinematics(
                all_target_coords,
                move_target=move_target,
                link_list=link_list,
                initial_angles=initial_angles_array,
                stop=50,
                thre=0.005,
                rthre=np.deg2rad(5.0),
                attempts_per_pose=1,
            )

            # Extract group joint angles from solutions
            n_joints = len(link_list)
            total_points = len(all_target_coords)
            initial_traj = np.zeros((total_points, n_joints))
            joint_indices = [
                robot_model.joint_list.index(link.joint)
                for link in link_list
            ]
            for i, (sol, success) in enumerate(
                zip(solutions, success_flags)
            ):
                if success:
                    initial_traj[i] = sol[joint_indices]
                else:
                    initial_traj[i] = all_initial_group_angles[i]

            # Force start/end to exact waypoint angles so that the
            # solver cache key is deterministic across repeated plans.
            initial_traj[0] = waypoint_angles[0]
            initial_traj[-1] = waypoint_angles[-1]

        return initial_traj, target_ee_positions, target_ee_rotations

    def _start_planning(self):
        """Start trajectory planning in a background thread."""
        if self._is_planning:
            return
        if len(self._waypoints) < 2:
            self._mp_status_text.content = (
                "**Need at least 2 waypoints to plan**"
            )
            return

        self._is_planning = True
        self._update_mp_status()

        self._planning_thread = threading.Thread(
            target=self._plan_trajectory, daemon=True
        )
        self._planning_thread.start()

    def _plan_trajectory(self):
        """Execute trajectory planning between waypoints (runs in background thread)."""
        try:
            from skrobot.planner.trajectory_optimization import TrajectoryProblem
            from skrobot.planner.trajectory_optimization.solvers import create_solver
            from skrobot.planner.trajectory_optimization.trajectory import interpolate_trajectory

            if not self._robot_models:
                return

            robot_id = next(iter(self._robot_models))
            robot_model = self._robot_models[robot_id]

            # Determine link_list and move_target from selected planning group
            link_list = None
            move_target = None
            group_name = self._selected_planning_group

            if group_name and robot_id in self._ik_targets:
                target = self._ik_targets[robot_id].get(group_name)
                if target is not None:
                    link_list = target['link_list']
                    move_target = target['end_coords']

            if link_list is None:
                # Fallback: use the first available IK group
                if robot_id in self._ik_targets and self._ik_targets[robot_id]:
                    first_group = next(iter(self._ik_targets[robot_id]))
                    target = self._ik_targets[robot_id][first_group]
                    link_list = target['link_list']
                    move_target = target['end_coords']
                    group_name = first_group

            if link_list is None:
                self._mp_status_text.content = "**No planning group available**"
                self._is_planning = False
                return

            # Get planning parameters from GUI
            n_points = int(self._mp_n_points.value)
            smoothness_w = float(self._mp_smoothness_weight.value)
            collision_w = float(self._mp_collision_weight.value)
            max_iters = int(self._mp_max_iterations.value)
            solver_type = self._mp_solver_dropdown.value
            use_self_collision = self._mp_self_collision.value
            use_cartesian = self._mp_cartesian_interp.value

            # Extract group joint angles for each waypoint
            waypoint_angles = []
            for wp in self._waypoints:
                if group_name in wp['group_angles']:
                    waypoint_angles.append(wp['group_angles'][group_name])
                else:
                    # Extract from full angle_vector
                    with self._preserved_angle_vector(
                        robot_model, update_links=False
                    ):
                        robot_model.angle_vector(wp['angle_vector'])
                        angles = np.array(
                            [link.joint.joint_angle() for link in link_list]
                        )
                        waypoint_angles.append(angles)

            # Detect world obstacles
            world_obstacles = self._detect_world_obstacles()

            # Determine collision link list (use the planning group's links)
            coll_link_list = list(link_list)

            # Build a single trajectory spanning all waypoints.
            # Intermediate waypoints are pinned via equality constraints
            # so that only one analyze()+solve() call is needed.
            n_segments = len(waypoint_angles) - 1
            points_per_seg = n_points
            total_points = (points_per_seg - 1) * n_segments + 1

            # Build initial trajectory
            cartesian_target_positions = None
            cartesian_target_rotations = None
            if use_cartesian and move_target is not None:
                initial_traj, cartesian_target_positions, \
                    cartesian_target_rotations = \
                    self._build_cartesian_initial_trajectory(
                        robot_model, link_list, move_target,
                        waypoint_angles, n_segments, points_per_seg,
                    )
            else:
                # Joint-space linear interpolation
                segments = []
                for seg_idx in range(n_segments):
                    seg = interpolate_trajectory(
                        waypoint_angles[seg_idx],
                        waypoint_angles[seg_idx + 1],
                        points_per_seg,
                    )
                    if seg_idx > 0:
                        seg = seg[1:]
                    segments.append(seg)
                initial_traj = np.concatenate(segments, axis=0)

            problem = TrajectoryProblem(
                robot_model=robot_model,
                link_list=link_list,
                n_waypoints=total_points,
                dt=0.1,
                move_target=move_target,
            )

            problem.add_smoothness_cost(weight=smoothness_w)
            problem.add_acceleration_cost(weight=smoothness_w * 0.1)

            if cartesian_target_positions is not None:
                cartesian_w = float(self._mp_cartesian_weight.value)
                problem.add_cartesian_path_cost(
                    target_positions=cartesian_target_positions,
                    target_rotations=cartesian_target_rotations,
                    weight=cartesian_w,
                    rotation_weight=1.0,
                )

            if world_obstacles and collision_w > 0:
                problem.add_collision_cost(
                    collision_link_list=coll_link_list,
                    world_obstacles=world_obstacles,
                    weight=collision_w,
                    activation_distance=0.05,
                )
                if use_self_collision:
                    problem.add_self_collision_cost(
                        weight=collision_w,
                        activation_distance=0.02,
                    )
            elif use_self_collision and collision_w > 0:
                dummy_obstacles = [{
                    'type': 'sphere',
                    'center': [0.0, 0.0, -1000.0],
                    'radius': 0.001,
                }]
                problem.add_collision_cost(
                    collision_link_list=coll_link_list,
                    world_obstacles=dummy_obstacles,
                    weight=0.0,
                    activation_distance=0.0,
                )
                problem.add_self_collision_cost(
                    weight=collision_w,
                    activation_distance=0.02,
                )

            # Pin intermediate waypoints as equality constraints
            for wp_i in range(1, n_segments):
                traj_idx = (points_per_seg - 1) * wp_i
                problem.add_waypoint_constraint(
                    traj_idx, waypoint_angles[wp_i]
                )

            if (self._cached_mp_solver is None
                    or self._cached_mp_solver_type != solver_type):
                solver = create_solver(
                    solver_type, max_iterations=max_iters, verbose=False
                )
                self._cached_mp_solver = solver
                self._cached_mp_solver_type = solver_type
            else:
                solver = self._cached_mp_solver
                solver.max_iterations = max_iters
            result = solver.solve(problem, initial_traj)
            self._planned_trajectory = result.trajectory

            # Store planning metadata
            self._planning_link_list = link_list
            self._planning_group_name = group_name
            self._planning_robot_id = robot_id

            # Render trajectory ghosts
            self._clear_trajectory_ghosts()
            if hasattr(self, '_mp_show_trajectory') and self._mp_show_trajectory.value:
                self._render_trajectory_ghosts()

            self._mp_status_text.content = (
                f"**Planning complete!** {len(self._planned_trajectory)} steps"
            )

        except Exception as e:
            self._mp_status_text.content = f"**Planning failed:** {e}"
            print(f"[Motion Planning] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_planning = False

    def _render_trajectory_ghosts(self):
        """Render ghost robots along the planned trajectory."""
        if self._planned_trajectory is None:
            return
        if not self._robot_models:
            return

        robot_id = self._planning_robot_id
        robot_model = self._robot_models.get(robot_id)
        if robot_model is None:
            return

        link_list = self._planning_link_list
        n_steps = len(self._planned_trajectory)

        # Show a subset of ghosts (every N-th step) to avoid clutter
        max_ghosts = 8
        step_interval = max(1, n_steps // max_ghosts)
        ghost_indices = list(range(0, n_steps, step_interval))
        # Always include last step
        if ghost_indices[-1] != n_steps - 1:
            ghost_indices.append(n_steps - 1)

        with self._preserved_angle_vector(robot_model):
            for ghost_idx, traj_idx in enumerate(ghost_indices):
                step_angles = self._planned_trajectory[traj_idx]
                # Apply group angles to robot
                for i, link in enumerate(link_list):
                    link.joint.joint_angle(float(step_angles[i]))
                av = robot_model.angle_vector().copy()

                # Compute opacity: fade from transparent to less transparent
                t = traj_idx / max(n_steps - 1, 1)
                opacity = 0.1 + 0.2 * t

                handles = self._render_ghost_robot(
                    robot_model, av,
                    ghost_id=f"traj_{ghost_idx}",
                    opacity=opacity,
                    color=(255, 165, 0),  # Orange for trajectory
                )
                self._trajectory_ghost_handles.append(handles)

    def _toggle_trajectory_visibility(self):
        """Toggle visibility of trajectory ghost robots."""
        if not hasattr(self, '_mp_show_trajectory'):
            return
        visible = self._mp_show_trajectory.value
        if visible and not self._trajectory_ghost_handles and self._planned_trajectory is not None:
            self._render_trajectory_ghosts()
        elif not visible:
            self._clear_trajectory_ghosts()

    def _toggle_animation(self):
        """Toggle animation playback."""
        if self._is_animating:
            self._stop_animation()
        else:
            self._start_animation()

    def _start_animation(self):
        """Start animation playback in a background thread."""
        if self._planned_trajectory is None:
            self._mp_status_text.content = "**No trajectory to animate**"
            return
        if self._is_animating:
            return

        self._is_animating = True
        self._mp_play_btn.name = "Stop"
        self._update_mp_status()

        self._animation_thread = threading.Thread(
            target=self._animate_trajectory, daemon=True
        )
        self._animation_thread.start()

    def _stop_animation(self):
        """Stop animation playback."""
        self._is_animating = False
        self._mp_play_btn.name = "Play"
        self._update_mp_status()

    def _animate_trajectory(self):
        """Animate the planned trajectory step by step (runs in background thread)."""
        try:
            if self._planned_trajectory is None:
                return

            robot_id = self._planning_robot_id
            robot_model = self._robot_models.get(robot_id)
            if robot_model is None:
                return

            link_list = self._planning_link_list
            n_steps = len(self._planned_trajectory)

            # Start from current progress position
            start_progress = self._mp_progress.value
            start_idx = int(start_progress * (n_steps - 1))

            for step_idx in range(start_idx, n_steps):
                if not self._is_animating:
                    break

                step_angles = self._planned_trajectory[step_idx]

                # Apply joint angles
                for i, link in enumerate(link_list):
                    link.joint.joint_angle(float(step_angles[i]))

                self.redraw()
                self._sync_joint_sliders(robot_id)

                # Update progress slider
                progress = step_idx / max(n_steps - 1, 1)
                with self._ik_update_guard():
                    self._mp_progress.value = progress

                # Compute sleep based on speed
                speed = self._mp_speed.value
                time.sleep(0.05 / speed)

        except Exception as e:
            print(f"[Animation] Error: {e}")
        finally:
            self._is_animating = False
            self._mp_play_btn.name = "Play"
            self._update_mp_status()

    def _scrub_trajectory(self):
        """Jump to a specific position in the trajectory based on progress slider."""
        if self._updating_from_ik:
            return
        if self._planned_trajectory is None:
            return
        if self._is_animating:
            return

        robot_id = self._planning_robot_id
        robot_model = self._robot_models.get(robot_id)
        if robot_model is None:
            return

        link_list = self._planning_link_list
        n_steps = len(self._planned_trajectory)
        progress = self._mp_progress.value

        step_idx = int(progress * (n_steps - 1))
        step_idx = np.clip(step_idx, 0, n_steps - 1)

        step_angles = self._planned_trajectory[step_idx]
        for i, link in enumerate(link_list):
            link.joint.joint_angle(float(step_angles[i]))

        self.redraw()
        self._sync_joint_sliders(robot_id)

    def _setup_motion_planning_callbacks(self, robot_model):
        """Set up motion planning integration after a robot is added.

        Parameters
        ----------
        robot_model : RobotModel
            The robot model that was just added.
        """
        robot_id = id(robot_model)

        # Populate planning group dropdown from IK groups
        if robot_id in self._ik_targets and self._ik_targets[robot_id]:
            group_names = list(self._ik_targets[robot_id].keys())
            options = ["(none)"] + group_names
            self._mp_group_dropdown.options = options
            if len(group_names) == 1:
                self._mp_group_dropdown.value = group_names[0]
                self._selected_planning_group = group_names[0]
            elif len(group_names) > 1:
                # Default to first arm-like group if available
                arm_groups = [g for g in group_names if 'arm' in g.lower()]
                if arm_groups:
                    self._mp_group_dropdown.value = arm_groups[0]
                    self._selected_planning_group = arm_groups[0]
                else:
                    self._mp_group_dropdown.value = group_names[0]
                    self._selected_planning_group = group_names[0]

    def draw_grid(self, width: float = 20.0, height: float = -0.001):
        self._server.scene.add_grid(
            "/grid",
            width=20.0,
            height=20.0,
            position=np.array([0.0, 0.0, -0.01]),
        )

    def _add_link(self, link: Link):
        assert isinstance(link, Link)
        link_id = str(id(link))
        if link_id in self._linkid_to_handle:
            return

        handle = None
        if isinstance(link, Sphere):
            # Although sphere can be treated as trimesh, naively rendering
            # it requires high cost. Therefore, we use an analytic sphere.
            color = link.visual_mesh.visual.face_colors[0, :3]
            alpha = link.visual_mesh.visual.face_colors[0, 3]
            if alpha > 1.0:
                alpha = alpha / 255.0
            handle = self._server.scene.add_icosphere(
                link.name,
                radius=link.radius,
                position=link.worldpos(),
                color=color,
                opacity=alpha)
        elif isinstance(link, Axis):
            handle = self._server.scene.add_frame(
                    link.name,
                    axes_length=link.axis_length,
                    axes_radius=link.axis_radius,
                    wxyz=matrix2quaternion(link.worldrot()),
                    position=link.worldpos(),
                )
        elif isinstance(link, PointCloudLink):
            mesh = link.visual_mesh
            assert isinstance(mesh, trimesh.PointCloud)
            if len(mesh.colors) > 0:
                colors = mesh.colors[:, :3]
            else:
                colors = np.zeros(3)
            self._server.scene.add_point_cloud(
                    link.name,
                    points=mesh.vertices,
                    colors=colors,
                    point_size=0.002,  # TODO(HiroIshida): configurable
                )
        elif isinstance(link, LineString):
            raise NotImplementedError("not implemented yet")
        else:
            mesh = link.concatenated_visual_mesh
            if mesh is not None:
                handle = self._server.scene.add_mesh_trimesh(
                        link.name,
                        mesh=mesh,
                        wxyz=matrix2quaternion(link.worldrot()),
                        position=link.worldpos(),
                    )

        if handle is not None:
            self._linkid_to_link[link_id] = link
            self._linkid_to_handle[link_id] = handle

    def _ensure_gui_initialized(self):
        """Initialize GUI section folders in the correct order."""
        if hasattr(self, '_gui_initialized'):
            return
        self._gui_initialized = True

        # Create section folders in desired order
        self._joint_angles_folder = self._server.gui.add_folder(
            "Joint Angles", expand_by_default=True
        )
        if self._enable_ik:
            self._ik_controls_folder = self._server.gui.add_folder(
                "IK Controls", expand_by_default=True
            )
            with self._ik_controls_folder:
                self._ik_constrain_rotation = self._server.gui.add_checkbox(
                    "Constrain Rotation", initial_value=True
                )
                self._batch_ik_samples = self._server.gui.add_slider(
                    "Batch IK Attempts",
                    min=1,
                    max=2000,
                    step=10,
                    initial_value=50,
                )
                self._server.gui.add_markdown(
                    "*JAX batch IK: faster interactive solving*"
                )
        self._export_folder = self._server.gui.add_folder(
            "Export Joint Angles", expand_by_default=False
        )

        if self._enable_motion_planning:
            self._setup_motion_planning_gui()

    def add(self, geometry: Union[Link, CascadedLink]):
        if isinstance(geometry, Link):
            self._add_link(geometry)
        elif isinstance(geometry, CascadedLink):
            for link in geometry.link_list:
                self._add_link(link)
            if isinstance(geometry, RobotModel):
                self._ensure_gui_initialized()
                self._add_joint_sliders(geometry)
                if self._enable_ik:
                    self._setup_ik_controls(geometry)
                if self._enable_motion_planning:
                    self._setup_motion_planning_callbacks(geometry)
        else:
            raise TypeError("geometry must be Link or CascadedLink")

    def show(self):
        host = self._server.get_host()
        port = self._server.get_port()
        # 0.0.0.0 is not a valid browser URL
        if host == "0.0.0.0":
            host = "localhost"
        url = f"http://{host}:{port}"
        webbrowser.open(url)

    def redraw(self):
        for link_id, handle in self._linkid_to_handle.items():
            link = self._linkid_to_link[link_id]
            self._update_handle_pose(
                handle, link.worldpos(), link.worldrot())

        # Sync all IK targets if IK is enabled
        if self._enable_ik:
            for robot_id in self._ik_targets:
                self._sync_ik_targets(robot_id)

    def delete(self, geometry: Union[Link, CascadedLink]):
        if isinstance(geometry, Link):
            links = [geometry]
        elif isinstance(geometry, CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError("geometry must be Link or CascadedLink")

        for link in links:
            link_id = str(id(link))
            if link_id not in self._linkid_to_handle:
                continue
            handle = self._linkid_to_handle[link_id]
            handle.remove()
            self._linkid_to_link.pop(link_id)
            self._linkid_to_handle.pop(link_id)
