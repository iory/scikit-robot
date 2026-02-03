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
    """

    def __init__(
        self,
        draw_grid: bool = True,
        enable_ik: bool = False,
    ):
        self._server = viser.ViserServer()
        self._linkid_to_handle = dict()
        self._linkid_to_link = dict()
        self._is_active = True
        self._joint_sliders = dict()
        self._joint_folders = dict()

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

    @property
    def is_active(self) -> bool:
        return self._is_active

    def close(self):
        self._is_active = False
        self._server.stop()

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
        # Map group names to possible attribute names
        attr_candidates = []
        if group_name == 'arm':
            attr_candidates = ['arm_end_coords', 'rarm_end_coords', 'end_coords']
        elif group_name == 'right_arm':
            attr_candidates = ['rarm_end_coords', 'right_arm_end_coords']
        elif group_name == 'left_arm':
            attr_candidates = ['larm_end_coords', 'left_arm_end_coords']
        elif group_name == 'right_leg':
            attr_candidates = ['rleg_end_coords', 'right_leg_end_coords']
        elif group_name == 'left_leg':
            attr_candidates = ['lleg_end_coords', 'left_leg_end_coords']
        elif group_name == 'head':
            attr_candidates = ['head_end_coords']
        elif group_name == 'torso':
            attr_candidates = ['torso_end_coords']
        else:
            attr_candidates = [f'{group_name}_end_coords']

        # Check direct attributes on robot model
        for attr_name in attr_candidates:
            if hasattr(robot_model, attr_name):
                end_coords = getattr(robot_model, attr_name)
                if isinstance(end_coords, CascadedCoords):
                    return end_coords

        # Check if robot model has a limb attribute with end_coords
        limb_attr_map = {
            'arm': ['arm', 'rarm'],
            'right_arm': ['rarm', 'right_arm'],
            'left_arm': ['larm', 'left_arm'],
            'right_leg': ['rleg', 'right_leg'],
            'left_leg': ['lleg', 'left_leg'],
            'head': ['head'],
            'torso': ['torso'],
        }
        limb_attrs = limb_attr_map.get(group_name, [group_name])

        for limb_attr in limb_attrs:
            try:
                if hasattr(robot_model, limb_attr):
                    limb = getattr(robot_model, limb_attr)
                    if hasattr(limb, 'end_coords'):
                        end_coords = limb.end_coords
                        if isinstance(end_coords, CascadedCoords):
                            return end_coords
            except NotImplementedError:
                # Some robot models raise NotImplementedError for unimplemented limbs
                continue

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
        # Map group names to possible limb attribute names
        limb_attr_map = {
            'arm': ['arm', 'rarm'],
            'right_arm': ['rarm', 'right_arm'],
            'left_arm': ['larm', 'left_arm'],
            'right_leg': ['rleg', 'right_leg'],
            'left_leg': ['lleg', 'left_leg'],
            'head': ['head'],
            'torso': ['torso'],
        }
        limb_attrs = limb_attr_map.get(group_name, [group_name])

        for limb_attr in limb_attrs:
            try:
                if hasattr(robot_model, limb_attr):
                    limb = getattr(robot_model, limb_attr)
                    if hasattr(limb, 'link_list') and limb.link_list:
                        return list(limb.link_list)
            except NotImplementedError:
                # Some robot models raise NotImplementedError for unimplemented limbs
                continue

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
        if self._updating_from_ik:
            return
        if self._is_throttled(robot_id, group_name):
            return

        target = self._get_ik_target(robot_id, group_name)
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
                handle.position = link.worldpos()
                handle.wxyz = matrix2quaternion(link.worldrot())

    def _solve_ik_from_numeric(self, robot_id: int, group_name: str):
        """Solve IK when numeric input fields are changed."""
        if self._updating_from_ik:
            return
        if self._is_throttled(robot_id, group_name):
            return

        target = self._get_ik_target(robot_id, group_name)
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
            self._updating_from_ik = True
            try:
                control = target['control']
                control.position = target_pos
                control.wxyz = matrix2quaternion(target_rot)
            finally:
                self._updating_from_ik = False
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

        self._updating_from_ik = True
        try:
            target['control'].position = pos
            target['control'].wxyz = matrix2quaternion(rot)
            self._update_pose_inputs(target, pos, rot)
        finally:
            self._updating_from_ik = False

    def _sync_numeric_inputs(self, robot_id: int, group_name: str):
        """Sync numeric input fields with the transform control position."""
        target = self._get_ik_target(robot_id, group_name)
        if target is None:
            return

        control = target['control']
        pos = np.array(control.position)
        rot = vtf.SO3(control.wxyz).as_matrix()

        self._updating_from_ik = True
        try:
            self._update_pose_inputs(target, pos, rot)
        finally:
            self._updating_from_ik = False

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

        self._updating_from_ik = True
        try:
            for joint_name, slider in self._joint_sliders.items():
                for joint in robot_model.joint_list:
                    if joint.name == joint_name:
                        angle = joint.joint_angle()
                        slider.value = float(np.clip(angle, slider.min, slider.max))
                        break
        finally:
            self._updating_from_ik = False

    def _sync_ik_targets(self, robot_id: int, exclude: Optional[str] = None):
        """Sync IK target positions with current end-effector poses."""
        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return

        self._updating_from_ik = True
        try:
            for name, target in robot_targets.items():
                if name == exclude:
                    continue
                pos = target['end_coords'].worldpos()
                rot = target['end_coords'].worldrot()
                target['control'].position = pos
                target['control'].wxyz = matrix2quaternion(rot)
                self._update_pose_inputs(target, pos, rot)
        finally:
            self._updating_from_ik = False

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
            handle.position = link.worldpos()
            handle.wxyz = matrix2quaternion(link.worldrot())

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
