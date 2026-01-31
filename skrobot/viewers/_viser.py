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
        self._jax_warning_shown = False

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

    def _solve_ik(self, robot_id: int, group_name: str):
        """Solve IK for a group when its target is moved."""
        if self._updating_from_ik:
            return

        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return

        target = robot_targets.get(group_name)
        if target is None:
            return

        robot_model = target['robot_model']
        control = target['control']
        target_pos = np.array(control.position)
        target_rot = vtf.SO3(control.wxyz).as_matrix()
        target_coords = Coordinates(pos=target_pos, rot=target_rot)

        constrain_rot = self._ik_constrain_rotation.value

        result = robot_model.inverse_kinematics(
            target_coords,
            link_list=target['link_list'],
            move_target=target['end_coords'],
            rotation_axis=constrain_rot,
            stop=30,
            revert_if_fail=True,
        )

        # If regular IK fails, try batch IK
        if result is False or result is None:
            result = self._solve_ik_batch(robot_id, group_name, target_pos, target_rot)
            if result is not False and result is not None:
                pass

        if result is not False and result is not None:
            self.redraw()
            self._sync_joint_sliders(robot_id)
            # Always exclude the current target to prevent it from moving
            # when IK doesn't reach the exact target position
            self._sync_ik_targets(robot_id, exclude=group_name)
            # Sync numeric inputs with the target position
            self._sync_numeric_inputs(robot_id, group_name)
        else:
            # IK failed: force update all link coordinates and redraw
            # This ensures cached worldcoords are recomputed after revert
            for link in robot_model.link_list:
                link.update(force=True)
            self.redraw()

    def _solve_ik_from_numeric(self, robot_id: int, group_name: str):
        """Solve IK when numeric input fields are changed."""
        if self._updating_from_ik:
            return

        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return

        target = robot_targets.get(group_name)
        if target is None:
            return

        robot_model = target['robot_model']

        # Get position from numeric inputs
        pos_inputs = target['pos_inputs']
        target_pos = np.array([
            pos_inputs[0].value,
            pos_inputs[1].value,
            pos_inputs[2].value
        ])

        # Get rotation from numeric inputs (degrees to radians)
        rot_inputs = target['rot_inputs']
        roll = np.deg2rad(rot_inputs[0].value)
        pitch = np.deg2rad(rot_inputs[1].value)
        yaw = np.deg2rad(rot_inputs[2].value)

        # Convert RPY to rotation matrix
        target_rot = rotation_matrix_from_rpy([yaw, pitch, roll])
        target_coords = Coordinates(pos=target_pos, rot=target_rot)

        constrain_rot = self._ik_constrain_rotation.value

        result = robot_model.inverse_kinematics(
            target_coords,
            link_list=target['link_list'],
            move_target=target['end_coords'],
            rotation_axis=constrain_rot,
            stop=30,
            revert_if_fail=True,
        )

        # If regular IK fails, try batch IK
        if result is False or result is None:
            result = self._solve_ik_batch(robot_id, group_name, target_pos, target_rot)

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
            # Sync other IK targets
            self._sync_ik_targets(robot_id, exclude=group_name)
        else:
            # IK failed: revert numeric inputs and control to current pose
            self._revert_ik_target(robot_id, group_name)

    def _revert_ik_target(self, robot_id: int, group_name: str):
        """Revert IK target to current end-effector pose when IK fails."""
        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return

        target = robot_targets.get(group_name)
        if target is None:
            return

        # Get current end-effector pose
        pos = target['end_coords'].worldpos()
        rot = target['end_coords'].worldrot()

        self._updating_from_ik = True
        try:
            # Revert transform control
            target['control'].position = pos
            target['control'].wxyz = matrix2quaternion(rot)

            # Revert numeric inputs
            if 'pos_inputs' in target:
                pos_inputs = target['pos_inputs']
                pos_inputs[0].value = float(pos[0])
                pos_inputs[1].value = float(pos[1])
                pos_inputs[2].value = float(pos[2])

            if 'rot_inputs' in target:
                roll, pitch, yaw = matrix2rpy(rot)
                rot_inputs = target['rot_inputs']
                rot_inputs[0].value = float(np.rad2deg(roll))
                rot_inputs[1].value = float(np.rad2deg(pitch))
                rot_inputs[2].value = float(np.rad2deg(yaw))
        finally:
            self._updating_from_ik = False

    def _sync_numeric_inputs(self, robot_id: int, group_name: str):
        """Sync numeric input fields with the transform control position."""
        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return

        target = robot_targets.get(group_name)
        if target is None:
            return

        control = target['control']
        pos = np.array(control.position)
        rot = vtf.SO3(control.wxyz).as_matrix()

        # Update position inputs
        self._updating_from_ik = True
        try:
            pos_inputs = target['pos_inputs']
            pos_inputs[0].value = float(pos[0])
            pos_inputs[1].value = float(pos[1])
            pos_inputs[2].value = float(pos[2])

            # Convert rotation matrix to RPY
            # matrix2rpy returns [roll, pitch, yaw]
            roll, pitch, yaw = matrix2rpy(rot)

            # Update rotation inputs (radians to degrees)
            rot_inputs = target['rot_inputs']
            rot_inputs[0].value = float(np.rad2deg(roll))
            rot_inputs[1].value = float(np.rad2deg(pitch))
            rot_inputs[2].value = float(np.rad2deg(yaw))
        finally:
            self._updating_from_ik = False

    def _solve_ik_batch(self, robot_id: int, group_name: str,
                        target_pos: np.ndarray, target_rot: np.ndarray):
        """Solve IK using JAX batch solver with random initial configurations.

        Parameters
        ----------
        robot_id : int
            Robot model ID.
        group_name : str
            Name of the IK group.
        target_pos : np.ndarray
            Target position (3,).
        target_rot : np.ndarray
            Target rotation matrix (3, 3).

        Returns
        -------
        np.ndarray or False
            Joint angles if successful, False otherwise.
        """
        if self._batch_ik_samples is None:
            return False

        robot_targets = self._ik_targets.get(robot_id)
        if robot_targets is None:
            return False

        target = robot_targets.get(group_name)
        if target is None:
            return False

        robot_model = target['robot_model']
        link_list = target['link_list']
        move_target = target['end_coords']

        # Get or create batch IK solver for this group
        robot_solvers = self._batch_ik_solvers.get(robot_id, {})
        if group_name not in robot_solvers:
            try:
                from skrobot.backend import list_backends
                if 'jax' not in list_backends():
                    if not self._jax_warning_shown:
                        print("[Batch IK] JAX not available. "
                              "Install with: pip install jax jaxlib")
                        self._jax_warning_shown = True
                    return False
                from skrobot.kinematics.differentiable import create_batch_ik_solver
                solver = create_batch_ik_solver(
                    robot_model, link_list, move_target,
                    backend_name='jax'
                )
                self._batch_ik_solvers[robot_id][group_name] = solver
            except Exception:
                return False
        else:
            solver = robot_solvers[group_name]

        # Get batch size from slider
        batch_size = int(self._batch_ik_samples.value)

        # Generate random initial configurations
        lower = np.array(solver.joint_limits_lower)
        upper = np.array(solver.joint_limits_upper)
        n_joints = solver.n_joints

        # Use current joint angles as one of the initial guesses
        current_angles = np.array([link.joint.joint_angle() for link in link_list])
        random_init = np.random.uniform(lower, upper, size=(batch_size - 1, n_joints))
        initial_angles = np.vstack([current_angles.reshape(1, -1), random_init])

        # Prepare target arrays (same target for all batch)
        target_positions = np.tile(target_pos, (batch_size, 1))
        target_rotations = np.tile(target_rot, (batch_size, 1, 1))

        # Rotation weight based on constraint setting
        constrain_rot = self._ik_constrain_rotation.value
        rot_weight = 0.1 if constrain_rot else 0.0

        # Solve batch IK
        try:
            solutions, success_flags, errors = solver(
                target_positions,
                target_rotations,
                initial_angles=initial_angles,
                max_iterations=100,
                learning_rate=0.1,
                pos_weight=1.0,
                rot_weight=rot_weight,
                pos_threshold=0.01,
            )

            # Convert JAX arrays to numpy if needed
            success_flags = np.asarray(success_flags)
            errors = np.asarray(errors)
            solutions = np.asarray(solutions)

            min_error = np.min(errors)
            best_idx = np.argmin(errors)

            # Find the best successful solution
            if np.any(success_flags):
                successful_indices = np.where(success_flags)[0]
                best_idx = successful_indices[np.argmin(errors[successful_indices])]
                best_solution = solutions[best_idx]

                # Apply the solution to the robot
                for i, link in enumerate(link_list):
                    link.joint.joint_angle(float(best_solution[i]))

                return best_solution
            else:
                # Even if no "success", use the best solution if error is reasonable
                if min_error < 0.02:  # 2cm threshold for fallback
                    best_solution = solutions[best_idx]
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

                # Update numeric inputs if they exist
                if 'pos_inputs' in target:
                    pos_inputs = target['pos_inputs']
                    pos_inputs[0].value = float(pos[0])
                    pos_inputs[1].value = float(pos[1])
                    pos_inputs[2].value = float(pos[2])

                if 'rot_inputs' in target:
                    # matrix2rpy returns [roll, pitch, yaw]
                    roll, pitch, yaw = matrix2rpy(rot)
                    rot_inputs = target['rot_inputs']
                    rot_inputs[0].value = float(np.rad2deg(roll))
                    rot_inputs[1].value = float(np.rad2deg(pitch))
                    rot_inputs[2].value = float(np.rad2deg(yaw))
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
                    "Batch IK Samples",
                    min=10,
                    max=500,
                    step=10,
                    initial_value=100,
                )
                self._server.gui.add_markdown(
                    "*Batch IK runs when regular IK fails*"
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
