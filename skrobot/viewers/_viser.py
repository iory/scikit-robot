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
from skrobot.model.joint import _MimicJointHook
from skrobot.model.joint import FixedJoint
from skrobot.model.link import Link
from skrobot.model.primitives import Axis
from skrobot.model.primitives import LineString
from skrobot.model.primitives import PointCloudLink
from skrobot.model.primitives import Sphere
from skrobot.model.robot_model import CascadedLink
from skrobot.model.robot_model import RobotModel


class ViserVisualizer:
    """Viser-based 3D visualizer for scikit-robot.

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
        self._ik_targets: Dict[str, dict] = {}
        self._robot_model: Optional[RobotModel] = None
        self._updating_from_ik = False

    @property
    def is_active(self) -> bool:
        return self._is_active

    def close(self):
        self._is_active = False
        self._server.stop()

    def _setup_ik_controls(self, robot_model: RobotModel):
        """Set up interactive IK controls for detected end-effectors."""
        from skrobot.urdf.robot_class_generator import generate_groups_from_geometry

        self._robot_model = robot_model

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

        # Add GUI
        self._server.gui.add_markdown("## IK Controls")
        self._ik_constrain_rotation = self._server.gui.add_checkbox(
            "Constrain Rotation", initial_value=True
        )

        # Create transform control for each group
        for group_name, (group_data, ec_info) in ik_groups.items():
            link_names = group_data.get('links', [])

            # Get link objects
            link_list = []
            for name in link_names:
                for link in robot_model.link_list:
                    if link.name == name:
                        link_list.append(link)
                        break

            if not link_list:
                continue

            # Create end_coords
            parent_link_name = ec_info.get('parent_link', link_names[-1])
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
                f"ik_target/{group_name}",
                scale=0.1,
                position=ee_pos,
                wxyz=matrix2quaternion(ee_rot),
            )

            # Add visibility checkbox for this target
            visibility_checkbox = self._server.gui.add_checkbox(
                f"Show {group_name}", initial_value=True
            )

            def make_visibility_callback(ctrl, checkbox):
                def callback(_):
                    ctrl.visible = checkbox.value
                return callback

            visibility_checkbox.on_update(
                make_visibility_callback(control, visibility_checkbox)
            )

            # Store target info
            self._ik_targets[group_name] = {
                'link_list': link_list,
                'end_coords': end_coords,
                'control': control,
                'visibility_checkbox': visibility_checkbox,
            }

            # Callback for when control is moved
            def make_ik_callback(gname):
                def callback(_):
                    self._solve_ik(gname)
                return callback

            control.on_update(make_ik_callback(group_name))

    def _solve_ik(self, group_name: str):
        """Solve IK for a group when its target is moved."""
        if self._robot_model is None or self._updating_from_ik:
            return

        target = self._ik_targets.get(group_name)
        if target is None:
            return

        control = target['control']
        target_pos = np.array(control.position)
        target_rot = vtf.SO3(control.wxyz).as_matrix()
        target_coords = Coordinates(pos=target_pos, rot=target_rot)

        constrain_rot = self._ik_constrain_rotation.value

        result = self._robot_model.inverse_kinematics(
            target_coords,
            link_list=target['link_list'],
            move_target=target['end_coords'],
            rotation_axis=constrain_rot,
            stop=30,
            revert_if_fail=True,
        )

        if result is not False and result is not None:
            self.redraw()
            self._sync_joint_sliders()
            self._sync_ik_targets(exclude=group_name)

    def _sync_joint_sliders(self):
        """Sync joint sliders with current robot state."""
        self._updating_from_ik = True
        try:
            for joint_name, slider in self._joint_sliders.items():
                for joint in self._robot_model.joint_list:
                    if joint.name == joint_name:
                        angle = joint.joint_angle()
                        slider.value = float(np.clip(angle, slider.min, slider.max))
                        break
        finally:
            self._updating_from_ik = False

    def _sync_ik_targets(self, exclude: Optional[str] = None):
        """Sync IK target positions with current end-effector poses."""
        self._updating_from_ik = True
        try:
            for name, target in self._ik_targets.items():
                if name == exclude:
                    continue
                pos = target['end_coords'].worldpos()
                rot = target['end_coords'].worldrot()
                target['control'].position = pos
                target['control'].wxyz = matrix2quaternion(rot)
        finally:
            self._updating_from_ik = False

    def _add_joint_sliders(self, robot_model: RobotModel):
        """Add GUI sliders for each joint in the robot model."""
        self._server.gui.add_markdown("## Joint Angles")

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
            # Skip joints whose child link has no visual/collision mesh
            child_link = joint.child_link
            if child_link is None:
                continue
            visual = child_link.visual_mesh
            collision = child_link.collision_mesh
            has_visual = visual is not None and (not isinstance(visual, list) or len(visual) > 0)
            has_collision = collision is not None and (not isinstance(collision, list) or len(collision) > 0)
            if not has_visual and not has_collision:
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

        # Create folders for each group
        self._joint_folders = {}
        for group_name, joints in joint_groups.items():
            with self._server.gui.add_folder(
                group_name,
                expand_by_default=True,
            ) as folder:
                self._joint_folders[group_name] = folder

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

                    def make_callback(j):
                        def callback(_):
                            if self._updating_from_ik:
                                return
                            j.joint_angle(self._joint_sliders[j.name].value)
                            self.redraw()
                            self._sync_ik_targets()
                        return callback

                    slider.on_update(make_callback(joint))
                    self._joint_sliders[joint.name] = slider

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

    def add(self, geometry: Union[Link, CascadedLink]):
        if isinstance(geometry, Link):
            self._add_link(geometry)
        elif isinstance(geometry, CascadedLink):
            for link in geometry.link_list:
                self._add_link(link)
            if isinstance(geometry, RobotModel):
                self._add_joint_sliders(geometry)
                if self._enable_ik:
                    self._setup_ik_controls(geometry)
        else:
            raise TypeError("geometry must be Link or CascadedLink")

    def show(self):
        host = self._server.get_host()
        port = self._server.get_port()
        url = f"http://{host}:{port}"
        webbrowser.open(url)

    def redraw(self):
        for link_id, handle in self._linkid_to_handle.items():
            link = self._linkid_to_link[link_id]
            handle.position = link.worldpos()
            handle.wxyz = matrix2quaternion(link.worldrot())

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
