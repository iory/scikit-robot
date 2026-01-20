from typing import Union
import webbrowser

import numpy as np
import trimesh
import viser

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
    def __init__(self, draw_grid: bool = True):
        self._server = viser.ViserServer()
        self._linkid_to_handle = dict()
        self._linkid_to_link = dict()
        self._is_active = True
        self._joint_sliders = dict()
        self._joint_folders = dict()

    @property
    def is_active(self) -> bool:
        return self._is_active

    def close(self):
        self._is_active = False
        self._server.stop()

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
                            j.joint_angle(self._joint_sliders[j.name].value)
                            self.redraw()
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
