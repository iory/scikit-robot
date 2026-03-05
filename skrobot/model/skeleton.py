"""Skeleton visualization model for robot kinematics.

This module provides SkeletonModel class that creates a visual representation
of robot joint positions and connections between them.
"""

import numpy as np
import trimesh

from skrobot.coordinates import Coordinates
from skrobot.model.link import Link
from skrobot.model.robot_model import CascadedLink


# Default colors for joint type markers (RGBA, values 0-255)
DEFAULT_JOINT_COLORS = {
    'revolute': (255, 0, 0, 255),       # Red
    'continuous': (0, 0, 255, 255),     # Blue
    'prismatic': (0, 255, 0, 255),      # Green
    'fixed': (128, 128, 128, 255),      # Gray
    'default': (0, 0, 0, 255),          # Black
}

DEFAULT_SKELETON_COLOR = (0, 0, 0, 255)  # Black


def _compute_auto_sizes(robot):
    """Compute marker/skeleton sizes based on robot geometry.

    Parameters
    ----------
    robot : RobotModel
        Robot model for size estimation.

    Returns
    -------
    dict
        Dictionary with 'marker_radius', 'marker_height', 'skeleton_radius'.
    """
    all_vertices = []

    for link in robot.link_list:
        if not hasattr(link, 'visual_mesh') or link.visual_mesh is None:
            continue
        meshes = link.visual_mesh
        if not isinstance(meshes, list):
            meshes = [meshes]
        try:
            T = link.worldcoords().T()
            if not np.all(np.isfinite(T)):
                continue
        except Exception:
            continue
        for m in meshes:
            if m is None or not hasattr(m, 'vertices') or len(m.vertices) == 0:
                continue
            verts = np.asarray(m.vertices)
            if not np.all(np.isfinite(verts)):
                continue
            verts_h = np.hstack([verts, np.ones((len(verts), 1))])
            verts_w = (T @ verts_h.T).T[:, :3]
            if np.all(np.isfinite(verts_w)):
                all_vertices.append(verts_w)

    if not all_vertices:
        return {
            'marker_radius': 0.03,
            'marker_height': 0.15,
            'skeleton_radius': 0.01,
        }

    all_vertices = np.vstack(all_vertices)
    L = np.max(all_vertices.max(axis=0) - all_vertices.min(axis=0))
    return {
        'marker_radius': float(np.clip(L * 0.025, 0.005, 0.1)),
        'marker_height': float(np.clip(L * 0.10, 0.02, 0.3)),
        'skeleton_radius': float(np.clip(L * 0.008, 0.002, 0.03)),
    }


class _ConnectorLink(Link):
    """Link that connects two joints with a cylinder mesh.

    The cylinder is rebuilt lazily when visual_mesh is accessed.
    """

    def __init__(self, parent_joint, target_joint, radius, color, name=None):
        super(_ConnectorLink, self).__init__(name=name)
        self._target_joint = target_joint
        self._radius = radius
        self._color = color
        self._cached_local_target = None
        self._mesh_dirty = True
        self._cylinder_mesh = None

        # Place at parent joint's child_link and assoc
        parent_wc = parent_joint.child_link.worldcoords()
        self.newcoords(
            Coordinates(pos=parent_wc.worldpos(), rot=parent_wc.worldrot()),
            check_validity=False)
        parent_joint.child_link.assoc(self, force=True)

        # Compute initial target position
        self.update(force=True)
        self._visual_mesh_changed = False

    def update(self, force=False):
        """Update coordinates and mark mesh dirty if target moved."""
        super(_ConnectorLink, self).update(force)
        target_world = self._target_joint.child_link.worldpos()
        local_target = self._worldcoords.inverse_transform_vector(target_world)
        if self._cached_local_target is not None and np.allclose(
                local_target, self._cached_local_target, atol=1e-6):
            return
        self._cached_local_target = local_target.copy()
        self._mesh_dirty = True

    @property
    def visual_mesh(self):
        """Lazily rebuild cylinder mesh when accessed."""
        if self._mesh_dirty and self._cached_local_target is not None:
            self._rebuild_cylinder()
        return self._cylinder_mesh

    @visual_mesh.setter
    def visual_mesh(self, value):
        self._cylinder_mesh = value
        self._mesh_dirty = False

    @property
    def concatenated_visual_mesh(self):
        """Return the lazily-built cylinder mesh."""
        return self.visual_mesh

    def _rebuild_cylinder(self):
        """Rebuild cylinder mesh to connect to target."""
        local_target = self._cached_local_target
        length = np.linalg.norm(local_target)
        if length < 1e-6:
            cyl = trimesh.creation.cylinder(radius=self._radius, height=1e-6)
        else:
            cyl = trimesh.creation.cylinder(radius=self._radius, height=length)
            c = Coordinates(pos=local_target / 2)
            c.align_axis_to_direction(local_target / length)
            cyl.apply_transform(c.T())
        cyl.visual.face_colors = self._color
        self._cylinder_mesh = cyl
        self._mesh_dirty = False


class SkeletonModel(CascadedLink):
    """Skeleton visualization model for robot kinematics.

    Creates cylinder markers at each joint position and connector lines
    between parent-child joints. The markers and connectors are assoc'd
    to the robot's links, so they automatically follow joint movements.

    Parameters
    ----------
    robot : RobotModel or CascadedLink
        Robot model to visualize.
    marker_radius : float, optional
        Radius of joint marker cylinders. Auto-calculated if None.
    marker_height : float, optional
        Height of joint marker cylinders. Auto-calculated if None.
    skeleton_radius : float, optional
        Radius of skeleton connector cylinders. Auto-calculated if None.
    joint_colors : dict, optional
        Colors for joint types. Keys: 'revolute', 'continuous', 'prismatic',
        'fixed'. Values: RGBA tuples (0-255).
    skeleton_color : tuple, optional
        RGBA color for skeleton lines (0-255). Default is black.
    with_joint_markers : bool
        Whether to show joint markers. Default is True.
    with_skeleton : bool
        Whether to show skeleton lines. Default is True.

    Examples
    --------
    >>> from skrobot.models import Fetch
    >>> from skrobot.model import SkeletonModel
    >>> from skrobot.viewers import TrimeshSceneViewer
    >>>
    >>> robot = Fetch()
    >>> skeleton = SkeletonModel(robot)
    >>>
    >>> viewer = TrimeshSceneViewer()
    >>> viewer.add(robot)
    >>> viewer.add(skeleton)
    >>> viewer.show()

    >>> # Show only skeleton without robot mesh
    >>> viewer2 = TrimeshSceneViewer()
    >>> viewer2.add(skeleton)
    >>> viewer2.show()
    """

    def __init__(self, robot, marker_radius=None, marker_height=None,
                 skeleton_radius=None, joint_colors=None, skeleton_color=None,
                 with_joint_markers=True, with_skeleton=True):
        super(SkeletonModel, self).__init__()

        # Auto-calculate sizes if not provided
        sizes = _compute_auto_sizes(robot)
        if marker_radius is None:
            marker_radius = sizes['marker_radius']
        if marker_height is None:
            marker_height = sizes['marker_height']
        if skeleton_radius is None:
            skeleton_radius = sizes['skeleton_radius']

        # Setup colors
        colors = dict(DEFAULT_JOINT_COLORS)
        if joint_colors is not None:
            colors.update(joint_colors)
        if skeleton_color is None:
            skeleton_color = DEFAULT_SKELETON_COLOR

        # Build parent joint mapping
        movable_joints = set(robot.joint_list)
        parent_joint_map = {}
        for joint in robot.joint_list:
            link = joint.parent_link
            while link is not None:
                if link.joint in movable_joints:
                    parent_joint_map[joint] = link.joint
                    break
                link = link.parent_link

        links = []

        # Create joint markers
        if with_joint_markers:
            for joint in robot.joint_list:
                jtype = joint.type
                marker = Link(name='skeleton_marker_{}'.format(joint.name))

                # Position at joint and align to axis
                c = Coordinates(pos=joint.child_link.worldpos())
                c.align_axis_to_direction(joint.world_axis)
                marker.newcoords(c, check_validity=False)

                # Create cylinder mesh
                cyl = trimesh.creation.cylinder(
                    radius=marker_radius, height=marker_height)
                cyl.visual.face_colors = colors.get(jtype, colors['default'])
                marker.visual_mesh = cyl
                marker._visual_mesh_changed = False

                # Assoc to robot
                joint.child_link.assoc(marker, force=True)
                links.append(marker)

        # Create skeleton connectors
        if with_skeleton:
            for joint in robot.joint_list:
                if joint not in parent_joint_map:
                    continue
                parent_joint = parent_joint_map[joint]
                connector = _ConnectorLink(
                    parent_joint=parent_joint,
                    target_joint=joint,
                    radius=skeleton_radius,
                    color=skeleton_color,
                    name='skeleton_connector_{}'.format(joint.name))
                links.append(connector)

        self.link_list = links

    def detach(self):
        """Dissociate all links from the robot."""
        for link in self.link_list:
            if link.parent is not None:
                link.parent.dissoc(link)
