try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np

import skrobot
from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates import CascadedCoords


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 collision_mesh=None,
                 visual_mesh=None,
                 mass=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self.centroid = centroid
        self.joint = None
        self._child_links = []
        self._parent_link = None
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)
        self.inertia_tensor = inertia_tensor
        self._collision_mesh = collision_mesh
        self.visual_mesh = visual_mesh
        if visual_mesh is not None:
            trimesh = _lazy_trimesh()
            self._concatenated_visual_mesh = trimesh.util.concatenate(
                self._visual_mesh)
        else:
            self._concatenated_visual_mesh = None
        self._visual_mesh_changed = False
        self._sdf = None

        # Dynamics properties
        self.mass = mass if mass is not None else 1.0  # kg

        # Velocity and acceleration states for dynamics computation
        self.angular_velocity = np.zeros(3)    # rad/s
        self.spatial_velocity = np.zeros(3)    # m/s
        self.angular_acceleration = np.zeros(3)  # rad/s^2
        self.spatial_acceleration = np.zeros(3)  # m/s^2

        # External forces and moments for dynamics computation
        self.ext_force = np.zeros(3)   # N
        self.ext_moment = np.zeros(3)  # Nm

        # Internal force/moment accumulation for inverse dynamics
        self._internal_force = np.zeros(3)   # N
        self._internal_moment = np.zeros(3)  # Nm

    @property
    def parent_link(self):
        return self._parent_link

    @property
    def child_links(self):
        return self._child_links

    def add_joint(self, j):
        self.joint = j

    def delete_joint(self):
        self.joint = None

    def add_child_link(self, child_link):
        """Add child link."""
        if child_link is not None and child_link not in self._child_links:
            self._child_links.append(child_link)

    def del_child_link(self, link):
        self._child_links.remove(link)

    def add_parent_link(self, parent_link):
        self._parent_link = parent_link

    def del_parent_link(self):
        self._parent_link = None

    @property
    def collision_mesh(self):
        """Return collision mesh

        Returns
        -------
        self._collision_mesh : trimesh.base.Trimesh
            A single collision mesh for the link.
            specified in the link frame,
            or None if there is not one.
        """
        return self._collision_mesh

    @collision_mesh.setter
    def collision_mesh(self, mesh):
        """Setter of collision mesh

        Parameters
        ----------
        mesh : trimesh.base.Trimesh
            A single collision mesh for the link.
            specified in the link frame,
            or None if there is not one.
        """
        if mesh is None or (isinstance(mesh, Sequence) and len(mesh) == 0):
            self._collision_mesh = None
            return
        trimesh = _lazy_trimesh()
        if mesh is not None and \
           not isinstance(mesh, trimesh.base.Trimesh):
            raise TypeError('input mesh is should be trimesh.base.Trimesh, '
                            'get type {}'.format(type(mesh)))
        self._collision_mesh = mesh

    @property
    def visual_mesh(self):
        """Return visual mesh

        Returns
        -------
        self._visual_mesh : None, trimesh.base.Trimesh, or
                            sequence of trimesh.Trimesh
            A set of visual meshes for the link in the link frame.
        """
        return self._visual_mesh

    @visual_mesh.setter
    def visual_mesh(self, mesh):
        """Setter of visual mesh

        Parameters
        ----------
        mesh : None, trimesh.Trimesh, sequence of trimesh.Trimesh,
               trimesh.points.PointCloud or str
            A set of visual meshes for the link in the link frame.
        """
        if mesh is None or (isinstance(mesh, Sequence) and len(mesh) == 0):
            self._visual_mesh = mesh
            self._concatenated_visual_mesh = None
            self._visual_mesh_changed = True
            return
        trimesh = _lazy_trimesh()
        if not (isinstance(mesh, trimesh.Trimesh)
                or (isinstance(mesh, Sequence)
                    and all(isinstance(m, trimesh.Trimesh) for m in mesh))
                or isinstance(mesh, trimesh.points.PointCloud)
                or isinstance(mesh, trimesh.path.path.Path3D)
                or isinstance(mesh, str)):
            raise TypeError(
                'mesh must be None, trimesh.Trimesh, sequence of '
                'trimesh.Trimesh, trimesh.points.PointCloud '
                'or path of mesh file, but got: {}'.format(type(mesh)))
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)
        self._visual_mesh = mesh
        self._concatenated_visual_mesh = trimesh.util.concatenate(mesh)
        self._visual_mesh_changed = True

    @property
    def concatenated_visual_mesh(self):
        """Concatenated visual mesh for visualization.

        Returns
        -------
        self._concatenated_visual_mesh : None, trimesh.base.Trimesh
            A concatenated visual meshes for the link in the link frame.
        """
        return self._concatenated_visual_mesh

    @property
    def visual_mesh_changed(self):
        return self._visual_mesh_changed

    @property
    def colors(self):
        if self._concatenated_visual_mesh is not None:
            return self._concatenated_visual_mesh.visual.face_colors
        else:
            return None

    def set_color(self, color):
        mesh = self._concatenated_visual_mesh
        if mesh is None:
            return
        color = np.array(color)
        if color.ndim == 2:
            mesh.visual.face_colors = color
        else:
            n_facet = len(mesh.visual.face_colors)
            mesh.visual.face_colors = np.array([color] * n_facet)
        self._visual_mesh_changed = True

    def reset_color(self):
        trimesh = _lazy_trimesh()
        concat_mesh = trimesh.util.concatenate(self._visual_mesh)
        self._concatenated_visual_mesh.visual.face_colors = \
            concat_mesh.visual.face_colors
        self._visual_mesh_changed = True

    def set_alpha(self, alpha):
        """Set alpha (transparency) value for visual mesh.

        Parameters
        ----------
        alpha : float
            Alpha value between 0.0 (transparent) and 1.0 (opaque)
        """
        mesh = self._concatenated_visual_mesh
        if mesh is None:
            return
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure alpha is in valid range
        # Update alpha channel (4th component) of all face colors
        mesh.visual.face_colors[:, 3] = np.round(alpha * 255).astype(np.uint8)
        self._visual_mesh_changed = True

    @property
    def sdf(self):
        """Return signed distance function.

        Returns
        -------
        self._sdf : None or skrobot.sdf.SignedDistanceFunction
            signed distance function.
        """
        return self._sdf

    @sdf.setter
    def sdf(self, sdf):
        """Setter of sdf.

        Parameters
        ----------
        sdf : skrobot.sdf.SignedDistanceFunction
            signed distance function.
        """
        if not isinstance(sdf, skrobot.sdf.SignedDistanceFunction):
            raise TypeError('sdf must be skrobot.sdf.SignedDistanceFunction.'
                            ' but is {}'.format(type(sdf)))
        self._sdf = sdf

    def set_mass_properties(self, mass, centroid=None, inertia_tensor=None):
        """Set mass properties for dynamics computation.

        Parameters
        ----------
        mass : float
            Link mass in [kg].
        centroid : np.ndarray, optional
            Center of mass position in link coordinates [m].
            If None, defaults to [0, 0, 0].
        inertia_tensor : np.ndarray, optional
            3x3 inertia tensor about centroid in [kg*m^2].
            If None, defaults to identity matrix.
        """
        self.mass = mass
        if centroid is not None:
            self.centroid = np.array(centroid)
        if inertia_tensor is not None:
            self.inertia_tensor = np.array(inertia_tensor)

    def clear_external_wrench(self):
        """Clear external forces and moments applied to this link."""
        self.ext_force.fill(0.0)
        self.ext_moment.fill(0.0)

    def apply_external_wrench(self, force=None, moment=None, point=None):
        """Apply external force and/or moment to this link.

        Parameters
        ----------
        force : np.ndarray, optional
            Force vector in world coordinates [N].
        moment : np.ndarray, optional
            Moment vector in world coordinates [Nm].
        point : np.ndarray, optional
            Point of force application in world coordinates [m].
            If provided, generates additional moment from force.
        """
        if force is not None:
            force = np.array(force)
            self.ext_force += force

            # If point is specified, add moment due to force offset
            if point is not None:
                point = np.array(point)
                link_pos = self.worldpos()
                r = point - link_pos
                self.ext_moment += np.cross(r, force)

        if moment is not None:
            self.ext_moment += np.array(moment)

    def calc_center_of_mass_jacobian(self, root_link, joint_list):
        """Calculate Jacobian matrix for center of mass.

        Parameters
        ----------
        root_link : Link
            Root link of the kinematic chain.
        joint_list : list of Joint
            List of joints affecting this link.

        Returns
        -------
        jacobian : np.ndarray
            3xn Jacobian matrix mapping joint velocities to CoM velocity.
        """
        if self.centroid is None:
            com_world = self.worldpos()
        else:
            com_world = self.worldpos() + self.worldrot().dot(self.centroid)

        jacobian = np.zeros((3, len(joint_list)))

        for i, joint in enumerate(joint_list):
            if joint is None:
                continue

            # Check if this joint affects this link
            current_link = self
            is_relevant = False
            while current_link is not None:
                if hasattr(current_link, 'joint') and current_link.joint == joint:
                    is_relevant = True
                    break
                current_link = current_link.parent_link

            if is_relevant:
                joint_pos = joint.parent_link.worldpos()
                joint_axis = joint.parent_link.worldrot().dot(joint.axis)

                if joint.__class__.__name__ == 'LinearJoint':
                    jacobian[:, i] = joint_axis
                else:  # rotational joint
                    r = com_world - joint_pos
                    jacobian[:, i] = np.cross(joint_axis, r)

        return jacobian


def _find_link_path(src_link, target_link, previous_link=None,
                    include_target=False):
    if src_link == target_link:
        if include_target:
            return [target_link], True
        else:
            return [], True
    paths = []
    links = []
    if hasattr(src_link, 'parent'):
        links = [src_link.parent]
    if hasattr(src_link, '_descendants'):
        links.extend(src_link._descendants)
    for next_link in links:
        if next_link is None or next_link == previous_link:
            continue
        path, succ = _find_link_path(next_link, target_link, src_link)
        if succ is True:
            paths.append(next_link)
            paths.extend(path)
            return paths, True
    return [], False


def find_link_path(src_link, target_link, include_source=True,
                   include_target=True):
    """Find paths of src_link to target_link

    Parameters
    ----------
    src_link : skrobot.model.link.Link
        source link.
    target_link : skrobot.model.link.Link
        target link.
    include_source : bool
        If `True`, return link list includes `src_link`.
    include_target : bool
        If `True`, return link list includes `target_link`.

    Returns
    -------
    ret : tuple(List[skrobot.model.link.Link], bool)
        If the links are connected, return Link list and `True`.
        Otherwise, return an empty list and `False`.
    """
    paths, succ = _find_link_path(src_link, target_link,
                                  include_target=include_target)
    if succ:
        if include_source:
            return [src_link] + paths, succ
        else:
            return paths, succ
    else:
        return [], False
