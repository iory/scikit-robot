try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np
import trimesh

import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import outer_product_matrix


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 collision_mesh=None,
                 visual_mesh=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self.centroid = centroid
        self.joint = None
        self._child_links = []
        self._parent_link = None
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)
        self._collision_mesh = collision_mesh
        self._visual_mesh = visual_mesh
        self._sdf = None

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
        if not (mesh is None
                or isinstance(mesh, trimesh.Trimesh)
                or (isinstance(mesh, Sequence)
                    and all(isinstance(m, trimesh.Trimesh) for m in mesh))
                or isinstance(mesh, trimesh.points.PointCloud)
                or isinstance(mesh, str)):
            raise TypeError(
                'mesh must be None, trimesh.Trimesh, sequence of '
                'trimesh.Trimesh, trimesh.points.PointCloud '
                'or path of mesh file, but got: {}'.format(type(mesh)))
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)
        self._visual_mesh = mesh

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

    def inverse_dynamics(
            self, debug_view=False, tmp_va=None, tmp_vb=None, tmp_vc=None,
            tmp_ma=None, tmp_mb=None, tmp_mc=None, tmp_md=None):
        if tmp_va is None:
            tmp_va = np.zeros(3)
        if tmp_vb is None:
            tmp_vb = np.zeros(3)
        if tmp_vc is None:
            tmp_vc = np.zeros(3)
        if tmp_ma is None:
            tmp_ma = np.zeros((3, 3))
        if tmp_mb is None:
            tmp_mb = np.zeros((3, 3))
        if tmp_mc is None:
            tmp_mc = np.zeros((3, 3))
        if tmp_md is None:
            tmp_md = np.zeros((3, 3))

        if debug_view:
            print(f";; inverse-dynamics link = {self.name}")

        w = 1e-3 * self.weight  # [g] -> [kg]
        fg = -1.0 * w * 1e-3 * g_vec  # [N]
        c = 1e-3 * self.centroid  # [m]
        iner = self.worldrot @ (1e-9 * self.inertia_tensor) @ self.worldrot.T  # [g mm^2] -> [kg m^2]
        c_hat = outer_product_matrix(c)
        I = iner + w * c_hat @ c_hat.T  # [kg m^2]

        momentum += w * (self.spatial_velocity + np.cross(self.angular_velocity, c))
        angular_momentum += (w * np.cross(c, self.spatial_velocity) +
                             I @ self.angular_velocity)
        force += (w * (self.spatial_acceleration + np.cross(self.angular_acceleration, c)) +
                  np.cross(self.angular_velocity, momentum))
        moment += (w * np.cross(c, self.spatial_acceleration) +
                   I @ self.angular_acceleration +
                   np.cross(self.spatial_velocity, momentum) +
                   np.cross(self.angular_velocity, angular_momentum))

        # use ext_force and ext_moment
        force -= (fg + self.ext_force)
        moment -= (np.cross(c, fg) + self.ext_moment)

        # propagation of force and moment from child-links
        for child in self.child_links:
            child.inverse_dynamics(debug_view=debug_view, tmp_va=tmp_va, tmp_vb=tmp_vb,
                                   tmp_vc=tmp_vc, tmp_ma=tmp_ma, tmp_mb=tmp_mb,
                                   tmp_mc=tmp_mc, tmp_md=tmp_md)
            force += child.force
            moment += child.moment

        # exclude if root-link
        if self.joint and self.parent_link and isinstance(self._parent, Link):
            joint_torque = (
                self.spatial_velocity_jacobian @ force +
                self.angular_velocity_jacobian @ moment)
            self.joint.joint_torque(joint_torque)


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
