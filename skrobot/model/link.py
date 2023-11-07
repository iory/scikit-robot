try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np
import trimesh
from scipy.constants import g

import skrobot
from skrobot.coordinates import CascadedCoords
from skrobot.coordinates.math import outer_product_matrix
from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.math import convert_to_axis_vector
from skrobot.coordinates.math import inverse_rodrigues


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 collision_mesh=None,
                 visual_mesh=None,
                 weight=None,
                 force=None,
                 moment=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self._centroid = centroid
        self._weight = weight
        self.joint = None
        self._child_links = []
        self._parent_link = None
        self._force = force
        self._moment = moment
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)
        self._inertia_tensor = inertia_tensor
        self._collision_mesh = collision_mesh
        self._visual_mesh = visual_mesh
        self._sdf = None

        self._angular_velocity = np.zeros(3) # [rad/s]
        self._angular_acceleration = np.zeros(3) # [rad/s^2]
        self._spatial_velocity = np.zeros(3) # [m/s]
        self._spatial_acceleration = np.zeros(3) # [m/s^2]
        self._angular_momentum = np.zeros(3) # [kg m^2/s]
        self._momentum = np.zeros(3) # [kg m/s]
        self._angular_momentum_velocity = np.zeros(3) # [kg m^2/s^2]
        self._momentum_velocity = np.zeros(3) # [kg m^2/s^2]
        self._force = np.zeros(3) # [N] = [kg m/s^2]
        self._moment = np.zeros(3) # [Nm] = [kg m^2/s^2]
        self._ext_force = np.zeros(3) # [N] = [kg m/s^2]
        self._ext_moment = np.zeros(3) # [Nm] = [kg m^2/s^2]

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
            self, tmp_va=None, tmp_vb=None, tmp_vc=None,
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

        m = self._weight
        f_g = - 1.0 * m * np.array([0, 0, g])  # [N]
        c = self._centroid  # [m]
        iner = self.worldrot() @ (self._inertia_tensor) @ self.worldrot().T  # [kg m^2]
        c_hat = outer_product_matrix(c)
        I = iner + m * c_hat @ c_hat.T  # [kg m^2]

        # import ipdb
        # ipdb.set_trace()
        self._momentum += m * (self._spatial_velocity + np.cross(self._angular_velocity, c))
        self._angular_momentum += (m * np.cross(c, self._spatial_velocity) +
                             I @ self._angular_velocity)
        self._force += (m * (self._spatial_acceleration + np.cross(self._angular_acceleration, c)) +
                        np.cross(self._angular_velocity, self._momentum))
        self._moment += (m * np.cross(c, self._spatial_acceleration) +
                   I @ self._angular_acceleration +
                   np.cross(self._spatial_velocity, self._momentum) +
                   np.cross(self._angular_velocity, self._angular_momentum))

        # use ext_force and ext_moment
        self._force -= f_g # + self.ext_force
        self._moment -= np.cross(c, f_g) # + self.ext_moment

        # propagation of force and moment from child-links
        for child in self.child_links:
            child.inverse_dynamics(tmp_va=tmp_va, tmp_vb=tmp_vb,
                                   tmp_vc=tmp_vc, tmp_ma=tmp_ma, tmp_mb=tmp_mb,
                                   tmp_mc=tmp_mc, tmp_md=tmp_md)
            self._force += child._force
            self._moment += child._moment

        print(self._force, self._moment)
        # exclude if root-link
        if self.joint and self.parent_link and isinstance(self._parent, Link):
            joint_torque = (
                self._spatial_velocity_jacobian @ self._force +
                self.angular_velocity_jacobian @ self._moment)
            print(joint_torque)
            self.joint.joint_torque(joint_torque)

    def calc_torque_without_ext_wrench(
            self, debug_view=None, calc_statics_p=True, dt=0.005, av=None, root_coords=None):
        if av is None:
            av = self.angle_vector()
        if root_coords is None:
            root_coords = self.links[0].copy_worldcoords()
        kwargs = {}
        if not calc_statics_p:
            ret_rc = self.calc_root_coords_vel_acc_from_pos(dt, root_coords)
            ret_av = self.calc_av_vel_acc_from_pos(dt, av)
            kwargs = {
                'jvv': ret_av.get('joint_velocity_vector'),
                'jav': ret_av.get('joint_acceleration_vector'),
                'root_spacial_velocity': ret_rc.get('root_spacial_velocity'),
                'root_angular_velocity': ret_rc.get('root_angular_velocity'),
                'root_spacial_acceleration': ret_rc.get('root_spacial_acceleration'),
                'root_angular_acceleration': ret_rc.get('root_angular_acceleration')
            }
        return self.calc_torque_from_vel_acc(debug_view=debug_view, **kwargs)

    def calc_root_coords_vel_acc_from_pos(self, dt, root_coords):
        dt_inv = 1.0 / dt

        if self.prev_root_coords is None:
            self.prev_root_coords = root_coords

        root_rot = root_coords.rotation_matrix()
        prev_root_rot = self.prev_root_coords.rotation_matrix()
        rw = dt_inv * self.prev_root_coords.rotate_vector(
            inverse_rodrigues(prev_root_rot.T @ root_rot,
                              return_angular_velocity=True)
        )

        root_pos = root_coords.translation_vector() * 1e-3
        prev_root_pos = self.prev_root_coords.translation_vector() * 1e-3

        if np.linalg.norm(rw) < 5e-3:
            rv = dt_inv * (root_pos - prev_root_pos)
        else:
            # Implementation of the complex velocity calculation goes here

        if self.prev_root_v is None:
            self.prev_root_v = rv
        if self.prev_root_w is None:
            self.prev_root_w = rw

        rwa = dt_inv * (rw - self.prev_root_w)
        # First order approximation
        sp_rva = (dt_inv * (rv - self.prev_root_v)) - np.cross(rwa, root_pos)

        # Store the previous state
        self.prev_root_coords = root_coords.copy()
        self.prev_root_v = rv
        self.prev_root_w = rw

        return {
            'root_spacial_velocity': rv - np.cross(rw, root_coords.translation_vector() * 1e-3),
            'root_angular_velocity': rw,
            'root_spacial_acceleration': sp_rva,
            'root_angular_acceleration': rwa
        }

    def calc_torque_from_vel_acc(self, debug_view=None, jvv=None, jav=None,
                                 root_spacial_velocity=np.zeros(3),
                                 root_angular_velocity=np.zeros(3),
                                 root_spacial_acceleration=np.zeros(3),
                                 root_angular_acceleration=np.zeros(3),
                                 calc_torque_buffer_args=None):
        if jvv is None:
            jvv = np.zeros(len(self.joint_list))
        if jav is None:
            jav = np.zeros(len(self.joint_list))

        torque_vector = np.zeros(len(self.joint_list))

        def all_child_links(link, func):
            func(link)
            for child in link.child_links:
                all_child_links(child, func)

        all_child_links(self.links[0], lambda l: l.set_spacial_and_angular_velocity_jacobian())

        for i, jnt in enumerate(self.joint_list):
            jnt.joint_velocity = jvv[i]
            jnt.joint_acceleration = jav[i]

        self.links[0].spacial_acceleration = root_spacial_acceleration
        self.links[0].angular_acceleration = root_angular_acceleration
        self.links[0].spacial_velocity = root_spacial_velocity
        self.links[0].angular_velocity = root_angular_velocity

        # Recursive kinematics and dynamics computation
        self.links[0].forward_all_kinematics(debug_view, *calc_torque_buffer_args[:2])
        self.links[0].inverse_dynamics(debug_view, *calc_torque_buffer_args)

        # Reset external forces and moments
        all_child_links(self.links[0], lambda l: l.reset_external_forces_and_moments())

        for i, jnt in enumerate(self.joint_list):
            torque_vector[i] = jnt.joint_torque

        return torque_vector

    def forward_all_kinematics(self):
        if self.parent_link and isinstance(self.parent, Link) and self.joint.__class__.__name__ != 'FixedJoint':
            paxis = self.joint.axis

            world_coords = self.parent_link.copy_worldcoords().transform(self.joint.default_coords)
            ax = normalize_vector(world_coords.rotate_vector(paxis))

            svj = self.joint.calc_spatial_velocity_jacobian(ax)
            avj = self.joint.calc_angular_velocity_jacobian(ax)

            print(ax, svj, avj)

            self._angular_velocity = self.parent_link._angular_velocity + svj * self.joint.joint_velocity
            self._spatial_velocity = self.parent_link._spatial_velocity + avj * self.joint.joint_velocity

            saj = self.joint.calc_spatial_acceleration_jacobian(svj, avj)
            self._spatial_acceleration = (self.parent_link._spatial_acceleration +
                                         saj * self.joint.joint_velocity +
                                         svj * self.joint.joint_acceleration)

            aaj = self.joint.calc_angular_acceleration_jacobian(avj)
            self._angular_acceleration = (self.parent_link._angular_acceleration +
                                          aaj * self.joint.joint_velocity +
                                         avj * self.joint.joint_acceleration)
            print(f'{self._angular_velocity} {self._angular_acceleration} {self._spatial_velocity} {self._spatial_acceleration}')

        for child_link in self.child_links:
            child_link.forward_all_kinematics()


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
