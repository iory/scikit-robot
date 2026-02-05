"""Manipulability ellipsoid visualization for Viser viewer."""

import warnings

import numpy as np
import trimesh.creation


class ManipulabilityEllipse:
    """Helper class to visualize the manipulability ellipsoid for a robot link.

    The manipulability ellipsoid visualizes the velocity manipulability
    measure (Yoshikawa, 1985) as a 3D ellipsoid. The ellipsoid's principal
    axes represent the directions of highest manipulability, and their
    lengths represent the magnitude of manipulability in each direction.

    Parameters
    ----------
    server : viser.ViserServer or viser.ClientHandle
        The Viser server or client handle.
    robot_model : RobotModel
        The scikit-robot robot model.
    link_list : list
        List of links forming the kinematic chain for Jacobian computation.
    root_node_name : str
        The base name for the ellipsoid mesh in the Viser scene.
    target_link : Link, optional
        The link to visualize the ellipsoid for initially.
    scaling_factor : float
        Scaling factor applied to the ellipsoid dimensions.
    visible : bool
        Initial visibility state.
    wireframe : bool
        Whether to render the ellipsoid as a wireframe.
    color : tuple
        The color of the ellipsoid mesh as (R, G, B).

    Attributes
    ----------
    manipulability : float
        The current manipulability measure (Yoshikawa's measure).

    Examples
    --------
    >>> from skrobot.viewers import ViserViewer
    >>> from skrobot.viewers._manipulability_ellipse import ManipulabilityEllipse
    >>> from skrobot.models import Kuka
    >>> robot = Kuka()
    >>> viewer = ViserViewer()
    >>> viewer.add(robot)
    >>> ellipse = ManipulabilityEllipse(
    ...     viewer._server, robot, robot.rarm.link_list,
    ...     target_link=robot.rarm_end_coords.parent)
    >>> ellipse.update()  # Update ellipsoid based on current joint angles
    """

    def __init__(
        self,
        server,
        robot_model,
        link_list,
        root_node_name="/manipulability",
        target_link=None,
        scaling_factor=0.2,
        visible=True,
        wireframe=True,
        color=(200, 200, 255),
    ):
        self._server = server
        self._robot_model = robot_model
        self._link_list = link_list
        self._root_node_name = root_node_name
        self._target_link = target_link
        self._scaling_factor = scaling_factor
        self._visible = visible
        self._wireframe = wireframe
        self._color = color

        self._base_manip_sphere = trimesh.creation.icosphere(radius=1.0)
        self._mesh_handle = None

        # Create mesh handle
        self._create_mesh_handle()

        self.manipulability = 0.0

    def _create_mesh_handle(self):
        """Creates or recreates the mesh handle in the Viser scene."""
        if self._mesh_handle is not None:
            self._mesh_handle.remove()

        # Create with dummy data initially, will be updated
        self._mesh_handle = self._server.scene.add_mesh_simple(
            self._root_node_name,
            vertices=np.zeros((1, 3), dtype=np.float32),
            faces=np.zeros((1, 3), dtype=np.uint32),
            color=self._color,
            wireframe=self._wireframe,
            visible=self._visible,
        )

        # Viser version compatibility
        if hasattr(self._mesh_handle, "cast_shadow"):
            self._mesh_handle.cast_shadow = False

    def set_target_link(self, link):
        """Sets the target link for which to display the ellipsoid.

        Parameters
        ----------
        link : Link or None
            The target link, or None to disable.
        """
        if link is None:
            self._target_link = None
            self.set_visibility(False)
        else:
            if link in self._link_list:
                self._target_link = link
                if self._mesh_handle is not None and self._visible:
                    self._mesh_handle.visible = True
            else:
                warnings.warn(
                    f"Link '{link.name}' not in link_list.",
                    stacklevel=2
                )
                self._target_link = None
                self.set_visibility(False)

    def _compute_jacobian(self):
        """Compute the position Jacobian for the target link.

        Returns
        -------
        jacobian : ndarray
            Position Jacobian matrix (3, n_joints).
        """
        if self._target_link is None:
            return None

        move_target = self._target_link
        jacobian = self._robot_model.calc_jacobian_from_link_list(
            move_target=move_target,
            link_list=self._link_list,
            position_mask=[True, True, True],
            rotation_mask=[False, False, False],
        )
        return jacobian

    def update(self):
        """Updates the ellipsoid based on the current joint configuration."""
        if (
            self._target_link is None
            or not self._visible
            or self._mesh_handle is None
        ):
            if self._mesh_handle is not None and self._mesh_handle.visible:
                self._mesh_handle.visible = False
            return

        if not self._mesh_handle.visible:
            self._mesh_handle.visible = True

        try:
            # Compute Jacobian
            jacobian = self._compute_jacobian()
            if jacobian is None:
                return

            # Compute manipulability (Yoshikawa's measure)
            JJT = jacobian @ jacobian.T
            det_JJT = np.linalg.det(JJT)
            self.manipulability = np.sqrt(max(0.0, det_JJT))

            # Eigendecomposition for ellipsoid
            cov_matrix = JJT
            vals, vecs = np.linalg.eigh(cov_matrix)
            vals = np.maximum(vals, 1e-9)  # Clamp for stability

            # Get target link pose
            target_pos = self._target_link.worldpos()

            # Create and transform ellipsoid mesh
            ellipsoid_mesh = self._base_manip_sphere.copy()
            tf = np.eye(4)
            tf[:3, :3] = vecs  # Rotation from eigenvectors
            tf[:3, 3] = target_pos  # Translation to link origin

            # Apply scaling according to eigenvalues
            ellipsoid_mesh.apply_scale(np.sqrt(vals) * self._scaling_factor)
            ellipsoid_mesh.apply_transform(tf)

            # Update Viser mesh
            self._mesh_handle.vertices = np.array(
                ellipsoid_mesh.vertices, dtype=np.float32
            )
            self._mesh_handle.faces = np.array(
                ellipsoid_mesh.faces, dtype=np.uint32
            )

        except Exception as e:
            warnings.warn(
                f"Failed to update manipulability ellipsoid: {e}",
                stacklevel=2
            )
            if self._mesh_handle is not None:
                self._mesh_handle.visible = False

    def set_visibility(self, visible):
        """Sets the visibility of the ellipsoid mesh.

        Parameters
        ----------
        visible : bool
            Whether the ellipsoid should be visible.
        """
        self._visible = visible
        if self._mesh_handle is not None:
            if visible and self._target_link is not None:
                self.update()
            elif self._mesh_handle.visible != visible:
                self._mesh_handle.visible = visible

    def set_scaling_factor(self, scaling_factor):
        """Sets the scaling factor for the ellipsoid.

        Parameters
        ----------
        scaling_factor : float
            New scaling factor.
        """
        self._scaling_factor = scaling_factor
        if self._visible and self._target_link is not None:
            self.update()

    def set_color(self, color):
        """Sets the color of the ellipsoid.

        Parameters
        ----------
        color : tuple
            New color as (R, G, B).
        """
        self._color = color
        if self._mesh_handle is not None:
            self._mesh_handle.color = color

    def remove(self):
        """Removes the ellipsoid mesh from the Viser scene."""
        if self._mesh_handle is not None:
            self._mesh_handle.remove()
            self._mesh_handle = None
        self._target_link = None
