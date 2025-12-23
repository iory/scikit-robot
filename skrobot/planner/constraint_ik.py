"""Constraint-based Inverse Kinematics using SLSQP.

This module provides IK solvers for reaching faces (planes) and lines,
as well as statics-aware IK that considers joint torques.
"""

import numpy as np
import scipy.optimize

from skrobot.coordinates import Coordinates
from skrobot.planner.utils import scipinize


class FaceTarget:
    """Target constraint for reaching a planar face.

    The end-effector should reach any point on the face while
    optionally constraining the approach direction (normal).

    Parameters
    ----------
    vertices : list of np.ndarray, optional
        Four vertices defining a rectangular face (in world coordinates).
        Shape: [(3,), (3,), (3,), (3,)]
    center : np.ndarray, optional
        Center position of the face. Shape: (3,)
    normal : np.ndarray, optional
        Normal vector of the face (will be normalized). Shape: (3,)
    x_axis : np.ndarray, optional
        X direction on the face plane. If None, computed automatically.
    x_length : float, optional
        Length of the face in x direction (half-width from center).
    y_length : float, optional
        Length of the face in y direction (half-height from center).
    margin : float, optional
        Margin from face edges in meters. Default is 0.0.
    normal_tolerance : float, optional
        Allowed angular deviation from normal in radians. Default is 0.0.
    approach_axis : str, optional
        Which axis of the end-effector should align with normal.
        Options: 'x', 'y', 'z', '-x', '-y', '-z'. Default is 'z'.

    Examples
    --------
    >>> import numpy as np
    >>> # Create a face from vertices
    >>> vertices = [
    ...     np.array([0.5, -0.1, 0.5]),
    ...     np.array([0.5, 0.1, 0.5]),
    ...     np.array([0.5, 0.1, 0.7]),
    ...     np.array([0.5, -0.1, 0.7])
    ... ]
    >>> face = FaceTarget(vertices=vertices)
    >>>
    >>> # Create a face from center, normal, and size
    >>> face = FaceTarget(
    ...     center=np.array([0.5, 0.0, 0.6]),
    ...     normal=np.array([1.0, 0.0, 0.0]),
    ...     x_length=0.1,
    ...     y_length=0.1
    ... )
    """

    def __init__(self,
                 vertices=None,
                 center=None,
                 normal=None,
                 x_axis=None,
                 x_length=None,
                 y_length=None,
                 margin=0.0,
                 normal_tolerance=0.0,
                 approach_axis='z'):
        if vertices is not None:
            vertices = [np.array(v) for v in vertices]
            self.center = np.mean(vertices, axis=0)
            v01 = vertices[1] - vertices[0]
            v03 = vertices[3] - vertices[0]
            self.x_axis = v01 / np.linalg.norm(v01)
            self.y_axis = v03 / np.linalg.norm(v03)
            self.normal = np.cross(self.x_axis, self.y_axis)
            self.normal = self.normal / np.linalg.norm(self.normal)
            self.x_length = np.linalg.norm(v01) / 2
            self.y_length = np.linalg.norm(v03) / 2
        elif center is not None and normal is not None:
            self.center = np.array(center)
            self.normal = np.array(normal)
            self.normal = self.normal / np.linalg.norm(self.normal)
            if x_axis is not None:
                self.x_axis = np.array(x_axis)
                self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)
            else:
                if abs(self.normal[2]) < 0.9:
                    self.x_axis = np.cross(np.array([0, 0, 1]), self.normal)
                else:
                    self.x_axis = np.cross(np.array([0, 1, 0]), self.normal)
                self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)
            self.y_axis = np.cross(self.normal, self.x_axis)
            self.x_length = x_length if x_length is not None else 0.1
            self.y_length = y_length if y_length is not None else 0.1
        else:
            raise ValueError(
                "Either 'vertices' or 'center' and 'normal' must be provided")

        self.margin = margin
        self.normal_tolerance = normal_tolerance
        self.approach_axis = approach_axis
        self._approach_sign = -1 if approach_axis.startswith('-') else 1
        self._approach_idx = {'x': 0, 'y': 1, 'z': 2}[
            approach_axis.lstrip('-')]

        self.rotation_matrix = np.column_stack(
            [self.x_axis, self.y_axis, self.normal])

    def get_target_coords(self, local_xy=None):
        """Get target coordinates for a specific point on the face.

        Parameters
        ----------
        local_xy : np.ndarray, optional
            Local (x, y) coordinates on the face. If None, returns center.

        Returns
        -------
        coords : Coordinates
            Target coordinates with position on face and rotation
            aligned with face normal.
        """
        if local_xy is None:
            local_xy = np.array([0.0, 0.0])
        pos = (self.center
               + local_xy[0] * self.x_axis
               + local_xy[1] * self.y_axis)
        rot = self._compute_rotation_matrix()
        return Coordinates(pos=pos, rot=rot)

    def _compute_rotation_matrix(self):
        """Compute rotation matrix for end-effector approaching the face."""
        axis_map = np.eye(3)
        target_rot = np.zeros((3, 3))
        approach_col = self._approach_idx
        normal_dir = -self._approach_sign * self.normal
        target_rot[:, approach_col] = normal_dir
        other_axes = [i for i in range(3) if i != approach_col]
        target_rot[:, other_axes[0]] = self.x_axis
        target_rot[:, other_axes[1]] = np.cross(
            target_rot[:, approach_col], target_rot[:, other_axes[0]])
        return target_rot

    def compute_error(self, ee_pos, ee_rot):
        """Compute task error for face constraint.

        Parameters
        ----------
        ee_pos : np.ndarray
            End-effector position. Shape: (3,)
        ee_rot : np.ndarray
            End-effector rotation matrix. Shape: (3, 3)

        Returns
        -------
        error : np.ndarray
            Error vector. First element is distance along normal,
            next two are position bounds violations (if any),
            last element is normal alignment error.
        """
        rel_pos = ee_pos - self.center
        dist_normal = np.dot(rel_pos, self.normal)
        local_x = np.dot(rel_pos, self.x_axis)
        local_y = np.dot(rel_pos, self.y_axis)

        x_limit = self.x_length - self.margin
        y_limit = self.y_length - self.margin
        x_violation = max(0, abs(local_x) - x_limit)
        y_violation = max(0, abs(local_y) - y_limit)

        approach_vec = ee_rot[:, self._approach_idx] * self._approach_sign
        normal_alignment = np.dot(approach_vec, -self.normal)
        normal_angle_error = np.arccos(np.clip(normal_alignment, -1, 1))
        normal_error = max(0, normal_angle_error - self.normal_tolerance)

        return np.array([dist_normal, x_violation, y_violation, normal_error])

    def compute_jacobian(self, ee_pos, ee_rot, fk_jacobian):
        """Compute Jacobian of face constraint with respect to joint angles.

        Parameters
        ----------
        ee_pos : np.ndarray
            End-effector position. Shape: (3,)
        ee_rot : np.ndarray
            End-effector rotation matrix. Shape: (3, 3)
        fk_jacobian : np.ndarray
            Forward kinematics Jacobian. Shape: (6, n_dof)

        Returns
        -------
        jacobian : np.ndarray
            Constraint Jacobian. Shape: (4, n_dof)
        """
        n_dof = fk_jacobian.shape[1]
        jacobian = np.zeros((4, n_dof))

        jacobian[0, :] = self.normal @ fk_jacobian[:3, :]

        rel_pos = ee_pos - self.center
        local_x = np.dot(rel_pos, self.x_axis)
        local_y = np.dot(rel_pos, self.y_axis)
        x_limit = self.x_length - self.margin
        y_limit = self.y_length - self.margin

        if abs(local_x) > x_limit:
            jacobian[1, :] = np.sign(local_x) * self.x_axis @ fk_jacobian[:3, :]
        if abs(local_y) > y_limit:
            jacobian[2, :] = np.sign(local_y) * self.y_axis @ fk_jacobian[:3, :]

        return jacobian


class LineTarget:
    """Target constraint for reaching a line segment.

    The end-effector should reach any point on the line while
    optionally constraining the approach direction.

    Parameters
    ----------
    start : np.ndarray
        Start point of the line segment. Shape: (3,)
    end : np.ndarray
        End point of the line segment. Shape: (3,)
    margin : float, optional
        Margin from line ends in meters. Default is 0.0.
    direction_axis : str, optional
        Which axis of the end-effector should align with line direction.
        Options: 'x', 'y', 'z', '-x', '-y', '-z', None.
        If None, no direction constraint. Default is None.
    normal_tolerance : float, optional
        Allowed angular deviation from direction in radians. Default is 0.0.

    Examples
    --------
    >>> import numpy as np
    >>> line = LineTarget(
    ...     start=np.array([0.5, -0.2, 0.5]),
    ...     end=np.array([0.5, 0.2, 0.5]),
    ...     direction_axis='x'
    ... )
    """

    def __init__(self,
                 start,
                 end,
                 margin=0.0,
                 direction_axis=None,
                 normal_tolerance=0.0):
        self.start = np.array(start)
        self.end = np.array(end)
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        self.direction = self.direction / self.length
        self.center = (self.start + self.end) / 2
        self.margin = margin
        self.normal_tolerance = normal_tolerance
        self.direction_axis = direction_axis

        if direction_axis is not None:
            self._dir_sign = -1 if direction_axis.startswith('-') else 1
            self._dir_idx = {'x': 0, 'y': 1, 'z': 2}[
                direction_axis.lstrip('-')]
        else:
            self._dir_sign = 1
            self._dir_idx = None

        if abs(self.direction[2]) < 0.9:
            self.normal1 = np.cross(self.direction, np.array([0, 0, 1]))
        else:
            self.normal1 = np.cross(self.direction, np.array([0, 1, 0]))
        self.normal1 = self.normal1 / np.linalg.norm(self.normal1)
        self.normal2 = np.cross(self.direction, self.normal1)

    def get_target_coords(self, t=0.5):
        """Get target coordinates for a specific point on the line.

        Parameters
        ----------
        t : float
            Parameter along the line (0=start, 1=end, 0.5=center).

        Returns
        -------
        coords : Coordinates
            Target coordinates with position on line.
        """
        pos = self.start + t * (self.end - self.start)
        rot = np.column_stack([self.direction, self.normal1, self.normal2])
        return Coordinates(pos=pos, rot=rot)

    def compute_error(self, ee_pos, ee_rot):
        """Compute task error for line constraint.

        Parameters
        ----------
        ee_pos : np.ndarray
            End-effector position. Shape: (3,)
        ee_rot : np.ndarray
            End-effector rotation matrix. Shape: (3, 3)

        Returns
        -------
        error : np.ndarray
            Error vector containing:
            - distance from line in normal1 direction
            - distance from line in normal2 direction
            - position bounds violation (if outside line segment)
            - direction alignment error (if direction_axis is set)
        """
        rel_pos = ee_pos - self.start
        proj_length = np.dot(rel_pos, self.direction)
        dist_normal1 = np.dot(rel_pos, self.normal1)
        dist_normal2 = np.dot(rel_pos, self.normal2)

        min_t = self.margin
        max_t = self.length - self.margin
        t_violation = 0.0
        if proj_length < min_t:
            t_violation = min_t - proj_length
        elif proj_length > max_t:
            t_violation = proj_length - max_t

        dir_error = 0.0
        if self._dir_idx is not None:
            ee_axis = ee_rot[:, self._dir_idx] * self._dir_sign
            alignment = np.dot(ee_axis, self.direction)
            angle_error = np.arccos(np.clip(abs(alignment), 0, 1))
            dir_error = max(0, angle_error - self.normal_tolerance)

        return np.array([dist_normal1, dist_normal2, t_violation, dir_error])

    def compute_jacobian(self, ee_pos, ee_rot, fk_jacobian):
        """Compute Jacobian of line constraint with respect to joint angles.

        Parameters
        ----------
        ee_pos : np.ndarray
            End-effector position. Shape: (3,)
        ee_rot : np.ndarray
            End-effector rotation matrix. Shape: (3, 3)
        fk_jacobian : np.ndarray
            Forward kinematics Jacobian. Shape: (6, n_dof)

        Returns
        -------
        jacobian : np.ndarray
            Constraint Jacobian. Shape: (4, n_dof)
        """
        n_dof = fk_jacobian.shape[1]
        jacobian = np.zeros((4, n_dof))

        jacobian[0, :] = self.normal1 @ fk_jacobian[:3, :]
        jacobian[1, :] = self.normal2 @ fk_jacobian[:3, :]

        rel_pos = ee_pos - self.start
        proj_length = np.dot(rel_pos, self.direction)
        min_t = self.margin
        max_t = self.length - self.margin

        if proj_length < min_t:
            jacobian[2, :] = -self.direction @ fk_jacobian[:3, :]
        elif proj_length > max_t:
            jacobian[2, :] = self.direction @ fk_jacobian[:3, :]

        return jacobian


def compute_fk_jacobian(robot, move_target, joint_list):
    """Compute forward kinematics Jacobian.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    move_target : Coordinates
        End-effector coordinates
    joint_list : list
        List of joints to compute Jacobian for

    Returns
    -------
    jacobian : np.ndarray
        Jacobian matrix. Shape: (6, n_dof)
    """
    n_dof = len(joint_list)
    jacobian = np.zeros((6, n_dof))
    ee_pos = move_target.worldpos()

    for j, joint in enumerate(joint_list):
        jtype = joint.joint_type
        if jtype in ('rotational', 'revolute'):
            joint_pos = joint.child_link.worldpos()
            axis = joint.axis
            if isinstance(axis, str):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
                joint_axis = joint.child_link.worldrot()[:, axis_idx].flatten()
            elif isinstance(axis, np.ndarray):
                joint_axis = joint.child_link.worldrot().dot(axis).flatten()
            else:
                joint_axis = np.array(axis)
            jacobian[:3, j] = np.cross(joint_axis, ee_pos - joint_pos)
            jacobian[3:6, j] = joint_axis
        elif jtype in ('linear', 'prismatic'):
            axis = joint.axis
            if isinstance(axis, str):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
                joint_axis = joint.child_link.worldrot()[:, axis_idx].flatten()
            elif isinstance(axis, np.ndarray):
                joint_axis = joint.child_link.worldrot().dot(axis).flatten()
            else:
                joint_axis = np.array(axis)
            jacobian[:3, j] = joint_axis
    return jacobian


def solve_ik_with_constraint(
        robot,
        target,
        move_target=None,
        link_list=None,
        rotation_axis=True,
        translation_axis=True,
        posture_weight=1e-3,
        stop=100,
        thre=0.001,
        rthre=np.deg2rad(1.0),
        slsqp_options=None,
        verbose=False):
    """Solve inverse kinematics with face or line constraint.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    target : FaceTarget, LineTarget, or Coordinates
        Target constraint. If Coordinates, uses standard IK.
    move_target : Coordinates, optional
        End-effector coordinates. If None, uses robot.end_coords.
    link_list : list, optional
        Link list for IK. If None, automatically determined.
    rotation_axis : bool or str
        Rotation constraint for standard targets.
    translation_axis : bool or str
        Translation constraint for standard targets.
    posture_weight : float
        Weight for posture regularization (minimize joint changes).
    stop : int
        Maximum iterations.
    thre : float
        Position threshold in meters.
    rthre : float
        Rotation threshold in radians.
    slsqp_options : dict, optional
        Options for scipy SLSQP optimizer.
    verbose : bool
        If True, print optimization progress.

    Returns
    -------
    success : bool
        True if IK succeeded.
    """
    if move_target is None:
        move_target = robot.end_coords
    if link_list is None:
        link_list = robot.link_lists(move_target.parent)

    joint_list = robot.joint_list_from_link_list(link_list,
                                                 ignore_fixed_joint=True)
    n_dof = len(joint_list)

    initial_angles = np.array([j.joint_angle() for j in joint_list])
    joint_limits = np.array([[j.min_angle, j.max_angle] for j in joint_list])

    def set_joint_angles(angles):
        for j, joint in enumerate(joint_list):
            joint.joint_angle(angles[j])

    def objective(x):
        diff = x - initial_angles
        f = 0.5 * posture_weight * np.dot(diff, diff)
        grad = posture_weight * diff
        return f, grad

    if isinstance(target, (FaceTarget, LineTarget)):
        def constraint(x):
            set_joint_angles(x)
            ee_pos = move_target.worldpos()
            ee_rot = move_target.worldrot()
            fk_jac = compute_fk_jacobian(robot, move_target, joint_list)

            error = target.compute_error(ee_pos, ee_rot)
            jac = target.compute_jacobian(ee_pos, ee_rot, fk_jac)

            if isinstance(target, FaceTarget):
                c = np.array([
                    thre - abs(error[0]),
                    -error[1],
                    -error[2],
                    rthre - error[3]
                ])
                c_jac = np.zeros((4, n_dof))
                c_jac[0, :] = -np.sign(error[0]) * jac[0, :]
                c_jac[1, :] = -jac[1, :]
                c_jac[2, :] = -jac[2, :]
            else:
                c = np.array([
                    thre - abs(error[0]),
                    thre - abs(error[1]),
                    -error[2],
                    rthre - error[3]
                ])
                c_jac = np.zeros((4, n_dof))
                c_jac[0, :] = -np.sign(error[0]) * jac[0, :]
                c_jac[1, :] = -np.sign(error[1]) * jac[1, :]
                c_jac[2, :] = -jac[2, :]

            return c, c_jac
    else:
        target_coords = target

        def constraint(x):
            set_joint_angles(x)
            ee_pos = move_target.worldpos()
            ee_rot = move_target.worldrot()
            target_pos = target_coords.worldpos()
            target_rot = target_coords.worldrot()
            fk_jac = compute_fk_jacobian(robot, move_target, joint_list)

            pos_error = target_pos - ee_pos
            rel_rot = np.dot(target_rot, ee_rot.T)
            trace_val = np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)
            angle = np.arccos(trace_val)
            if angle > 1e-6:
                axis = np.array([
                    rel_rot[2, 1] - rel_rot[1, 2],
                    rel_rot[0, 2] - rel_rot[2, 0],
                    rel_rot[1, 0] - rel_rot[0, 1]
                ]) / (2 * np.sin(angle))
                rot_error = axis * angle
            else:
                rot_error = np.zeros(3)

            pos_norm = np.linalg.norm(pos_error)
            rot_norm = np.linalg.norm(rot_error)

            c = np.array([thre - pos_norm, rthre - rot_norm])
            c_jac = np.zeros((2, n_dof))
            if pos_norm > 1e-8:
                c_jac[0, :] = (pos_error / pos_norm) @ fk_jac[:3, :]
            if rot_norm > 1e-8:
                c_jac[1, :] = (rot_error / rot_norm) @ fk_jac[3:6, :]

            return c, c_jac

    if slsqp_options is None:
        slsqp_options = {'ftol': 1e-6, 'disp': verbose, 'maxiter': stop}

    bounds = list(zip(joint_limits[:, 0], joint_limits[:, 1]))

    obj_scipy, obj_jac_scipy = scipinize(objective)
    ineq_scipy, ineq_jac_scipy = scipinize(constraint)

    ineq_dict = {
        'type': 'ineq',
        'fun': ineq_scipy,
        'jac': ineq_jac_scipy
    }

    result = scipy.optimize.minimize(
        obj_scipy, initial_angles,
        method='SLSQP',
        jac=obj_jac_scipy,
        bounds=bounds,
        constraints=[ineq_dict],
        options=slsqp_options)

    set_joint_angles(result.x)

    ee_pos = move_target.worldpos()
    ee_rot = move_target.worldrot()

    if isinstance(target, (FaceTarget, LineTarget)):
        error = target.compute_error(ee_pos, ee_rot)
        if isinstance(target, FaceTarget):
            # Check position constraints with some tolerance
            success = (abs(error[0]) < thre * 3
                       and error[1] < thre * 2
                       and error[2] < thre * 2
                       and error[3] < rthre * 2)
        else:
            success = (abs(error[0]) < thre * 3
                       and abs(error[1]) < thre * 3
                       and error[2] < thre * 2)
    else:
        pos_error = np.linalg.norm(target.worldpos() - ee_pos)
        success = pos_error < thre * 3

    # Also consider optimizer success
    if result.success:
        success = True

    if verbose:
        print(f"IK {'succeeded' if success else 'failed'}")
        if isinstance(target, (FaceTarget, LineTarget)):
            print(f"  Error: {error}")

    if not success:
        set_joint_angles(initial_angles)
        return False

    return success


def compute_gravity_torque(robot, joint_list, gravity=np.array([0, 0, -9.81])):
    """Compute joint torques due to gravity.

    Uses Newton-Euler method to compute static torques.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    joint_list : list
        List of joints to compute torques for
    gravity : np.ndarray
        Gravity vector. Shape: (3,)

    Returns
    -------
    torques : np.ndarray
        Gravity torques. Shape: (n_dof,)
    """
    n_dof = len(joint_list)
    torques = np.zeros(n_dof)

    for j, joint in enumerate(joint_list):
        child_link = joint.child_link
        descendants = [child_link]
        queue = [child_link]
        while queue:
            link = queue.pop(0)
            for child in getattr(link, 'child_links', []):
                if child not in descendants:
                    descendants.append(child)
                    queue.append(child)

        total_torque = 0.0
        joint_pos = child_link.worldpos()

        jtype = joint.joint_type
        if jtype in ('rotational', 'revolute'):
            axis = joint.axis
            if isinstance(axis, str):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
                joint_axis = child_link.worldrot()[:, axis_idx].flatten()
            elif isinstance(axis, np.ndarray):
                joint_axis = child_link.worldrot().dot(axis).flatten()
            else:
                joint_axis = np.array(axis)

            for link in descendants:
                if hasattr(link, 'weight') and link.weight > 0:
                    mass = link.weight
                    if hasattr(link, 'centroid'):
                        com = link.centroid
                    else:
                        com = link.worldpos()

                    force = mass * gravity
                    r = com - joint_pos
                    moment = np.cross(r, force)
                    total_torque += np.dot(moment, joint_axis)

        elif jtype in ('linear', 'prismatic'):
            axis = joint.axis
            if isinstance(axis, str):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
                joint_axis = child_link.worldrot()[:, axis_idx].flatten()
            elif isinstance(axis, np.ndarray):
                joint_axis = child_link.worldrot().dot(axis).flatten()
            else:
                joint_axis = np.array(axis)

            for link in descendants:
                if hasattr(link, 'weight') and link.weight > 0:
                    mass = link.weight
                    force = mass * gravity
                    total_torque += np.dot(force, joint_axis)

        torques[j] = total_torque

    return torques


def compute_contact_torque(robot, joint_list, contact_coords, contact_wrench):
    """Compute joint torques due to contact force.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    joint_list : list
        List of joints
    contact_coords : Coordinates
        Contact point coordinates
    contact_wrench : np.ndarray
        Contact wrench [fx, fy, fz, mx, my, mz] in world frame. Shape: (6,)

    Returns
    -------
    torques : np.ndarray
        Contact-induced torques. Shape: (n_dof,)
    """
    n_dof = len(joint_list)
    torques = np.zeros(n_dof)

    contact_pos = contact_coords.worldpos()
    contact_parent = contact_coords.parent

    link_chain = []
    current = contact_parent
    while current is not None:
        link_chain.append(current)
        current = getattr(current, 'parent_link', None)

    force = contact_wrench[:3]
    moment = contact_wrench[3:6]

    for j, joint in enumerate(joint_list):
        child_link = joint.child_link
        if child_link not in link_chain:
            continue

        joint_pos = child_link.worldpos()

        jtype = joint.joint_type
        if jtype in ('rotational', 'revolute'):
            axis = joint.axis
            if isinstance(axis, str):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
                joint_axis = child_link.worldrot()[:, axis_idx].flatten()
            elif isinstance(axis, np.ndarray):
                joint_axis = child_link.worldrot().dot(axis).flatten()
            else:
                joint_axis = np.array(axis)

            r = contact_pos - joint_pos
            moment_from_force = np.cross(r, force)
            total_moment = moment_from_force + moment
            torques[j] = np.dot(total_moment, joint_axis)

        elif jtype in ('linear', 'prismatic'):
            axis = joint.axis
            if isinstance(axis, str):
                axis_idx = {'x': 0, 'y': 1, 'z': 2}.get(axis, 0)
                joint_axis = child_link.worldrot()[:, axis_idx].flatten()
            elif isinstance(axis, np.ndarray):
                joint_axis = child_link.worldrot().dot(axis).flatten()
            else:
                joint_axis = np.array(axis)
            torques[j] = np.dot(force, joint_axis)

    return torques


class ContactConstraint:
    """Contact constraint for statics-aware IK.

    Parameters
    ----------
    contact_coords : Coordinates
        Contact point coordinates
    friction_coeff : float
        Friction coefficient. Default is 0.5.
    max_normal_force : float
        Maximum normal force in N. Default is 1000.
    min_normal_force : float
        Minimum normal force in N. Default is 0.
    contact_normal : np.ndarray
        Contact surface normal (pointing into the surface).
        If None, uses z-axis of contact_coords.

    Examples
    --------
    >>> contact = ContactConstraint(
    ...     contact_coords=robot.rleg_end_coords,
    ...     friction_coeff=0.5,
    ...     max_normal_force=500
    ... )
    """

    def __init__(self,
                 contact_coords,
                 friction_coeff=0.5,
                 max_normal_force=1000.0,
                 min_normal_force=0.0,
                 contact_normal=None):
        self.contact_coords = contact_coords
        self.friction_coeff = friction_coeff
        self.max_normal_force = max_normal_force
        self.min_normal_force = min_normal_force
        self._contact_normal = contact_normal

    @property
    def contact_normal(self):
        if self._contact_normal is not None:
            return self._contact_normal
        return self.contact_coords.worldrot()[:, 2]

    def check_wrench_feasibility(self, wrench):
        """Check if contact wrench satisfies friction cone constraint.

        Parameters
        ----------
        wrench : np.ndarray
            Contact wrench [fx, fy, fz, mx, my, mz]. Shape: (6,)

        Returns
        -------
        feasible : bool
            True if wrench is feasible.
        margin : float
            Margin to constraint boundary (positive = feasible).
        """
        force = wrench[:3]
        normal = self.contact_normal
        fn = np.dot(force, normal)
        ft = force - fn * normal
        ft_norm = np.linalg.norm(ft)

        normal_ok = self.min_normal_force <= fn <= self.max_normal_force
        friction_ok = ft_norm <= self.friction_coeff * fn

        if fn > 0:
            margin = min(
                fn - self.min_normal_force,
                self.max_normal_force - fn,
                self.friction_coeff * fn - ft_norm
            )
        else:
            margin = fn

        return normal_ok and friction_ok, margin


def solve_statics_ik(
        robot,
        target_coords_list,
        move_target_list,
        contact_list,
        link_list=None,
        external_forces=None,
        gravity=np.array([0, 0, -9.81]),
        torque_weight=1e-3,
        posture_weight=1e-4,
        stop=100,
        thre=0.001,
        rthre=np.deg2rad(1.0),
        slsqp_options=None,
        verbose=False):
    """Solve inverse kinematics with static equilibrium constraints.

    This function solves IK while ensuring force/torque balance and
    minimizing joint torques.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance
    target_coords_list : list of Coordinates
        Target coordinates for each end-effector.
    move_target_list : list of Coordinates
        End-effector coordinates (attached to robot).
    contact_list : list of ContactConstraint
        Contact constraints for equilibrium.
    link_list : list, optional
        Link list for IK. If None, automatically determined.
    external_forces : list of tuple, optional
        External forces as [(coords, wrench), ...].
    gravity : np.ndarray
        Gravity vector. Default is [0, 0, -9.81].
    torque_weight : float
        Weight for torque minimization in objective.
    posture_weight : float
        Weight for posture regularization.
    stop : int
        Maximum iterations.
    thre : float
        Position threshold in meters.
    rthre : float
        Rotation threshold in radians.
    slsqp_options : dict, optional
        Options for scipy SLSQP optimizer.
    verbose : bool
        If True, print optimization progress.

    Returns
    -------
    success : bool
        True if IK succeeded.
    contact_wrenches : list of np.ndarray
        Optimized contact wrenches for each contact.
    joint_torques : np.ndarray
        Resulting joint torques.
    """
    if link_list is None:
        all_links = []
        for mt in move_target_list:
            all_links.extend(robot.link_lists(mt.parent))
        link_list = list(set(all_links))

    joint_list = robot.joint_list_from_link_list(link_list,
                                                 ignore_fixed_joint=True)
    n_dof = len(joint_list)
    n_contacts = len(contact_list)
    n_targets = len(target_coords_list)

    n_wrench_vars = n_contacts * 6
    n_total_vars = n_dof + n_wrench_vars

    initial_angles = np.array([j.joint_angle() for j in joint_list])
    initial_wrenches = np.zeros(n_wrench_vars)

    for i, contact in enumerate(contact_list):
        normal = contact.contact_normal
        initial_wrenches[i * 6:i * 6 + 3] = normal * 10.0

    x0 = np.concatenate([initial_angles, initial_wrenches])

    joint_limits = np.array([[j.min_angle, j.max_angle] for j in joint_list])
    wrench_bounds = []
    for contact in contact_list:
        wrench_bounds.extend([
            (-contact.max_normal_force, contact.max_normal_force),
            (-contact.max_normal_force, contact.max_normal_force),
            (contact.min_normal_force, contact.max_normal_force),
            (-100, 100),
            (-100, 100),
            (-100, 100)
        ])

    bounds = (list(zip(joint_limits[:, 0], joint_limits[:, 1]))
              + wrench_bounds)

    def set_joint_angles(angles):
        for j, joint in enumerate(joint_list):
            joint.joint_angle(angles[j])

    def objective(x):
        angles = x[:n_dof]
        wrenches = x[n_dof:].reshape(n_contacts, 6)

        set_joint_angles(angles)

        gravity_torque = compute_gravity_torque(robot, joint_list, gravity)
        contact_torques = np.zeros(n_dof)
        for i, (contact, wrench) in enumerate(zip(contact_list, wrenches)):
            ct = compute_contact_torque(
                robot, joint_list, contact.contact_coords, wrench)
            contact_torques += ct

        total_torque = gravity_torque + contact_torques
        torque_cost = torque_weight * np.dot(total_torque, total_torque)

        angle_diff = angles - initial_angles
        posture_cost = posture_weight * np.dot(angle_diff, angle_diff)

        f = torque_cost + posture_cost
        grad = np.zeros(n_total_vars)
        grad[:n_dof] = 2 * posture_weight * angle_diff

        return f, grad

    def ik_constraint(x):
        angles = x[:n_dof]
        set_joint_angles(angles)

        constraints = []
        jacobians = []

        for i, (target, move_target) in enumerate(
                zip(target_coords_list, move_target_list)):
            ee_pos = move_target.worldpos()
            ee_rot = move_target.worldrot()
            target_pos = target.worldpos()
            target_rot = target.worldrot()

            pos_error = target_pos - ee_pos
            pos_norm = np.linalg.norm(pos_error)
            constraints.append(thre - pos_norm)

            rel_rot = np.dot(target_rot, ee_rot.T)
            trace_val = np.clip((np.trace(rel_rot) - 1) / 2, -1, 1)
            angle = np.arccos(trace_val)
            constraints.append(rthre - angle)

        c = np.array(constraints)
        c_jac = np.zeros((len(constraints), n_total_vars))

        return c, c_jac

    def equilibrium_constraint(x):
        angles = x[:n_dof]
        wrenches = x[n_dof:].reshape(n_contacts, 6)

        set_joint_angles(angles)

        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        com = robot.centroid()
        if hasattr(robot, 'weight'):
            total_mass = robot.weight
        else:
            total_mass = sum(getattr(link, 'weight', 0)
                             for link in robot.link_list)

        gravity_force = total_mass * gravity
        total_force += gravity_force
        total_moment += np.cross(com, gravity_force)

        if external_forces:
            for coords, wrench in external_forces:
                pos = coords.worldpos()
                total_force += wrench[:3]
                total_moment += np.cross(pos, wrench[:3]) + wrench[3:6]

        for contact, wrench in zip(contact_list, wrenches):
            pos = contact.contact_coords.worldpos()
            total_force += wrench[:3]
            total_moment += np.cross(pos, wrench[:3]) + wrench[3:6]

        force_balance = 0.1 - np.abs(total_force)
        moment_balance = 0.1 - np.abs(total_moment)

        c = np.concatenate([force_balance, moment_balance])
        c_jac = np.zeros((6, n_total_vars))

        return c, c_jac

    def friction_constraint(x):
        wrenches = x[n_dof:].reshape(n_contacts, 6)

        constraints = []
        for contact, wrench in zip(contact_list, wrenches):
            force = wrench[:3]
            normal = contact.contact_normal
            fn = np.dot(force, normal)
            ft = force - fn * normal
            ft_norm = np.linalg.norm(ft)

            constraints.append(fn - contact.min_normal_force)
            constraints.append(contact.max_normal_force - fn)
            constraints.append(contact.friction_coeff * fn - ft_norm)

        c = np.array(constraints)
        c_jac = np.zeros((len(constraints), n_total_vars))

        return c, c_jac

    if slsqp_options is None:
        slsqp_options = {'ftol': 1e-6, 'disp': verbose, 'maxiter': stop}

    obj_scipy, obj_jac_scipy = scipinize(objective)
    ik_scipy, ik_jac_scipy = scipinize(ik_constraint)
    eq_scipy, eq_jac_scipy = scipinize(equilibrium_constraint)
    fr_scipy, fr_jac_scipy = scipinize(friction_constraint)

    constraints = [
        {'type': 'ineq', 'fun': ik_scipy, 'jac': ik_jac_scipy},
        {'type': 'ineq', 'fun': eq_scipy, 'jac': eq_jac_scipy},
        {'type': 'ineq', 'fun': fr_scipy, 'jac': fr_jac_scipy}
    ]

    result = scipy.optimize.minimize(
        obj_scipy, x0,
        method='SLSQP',
        jac=obj_jac_scipy,
        bounds=bounds,
        constraints=constraints,
        options=slsqp_options)

    final_angles = result.x[:n_dof]
    final_wrenches = result.x[n_dof:].reshape(n_contacts, 6)

    set_joint_angles(final_angles)

    success = True
    for i, (target, move_target) in enumerate(
            zip(target_coords_list, move_target_list)):
        pos_error = np.linalg.norm(target.worldpos() - move_target.worldpos())
        if pos_error > thre * 2:
            success = False
            break

    gravity_torque = compute_gravity_torque(robot, joint_list, gravity)
    contact_torques = np.zeros(n_dof)
    for contact, wrench in zip(contact_list, final_wrenches):
        ct = compute_contact_torque(
            robot, joint_list, contact.contact_coords, wrench)
        contact_torques += ct
    joint_torques = gravity_torque + contact_torques

    if verbose:
        print(f"Statics IK {'succeeded' if success else 'failed'}")
        print(f"  Max joint torque: {np.max(np.abs(joint_torques)):.2f} Nm")
        for i, wrench in enumerate(final_wrenches):
            print(f"  Contact {i} force: {wrench[:3]}")

    if not success:
        set_joint_angles(initial_angles)

    return success, list(final_wrenches), joint_torques
