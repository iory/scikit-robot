"""Forward kinematics utilities for trajectory optimization.

This module provides backend-agnostic FK computation functions
shared across different solvers (scipy, jaxls, gradient_descent).
"""

from skrobot.backend import rodrigues_rotation
from skrobot.kinematics.differentiable import pose_error_se3_log as pose_error_log
from skrobot.kinematics.differentiable import rotation_error_so3_log as rotation_error_log


def build_fk_functions(fk_data, backend):
    """Build forward kinematics helper functions.

    Parameters
    ----------
    fk_data : dict
        FK parameters including:
        - link_translations: (n_joints, 3) link translations
        - link_rotations: (n_joints, 3, 3) link rotations
        - joint_axes: (n_joints, 3) joint axes
        - base_position: (3,) base position
        - base_rotation: (3, 3) base rotation
        - n_joints: int
        - collision_link_to_chain_idx: (n_coll_links,) indices
        - collision_link_offsets_pos: (n_coll_links, 3)
        - collision_link_offsets_rot: (n_coll_links, 3, 3)
        - sphere_centers_local: (n_spheres, 3) local positions
        - collision_link_indices: (n_spheres,) link index per sphere
    backend : module
        Array module (numpy, jax.numpy, or skrobot backend).

    Returns
    -------
    tuple
        (get_link_transforms, get_sphere_positions) functions.
    """
    xp = backend

    link_trans = fk_data['link_translations']
    link_rots = fk_data['link_rotations']
    joint_axes = fk_data['joint_axes']
    base_pos = fk_data['base_position']
    base_rot = fk_data['base_rotation']
    n_joints = fk_data['n_joints']
    ref_angles = fk_data.get('ref_angles')

    coll_link_idx = fk_data.get('collision_link_to_chain_idx')
    coll_offsets_pos = fk_data.get('collision_link_offsets_pos')
    coll_offsets_rot = fk_data.get('collision_link_offsets_rot')
    sphere_centers = fk_data.get('sphere_centers_local')
    sphere_link_indices = fk_data.get('collision_link_indices')

    def get_link_transforms(angles):
        """Compute link transforms for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        tuple
            (positions, rotations) arrays of shape
            (n_joints, 3) and (n_joints, 3, 3).
        """
        positions = []
        rotations = []
        current_pos = base_pos
        current_rot = base_rot

        for i in range(n_joints):
            current_pos = current_pos + current_rot @ link_trans[i]
            current_rot = current_rot @ link_rots[i]
            # Subtract ref_angles because link_rots already includes
            # the rotation at the reference configuration
            delta = angles[i]
            if ref_angles is not None:
                delta = delta - ref_angles[i]
            joint_rot = rodrigues_rotation(xp, joint_axes[i], delta)
            current_rot = current_rot @ joint_rot
            positions.append(current_pos)
            rotations.append(current_rot)

        return xp.stack(positions), xp.stack(rotations)

    ee_offset_pos = fk_data.get('ee_offset_position')
    ee_offset_rot = fk_data.get('ee_offset_rotation')

    def get_ee_position(angles):
        """Compute end-effector position for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        array
            End-effector position in world frame (3,).
        """
        pos, _ = get_ee_pose(angles)
        return pos

    def get_ee_pose(angles):
        """Compute end-effector position and rotation for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        position : array
            End-effector position in world frame (3,).
        rotation : array
            End-effector rotation matrix in world frame (3, 3).
        """
        positions, rotations = get_link_transforms(angles)
        last_pos = positions[-1]
        last_rot = rotations[-1]
        if ee_offset_pos is not None:
            ee_pos = last_pos + last_rot @ ee_offset_pos
        else:
            ee_pos = last_pos
        if ee_offset_rot is not None:
            ee_rot = last_rot @ ee_offset_rot
        else:
            ee_rot = last_rot
        return ee_pos, ee_rot

    def get_sphere_positions(angles):
        """Compute collision sphere positions for given joint angles.

        Spheres approximate collision geometries (spheres or capsules)
        attached to robot links.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).

        Returns
        -------
        array
            Sphere positions in world frame (n_spheres, 3).
        """
        if sphere_centers is None:
            return xp.zeros((0, 3))

        link_positions, link_rotations = get_link_transforms(angles)

        chain_idx = coll_link_idx[sphere_link_indices]
        sphere_link_pos = link_positions[chain_idx]
        sphere_link_rot = link_rotations[chain_idx]

        offsets_pos = coll_offsets_pos[sphere_link_indices]
        offsets_rot = coll_offsets_rot[sphere_link_indices]

        local = xp.einsum('ijk,ik->ij', offsets_rot, sphere_centers) \
            + offsets_pos
        world = sphere_link_pos \
            + xp.einsum('ijk,ik->ij', sphere_link_rot, local)
        return world

    return get_link_transforms, get_sphere_positions, get_ee_position, get_ee_pose


def compute_sphere_obstacle_distances(sphere_positions, sphere_radii,
                                       obstacle_centers, obstacle_radii,
                                       backend):
    """Compute signed distances between collision spheres and obstacles.

    Parameters
    ----------
    sphere_positions : array
        Collision sphere positions (n_spheres, 3).
    sphere_radii : array
        Collision sphere radii (n_spheres,).
    obstacle_centers : array
        Obstacle centers (n_obstacles, 3).
    obstacle_radii : array
        Obstacle radii (n_obstacles,).
    backend : module
        Array module.

    Returns
    -------
    array
        Signed distances (n_spheres, n_obstacles).
        Positive = separated, negative = penetrating.
    """
    xp = backend
    # sphere_positions: (n_spheres, 3)
    # obstacle_centers: (n_obstacles, 3)
    diff = sphere_positions[:, None, :] - obstacle_centers[None, :, :]
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    signed_dists = dists - sphere_radii[:, None] - obstacle_radii[None, :]
    return signed_dists


def compute_self_collision_distances(sphere_positions, sphere_radii,
                                     pairs_i, pairs_j, backend):
    """Compute signed distances for self-collision pairs.

    Parameters
    ----------
    sphere_positions : array
        Collision sphere positions (n_spheres, 3).
    sphere_radii : array
        Collision sphere radii (n_spheres,).
    pairs_i : array
        First sphere indices for each pair.
    pairs_j : array
        Second sphere indices for each pair.
    backend : module
        Array module.

    Returns
    -------
    array
        Signed distances for each pair.
    """
    xp = backend
    pos_i = sphere_positions[pairs_i]
    pos_j = sphere_positions[pairs_j]
    rad_i = sphere_radii[pairs_i]
    rad_j = sphere_radii[pairs_j]

    diff = pos_i - pos_j
    dists = xp.sqrt(xp.sum(diff ** 2, axis=-1) + 1e-10)
    signed_dists = dists - rad_i - rad_j
    return signed_dists


def rotation_error_vector(actual_rot, target_rot, backend):
    """Compute rotation error vector from anti-symmetric part of R_err.

    Extracts three independent components from the anti-symmetric part
    of ``actual_rot @ target_rot^T``.  The resulting 3-vector is zero
    when the two rotations are identical.

    Parameters
    ----------
    actual_rot : array
        Actual rotation matrix (3, 3).
    target_rot : array
        Target rotation matrix (3, 3).
    backend : module
        Array module (numpy or jax.numpy).

    Returns
    -------
    array
        Rotation error vector (3,).
    """
    xp = backend
    R_err = xp.matmul(actual_rot, xp.transpose(target_rot))
    return xp.stack([
        R_err[1, 0] - R_err[0, 1],
        R_err[2, 0] - R_err[0, 2],
        R_err[2, 1] - R_err[1, 2],
    ])


def compute_collision_residuals(signed_distances, activation_distance, backend):
    """Convert signed distances to collision residuals.

    Parameters
    ----------
    signed_distances : array
        Signed distances (positive = separated).
    activation_distance : float
        Distance threshold for activation.
    backend : module
        Array module.

    Returns
    -------
    array
        Collision residuals (positive when too close).
    """
    xp = backend
    return xp.maximum(0.0, activation_distance - signed_distances)


def prepare_fk_data(problem, backend):
    """Prepare FK data dictionary from problem definition.

    Parameters
    ----------
    problem : TrajectoryProblem
        Trajectory optimization problem.
    backend : module
        Array module for array conversion.

    Returns
    -------
    dict
        FK data dictionary for build_fk_functions().
    """
    xp = backend
    fk_params = problem.fk_params

    fk_data = {
        'link_translations': xp.array(fk_params['link_translations']),
        'link_rotations': xp.array(fk_params['link_rotations']),
        'joint_axes': xp.array(fk_params['joint_axes']),
        'base_position': xp.array(fk_params['base_position']),
        'base_rotation': xp.array(fk_params['base_rotation']),
        'n_joints': fk_params['n_joints'],
        'ee_offset_position': xp.array(fk_params['ee_offset_position']),
        'ee_offset_rotation': xp.array(fk_params['ee_offset_rotation']),
        'ref_angles': xp.array(fk_params['ref_angles']),
    }

    # Add collision data if available
    if problem.collision_spheres is not None:
        fk_data['collision_link_to_chain_idx'] = xp.array(
            problem.collision_link_to_chain_idx)
        fk_data['collision_link_offsets_pos'] = xp.array(
            problem.collision_link_offsets_pos)
        fk_data['collision_link_offsets_rot'] = xp.array(
            problem.collision_link_offsets_rot)
        fk_data['sphere_centers_local'] = xp.array(
            problem.collision_spheres['sphere_centers_local'])
        fk_data['sphere_radii'] = xp.array(
            problem.collision_spheres['sphere_radii'])
        fk_data['collision_link_indices'] = xp.array(
            problem.collision_spheres['link_indices'])

    return fk_data


__all__ = [
    'build_fk_functions',
    'rotation_error_vector',
    'rotation_error_log',
    'pose_error_log',
    'compute_sphere_obstacle_distances',
    'compute_self_collision_distances',
    'compute_collision_residuals',
    'prepare_fk_data',
]
