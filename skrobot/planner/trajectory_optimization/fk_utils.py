"""Forward kinematics utilities for trajectory optimization.

This module provides backend-agnostic FK computation functions
shared across different solvers (scipy, jaxls, gradient_descent).
"""

from skrobot.backend import rodrigues_rotation
from skrobot.kinematics.differentiable import pose_error_se3_log as pose_error_log
from skrobot.kinematics.differentiable import rotation_error_so3_log as rotation_error_log


def fk_runtime_arrays(fk_data):
    """Return only the array-valued entries of ``fk_data``.

    The dict also holds ``n_joints`` (int) and ``joint_types`` (list of str),
    which cannot be passed to a jitted function. Those describe the structure
    of the computation rather than its inputs, so ``build_fk_functions``
    captures them at build time and only the arrays need to travel.

    Parameters
    ----------
    fk_data : dict
        Output of :func:`prepare_fk_data`.

    Returns
    -------
    dict
        Array-valued entries, ready to pass to a jitted or vmapped function.
    """
    return {key: value for key, value in fk_data.items()
            if value is not None
            and hasattr(value, 'shape') and hasattr(value, 'dtype')}


def build_fk_functions(fk_data, backend, parameterized=False):
    """Build forward kinematics helper functions.

    Parameters
    ----------
    fk_data : dict
        FK parameters including:
        - link_translations: (n_joints, 3) link translations
        - link_rotations: (n_joints, 3, 3) link rotations
        - joint_axes: (n_joints, 3) joint axes
        - joint_types: (n_joints,) joint type strings. A 'prismatic' joint
          translates along its axis; anything else rotates about it. If the
          key is absent every joint is treated as rotational.
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
    parameterized : bool
        If ``False`` (default) the returned functions take ``(angles)`` and
        read the arrays from ``fk_data``, which is the historical contract.
        If ``True`` they take ``(angles, fk)`` and read the arrays from
        ``fk``, so a single traced function serves every problem with the
        same structure. Use :func:`fk_runtime_arrays` to build ``fk``.

        Only values may vary between calls. The structure -- joint count,
        joint types, and which optional entries are present -- is still read
        from ``fk_data`` when the functions are built, because it decides
        the shape of the computation graph.

    Returns
    -------
    tuple
        (get_link_transforms, get_sphere_positions, get_ee_position,
        get_ee_pose) functions.
    """
    xp = backend

    # Structure: fixes the shape of the graph, so it stays constant.
    n_joints = fk_data['n_joints']
    joint_types = fk_data.get('joint_types')
    has_ref = fk_data.get('ref_angles') is not None
    has_ee_pos = fk_data.get('ee_offset_position') is not None
    has_ee_rot = fk_data.get('ee_offset_rotation') is not None
    has_spheres = fk_data.get('sphere_centers_local') is not None
    has_static = fk_data.get('collision_link_is_static') is not None

    def get_link_transforms(angles, fk):
        """Compute link transforms for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).
        fk : dict
            FK arrays.

        Returns
        -------
        tuple
            (positions, rotations) arrays of shape
            (n_joints, 3) and (n_joints, 3, 3).
        """
        link_trans = fk['link_translations']
        link_rots = fk['link_rotations']
        joint_axes = fk['joint_axes']
        ref_angles = fk['ref_angles'] if has_ref else None

        positions = []
        rotations = []
        current_pos = fk['base_position']
        current_rot = fk['base_rotation']

        for i in range(n_joints):
            current_pos = current_pos + current_rot @ link_trans[i]
            current_rot = current_rot @ link_rots[i]
            # Subtract ref_angles because link_rots already includes
            # the rotation at the reference configuration
            delta = angles[i]
            if ref_angles is not None:
                delta = delta - ref_angles[i]
            if joint_types is not None and joint_types[i] == 'prismatic':
                current_pos = current_pos \
                    + current_rot @ (joint_axes[i] * delta)
            else:
                joint_rot = rodrigues_rotation(xp, joint_axes[i], delta)
                current_rot = current_rot @ joint_rot
            positions.append(current_pos)
            rotations.append(current_rot)

        return xp.stack(positions), xp.stack(rotations)

    def get_ee_pose(angles, fk):
        """Compute end-effector position and rotation.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).
        fk : dict
            FK arrays.

        Returns
        -------
        position : array
            End-effector position in world frame (3,).
        rotation : array
            End-effector rotation matrix in world frame (3, 3).
        """
        positions, rotations = get_link_transforms(angles, fk)
        last_pos = positions[-1]
        last_rot = rotations[-1]
        if has_ee_pos:
            ee_pos = last_pos + last_rot @ fk['ee_offset_position']
        else:
            ee_pos = last_pos
        if has_ee_rot:
            ee_rot = last_rot @ fk['ee_offset_rotation']
        else:
            ee_rot = last_rot
        return ee_pos, ee_rot

    def get_ee_position(angles, fk):
        """Compute end-effector position for given joint angles.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).
        fk : dict
            FK arrays.

        Returns
        -------
        array
            End-effector position in world frame (3,).
        """
        pos, _ = get_ee_pose(angles, fk)
        return pos

    def get_collision_link_transforms(angles, fk):
        """Compute world transforms of the collision links.

        Links that the optimized chain cannot move (see
        ``collision_link_is_static``) are placed from the base frame instead
        of from a chain link.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).
        fk : dict
            FK arrays.

        Returns
        -------
        tuple
            (positions, rotations) arrays of shape (n_coll_links, 3) and
            (n_coll_links, 3, 3) in the world frame.
        """
        link_positions, link_rotations = get_link_transforms(angles, fk)
        coll_link_idx = fk['collision_link_to_chain_idx']
        chain_pos = link_positions[coll_link_idx]
        chain_rot = link_rotations[coll_link_idx]

        if has_static:
            static = fk['collision_link_is_static'][:, None]
            base_pos = fk['base_position']
            base_rot = fk['base_rotation']
            chain_pos = xp.where(static, base_pos[None, :], chain_pos)
            chain_rot = xp.where(static[:, :, None], base_rot[None, :, :],
                                 chain_rot)

        world_pos = chain_pos + xp.einsum(
            'cij,cj->ci', chain_rot, fk['collision_link_offsets_pos'])
        world_rot = xp.einsum(
            'cij,cjk->cik', chain_rot, fk['collision_link_offsets_rot'])
        return world_pos, world_rot

    def get_sphere_positions(angles, fk):
        """Compute collision sphere positions for given joint angles.

        Spheres approximate collision geometries (spheres or capsules)
        attached to robot links.

        Parameters
        ----------
        angles : array
            Joint angles (n_joints,).
        fk : dict
            FK arrays.

        Returns
        -------
        array
            Sphere positions in world frame (n_spheres, 3).
        """
        if not has_spheres:
            return xp.zeros((0, 3))

        link_pos, link_rot = get_collision_link_transforms(angles, fk)
        sphere_link_indices = fk['collision_link_indices']
        sphere_pos = link_pos[sphere_link_indices]
        sphere_rot = link_rot[sphere_link_indices]
        return sphere_pos + xp.einsum(
            'ijk,ik->ij', sphere_rot, fk['sphere_centers_local'])

    # Exposed through build_collision_link_transform_fn rather than the return
    # tuple, whose arity is part of this module's public API.
    get_sphere_positions.collision_link_transforms = \
        get_collision_link_transforms

    if parameterized:
        return (get_link_transforms, get_sphere_positions, get_ee_position,
                get_ee_pose)

    def _bind(fn):
        def bound(angles):
            return fn(angles, fk_data)
        bound.__name__ = fn.__name__
        bound.__doc__ = fn.__doc__
        return bound

    bound_link = _bind(get_link_transforms)
    bound_spheres = _bind(get_sphere_positions)
    bound_ee_pos = _bind(get_ee_position)
    bound_ee_pose = _bind(get_ee_pose)
    bound_spheres.collision_link_transforms = _bind(
        get_collision_link_transforms)
    return bound_link, bound_spheres, bound_ee_pos, bound_ee_pose


def build_collision_link_transform_fn(fk_data, backend,
                                      parameterized=False):
    """Build ``angles -> (positions, rotations)`` for the collision links.

    The returned callable applies each collision link's offset from its
    kinematic-chain ancestor, and places links the chain cannot move (see
    ``collision_link_is_static``) from the base frame instead.

    Parameters
    ----------
    fk_data : dict
        Output of :func:`prepare_fk_data`.  Must contain the collision link
        entries, i.e. ``add_collision_cost`` must have been called.
    backend : module
        Array module (``numpy`` or ``jax.numpy``).

    Returns
    -------
    callable
        ``f(angles) -> (positions, rotations)`` of shape (n_coll_links, 3)
        and (n_coll_links, 3, 3) in the world frame. With
        ``parameterized=True`` the signature is ``f(angles, fk)``.
    """
    _, get_sphere_positions, _, _ = build_fk_functions(
        fk_data, backend, parameterized=parameterized)
    return get_sphere_positions.collision_link_transforms


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


def build_chain_link_transforms_with_base(fk_data, backend):
    """Like :func:`build_fk_functions`'s ``get_link_transforms`` but the
    base pose is an *argument*, not a closure-captured constant.

    Needed for trajectory optimisation where the base translation /
    rotation are part of the per-waypoint variable (floating-base DoF).

    Parameters
    ----------
    fk_data : dict
        Same FK data dict consumed by :func:`build_fk_functions`.
    backend : module
        Array module.

    Returns
    -------
    callable
        ``get_link_transforms(angles, base_pos, base_rot)`` returning
        ``(positions, rotations)``.
    """
    xp = backend
    link_trans = fk_data['link_translations']
    link_rots = fk_data['link_rotations']
    joint_axes = fk_data['joint_axes']
    n_joints = fk_data['n_joints']
    ref_angles = fk_data.get('ref_angles')

    def get_link_transforms(angles, base_pos, base_rot):
        positions = []
        rotations = []
        current_pos = base_pos
        current_rot = base_rot
        for i in range(n_joints):
            current_pos = current_pos + current_rot @ link_trans[i]
            current_rot = current_rot @ link_rots[i]
            delta = angles[i]
            if ref_angles is not None:
                delta = delta - ref_angles[i]
            joint_rot = rodrigues_rotation(xp, joint_axes[i], delta)
            current_rot = current_rot @ joint_rot
            positions.append(current_pos)
            rotations.append(current_rot)
        return xp.stack(positions), xp.stack(rotations)

    return get_link_transforms


def build_chain_ee_pose_with_base(fk_data, backend):
    """EE-pose function with explicit base pose argument.

    Counterpart to :func:`build_chain_link_transforms_with_base`. The
    EE offset (last_link -> move_target) baked into ``fk_data`` is
    applied at the end.
    """
    get_link_transforms = build_chain_link_transforms_with_base(fk_data, backend)
    ee_off_pos = fk_data.get('ee_offset_position')
    ee_off_rot = fk_data.get('ee_offset_rotation')

    def get_ee_pose(angles, base_pos, base_rot):
        positions, rotations = get_link_transforms(angles, base_pos, base_rot)
        last_pos = positions[-1]
        last_rot = rotations[-1]
        if ee_off_pos is not None:
            ee_pos = last_pos + last_rot @ ee_off_pos
        else:
            ee_pos = last_pos
        if ee_off_rot is not None:
            ee_rot = last_rot @ ee_off_rot
        else:
            ee_rot = last_rot
        return ee_pos, ee_rot

    return get_ee_pose


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
        # Kept as a Python list: it selects the branch in
        # build_fk_functions rather than taking part in the arithmetic.
        'joint_types': list(fk_params['joint_types']),
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
        fk_data['collision_link_is_static'] = xp.array(
            problem.collision_link_is_static)
        fk_data['sphere_centers_local'] = xp.array(
            problem.collision_spheres['sphere_centers_local'])
        fk_data['sphere_radii'] = xp.array(
            problem.collision_spheres['sphere_radii'])
        fk_data['collision_link_indices'] = xp.array(
            problem.collision_spheres['link_indices'])

    return fk_data


__all__ = [
    'build_collision_link_transform_fn',
    'build_fk_functions',
    'rotation_error_vector',
    'rotation_error_log',
    'pose_error_log',
    'compute_sphere_obstacle_distances',
    'compute_self_collision_distances',
    'compute_collision_residuals',
    'prepare_fk_data',
]
