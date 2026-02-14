"""Backend-agnostic differentiable kinematics.

This module provides forward kinematics and Jacobian computation that works
with any differentiable backend (NumPy, JAX, PyTorch).
"""

import numpy as np

from skrobot.coordinates.math import normalize_axis_mask


# =============================================================================
# Lie Group Utilities for Rotation Error
# =============================================================================

def rotation_error_so3_log(actual_rot, target_rot):
    """Compute rotation error using SO(3) logarithmic map.

    Uses jaxlie's SO3 logarithmic map to compute the rotation error
    as an axis-angle vector. This provides better numerical properties
    than the anti-symmetric extraction method, especially for large rotations.

    The error is computed as: log(actual_rot^T @ target_rot)
    which gives the rotation needed to go from actual to target in the
    local (body) frame.

    Parameters
    ----------
    actual_rot : jax.Array
        Actual rotation matrix (3, 3).
    target_rot : jax.Array
        Target rotation matrix (3, 3).

    Returns
    -------
    jax.Array
        Rotation error vector (3,) in axis-angle representation.
        The magnitude is the rotation angle in radians.

    Notes
    -----
    This function requires jaxlie to be installed. It is designed
    for use with JAX-based solvers.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> actual = jnp.eye(3)
    >>> target = jnp.eye(3)
    >>> err = rotation_error_so3_log(actual, target)
    >>> jnp.allclose(err, jnp.zeros(3))
    True
    """
    import jaxlie

    actual_so3 = jaxlie.SO3.from_matrix(actual_rot)
    target_so3 = jaxlie.SO3.from_matrix(target_rot)
    # Compute error in local frame: (actual^{-1} @ target).log()
    # This gives the rotation from actual to target
    error_so3 = actual_so3.inverse() @ target_so3
    return error_so3.log()


def pose_error_se3_log(actual_pos, actual_rot, target_pos, target_rot):
    """Compute pose error using SE(3) logarithmic map.

    Uses jaxlie's SE3 logarithmic map to compute the full pose error
    as a 6D vector (3D translation + 3D rotation). This provides a
    unified error representation on the SE(3) manifold.

    Parameters
    ----------
    actual_pos : jax.Array
        Actual position (3,).
    actual_rot : jax.Array
        Actual rotation matrix (3, 3).
    target_pos : jax.Array
        Target position (3,).
    target_rot : jax.Array
        Target rotation matrix (3, 3).

    Returns
    -------
    jax.Array
        Pose error vector (6,) where:
        - [0:3] is the translation error
        - [3:6] is the rotation error (axis-angle)

    Notes
    -----
    This function requires jaxlie to be installed. The SE(3) logarithmic
    map provides a mathematically principled way to measure pose differences
    that respects the geometry of rigid body transformations.
    """
    import jaxlie

    actual_so3 = jaxlie.SO3.from_matrix(actual_rot)
    target_so3 = jaxlie.SO3.from_matrix(target_rot)
    actual_se3 = jaxlie.SE3.from_rotation_and_translation(actual_so3, actual_pos)
    target_se3 = jaxlie.SE3.from_rotation_and_translation(target_so3, target_pos)
    # Compute error: (actual^{-1} @ target).log()
    error_se3 = actual_se3.inverse() @ target_se3
    return error_se3.log()


def rotation_error_so3_log_batch(actual_rot_batch, target_rot_batch):
    """Compute batched rotation error using SO(3) logarithmic map.

    Vectorized version of rotation_error_so3_log for batch processing.

    Parameters
    ----------
    actual_rot_batch : jax.Array
        Actual rotation matrices (batch, 3, 3).
    target_rot_batch : jax.Array
        Target rotation matrices (batch, 3, 3).

    Returns
    -------
    jax.Array
        Rotation error vectors (batch, 3) in axis-angle representation.

    Notes
    -----
    This function requires jaxlie to be installed. Uses jax.vmap for
    efficient batch processing.
    """
    import jax
    import jaxlie

    def single_rot_error(actual_rot, target_rot):
        actual_so3 = jaxlie.SO3.from_matrix(actual_rot)
        target_so3 = jaxlie.SO3.from_matrix(target_rot)
        error_so3 = actual_so3.inverse() @ target_so3
        return error_so3.log()

    return jax.vmap(single_rot_error)(actual_rot_batch, target_rot_batch)


# =============================================================================
# Helper Functions for IK Solvers
# =============================================================================

def _create_mirror_rotation_matrix(axis, backend=None):
    """Create a 180-degree rotation matrix around the specified axis.

    Parameters
    ----------
    axis : str
        Axis to rotate around ('x', 'y', or 'z').
    backend : object, optional
        Backend to use. If None, returns numpy array.

    Returns
    -------
    array
        3x3 rotation matrix, or None if axis is None.
    """
    if axis is None:
        return None

    if axis == 'x':
        mat = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    elif axis == 'y':
        mat = np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., -1.]])
    elif axis == 'z':
        mat = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
    else:
        return None

    if backend is not None:
        return backend.array(mat)
    return mat


def _get_mimic_joint_info(fk_params):
    """Extract mimic joint information from FK parameters.

    Parameters
    ----------
    fk_params : dict
        FK parameters from extract_fk_parameters.

    Returns
    -------
    tuple
        (mimic_parent_indices, mimic_multipliers, mimic_offsets, non_mimic_indices, n_opt)
    """
    n_joints = fk_params['n_joints']
    mimic_parent_indices = fk_params.get(
        'mimic_parent_indices', np.array([-1] * n_joints, dtype=np.int32))
    mimic_multipliers = fk_params.get('mimic_multipliers', np.ones(n_joints))
    mimic_offsets = fk_params.get('mimic_offsets', np.zeros(n_joints))

    non_mimic_indices = np.array(
        [i for i in range(n_joints) if mimic_parent_indices[i] < 0],
        dtype=np.int32)
    n_opt = len(non_mimic_indices)

    return mimic_parent_indices, mimic_multipliers, mimic_offsets, non_mimic_indices, n_opt


def _select_best_attempts(solutions, success_flags, errors, n_targets, attempts_per_pose,
                          n_joints, select_closest_to_initial=False, initial_angles=None):
    """Select the best solution from multiple attempts per target.

    Parameters
    ----------
    solutions : np.ndarray
        All solutions (n_targets * attempts_per_pose, n_joints).
    success_flags : np.ndarray
        Success flags (n_targets * attempts_per_pose,).
    errors : np.ndarray
        Combined errors (n_targets * attempts_per_pose,).
    n_targets : int
        Number of target poses.
    attempts_per_pose : int
        Number of attempts per target.
    n_joints : int
        Number of joints.
    select_closest_to_initial : bool
        If True, prefer solutions closest to initial angles.
    initial_angles : np.ndarray, optional
        Initial angles for distance comparison.

    Returns
    -------
    tuple
        (best_solutions, best_success, best_errors)
    """
    solutions = solutions.reshape(n_targets, attempts_per_pose, n_joints)
    success_flags = success_flags.reshape(n_targets, attempts_per_pose)
    errors = errors.reshape(n_targets, attempts_per_pose)

    if select_closest_to_initial and initial_angles is not None:
        init_angles = np.asarray(initial_angles)
        if init_angles.ndim == 1:
            init_angles = np.tile(init_angles, (n_targets, 1))

        best_indices = []
        err_threshold = 0.02  # Consider solutions with error < 2cm as valid
        for i in range(n_targets):
            # First attempt (index 0) starts from current angles
            first_success = success_flags[i, 0] or errors[i, 0] < err_threshold
            if first_success:
                best_idx = 0
            else:
                valid_mask = success_flags[i] | (errors[i] < err_threshold)
                if np.any(valid_mask):
                    distances = np.linalg.norm(
                        solutions[i, valid_mask] - init_angles[i], axis=1)
                    valid_indices = np.where(valid_mask)[0]
                    best_idx = valid_indices[np.argmin(distances)]
                else:
                    best_idx = np.argmin(errors[i])
            best_indices.append(best_idx)
        best_indices = np.array(best_indices)
    else:
        best_indices = np.argmin(errors, axis=1)

    target_indices = np.arange(n_targets)
    best_solutions = solutions[target_indices, best_indices]
    best_success = success_flags[target_indices, best_indices]
    best_errors = errors[target_indices, best_indices]

    return best_solutions, best_success, best_errors


def extract_fk_parameters(robot_model, link_list, move_target):
    """Extract FK parameters from a robot model for differentiable computation.

    This function extracts all static transforms and joint information needed
    to implement FK in a differentiable way, enabling autodiff support.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    link_list : list
        List of links in the kinematic chain.
    move_target : CascadedCoords
        End effector coordinates.

    Returns
    -------
    dict
        Dictionary containing:
        - 'n_joints': Number of joints
        - 'link_translations': Static translations for each link (n_joints, 3)
        - 'link_rotations': Static rotations for each link (n_joints, 3, 3)
        - 'joint_axes': Joint rotation axes (n_joints, 3)
        - 'joint_types': Joint types (list of str)
        - 'joint_limits_lower': Lower joint limits (n_joints,)
        - 'joint_limits_upper': Upper joint limits (n_joints,)
        - 'base_position': Base link world position (3,)
        - 'base_rotation': Base link world rotation (3, 3)
        - 'ee_offset_position': End effector offset from last link (3,)
        - 'ee_offset_rotation': End effector offset rotation (3, 3)

    Examples
    --------
    >>> from skrobot.kinematics.differentiable import extract_fk_parameters
    >>> fk_params = extract_fk_parameters(robot, link_list, move_target)
    >>> print(fk_params['n_joints'])
    7
    """
    n_joints = len(link_list)

    # Save original joint angles
    original_angles = [link.joint.joint_angle() for link in link_list]

    # Extract mimic joint information
    # For each joint, store (parent_index, multiplier, offset) if it's a mimic joint
    # parent_index is -1 if not a mimic joint
    mimic_info = []
    joint_name_to_index = {}

    # First pass: build joint name to index mapping
    for i, link in enumerate(link_list):
        joint_name_to_index[link.joint.name] = i

    # Second pass: identify mimic joints
    for i, link in enumerate(link_list):
        joint = link.joint
        if hasattr(joint, 'mimic') and joint.mimic is not None:
            parent_joint = joint.mimic.get('joint')
            if parent_joint is not None and parent_joint.name in joint_name_to_index:
                parent_idx = joint_name_to_index[parent_joint.name]
                multiplier = joint.mimic.get('multiplier', 1.0)
                offset = joint.mimic.get('offset', 0.0)
                mimic_info.append((parent_idx, multiplier, offset))
            else:
                mimic_info.append((-1, 1.0, 0.0))
        else:
            mimic_info.append((-1, 1.0, 0.0))

    # Determine reference angles for each joint.
    # Use 0 if valid, otherwise use the closest limit.
    # This is needed because some joints (e.g., Panda joint4) have limits
    # that don't include 0.
    ref_angles = []
    joint_limits_lower = []
    joint_limits_upper = []

    for i, link in enumerate(link_list):
        joint = link.joint

        # Get joint limits
        min_angle = getattr(joint, 'min_angle', None)
        max_angle = getattr(joint, 'max_angle', None)

        if min_angle is None or not np.isfinite(min_angle):
            min_angle = -np.pi
        if max_angle is None or not np.isfinite(max_angle):
            max_angle = np.pi

        joint_limits_lower.append(min_angle)
        joint_limits_upper.append(max_angle)

        # For mimic joints, ref_angle is computed from parent's ref_angle
        parent_idx, multiplier, offset = mimic_info[i]
        if parent_idx >= 0:
            # Mimic joint: ref_angle = parent_ref * multiplier + offset
            parent_ref = ref_angles[parent_idx]
            ref_angle = parent_ref * multiplier + offset
        else:
            # Regular joint: choose reference angle
            if min_angle <= 0 <= max_angle:
                ref_angle = 0.0
            elif 0 < min_angle:
                ref_angle = min_angle
            else:  # max_angle < 0
                ref_angle = max_angle

        ref_angles.append(ref_angle)

    ref_angles = np.array(ref_angles)

    # Set joints to their reference angles
    # Skip mimic joints - they are set automatically when parent joint is set
    for i, link in enumerate(link_list):
        parent_idx, _, _ = mimic_info[i]
        if parent_idx < 0:  # Not a mimic joint
            link.joint.joint_angle(ref_angles[i])

    # Get base link world transform (parent of first link in chain)
    first_link = link_list[0]
    if first_link.parent is not None:
        base_position = first_link.parent.worldpos().copy()
        base_rotation = first_link.parent.worldrot().copy()
    else:
        base_position = np.zeros(3)
        base_rotation = np.eye(3)

    # Extract static transforms for each link using worldcoords()
    # This properly handles fixed links between movable links in link_list.
    # The transforms are computed at the reference configuration.
    link_translations = []
    link_rotations = []
    joint_axes = []
    joint_types = []

    for i, link in enumerate(link_list):
        joint = link.joint

        if i == 0:
            # First link: transform from base (parent of first link)
            from_coords = first_link.parent.worldcoords() if first_link.parent else None
        else:
            # Subsequent links: transform from previous link in list
            from_coords = link_list[i - 1].worldcoords()

        # Compute relative transform at reference configuration.
        link_coords = link.worldcoords()

        if from_coords is not None:
            # Relative transform: from_coords^-1 * link_coords
            rel_coords = from_coords.inverse_transformation().transform(link_coords)
            trans = rel_coords.worldpos().copy()
            rot = rel_coords.worldrot().copy()
        else:
            # No parent, use link's world coordinates directly
            trans = link_coords.worldpos().copy()
            rot = link_coords.worldrot().copy()

        link_translations.append(trans)
        link_rotations.append(rot)

        # Joint info
        joint_axes.append(np.array(joint.axis))
        joint_types.append(joint.joint_type)

    # Get end effector offset from last link
    last_link = link_list[-1]

    # Compute relative transform from last link to move_target at ref config
    rel_coords = last_link.worldcoords().inverse_transformation().transform(
        move_target.worldcoords()
    )
    ee_offset_position = rel_coords.worldpos().copy()
    ee_offset_rotation = rel_coords.worldrot().copy()

    # Restore original angles
    for i, link in enumerate(link_list):
        link.joint.joint_angle(original_angles[i])

    # Convert mimic_info to arrays for JAX compatibility
    mimic_parent_indices = np.array([m[0] for m in mimic_info], dtype=np.int32)
    mimic_multipliers = np.array([m[1] for m in mimic_info])
    mimic_offsets = np.array([m[2] for m in mimic_info])

    return {
        'n_joints': n_joints,
        'link_translations': np.array(link_translations),
        'link_rotations': np.array(link_rotations),
        'joint_axes': np.array(joint_axes),
        'joint_types': joint_types,
        'joint_limits_lower': np.array(joint_limits_lower),
        'joint_limits_upper': np.array(joint_limits_upper),
        'ref_angles': ref_angles,
        'base_position': base_position,
        'base_rotation': base_rotation,
        'ee_offset_position': ee_offset_position,
        'ee_offset_rotation': ee_offset_rotation,
        'mimic_parent_indices': mimic_parent_indices,
        'mimic_multipliers': mimic_multipliers,
        'mimic_offsets': mimic_offsets,
    }


def forward_kinematics(backend, joint_angles, fk_params):
    """Compute forward kinematics using the specified backend.

    This function computes the position and orientation of each link
    in the kinematic chain.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles : array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters(), containing:
        - 'link_translations': (n_joints, 3)
        - 'link_rotations': (n_joints, 3, 3)
        - 'joint_axes': (n_joints, 3)
        - 'base_position': (3,)
        - 'base_rotation': (3, 3)

    Returns
    -------
    positions : array
        Link positions (n_joints, 3).
    rotations : array
        Link rotations (n_joints, 3, 3).

    Examples
    --------
    >>> from skrobot.backend import get_backend
    >>> from skrobot.kinematics.differentiable import extract_fk_parameters
    >>> backend = get_backend('jax')
    >>> fk_params = extract_fk_parameters(robot, link_list, move_target)
    >>> q = backend.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    >>> positions, rotations = forward_kinematics(backend, q, fk_params)
    """
    n_joints = fk_params['n_joints']
    translations = backend.array(fk_params['link_translations'])
    local_rotations = backend.array(fk_params['link_rotations'])
    axes = backend.array(fk_params['joint_axes'])
    base_pos = backend.array(fk_params['base_position'])
    base_rot = backend.array(fk_params['base_rotation'])
    ref_angles = backend.array(fk_params['ref_angles'])

    # Handle mimic joints: compute effective joint angles
    # For mimic joints, angle = parent_angle * multiplier + offset
    mimic_parent_indices = fk_params.get('mimic_parent_indices')
    # Check if any mimic joints exist (using numpy, before JAX tracing)
    has_mimic_joints = (mimic_parent_indices is not None
                        and np.any(np.asarray(mimic_parent_indices) >= 0))
    if has_mimic_joints:
        mimic_multipliers = backend.array(fk_params['mimic_multipliers'])
        mimic_offsets = backend.array(fk_params['mimic_offsets'])

        # Create effective joint angles array
        # For non-mimic joints (parent_index == -1), use original angle
        # For mimic joints, compute from parent
        effective_angles = []
        for i in range(n_joints):
            parent_idx = mimic_parent_indices[i]
            if parent_idx >= 0:
                # Mimic joint: angle = parent_angle * multiplier + offset
                parent_angle = joint_angles[parent_idx]
                effective_angle = parent_angle * mimic_multipliers[i] + mimic_offsets[i]
                effective_angles.append(effective_angle)
            else:
                # Regular joint
                effective_angles.append(joint_angles[i])
        joint_angles = backend.stack(effective_angles)

    positions = []
    rotations = []

    current_pos = base_pos
    current_rot = base_rot

    joint_types = fk_params['joint_types']

    for i in range(n_joints):
        # Apply link static transform (computed at reference configuration)
        current_pos = current_pos + backend.matmul(current_rot, translations[i])
        current_rot = backend.matmul(current_rot, local_rotations[i])

        # Apply joint motion as delta from reference angle
        # The static transforms include the reference configuration, so we apply
        # only the difference (joint_angles[i] - ref_angles[i])
        delta_angle = joint_angles[i] - ref_angles[i]

        if joint_types[i] == 'prismatic':
            # Prismatic joint: translate along axis
            current_pos = current_pos + backend.matmul(current_rot, axes[i] * delta_angle)
        else:
            # Revolute joint: rotate around axis
            joint_rot = _axis_angle_to_matrix(backend, axes[i], delta_angle)
            current_rot = backend.matmul(current_rot, joint_rot)

        positions.append(current_pos)
        rotations.append(current_rot)

    return backend.stack(positions), backend.stack(rotations)


def forward_kinematics_ee(backend, joint_angles, fk_params):
    """Compute end-effector pose using the specified backend.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles : array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters().

    Returns
    -------
    position : array
        End-effector position (3,).
    rotation : array
        End-effector rotation matrix (3, 3).

    Examples
    --------
    >>> backend = get_backend('jax')
    >>> pos, rot = forward_kinematics_ee(backend, q, fk_params)
    """
    positions, rotations = forward_kinematics(backend, joint_angles, fk_params)

    # Apply end-effector offset if present
    ee_pos = positions[-1]
    ee_rot = rotations[-1]

    if 'ee_offset_position' in fk_params:
        ee_offset_pos = backend.array(fk_params['ee_offset_position'])
        ee_offset_rot = backend.array(fk_params['ee_offset_rotation'])

        ee_pos = ee_pos + backend.matmul(ee_rot, ee_offset_pos)
        ee_rot = backend.matmul(ee_rot, ee_offset_rot)

    return ee_pos, ee_rot


def compute_jacobian_analytical(backend, joint_angles, fk_params):
    """Compute position Jacobian analytically (no autodiff).

    This is faster than compute_jacobian() for backends without
    efficient autodiff (e.g., NumPy with numerical differentiation).

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles : array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters().

    Returns
    -------
    jacobian : array
        Position Jacobian matrix (3, n_joints).
    ee_position : array
        End-effector position (3,).
    ee_rotation : array
        End-effector rotation matrix (3, 3).
    """
    n_joints = fk_params['n_joints']
    translations = backend.array(fk_params['link_translations'])
    local_rotations = backend.array(fk_params['link_rotations'])
    axes = backend.array(fk_params['joint_axes'])
    base_pos = backend.array(fk_params['base_position'])
    base_rot = backend.array(fk_params['base_rotation'])
    ref_angles = backend.array(fk_params['ref_angles'])
    joint_types = fk_params['joint_types']

    # Handle mimic joints: compute effective joint angles
    mimic_parent_indices = fk_params.get('mimic_parent_indices')
    if mimic_parent_indices is not None:
        mimic_multipliers = backend.array(fk_params['mimic_multipliers'])
        mimic_offsets = backend.array(fk_params['mimic_offsets'])

        # Create effective joint angles array
        effective_angles = []
        for i in range(n_joints):
            parent_idx = mimic_parent_indices[i]
            if parent_idx >= 0:
                # Mimic joint: angle = parent_angle * multiplier + offset
                parent_angle = joint_angles[parent_idx]
                effective_angle = parent_angle * mimic_multipliers[i] + mimic_offsets[i]
                effective_angles.append(effective_angle)
            else:
                # Regular joint
                effective_angles.append(joint_angles[i])
        joint_angles = backend.stack(effective_angles)

    positions = []
    z_axes = []

    current_pos = base_pos
    current_rot = base_rot

    for i in range(n_joints):
        # Apply link static transform
        current_pos = current_pos + backend.matmul(current_rot, translations[i])
        current_rot = backend.matmul(current_rot, local_rotations[i])

        # Store position and z-axis before joint motion
        positions.append(current_pos)
        z_axes.append(backend.matmul(current_rot, axes[i]))

        # Apply joint motion as delta from reference angle
        delta_angle = joint_angles[i] - ref_angles[i]
        if joint_types[i] == 'prismatic':
            # Prismatic joint: translate along axis
            current_pos = current_pos + backend.matmul(current_rot, axes[i] * delta_angle)
        else:
            # Revolute joint: rotate around axis
            joint_rot = _axis_angle_to_matrix(backend, axes[i], delta_angle)
            current_rot = backend.matmul(current_rot, joint_rot)

    # Apply end-effector offset
    ee_offset_pos = backend.array(fk_params['ee_offset_position'])
    ee_offset_rot = backend.array(fk_params['ee_offset_rotation'])
    ee_pos = current_pos + backend.matmul(current_rot, ee_offset_pos)
    ee_rot = backend.matmul(current_rot, ee_offset_rot)

    # Compute Jacobian
    # For revolute joints: J[:, i] = z_i × (p_ee - p_i)
    # For prismatic joints: J[:, i] = z_i (axis direction)
    J = backend.zeros((3, n_joints))
    for i in range(n_joints):
        z = z_axes[i]
        if joint_types[i] == 'prismatic':
            # Prismatic: Jacobian column is the axis direction
            if backend.name == 'jax':
                J = J.at[:, i].set(z)
            else:
                J[:, i] = z
        else:
            # Revolute: Jacobian column is z × r
            r = ee_pos - positions[i]
            cross_prod = backend.cross(z, r)
            if backend.name == 'jax':
                J = J.at[:, i].set(cross_prod)
            else:
                J[:, i] = cross_prod

    return J, ee_pos, ee_rot


def compute_jacobian_analytical_batch(joint_angles_batch, fk_params):
    """Compute position Jacobian analytically for a batch of configurations.

    This is a NumPy-optimized batch implementation that processes all
    configurations in parallel using vectorized operations.

    Parameters
    ----------
    joint_angles_batch : np.ndarray
        Joint angles (batch_size, n_joints).
    fk_params : dict
        FK parameters from extract_fk_parameters().

    Returns
    -------
    jacobians : np.ndarray
        Position Jacobian matrices (batch_size, 3, n_joints).
    ee_positions : np.ndarray
        End-effector positions (batch_size, 3).
    ee_rotations : np.ndarray
        End-effector rotation matrices (batch_size, 3, 3).
    """
    import numpy as np

    batch_size = joint_angles_batch.shape[0]
    n_joints = fk_params['n_joints']
    translations = np.asarray(fk_params['link_translations'])
    local_rotations = np.asarray(fk_params['link_rotations'])
    axes = np.asarray(fk_params['joint_axes'])
    base_pos = np.asarray(fk_params['base_position'])
    base_rot = np.asarray(fk_params['base_rotation'])
    ref_angles = np.asarray(fk_params['ref_angles'])
    ee_offset_pos = np.asarray(fk_params['ee_offset_position'])
    ee_offset_rot = np.asarray(fk_params['ee_offset_rotation'])
    joint_types = fk_params['joint_types']

    # Handle mimic joints: compute effective joint angles
    mimic_parent_indices = fk_params.get('mimic_parent_indices')
    if mimic_parent_indices is not None:
        mimic_multipliers = np.asarray(fk_params['mimic_multipliers'])
        mimic_offsets = np.asarray(fk_params['mimic_offsets'])

        # Create effective joint angles array
        effective_angles = joint_angles_batch.copy()
        for i in range(n_joints):
            parent_idx = mimic_parent_indices[i]
            if parent_idx >= 0:
                # Mimic joint: angle = parent_angle * multiplier + offset
                effective_angles[:, i] = (
                    joint_angles_batch[:, parent_idx] * mimic_multipliers[i]
                    + mimic_offsets[i]
                )
        joint_angles_batch = effective_angles

    # Initialize batch arrays
    # current_pos: (batch_size, 3)
    # current_rot: (batch_size, 3, 3)
    current_pos = np.tile(base_pos, (batch_size, 1))
    current_rot = np.tile(base_rot, (batch_size, 1, 1))

    # Store positions and z-axes for Jacobian computation
    positions = np.zeros((n_joints, batch_size, 3))
    z_axes = np.zeros((n_joints, batch_size, 3))

    for i in range(n_joints):
        # Apply link static transform
        # current_pos += current_rot @ translations[i]
        current_pos = current_pos + np.einsum('bij,j->bi', current_rot, translations[i])
        # current_rot = current_rot @ local_rotations[i]
        current_rot = np.einsum('bij,jk->bik', current_rot, local_rotations[i])

        # Store position and z-axis before joint motion
        positions[i] = current_pos
        # z_axes[i] = current_rot @ axes[i]
        z_axes[i] = np.einsum('bij,j->bi', current_rot, axes[i])

        # Apply joint motion as delta from reference angle (batched)
        delta = joint_angles_batch[:, i] - ref_angles[i]  # (batch_size,)
        axis = axes[i]  # (3,)

        if joint_types[i] == 'prismatic':
            # Prismatic joint: translate along axis
            # current_pos += current_rot @ (axis * delta)
            axis_scaled = axis[None, :] * delta[:, None]  # (batch_size, 3)
            current_pos = current_pos + np.einsum('bij,bj->bi', current_rot, axis_scaled)
        else:
            # Revolute joint: rotate around axis
            # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
            # where K is skew-symmetric matrix of axis
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            K2 = K @ K

            sin_theta = np.sin(delta)[:, None, None]  # (batch_size, 1, 1)
            cos_theta = np.cos(delta)[:, None, None]
            I = np.eye(3)

            # joint_rot: (batch_size, 3, 3)
            joint_rot = I + sin_theta * K + (1 - cos_theta) * K2

            # current_rot = current_rot @ joint_rot
            current_rot = np.einsum('bij,bjk->bik', current_rot, joint_rot)

    # Apply end-effector offset
    ee_pos = current_pos + np.einsum('bij,j->bi', current_rot, ee_offset_pos)
    ee_rot = np.einsum('bij,jk->bik', current_rot, ee_offset_rot)

    # Compute Jacobian
    # For revolute joints: J[:, :, i] = z_i × (p_ee - p_i)
    # For prismatic joints: J[:, :, i] = z_i (axis direction)
    jacobians = np.zeros((batch_size, 3, n_joints))
    for i in range(n_joints):
        z = z_axes[i]  # (batch_size, 3)
        if joint_types[i] == 'prismatic':
            # Prismatic: Jacobian column is the axis direction
            jacobians[:, :, i] = z
        else:
            # Revolute: Jacobian column is z × r
            r = ee_pos - positions[i]  # (batch_size, 3)
            cross = np.cross(z, r)
            jacobians[:, :, i] = cross

    return jacobians, ee_pos, ee_rot


def compute_jacobian(backend, joint_angles, fk_params, link_index: int = -1):
    """Compute Jacobian matrix using automatic differentiation.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles : array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters().
    link_index : int
        Index of the link to compute Jacobian for. -1 for end-effector.

    Returns
    -------
    jacobian : array
        Jacobian matrix (3, n_joints) for position only.

    Examples
    --------
    >>> backend = get_backend('jax')
    >>> J = compute_jacobian(backend, q, fk_params)
    >>> # J is (3, 7) for 7-DOF arm
    """
    def position_fn(q):
        positions, _ = forward_kinematics(backend, q, fk_params)
        return positions[link_index]

    jac_fn = backend.jacobian(position_fn)
    return jac_fn(joint_angles)


def compute_full_jacobian(backend, joint_angles, fk_params, link_index: int = -1):
    """Compute full 6-DOF Jacobian (position + orientation).

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles : array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters().
    link_index : int
        Index of the link. -1 for end-effector.

    Returns
    -------
    jacobian : array
        Full Jacobian matrix (6, n_joints).
        First 3 rows are linear velocity, last 3 are angular velocity.
    """
    def pose_fn(q):
        positions, rotations = forward_kinematics(backend, q, fk_params)
        pos = positions[link_index]
        rot = rotations[link_index]
        # Return position and rotation in a differentiable form
        # Use rotation matrix columns for orientation representation
        return backend.concatenate([pos, rot[:, 0], rot[:, 1], rot[:, 2]])

    jac_fn = backend.jacobian(pose_fn)
    full_jac = jac_fn(joint_angles)  # (12, n_joints)

    # Extract position Jacobian (first 3 rows) and angular Jacobian (approximation)
    # For proper angular velocity Jacobian, we'd need more complex computation
    pos_jac = full_jac[:3, :]

    # Approximate angular Jacobian from rotation matrix derivatives
    # This is a simplified version - full implementation would use SO(3) math
    rot_jac = full_jac[3:6, :]  # Use first rotation column derivative

    return backend.concatenate([
        backend.expand_dims(pos_jac, 0),
        backend.expand_dims(rot_jac, 0)
    ]).reshape((6, -1))


def solve_ik_gradient_descent(
    backend,
    target_position,
    target_rotation,
    fk_params,
    initial_angles=None,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    pos_weight: float = 1.0,
    rot_weight: float = 0.1,
    tolerance: float = 1e-6,
):
    """Solve inverse kinematics using gradient descent.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    target_position : array
        Target end-effector position (3,).
    target_rotation : array
        Target end-effector rotation matrix (3, 3).
    fk_params : dict
        FK parameters from extract_fk_parameters().
    initial_angles : array, optional
        Initial joint angles. If None, uses zeros.
    max_iterations : int
        Maximum number of iterations.
    learning_rate : float
        Gradient descent step size.
    pos_weight : float
        Weight for position error.
    rot_weight : float
        Weight for rotation error.
    tolerance : float
        Convergence tolerance.

    Returns
    -------
    joint_angles : array
        Solved joint angles (n_joints,).
    success : bool
        Whether IK converged.
    final_error : float
        Final position error.

    Examples
    --------
    >>> backend = get_backend('jax')
    >>> target_pos = backend.array([0.5, 0.0, 0.5])
    >>> target_rot = backend.eye(3)
    >>> q, success, error = solve_ik_gradient_descent(
    ...     backend, target_pos, target_rot, fk_params
    ... )
    """
    joint_limits_lower = backend.array(fk_params['joint_limits_lower'])
    joint_limits_upper = backend.array(fk_params['joint_limits_upper'])

    if initial_angles is None:
        # Start at middle of joint limits
        initial_angles = (joint_limits_lower + joint_limits_upper) / 2
    else:
        initial_angles = backend.array(initial_angles)

    target_position = backend.array(target_position)
    target_rotation = backend.array(target_rotation)

    # Use SO(3) log rotation error if JAX backend is available
    use_so3_log = backend.name == 'jax'

    def loss_fn(q):
        pos, rot = forward_kinematics_ee(backend, q, fk_params)
        pos_error = backend.sum((pos - target_position) ** 2)
        if use_so3_log:
            # Use SO(3) logarithmic map for rotation error
            rot_err_vec = rotation_error_so3_log(rot, target_rotation)
            rot_error = backend.sum(rot_err_vec ** 2)
        else:
            # Fallback to matrix difference (less accurate for large rotations)
            rot_error = backend.sum((rot - target_rotation) ** 2)
        return pos_weight * pos_error + rot_weight * rot_error

    grad_fn = backend.gradient(loss_fn)

    # Compile for speed if supported
    compiled_loss = backend.compile(loss_fn)
    compiled_grad = backend.compile(grad_fn)

    angles = initial_angles
    prev_loss = float('inf')

    for i in range(max_iterations):
        loss = compiled_loss(angles)
        grad = compiled_grad(angles)

        # Update
        angles = angles - learning_rate * grad

        # Apply joint limits
        angles = backend.clip(angles, joint_limits_lower, joint_limits_upper)

        # Check convergence
        loss_np = float(backend.to_numpy(loss))
        if abs(prev_loss - loss_np) < tolerance:
            break
        prev_loss = loss_np

    # Compute final error
    final_pos, _ = forward_kinematics_ee(backend, angles, fk_params)
    final_error = float(backend.to_numpy(
        backend.sqrt(backend.sum((final_pos - target_position) ** 2))
    ))

    success = final_error < 0.01  # 1cm threshold

    return angles, success, final_error


def batch_forward_kinematics(backend, joint_angles_batch, fk_params):
    """Compute forward kinematics for a batch of configurations.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles_batch : array
        Batch of joint angles (batch_size, n_joints).
    fk_params : dict
        FK parameters from extract_fk_parameters().

    Returns
    -------
    positions : array
        Link positions (batch_size, n_joints, 3).
    rotations : array
        Link rotations (batch_size, n_joints, 3, 3).
    """
    def single_fk(q):
        return forward_kinematics(backend, q, fk_params)

    # Use vmap if available
    batched_fk = backend.vmap(single_fk)
    return batched_fk(joint_angles_batch)


def batch_solve_ik(
    backend,
    target_positions,
    target_rotations,
    fk_params,
    initial_angles=None,
    max_iterations: int = 100,
    learning_rate: float = 0.1,
    pos_weight: float = 1.0,
    rot_weight: float = 0.1,
    pos_threshold: float = 0.001,
):
    """Solve batch IK using gradient descent.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    target_positions : array
        Target positions (batch_size, 3).
    target_rotations : array
        Target rotations (batch_size, 3, 3).
    fk_params : dict
        FK parameters from extract_fk_parameters().
    initial_angles : array, optional
        Initial angles (batch_size, n_joints) or (n_joints,).
    max_iterations : int
        Maximum iterations per solve.
    learning_rate : float
        Gradient descent step size.
    pos_weight : float
        Position error weight.
    rot_weight : float
        Rotation error weight.
    pos_threshold : float
        Position error threshold for success determination.

    Returns
    -------
    solutions : array
        Solved joint angles (batch_size, n_joints).
    success_flags : array
        Success flags (batch_size,).
    errors : array
        Final position errors (batch_size,).
    """
    batch_size = target_positions.shape[0]

    joint_limits_lower = backend.array(fk_params['joint_limits_lower'])
    joint_limits_upper = backend.array(fk_params['joint_limits_upper'])

    if initial_angles is None:
        init = (joint_limits_lower + joint_limits_upper) / 2
        initial_angles = backend.stack([init] * batch_size)
    else:
        initial_angles = backend.array(initial_angles)
        if len(initial_angles.shape) == 1:
            initial_angles = backend.stack([initial_angles] * batch_size)

    target_positions = backend.array(target_positions)
    target_rotations = backend.array(target_rotations)

    def solve_single(init_q, target_pos, target_rot):
        def loss_fn(q):
            pos, rot = forward_kinematics_ee(backend, q, fk_params)
            pos_error = backend.sum((pos - target_pos) ** 2)
            rot_error = backend.sum((rot - target_rot) ** 2)
            return pos_weight * pos_error + rot_weight * rot_error

        grad_fn = backend.gradient(loss_fn)

        q = init_q
        for _ in range(max_iterations):
            grad = grad_fn(q)
            q = q - learning_rate * grad
            q = backend.clip(q, joint_limits_lower, joint_limits_upper)

        final_pos, _ = forward_kinematics_ee(backend, q, fk_params)
        error = backend.sqrt(backend.sum((final_pos - target_pos) ** 2))

        return q, error

    # Vectorize if possible
    if backend.supports_jit:
        batched_solve = backend.vmap(solve_single)
        solutions, errors = batched_solve(
            initial_angles, target_positions, target_rotations
        )
    else:
        # Sequential fallback
        solutions = []
        errors = []
        for i in range(batch_size):
            sol, err = solve_single(
                initial_angles[i], target_positions[i], target_rotations[i]
            )
            solutions.append(sol)
            errors.append(err)
        solutions = backend.stack(solutions)
        errors = backend.stack(errors)

    success_flags = errors < pos_threshold

    return solutions, success_flags, errors


def _create_numpy_optimized_solver(fk_params):
    """Create an optimized batch IK solver using pure NumPy with Jacobian method.

    This uses the analytical geometric Jacobian with damped least-squares,
    matching the JAX Jacobian solver but implemented in batched NumPy.
    """
    n_joints = fk_params['n_joints']

    # Identify non-mimic joints using helper
    (mimic_parent_indices, mimic_multipliers, mimic_offsets,
     non_mimic_indices, n_opt) = _get_mimic_joint_info(fk_params)

    joint_limits_lower = fk_params['joint_limits_lower'][non_mimic_indices]
    joint_limits_upper = fk_params['joint_limits_upper'][non_mimic_indices]

    def _expand_to_full_angles_batch(opt_angles_batch):
        """Expand optimization variables to full joint angles (batched)."""
        batch_size = opt_angles_batch.shape[0]
        full_angles = np.zeros((batch_size, n_joints))

        # Fill non-mimic joints
        full_angles[:, non_mimic_indices] = opt_angles_batch

        # Fill mimic joints
        for i in range(n_joints):
            parent_idx = mimic_parent_indices[i]
            if parent_idx >= 0:
                full_angles[:, i] = (
                    full_angles[:, parent_idx] * mimic_multipliers[i] + mimic_offsets[i]
                )

        return full_angles

    def solve(target_positions, target_rotations,
              initial_angles=None,
              max_iterations=30,
              damping=0.01,
              pos_threshold=0.001,
              rot_threshold=0.1,
              position_mask=True,
              rotation_mask=True,
              rotation_mirror=None,
              attempts_per_pose=1,
              use_current_angles=True,
              select_closest_to_initial=False,
              **kwargs):
        """Solve batch IK using Jacobian-based damped least-squares in NumPy.

        Parameters
        ----------
        target_positions : array-like
            Target positions (N, 3).
        target_rotations : array-like
            Target rotation matrices (N, 3, 3).
        initial_angles : array-like, optional
            Initial joint angles.
        max_iterations : int
            Maximum iterations (default 30).
        damping : float
            Damping factor for least squares (default 0.01).
        pos_threshold : float
            Position error threshold for success.
        rot_threshold : float
            Rotation error threshold for success.
        position_mask, rotation_mask, rotation_mirror, attempts_per_pose,
        use_current_angles, select_closest_to_initial : same as JAX solver.
        """
        target_positions = np.asarray(target_positions)
        target_rotations = np.asarray(target_rotations)

        if target_positions.ndim == 1:
            target_positions = target_positions.reshape(1, 3)
            target_rotations = target_rotations.reshape(1, 3, 3)

        n_targets = target_positions.shape[0]

        # Parse masks
        pos_mask_arr = normalize_axis_mask(position_mask)
        rot_mask_arr = normalize_axis_mask(rotation_mask)

        pos_mask_sum = np.sum(pos_mask_arr)
        rot_mask_sum = np.sum(rot_mask_arr)
        has_pos_constraint = pos_mask_sum > 0
        has_rot_constraint = rot_mask_sum > 0

        # Determine active Jacobian rows
        # Position: rows 0,1,2; Rotation: rows 3,4,5
        # For rotation, always include all 3 rows when there's any constraint
        # because masking is done in local frame and rotated to world frame
        active_rows = []
        if has_pos_constraint:
            for i in range(3):
                if pos_mask_arr[i] > 0:
                    active_rows.append(i)
        if has_rot_constraint:
            active_rows.extend([3, 4, 5])
        active_rows = np.array(active_rows, dtype=np.int32)
        n_active = len(active_rows)

        # Create mirror rotation if specified
        mirror_rot = _create_mirror_rotation_matrix(rotation_mirror)

        # Handle attempts_per_pose
        if attempts_per_pose > 1:
            target_positions_expanded = np.repeat(
                target_positions, attempts_per_pose, axis=0)
            target_rotations_expanded = np.repeat(
                target_rotations, attempts_per_pose, axis=0)

            n_expanded = n_targets * attempts_per_pose
            if initial_angles is not None and use_current_angles:
                init_angles = np.asarray(initial_angles)
                if init_angles.ndim == 1:
                    init_angles = np.tile(init_angles, (n_targets, 1))
                init_opt_angles = np.zeros((n_expanded, n_opt))
                for t in range(n_targets):
                    init_opt_angles[t * attempts_per_pose] = (
                        init_angles[t, non_mimic_indices])
                    for a in range(1, attempts_per_pose):
                        idx = t * attempts_per_pose + a
                        init_opt_angles[idx] = np.random.uniform(
                            joint_limits_lower, joint_limits_upper)
            else:
                init_opt_angles = np.random.uniform(
                    joint_limits_lower, joint_limits_upper, (n_expanded, n_opt))
        else:
            target_positions_expanded = target_positions
            target_rotations_expanded = target_rotations
            n_expanded = n_targets

            if initial_angles is not None:
                init_angles = np.asarray(initial_angles)
                if init_angles.ndim == 1:
                    init_angles = np.tile(init_angles, (n_targets, 1))
                init_opt_angles = init_angles[:, non_mimic_indices]
            else:
                init_opt_angles = np.random.uniform(
                    joint_limits_lower, joint_limits_upper, (n_expanded, n_opt))

        target_pos = target_positions_expanded
        target_rot = target_rotations_expanded

        # Precompute damping matrix
        damping_eye = damping * np.eye(n_active)

        # Jacobian-based damped least-squares optimization loop
        opt_angles = init_opt_angles.copy()

        for iteration in range(max_iterations):
            full_angles = _expand_to_full_angles_batch(opt_angles)

            # Compute Jacobian, position, and rotation for all batch elements
            J_opt, pos, rot = compute_geometric_jacobian_batched_numpy(
                full_angles, fk_params, non_mimic_indices,
                mimic_parent_indices, mimic_multipliers)

            # Build error vector: (batch, 6)
            # Position error
            pos_err = (target_pos - pos) * pos_mask_arr  # (batch, 3)

            # Rotation error in local frame, masked, then rotated to world
            if has_rot_constraint:
                rot_T = np.transpose(rot, axes=(0, 2, 1))
                r_diff = np.matmul(rot_T, target_rot)

                rot_err_local = 0.5 * np.stack([
                    r_diff[:, 2, 1] - r_diff[:, 1, 2],
                    r_diff[:, 0, 2] - r_diff[:, 2, 0],
                    r_diff[:, 1, 0] - r_diff[:, 0, 1]
                ], axis=1)  # (batch, 3)

                if mirror_rot is not None:
                    target_rot_m = target_rot @ mirror_rot
                    r_diff_m = np.matmul(rot_T, target_rot_m)
                    rot_err_local_m = 0.5 * np.stack([
                        r_diff_m[:, 2, 1] - r_diff_m[:, 1, 2],
                        r_diff_m[:, 0, 2] - r_diff_m[:, 2, 0],
                        r_diff_m[:, 1, 0] - r_diff_m[:, 0, 1]
                    ], axis=1)

                    # Select mirrored where closer
                    frob_direct = np.sum((target_rot - rot) ** 2, axis=(1, 2))
                    frob_mirror = np.sum((target_rot_m - rot) ** 2, axis=(1, 2))
                    use_mirror = (frob_mirror < 0.8 * frob_direct)[:, None]
                    rot_err_local = np.where(
                        use_mirror, rot_err_local_m, rot_err_local)

                # Mask in local frame, then rotate to world for Jacobian
                rot_err_local_masked = rot_err_local * rot_mask_arr
                rot_err_world = np.einsum(
                    'bij,bj->bi', rot, rot_err_local_masked)
            else:
                rot_err_world = np.zeros_like(pos_err)

            # Full error vector: (batch, 6)
            full_err = np.concatenate([pos_err, rot_err_world], axis=1)

            # Select active rows
            err = full_err[:, active_rows]           # (batch, n_active)
            J = J_opt[:, active_rows, :]             # (batch, n_active, n_opt)

            # Damped least squares: Δq = J^T (J J^T + λI)^{-1} e
            # (batch, n_active, n_active)
            JJT = np.matmul(J, np.transpose(J, (0, 2, 1))) + damping_eye
            # (batch, n_active, 1) -> solve -> (batch, n_active, 1) -> squeeze
            solved = np.linalg.solve(
                JJT, err[..., np.newaxis]).squeeze(-1)
            # (batch, n_opt)
            delta_q = np.einsum('bji,bj->bi', J, solved)

            opt_angles = opt_angles + delta_q
            opt_angles = np.clip(opt_angles, joint_limits_lower, joint_limits_upper)

            # Early stopping: check convergence
            pos_err_sq = np.sum(
                ((target_pos - pos) * pos_mask_arr) ** 2, axis=1)
            if np.all(pos_err_sq < pos_threshold * pos_threshold):
                # Need to also check rotation for final convergence
                break

        # Final error computation
        full_angles = _expand_to_full_angles_batch(opt_angles)
        pos_final, rot_final = forward_kinematics_ee_batched_numpy(
            full_angles, fk_params)

        # Position error
        if has_pos_constraint:
            pos_err_final = np.sqrt(np.sum(
                ((target_pos - pos_final) * pos_mask_arr) ** 2, axis=1))
        else:
            pos_err_final = np.zeros(n_expanded)

        # Rotation error
        if has_rot_constraint:
            rot_T = np.transpose(rot_final, axes=(0, 2, 1))
            r_diff = np.matmul(rot_T, target_rot)
            rot_err_local = 0.5 * np.stack([
                r_diff[:, 2, 1] - r_diff[:, 1, 2],
                r_diff[:, 0, 2] - r_diff[:, 2, 0],
                r_diff[:, 1, 0] - r_diff[:, 0, 1]
            ], axis=1)
            rot_err_masked = rot_err_local * rot_mask_arr
            rot_err_final = np.sqrt(np.sum(rot_err_masked ** 2, axis=1))

            if mirror_rot is not None:
                target_rot_m = target_rot @ mirror_rot
                r_diff_m = np.matmul(rot_T, target_rot_m)
                rot_err_local_m = 0.5 * np.stack([
                    r_diff_m[:, 2, 1] - r_diff_m[:, 1, 2],
                    r_diff_m[:, 0, 2] - r_diff_m[:, 2, 0],
                    r_diff_m[:, 1, 0] - r_diff_m[:, 0, 1]
                ], axis=1)
                rot_err_masked_m = rot_err_local_m * rot_mask_arr
                rot_err_m = np.sqrt(np.sum(rot_err_masked_m ** 2, axis=1))
                rot_err_final = np.minimum(rot_err_final, rot_err_m)
        else:
            rot_err_final = np.zeros(n_expanded)

        combined_err = pos_err_final + rot_err_final
        if has_pos_constraint and has_rot_constraint:
            success = (pos_err_final < pos_threshold) & (
                rot_err_final < rot_threshold)
        elif has_pos_constraint:
            success = pos_err_final < pos_threshold
        elif has_rot_constraint:
            success = rot_err_final < rot_threshold
        else:
            success = np.ones(n_expanded, dtype=bool)

        # Select best attempt for each target
        if attempts_per_pose > 1:
            solutions, success_out, errors_out = _select_best_attempts(
                full_angles, success, combined_err, n_targets, attempts_per_pose,
                n_joints, select_closest_to_initial, initial_angles)
        else:
            solutions = full_angles
            success_out = success
            errors_out = combined_err

        return solutions, success_out, errors_out

    # Attach metadata
    solve.n_joints = n_joints
    solve.joint_limits_lower = fk_params['joint_limits_lower']
    solve.joint_limits_upper = fk_params['joint_limits_upper']
    solve.fk_params = fk_params
    solve.backend_name = 'numpy'

    return solve


def compute_geometric_jacobian_jax(joint_angles, fk_params, return_non_mimic=True):
    """Compute the geometric Jacobian using JAX.

    Computes the 6xN Jacobian matrix where:
    - Rows 0-2: Linear velocity Jacobian
    - Rows 3-5: Angular velocity Jacobian

    For revolute joint i: J_v[:,i] = z_i × (p_ee - p_i), J_ω[:,i] = z_i
    For prismatic joint i: J_v[:,i] = z_i, J_ω[:,i] = 0

    Parameters
    ----------
    joint_angles : jax array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters.
    return_non_mimic : bool
        If True, return Jacobian for non-mimic joints only (with mimic
        contributions folded in). Default True.

    Returns
    -------
    jacobian : jax array
        Geometric Jacobian (6, n_opt) where n_opt is non-mimic joint count.
    ee_pos : jax array
        End-effector position (3,).
    ee_rot : jax array
        End-effector rotation (3, 3).
    """
    import jax.numpy as jnp

    n_joints = fk_params['n_joints']
    joint_axes = fk_params['joint_axes']
    joint_types = fk_params['joint_types']
    link_translations = fk_params['link_translations']
    link_rotations = fk_params['link_rotations']
    base_position = fk_params['base_position']
    base_rotation = fk_params['base_rotation']
    ee_offset_position = fk_params['ee_offset_position']
    ee_offset_rotation = fk_params['ee_offset_rotation']
    ref_angles = fk_params['ref_angles']

    # Get mimic joint info
    mimic_parent_indices = fk_params.get(
        'mimic_parent_indices', np.array([-1] * n_joints))
    mimic_multipliers = fk_params.get('mimic_multipliers', np.ones(n_joints))
    mimic_offsets = fk_params.get('mimic_offsets', np.zeros(n_joints))

    # Handle mimic joints: compute effective joint angles
    effective_angles = []
    for i in range(n_joints):
        parent_idx = mimic_parent_indices[i]
        if parent_idx >= 0:
            # Mimic joint: angle = parent_angle * multiplier + offset
            parent_angle = joint_angles[parent_idx]
            effective_angle = parent_angle * mimic_multipliers[i] + mimic_offsets[i]
            effective_angles.append(effective_angle)
        else:
            # Regular joint
            effective_angles.append(joint_angles[i])
    joint_angles = jnp.stack(effective_angles)

    # Compute FK and joint positions/axes in world frame
    current_pos = jnp.array(base_position)
    current_rot = jnp.array(base_rotation)

    joint_positions = []
    joint_axes_world = []

    for i in range(n_joints):
        # Apply link transform
        link_trans = jnp.array(link_translations[i])
        link_rot = jnp.array(link_rotations[i])
        current_pos = current_pos + current_rot @ link_trans
        current_rot = current_rot @ link_rot

        # Store joint position and axis in world frame
        joint_positions.append(current_pos.copy())
        axis_local = jnp.array(joint_axes[i])
        joint_axes_world.append(current_rot @ axis_local)

        # Apply joint rotation/translation
        angle = joint_angles[i] - ref_angles[i]
        if joint_types[i] in ('revolute', 'continuous'):
            # Rodrigues formula for rotation
            axis = axis_local / (jnp.linalg.norm(axis_local) + 1e-10)
            K = jnp.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            joint_rot = (jnp.eye(3) + jnp.sin(angle) * K
                         + (1 - jnp.cos(angle)) * (K @ K))
            current_rot = current_rot @ joint_rot
        else:  # prismatic
            current_pos = current_pos + current_rot @ (axis_local * angle)

    # Apply end-effector offset
    ee_pos = current_pos + current_rot @ jnp.array(ee_offset_position)
    ee_rot = current_rot @ jnp.array(ee_offset_rotation)

    # Compute full Jacobian columns (all joints)
    J_v_full = jnp.zeros((3, n_joints))
    J_w_full = jnp.zeros((3, n_joints))

    for i in range(n_joints):
        z_i = joint_axes_world[i]
        p_i = joint_positions[i]

        if joint_types[i] in ('revolute', 'continuous'):
            # J_v = z × (p_ee - p_i)
            r = ee_pos - p_i
            j_v = jnp.cross(z_i, r)
            j_w = z_i
        else:  # prismatic
            j_v = z_i
            j_w = jnp.zeros(3)

        J_v_full = J_v_full.at[:, i].set(j_v)
        J_w_full = J_w_full.at[:, i].set(j_w)

    J_full = jnp.vstack([J_v_full, J_w_full])

    if not return_non_mimic:
        return J_full, ee_pos, ee_rot

    # Fold mimic joint contributions into parent joints
    # For mimic joint i with parent p: dq_i = multiplier * dq_p
    # So J[:,p] += multiplier * J[:,i]
    non_mimic_indices = [i for i in range(n_joints) if mimic_parent_indices[i] < 0]
    n_opt = len(non_mimic_indices)

    # Build mapping from full joint index to non-mimic index
    full_to_opt = {}
    for opt_idx, full_idx in enumerate(non_mimic_indices):
        full_to_opt[full_idx] = opt_idx

    # Create reduced Jacobian
    J_opt = jnp.zeros((6, n_opt))
    for i in range(n_joints):
        parent_idx = mimic_parent_indices[i]
        if parent_idx < 0:
            # Non-mimic joint
            opt_idx = full_to_opt[i]
            J_opt = J_opt.at[:, opt_idx].set(J_opt[:, opt_idx] + J_full[:, i])
        else:
            # Mimic joint - add contribution to parent
            if parent_idx in full_to_opt:
                opt_idx = full_to_opt[parent_idx]
                multiplier = mimic_multipliers[i]
                J_opt = J_opt.at[:, opt_idx].set(
                    J_opt[:, opt_idx] + multiplier * J_full[:, i])

    return J_opt, ee_pos, ee_rot


def _create_jax_jacobian_solver(fk_params, backend):
    """Create a JAX-based Jacobian IK solver with full JIT compilation.

    This solver uses damped least-squares with the analytical Jacobian,
    fully JIT-compiled using lax.fori_loop with batched operations for
    maximum performance.

    Parameters
    ----------
    fk_params : dict
        FK parameters from extract_fk_parameters.
    backend : object
        JAX backend instance.

    Returns
    -------
    callable
        Solver function.
    """
    import jax
    from jax import lax
    import jax.numpy as jnp

    n_joints = fk_params['n_joints']

    # Get mimic joint info using helper
    (mimic_parent_indices, mimic_multipliers, mimic_offsets,
     non_mimic_indices, n_opt) = _get_mimic_joint_info(fk_params)

    # Joint limits for non-mimic joints
    joint_limits_lower = jnp.array(fk_params['joint_limits_lower'][non_mimic_indices])
    joint_limits_upper = jnp.array(fk_params['joint_limits_upper'][non_mimic_indices])

    non_mimic_indices_jax = jnp.array(non_mimic_indices.astype(np.int32))

    # JIT cache for different parameters
    _jit_cache = {}

    def _create_batched_solver_fn(max_iterations, damping, pos_threshold, rot_threshold,
                                  pos_mask_arr, rot_mask_arr, rotation_mirror):
        """Create JIT-compiled batched Jacobian solver using fori_loop."""
        pos_mask = jnp.array(pos_mask_arr)
        rot_mask = jnp.array(rot_mask_arr)

        pos_mask_sum = float(np.sum(pos_mask_arr))
        rot_mask_sum = float(np.sum(rot_mask_arr))
        has_pos_constraint = pos_mask_sum > 0
        has_rot_constraint = rot_mask_sum > 0

        # Mirror rotation matrix
        mirror_rot_np = _create_mirror_rotation_matrix(rotation_mirror)
        mirror_rot = jnp.array(mirror_rot_np) if mirror_rot_np is not None else None

        # Determine active Jacobian rows
        active_rows = []
        if has_pos_constraint:
            for i in range(3):
                if pos_mask_arr[i] > 0:
                    active_rows.append(i)
        if has_rot_constraint:
            active_rows.extend([3, 4, 5])
        active_rows_jax = jnp.array(active_rows, dtype=jnp.int32)
        n_active = len(active_rows)

        damping_eye = damping * jnp.eye(n_active)

        # Precompute FK constants as JAX arrays for use inside JIT
        _translations = [jnp.array(fk_params['link_translations'][i])
                         for i in range(n_joints)]
        _local_rotations = [jnp.array(fk_params['link_rotations'][i])
                            for i in range(n_joints)]
        _axes = [jnp.array(fk_params['joint_axes'][i])
                 for i in range(n_joints)]
        _base_pos = jnp.array(fk_params['base_position'])
        _base_rot = jnp.array(fk_params['base_rotation'])
        _ref_angles = fk_params['ref_angles']
        _joint_types = fk_params['joint_types']
        _ee_offset_pos = jnp.array(fk_params['ee_offset_position'])
        _ee_offset_rot = jnp.array(fk_params['ee_offset_rotation'])

        # Precompute Rodrigues constants per joint
        _joint_K = []
        _joint_K2 = []
        for i in range(n_joints):
            axis = np.array(fk_params['joint_axes'][i])
            axis_norm = axis / (np.linalg.norm(axis) + 1e-10)
            K = np.array([
                [0, -axis_norm[2], axis_norm[1]],
                [axis_norm[2], 0, -axis_norm[0]],
                [-axis_norm[1], axis_norm[0], 0]
            ])
            _joint_K.append(jnp.array(K))
            _joint_K2.append(jnp.array(K @ K))

        # Precompute mimic joint info as Python ints for static conditionals
        _mimic_parents = [int(mimic_parent_indices[i]) for i in range(n_joints)]
        _mimic_mults = [float(mimic_multipliers[i]) for i in range(n_joints)]
        _mimic_offs = [float(mimic_offsets[i]) for i in range(n_joints)]

        # Build opt-index mapping
        _full_to_opt = {}
        for opt_idx, full_idx in enumerate(non_mimic_indices):
            _full_to_opt[int(full_idx)] = opt_idx

        def _compute_jac_fk(opt_angles_batch):
            """Batched Jacobian+FK with static data captured from closure."""
            batch_size = opt_angles_batch.shape[0]

            # Expand to full angles
            full_angles = jnp.zeros((batch_size, n_joints))
            full_angles = full_angles.at[:, non_mimic_indices_jax].set(
                opt_angles_batch)
            for i in range(n_joints):
                if _mimic_parents[i] >= 0:
                    full_angles = full_angles.at[:, i].set(
                        full_angles[:, _mimic_parents[i]] * _mimic_mults[i]
                        + _mimic_offs[i])

            # FK with joint position/axis tracking
            current_pos = jnp.tile(_base_pos, (batch_size, 1))
            current_rot = jnp.tile(_base_rot, (batch_size, 1, 1))

            joint_positions = []
            joint_axes_world = []

            for i in range(n_joints):
                current_pos = current_pos + jnp.einsum(
                    'bij,j->bi', current_rot, _translations[i])
                current_rot = jnp.einsum(
                    'bij,jk->bik', current_rot, _local_rotations[i])

                joint_positions.append(current_pos)
                joint_axes_world.append(jnp.einsum(
                    'bij,j->bi', current_rot, _axes[i]))

                delta = full_angles[:, i] - _ref_angles[i]
                if _joint_types[i] == 'prismatic':
                    current_pos = current_pos + jnp.einsum(
                        'bij,j,b->bi', current_rot, _axes[i], delta)
                else:
                    c = jnp.cos(delta)[:, None, None]
                    s = jnp.sin(delta)[:, None, None]
                    joint_rots = jnp.eye(3) + s * _joint_K[i] + (
                        1 - c) * _joint_K2[i]
                    current_rot = jnp.einsum(
                        'bij,bjk->bik', current_rot, joint_rots)

            ee_pos = current_pos + jnp.einsum(
                'bij,j->bi', current_rot, _ee_offset_pos)
            ee_rot = jnp.einsum(
                'bij,jk->bik', current_rot, _ee_offset_rot)

            # Jacobian columns
            J_v = jnp.zeros((batch_size, 3, n_joints))
            J_w = jnp.zeros((batch_size, 3, n_joints))
            for i in range(n_joints):
                z_i = joint_axes_world[i]
                p_i = joint_positions[i]
                if _joint_types[i] in ('revolute', 'continuous'):
                    jv = jnp.cross(z_i, ee_pos - p_i)
                    jw = z_i
                else:
                    jv = z_i
                    jw = jnp.zeros_like(z_i)
                J_v = J_v.at[:, :, i].set(jv)
                J_w = J_w.at[:, :, i].set(jw)

            J_full = jnp.concatenate([J_v, J_w], axis=1)

            # Fold mimic contributions
            J_opt = jnp.zeros((batch_size, 6, n_opt))
            for i in range(n_joints):
                if _mimic_parents[i] < 0:
                    opt_idx = _full_to_opt[i]
                    J_opt = J_opt.at[:, :, opt_idx].add(J_full[:, :, i])
                else:
                    parent = _mimic_parents[i]
                    if parent in _full_to_opt:
                        opt_idx = _full_to_opt[parent]
                        J_opt = J_opt.at[:, :, opt_idx].add(
                            _mimic_mults[i] * J_full[:, :, i])

            return J_opt, ee_pos, ee_rot, full_angles

        def solve_batched(init_opt_angles, target_pos, target_rot):
            """Solve batch IK using fori_loop with batched operations."""

            def body_fn(i, opt_angles):
                J_opt, pos, rot, _ = _compute_jac_fk(opt_angles)

                # Position error: (batch, 3)
                pos_err = (target_pos - pos) * pos_mask

                # Rotation error in local frame, masked, rotated to world
                # Use SO(3) logarithmic map for better accuracy
                if has_rot_constraint:
                    rot_err_local = rotation_error_so3_log_batch(rot, target_rot)

                    if mirror_rot is not None:
                        target_rot_m = target_rot @ mirror_rot
                        rot_err_local_m = rotation_error_so3_log_batch(
                            rot, target_rot_m)
                        frob_direct = jnp.sum(
                            (target_rot - rot) ** 2, axis=(1, 2))
                        frob_mirror = jnp.sum(
                            (target_rot_m - rot) ** 2, axis=(1, 2))
                        use_mirror = (frob_mirror < 0.8 * frob_direct)[
                            :, None]
                        rot_err_local = jnp.where(
                            use_mirror, rot_err_local_m, rot_err_local)

                    rot_err_local_masked = rot_err_local * rot_mask
                    rot_err_world = jnp.einsum(
                        'bij,bj->bi', rot, rot_err_local_masked)
                else:
                    rot_err_world = jnp.zeros_like(pos_err)

                # Full error: (batch, 6)
                full_err = jnp.concatenate([pos_err, rot_err_world], axis=1)

                # Select active rows
                err = full_err[:, active_rows_jax]
                J = J_opt[:, active_rows_jax, :]

                # Damped least squares: (batch, n_opt)
                JJT = jnp.matmul(
                    J, jnp.transpose(J, (0, 2, 1))) + damping_eye
                solved = jnp.linalg.solve(
                    JJT, err[..., jnp.newaxis]).squeeze(-1)
                delta_q = jnp.einsum('bji,bj->bi', J, solved)

                new_opt = opt_angles + delta_q
                new_opt = jnp.clip(
                    new_opt, joint_limits_lower, joint_limits_upper)
                return new_opt

            # Run fixed-iteration loop
            final_opt = lax.fori_loop(0, max_iterations, body_fn,
                                      init_opt_angles)

            # Compute final errors
            _, pos_final, rot_final, final_full = _compute_jac_fk(final_opt)

            # Position error
            pos_err_final = jnp.sqrt(jnp.sum(
                ((target_pos - pos_final) * pos_mask) ** 2, axis=1))

            # Rotation error using SO(3) logarithmic map
            if has_rot_constraint:
                rot_err_local = rotation_error_so3_log_batch(rot_final, target_rot)
                rot_err_masked = rot_err_local * rot_mask
                rot_err_final = jnp.sqrt(jnp.sum(
                    rot_err_masked ** 2, axis=1))

                if mirror_rot is not None:
                    target_rot_m = target_rot @ mirror_rot
                    rot_err_local_m = rotation_error_so3_log_batch(
                        rot_final, target_rot_m)
                    rot_err_m = jnp.sqrt(jnp.sum(
                        (rot_err_local_m * rot_mask) ** 2, axis=1))
                    rot_err_final = jnp.minimum(rot_err_final, rot_err_m)
            else:
                rot_err_final = jnp.zeros_like(pos_err_final)

            combined_err = pos_err_final + rot_err_final

            if has_pos_constraint and has_rot_constraint:
                success = (pos_err_final < pos_threshold) & (
                    rot_err_final < rot_threshold)
            elif has_pos_constraint:
                success = pos_err_final < pos_threshold
            elif has_rot_constraint:
                success = rot_err_final < rot_threshold
            else:
                success = jnp.ones(pos_err_final.shape[0], dtype=bool)

            return final_full, success, combined_err

        return jax.jit(solve_batched)

    def solve(target_positions, target_rotations,
              initial_angles=None,
              max_iterations=30,
              damping=0.01,
              pos_threshold=0.001,
              rot_threshold=0.1,
              position_mask=True,
              rotation_mask=True,
              rotation_mirror=None,
              attempts_per_pose=1,
              use_current_angles=True,
              select_closest_to_initial=False,
              **kwargs):
        """Solve batch IK using Jacobian-based method.

        Parameters
        ----------
        target_positions : array-like
            Target positions (N, 3).
        target_rotations : array-like
            Target rotation matrices (N, 3, 3).
        initial_angles : array-like, optional
            Initial joint angles.
        max_iterations : int
            Maximum iterations (default 30, typically sufficient).
        damping : float
            Damping factor for least squares (default 0.01).
        pos_threshold : float
            Position error threshold for success.
        rot_threshold : float
            Rotation error threshold for success.
        position_mask, rotation_mask, rotation_mirror, attempts_per_pose,
        use_current_angles, select_closest_to_initial : same as gradient descent solver.

        Returns
        -------
        tuple
            (solutions, success_flags, combined_errors)
        """
        target_positions = backend.array(np.asarray(target_positions, dtype=np.float64))
        target_rotations = backend.array(np.asarray(target_rotations, dtype=np.float64))
        n_targets = target_positions.shape[0]

        # Handle initial angles
        if initial_angles is None:
            init = (joint_limits_lower + joint_limits_upper) / 2
            base_initial_opt_angles = backend.stack([init] * n_targets)
        else:
            initial_angles = backend.array(np.asarray(initial_angles, dtype=np.float64))
            if len(initial_angles.shape) == 1:
                initial_angles = backend.stack([initial_angles] * n_targets)
            base_initial_opt_angles = initial_angles[:, non_mimic_indices]

        # Handle multiple attempts
        if attempts_per_pose > 1:
            target_positions_np = backend.to_numpy(target_positions)
            target_rotations_np = backend.to_numpy(target_rotations)
            expanded_target_positions_np = np.repeat(
                target_positions_np, attempts_per_pose, axis=0)
            expanded_target_rotations_np = np.repeat(
                target_rotations_np, attempts_per_pose, axis=0)

            lower_np = backend.to_numpy(joint_limits_lower)
            upper_np = backend.to_numpy(joint_limits_upper)
            base_initial_np = backend.to_numpy(base_initial_opt_angles)

            n_expanded = n_targets * attempts_per_pose
            if use_current_angles:
                all_initial = np.random.uniform(
                    lower_np, upper_np,
                    size=(n_targets, attempts_per_pose, n_opt))
                all_initial[:, 0, :] = base_initial_np
                all_initial = all_initial.reshape(n_expanded, n_opt)
            else:
                all_initial = np.random.uniform(
                    lower_np, upper_np, size=(n_expanded, n_opt))

            initial_opt_angles = backend.array(all_initial.astype(np.float64))
            target_positions_solve = backend.array(
                expanded_target_positions_np.astype(np.float64))
            target_rotations_solve = backend.array(
                expanded_target_rotations_np.astype(np.float64))
        else:
            initial_opt_angles = base_initial_opt_angles
            target_positions_solve = target_positions
            target_rotations_solve = target_rotations

        # Normalize masks
        pos_mask_arr = normalize_axis_mask(position_mask)
        rot_mask_arr = normalize_axis_mask(rotation_mask)

        # Get or create JIT-compiled solver
        cache_key = (max_iterations, damping, pos_threshold, rot_threshold,
                     tuple(pos_mask_arr), tuple(rot_mask_arr), rotation_mirror)
        if cache_key not in _jit_cache:
            _jit_cache[cache_key] = _create_batched_solver_fn(
                max_iterations, damping, pos_threshold, rot_threshold,
                pos_mask_arr, rot_mask_arr, rotation_mirror)

        solver_fn = _jit_cache[cache_key]

        # Solve
        all_solutions, all_success, all_errors = solver_fn(
            initial_opt_angles, target_positions_solve, target_rotations_solve)

        # Select best from multiple attempts
        if attempts_per_pose > 1:
            all_solutions_np = backend.to_numpy(all_solutions)
            all_success_np = backend.to_numpy(all_success)
            all_errors_np = backend.to_numpy(all_errors)

            solutions, success_flags, errors = _select_best_attempts(
                all_solutions_np, all_success_np, all_errors_np,
                n_targets, attempts_per_pose, n_joints,
                select_closest_to_initial, initial_angles)

            return (backend.array(solutions),
                    backend.array(success_flags),
                    backend.array(errors))
        else:
            return all_solutions, all_success, all_errors

    # Attach metadata
    solve.n_joints = n_joints
    solve.joint_limits_lower = fk_params['joint_limits_lower']
    solve.joint_limits_upper = fk_params['joint_limits_upper']
    solve.fk_params = fk_params
    solve.backend = backend
    solve.method = 'jacobian'

    return solve


def create_batch_ik_solver(robot_model, link_list, move_target,
                           backend_name='jax', method='jacobian'):
    """Create a high-performance batch IK solver.

    This is the recommended way to solve batch IK problems. It:
    1. Extracts FK parameters from the robot model
    2. Creates a backend-specific optimized solver
    3. Returns a batch solver that uses vmap and jit for parallelization

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    link_list : list
        List of links in the kinematic chain.
    move_target : CascadedCoords
        End effector coordinates.
    backend_name : str
        Backend to use ('jax', 'numpy'). Default is 'jax'.
    method : str
        IK solving method. Default is 'jacobian'.
        - 'jacobian': Uses damped least-squares with analytical Jacobian.
          Faster convergence (typically 30 iterations), fully JIT-compiled.
          Recommended for most use cases.
        - 'gradient_descent': Uses gradient descent optimization.
          More flexible for custom cost functions but slower convergence.

    Returns
    -------
    callable
        Batch IK solver function with signature:
        solve(target_positions, target_rotations, initial_angles=None,
              max_iterations=30, damping=0.01,  # for jacobian method
              position_mask=True, rotation_mask=True) -> (solutions, success, errors)

    Examples
    --------
    >>> # Jacobian-based solver (default, fastest)
    >>> solver = create_batch_ik_solver(robot, link_list, move_target)
    >>> solutions, success, errors = solver(target_positions, target_rotations)
    >>> # Gradient descent solver
    >>> solver = create_batch_ik_solver(robot, link_list, move_target,
    ...                                  method='gradient_descent')
    >>> solutions, success, errors = solver(
    ...     target_positions, target_rotations,
    ...     position_mask='xy',  # constrain x, y position only
    ...     rotation_mask='x',   # constrain x-axis direction only
    ... )
    """
    import numpy as np

    from skrobot.backend import get_backend

    # Extract FK parameters
    fk_params = extract_fk_parameters(robot_model, link_list, move_target)

    # Use optimized NumPy solver for numpy backend
    if backend_name == 'numpy':
        return _create_numpy_optimized_solver(fk_params)

    backend = get_backend(backend_name)

    # Use Jacobian-based solver for JAX (faster)
    if method == 'jacobian' and backend_name == 'jax':
        return _create_jax_jacobian_solver(fk_params, backend)

    # Fall through to gradient descent solver
    if method not in ('jacobian', 'gradient_descent'):
        raise ValueError(f"Unknown method: {method}. Use 'jacobian' or 'gradient_descent'.")

    n_joints = fk_params['n_joints']

    # Identify non-mimic joints (these are the actual optimization variables)
    (mimic_parent_indices, mimic_multipliers, mimic_offsets,
     non_mimic_indices, _) = _get_mimic_joint_info(fk_params)

    # Get joint limits for non-mimic joints only
    joint_limits_lower = backend.array(fk_params['joint_limits_lower'][non_mimic_indices])
    joint_limits_upper = backend.array(fk_params['joint_limits_upper'][non_mimic_indices])

    # JIT cache for different iteration counts
    _jit_cache = {}

    # Precompute arrays for expanding opt angles to full angles
    _non_mimic_indices_arr = backend.array(non_mimic_indices)
    _mimic_parent_indices_arr = backend.array(mimic_parent_indices)
    _mimic_multipliers_arr = backend.array(mimic_multipliers)
    _mimic_offsets_arr = backend.array(mimic_offsets)

    def _expand_to_full_angles(opt_angles):
        """Expand optimization variables to full joint angles including mimic joints."""
        # Create full angles array
        full_angles = backend.zeros(n_joints)

        # Fill in non-mimic joints
        for i, opt_idx in enumerate(non_mimic_indices):
            if backend.name == 'jax':
                full_angles = full_angles.at[opt_idx].set(opt_angles[i])
            else:
                full_angles[opt_idx] = opt_angles[i]

        # Fill in mimic joints based on their parent values
        for i in range(n_joints):
            parent_idx = mimic_parent_indices[i]
            if parent_idx >= 0:
                mimic_angle = full_angles[parent_idx] * mimic_multipliers[i] + mimic_offsets[i]
                if backend.name == 'jax':
                    full_angles = full_angles.at[i].set(mimic_angle)
                else:
                    full_angles[i] = mimic_angle

        return full_angles

    def _create_solver_fn(max_iterations, learning_rate, pos_weight, rot_weight,
                          pos_threshold, rot_threshold, pos_mask_arr, rot_mask_arr,
                          rotation_mirror):
        """Create a JIT-compiled solver for given parameters."""

        # Convert masks to backend arrays
        pos_mask = backend.array(pos_mask_arr)
        rot_mask = backend.array(rot_mask_arr)

        # Check mask sums for determining constraint types
        pos_mask_sum = np.sum(pos_mask_arr)
        rot_mask_sum = np.sum(rot_mask_arr)
        has_pos_constraint = pos_mask_sum > 0
        has_rot_constraint = rot_mask_sum > 0

        # Create 180° rotation matrix for mirror axis
        mirror_rot = _create_mirror_rotation_matrix(rotation_mirror, backend)

        # Use SO(3) log rotation error if JAX backend is available
        use_so3_log = backend.name == 'jax'

        def _compute_rot_err_local_gd(target_r, current_r):
            """Compute rotation error vector in local (current) frame.

            Uses SO(3) logarithmic map when JAX backend is available,
            otherwise falls back to anti-symmetric extraction.
            """
            if use_so3_log:
                # Use SO(3) logarithmic map for better accuracy
                return rotation_error_so3_log(current_r, target_r)
            else:
                # Fallback: anti-symmetric extraction
                r_diff = backend.matmul(backend.transpose(current_r), target_r)
                return 0.5 * backend.stack([
                    r_diff[2, 1] - r_diff[1, 2],
                    r_diff[0, 2] - r_diff[2, 0],
                    r_diff[1, 0] - r_diff[0, 1]
                ])

        def compute_rot_error_single(rot, target_rot):
            """Compute rotation error for a single target rotation."""
            # Compute in local frame and mask there
            rot_err_local = _compute_rot_err_local_gd(target_rot, rot)
            return backend.sum((rot_err_local * rot_mask) ** 2)

        def compute_rot_error_single_for_check(rot, target_rot):
            """Compute rotation error for convergence check."""
            # Compute in local frame and mask there
            rot_err_local = _compute_rot_err_local_gd(target_rot, rot)
            return backend.sqrt(backend.sum((rot_err_local * rot_mask) ** 2))

        def loss_fn(opt_angles, target_pos, target_rot):
            """Compute weighted pose error with mask constraints."""
            # Expand to full angles (including mimic joints)
            full_angles = _expand_to_full_angles(opt_angles)
            pos, rot = forward_kinematics_ee(backend, full_angles, fk_params)

            # Position error with mask
            pos_diff = pos - target_pos
            pos_err = backend.sum((pos_diff * pos_mask) ** 2)

            # Rotation error with mask (and optional mirror)
            if mirror_rot is not None:
                # Compute error for both normal and mirrored target
                rot_err_normal = compute_rot_error_single(rot, target_rot)
                target_rot_mirrored = backend.matmul(target_rot, mirror_rot)
                rot_err_mirror = compute_rot_error_single(rot, target_rot_mirrored)
                # Take minimum
                rot_err = backend.minimum(rot_err_normal, rot_err_mirror)
            else:
                rot_err = compute_rot_error_single(rot, target_rot)

            return pos_weight * pos_err + rot_weight * rot_err

        loss_and_grad = backend.value_and_grad(loss_fn)

        def compute_errors(opt_angles, target_pos, target_rot):
            """Compute position and rotation errors."""
            full_angles = _expand_to_full_angles(opt_angles)
            pos, rot = forward_kinematics_ee(backend, full_angles, fk_params)

            # Position error
            pos_diff = pos - target_pos
            pos_err = backend.sqrt(backend.sum((pos_diff * pos_mask) ** 2))

            # Rotation error (with optional mirror)
            if mirror_rot is not None:
                rot_err_normal = compute_rot_error_single_for_check(rot, target_rot)
                target_rot_mirrored = backend.matmul(target_rot, mirror_rot)
                rot_err_mirror = compute_rot_error_single_for_check(rot, target_rot_mirrored)
                rot_err = backend.minimum(rot_err_normal, rot_err_mirror)
            else:
                rot_err = compute_rot_error_single_for_check(rot, target_rot)

            return pos_err, rot_err

        def check_converged(pos_err, rot_err):
            """Check if converged based on mask constraints."""
            if has_pos_constraint and has_rot_constraint:
                return (pos_err < pos_threshold) & (rot_err < rot_threshold)
            elif has_pos_constraint:
                return pos_err < pos_threshold
            elif has_rot_constraint:
                return rot_err < rot_threshold
            else:
                return backend.array(True)

        def solve_single(init_opt_angles, target_pos, target_rot):
            """Solve IK for a single target with early stopping."""

            def cond_fn(state):
                """Continue if not converged and not at max iterations."""
                opt_angles, iteration, pos_err, rot_err = state
                not_converged = ~check_converged(pos_err, rot_err)
                not_max_iter = iteration < max_iterations
                return not_converged & not_max_iter

            def body_fn(state):
                """Single iteration step."""
                opt_angles, iteration, _, _ = state

                # Compute gradient and update
                loss, grad = loss_and_grad(opt_angles, target_pos, target_rot)
                new_opt_angles = opt_angles - learning_rate * grad
                new_opt_angles = backend.clip(
                    new_opt_angles, joint_limits_lower, joint_limits_upper
                )

                # Compute errors for convergence check
                pos_err, rot_err = compute_errors(new_opt_angles, target_pos, target_rot)

                return (new_opt_angles, iteration + 1, pos_err, rot_err)

            # Initial state: (angles, iteration=0, pos_err=inf, rot_err=inf)
            init_pos_err, init_rot_err = compute_errors(
                init_opt_angles, target_pos, target_rot
            )
            init_state = (init_opt_angles, 0, init_pos_err, init_rot_err)

            # Run optimization with early stopping
            final_opt_angles, final_iter, final_pos_err, final_rot_err = backend.while_loop(
                cond_fn, body_fn, init_state
            )

            # Expand to full angles
            final_full_angles = _expand_to_full_angles(final_opt_angles)

            # Check success
            success = check_converged(final_pos_err, final_rot_err)

            # Combined error for best solution selection
            combined_err = final_pos_err + final_rot_err

            return final_full_angles, success, combined_err

        # Vectorize over batch dimension (only batched args now)
        batched_solve = backend.vmap(solve_single)

        def solve_batch(init_opt_angles, target_positions, target_rotations):
            return batched_solve(init_opt_angles, target_positions, target_rotations)

        # JIT compile if supported
        return backend.compile(solve_batch)

    def solve(target_positions, target_rotations,
              initial_angles=None,
              max_iterations=100,
              learning_rate=0.1,
              pos_weight=1.0,
              rot_weight=0.1,
              pos_threshold=0.001,
              rot_threshold=0.1,
              position_mask=True,
              rotation_mask=True,
              rotation_mirror=None,
              attempts_per_pose=1,
              use_current_angles=True,
              select_closest_to_initial=False):
        """Solve batch IK.

        Parameters
        ----------
        target_positions : array-like
            Target positions (N, 3).
        target_rotations : array-like
            Target rotation matrices (N, 3, 3).
        initial_angles : array-like, optional
            Initial joint angles (N, n_joints) or (n_joints,).
            If attempts_per_pose > 1 and use_current_angles=True, this is used
            as the first attempt for each target.
        max_iterations : int
            Maximum gradient descent iterations.
        learning_rate : float
            Gradient descent step size.
        pos_weight : float
            Position error weight.
        rot_weight : float
            Rotation error weight.
        pos_threshold : float
            Position error threshold for success (in meters).
        rot_threshold : float
            Rotation error threshold for success.
            For single-axis constraint: 0 = aligned, 1 = perpendicular.
            For multi-axis constraint: Frobenius norm of rotation matrix diff.
            Default is 0.1.
        position_mask : bool, str, list, or numpy.ndarray
            Position constraint mask. Specifies which position axes to constrain.
            - True: constrain all axes (default)
            - False: no position constraint
            - 'x', 'y', 'z': constrain single axis
            - 'xy', 'xz', 'yz': constrain two axes
            - [1, 1, 0]: array form (1=constrain, 0=free)
        rotation_mask : bool, str, list, or numpy.ndarray
            Rotation constraint mask. Specifies which rotation axes to constrain.
            - True: constrain all rotation axes (default)
            - False: no rotation constraint
            - 'x', 'y', 'z': constrain single axis direction
            - 'xy', 'xz', 'yz': constrain two axis directions
            - [1, 1, 0]: array form (1=constrain, 0=free)
        rotation_mirror : str or None
            Allow 180° rotated orientation around specified axis.
            - None: no mirror (default)
            - 'x': allow 180° rotation around x-axis
            - 'y': allow 180° rotation around y-axis
            - 'z': allow 180° rotation around z-axis
            Useful when gripper orientation can be flipped (e.g., up or down).
        attempts_per_pose : int
            Number of IK attempts per target pose. Each attempt uses a different
            initial configuration. The best solution (lowest error) is returned.
            Default is 1 (single attempt).
        use_current_angles : bool
            If True and attempts_per_pose > 1, use initial_angles as the first
            attempt (like viser's strategy). Remaining attempts use random
            initial values. Default is True.

        Returns
        -------
        tuple
            (solutions, success_flags, combined_errors)
            combined_errors is the sum of position and rotation errors.
        """
        target_positions = backend.array(np.asarray(target_positions, dtype=np.float64))
        target_rotations = backend.array(np.asarray(target_rotations, dtype=np.float64))
        n_targets = target_positions.shape[0]
        n_opt_joints = len(non_mimic_indices)

        # Handle initial angles
        if initial_angles is None:
            # Use middle of joint limits as initial guess (for non-mimic joints only)
            init = (joint_limits_lower + joint_limits_upper) / 2
            base_initial_opt_angles = backend.stack([init] * n_targets)
        else:
            initial_angles = backend.array(np.asarray(initial_angles, dtype=np.float64))
            if len(initial_angles.shape) == 1:
                initial_angles = backend.stack([initial_angles] * n_targets)
            # Extract non-mimic joint angles
            base_initial_opt_angles = initial_angles[:, non_mimic_indices]

        # Generate multiple initial values if attempts_per_pose > 1
        if attempts_per_pose > 1:
            # Convert to numpy for expansion
            target_positions_np = backend.to_numpy(target_positions)
            target_rotations_np = backend.to_numpy(target_rotations)

            # Expand targets to (n_targets * attempts_per_pose)
            # Each target is repeated attempts_per_pose times
            expanded_target_positions_np = np.repeat(
                target_positions_np, attempts_per_pose, axis=0
            )
            expanded_target_rotations_np = np.repeat(
                target_rotations_np, attempts_per_pose, axis=0
            )

            # Generate initial angles for all attempts (vectorized)
            # Convert joint limits to numpy for random generation
            lower_np = backend.to_numpy(joint_limits_lower)
            upper_np = backend.to_numpy(joint_limits_upper)
            base_initial_opt_angles_np = backend.to_numpy(base_initial_opt_angles)

            n_expanded = n_targets * attempts_per_pose

            if use_current_angles:
                # First attempt uses provided angles, rest are random
                # Shape: (n_targets, attempts_per_pose, n_opt_joints)
                all_initial_opt_angles = np.random.uniform(
                    lower_np, upper_np,
                    size=(n_targets, attempts_per_pose, n_opt_joints)
                )
                # Set first attempt to provided initial angles
                all_initial_opt_angles[:, 0, :] = base_initial_opt_angles_np
                # Reshape to (n_expanded, n_opt_joints)
                all_initial_opt_angles = all_initial_opt_angles.reshape(
                    n_expanded, n_opt_joints
                )
            else:
                # All attempts are random
                all_initial_opt_angles = np.random.uniform(
                    lower_np, upper_np, size=(n_expanded, n_opt_joints)
                )

            initial_opt_angles = backend.array(
                all_initial_opt_angles.astype(np.float64))
            target_positions_for_solve = backend.array(
                expanded_target_positions_np.astype(np.float64))
            target_rotations_for_solve = backend.array(
                expanded_target_rotations_np.astype(np.float64))
        else:
            initial_opt_angles = base_initial_opt_angles
            target_positions_for_solve = target_positions
            target_rotations_for_solve = target_rotations

        # Normalize masks to arrays
        pos_mask_arr = normalize_axis_mask(position_mask)
        rot_mask_arr = normalize_axis_mask(rotation_mask)

        # Get or create JIT-compiled solver for these parameters
        # Include masks, thresholds, and mirror in cache key
        cache_key = (max_iterations, learning_rate, pos_weight, rot_weight,
                     pos_threshold, rot_threshold,
                     tuple(pos_mask_arr), tuple(rot_mask_arr), rotation_mirror)
        if cache_key not in _jit_cache:
            _jit_cache[cache_key] = _create_solver_fn(
                max_iterations, learning_rate, pos_weight, rot_weight,
                pos_threshold, rot_threshold, pos_mask_arr, rot_mask_arr,
                rotation_mirror
            )

        solver_fn = _jit_cache[cache_key]

        # Solve all (targets × attempts)
        all_solutions, all_success, all_errors = solver_fn(
            initial_opt_angles, target_positions_for_solve, target_rotations_for_solve
        )

        # If multiple attempts, select best solution for each target
        if attempts_per_pose > 1:
            # Convert to numpy for selection
            all_solutions_np = backend.to_numpy(all_solutions)
            all_success_np = backend.to_numpy(all_success)
            all_errors_np = backend.to_numpy(all_errors)

            # Reshape to (n_targets, attempts_per_pose, ...)
            all_solutions_np = all_solutions_np.reshape(
                n_targets, attempts_per_pose, n_joints
            )
            all_success_np = all_success_np.reshape(n_targets, attempts_per_pose)
            all_errors_np = all_errors_np.reshape(n_targets, attempts_per_pose)

            if select_closest_to_initial and initial_angles is not None:
                # Prefer first attempt (starts from current angles) if successful
                # Otherwise select from successful solutions the one closest to initial
                init_angles_np = np.asarray(initial_angles)
                if init_angles_np.ndim == 1:
                    init_angles_np = np.tile(init_angles_np, (n_targets, 1))

                best_indices = []
                err_threshold = 0.02  # Consider solutions with error < 2cm as valid
                for i in range(n_targets):
                    # First attempt (index 0) starts from current angles
                    first_success = all_success_np[i, 0] or all_errors_np[i, 0] < err_threshold
                    if first_success:
                        # Use first attempt if it succeeded
                        best_idx = 0
                    else:
                        # Find other successful attempts
                        valid_mask = all_success_np[i] | (all_errors_np[i] < err_threshold)
                        if np.any(valid_mask):
                            # Select the one closest to initial angles
                            distances = np.linalg.norm(
                                all_solutions_np[i, valid_mask] - init_angles_np[i], axis=1
                            )
                            valid_indices = np.where(valid_mask)[0]
                            best_idx = valid_indices[np.argmin(distances)]
                        else:
                            # Fall back to minimum error
                            best_idx = np.argmin(all_errors_np[i])
                    best_indices.append(best_idx)
                best_indices = np.array(best_indices)
            else:
                # Select best attempt for each target (lowest error) - vectorized
                best_indices = np.argmin(all_errors_np, axis=1)

            # Use advanced indexing instead of Python loop
            target_indices = np.arange(n_targets)
            solutions = all_solutions_np[target_indices, best_indices]
            success_flags = all_success_np[target_indices, best_indices]
            errors = all_errors_np[target_indices, best_indices]

            return backend.array(solutions), backend.array(success_flags), backend.array(errors)
        else:
            return all_solutions, all_success, all_errors

    # Create FK function for external use
    def fk_fn(angles):
        return forward_kinematics_ee(backend, angles, fk_params)

    # Attach parameters for reference
    solve.n_joints = n_joints
    solve.joint_limits_lower = fk_params['joint_limits_lower']
    solve.joint_limits_upper = fk_params['joint_limits_upper']
    solve.fk_fn = fk_fn
    solve.fk_params = fk_params
    solve.backend = backend
    solve.method = 'gradient_descent'

    return solve


def _axis_angle_to_matrix(backend, axis, angle):
    """Convert axis-angle representation to rotation matrix.

    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    axis : array
        Rotation axis (3,).
    angle : float or array
        Rotation angle in radians.

    Returns
    -------
    rotation : array
        Rotation matrix (3, 3).
    """
    # Normalize axis
    axis_norm = backend.sqrt(backend.sum(axis ** 2) + 1e-10)
    axis = axis / axis_norm

    # Skew-symmetric matrix
    K = backend.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    I = backend.eye(3)
    c = backend.cos(angle)
    s = backend.sin(angle)

    return I + s * K + (1 - c) * backend.matmul(K, K)


def _axis_angle_to_matrix_batched(axis, angles):
    """Convert axis-angle to rotation matrices for a batch of angles.

    Uses Rodrigues' formula with NumPy broadcasting for efficiency.

    Parameters
    ----------
    axis : numpy.ndarray
        Rotation axis (3,).
    angles : numpy.ndarray
        Rotation angles (batch,).

    Returns
    -------
    rotations : numpy.ndarray
        Rotation matrices (batch, 3, 3).
    """
    # Normalize axis
    axis = axis / (np.linalg.norm(axis) + 1e-10)

    # Skew-symmetric matrix K
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # K @ K precomputed
    K2 = K @ K

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    # angles: (batch,)
    c = np.cos(angles)[:, None, None]  # (batch, 1, 1)
    s = np.sin(angles)[:, None, None]  # (batch, 1, 1)

    I = np.eye(3)
    return I + s * K + (1 - c) * K2  # (batch, 3, 3)


def compute_geometric_jacobian_batched_numpy(
        joint_angles_batch, fk_params, non_mimic_indices=None,
        mimic_parent_indices=None, mimic_multipliers=None):
    """Compute the geometric Jacobian for a batch of joint angles using NumPy.

    Computes the 6xN_opt Jacobian matrix (with mimic joint folding) for each
    element in the batch, along with the end-effector pose.

    Parameters
    ----------
    joint_angles_batch : numpy.ndarray
        Joint angles (batch, n_joints).
    fk_params : dict
        FK parameters from extract_fk_parameters().
    non_mimic_indices : numpy.ndarray, optional
        Indices of non-mimic joints. Computed from fk_params if None.
    mimic_parent_indices : numpy.ndarray, optional
        Mimic parent indices. Computed from fk_params if None.
    mimic_multipliers : numpy.ndarray, optional
        Mimic multipliers. Computed from fk_params if None.

    Returns
    -------
    J_opt : numpy.ndarray
        Geometric Jacobian (batch, 6, n_opt).
    ee_pos : numpy.ndarray
        End-effector positions (batch, 3).
    ee_rot : numpy.ndarray
        End-effector rotations (batch, 3, 3).
    """
    batch_size = joint_angles_batch.shape[0]
    n_joints = fk_params['n_joints']

    translations = fk_params['link_translations']
    local_rotations = fk_params['link_rotations']
    axes = fk_params['joint_axes']
    base_pos = fk_params['base_position']
    base_rot = fk_params['base_rotation']
    ref_angles = fk_params['ref_angles']
    joint_types = fk_params['joint_types']
    ee_offset_pos = fk_params['ee_offset_position']
    ee_offset_rot = fk_params['ee_offset_rotation']

    # Get mimic joint info
    if non_mimic_indices is None or mimic_parent_indices is None:
        _mimic_parent = fk_params.get(
            'mimic_parent_indices', np.array([-1] * n_joints, dtype=np.int32))
        _mimic_mult = fk_params.get('mimic_multipliers', np.ones(n_joints))
        _non_mimic = np.array(
            [i for i in range(n_joints) if _mimic_parent[i] < 0],
            dtype=np.int32)
        if non_mimic_indices is None:
            non_mimic_indices = _non_mimic
        if mimic_parent_indices is None:
            mimic_parent_indices = _mimic_parent
        if mimic_multipliers is None:
            mimic_multipliers = _mimic_mult
    n_opt = len(non_mimic_indices)

    # Handle mimic joints in angles
    _fk_mimic_parent = fk_params.get('mimic_parent_indices')
    if _fk_mimic_parent is not None:
        _fk_mimic_mult = fk_params['mimic_multipliers']
        _fk_mimic_off = fk_params['mimic_offsets']
        effective_angles = joint_angles_batch.copy()
        for i in range(n_joints):
            parent_idx = _fk_mimic_parent[i]
            if parent_idx >= 0:
                effective_angles[:, i] = (
                    joint_angles_batch[:, parent_idx] * _fk_mimic_mult[i]
                    + _fk_mimic_off[i]
                )
        joint_angles_batch = effective_angles

    # Forward kinematics, tracking joint positions and axes in world frame
    current_pos = np.tile(base_pos, (batch_size, 1))
    current_rot = np.tile(base_rot, (batch_size, 1, 1))

    # Store joint positions and world-frame axes for Jacobian computation
    joint_positions = np.zeros((n_joints, batch_size, 3))
    joint_axes_world = np.zeros((n_joints, batch_size, 3))

    for i in range(n_joints):
        # Apply link static transform
        current_pos = current_pos + np.einsum(
            'bij,j->bi', current_rot, translations[i])
        current_rot = np.einsum(
            'bij,jk->bik', current_rot, local_rotations[i])

        # Store joint position and axis in world frame
        joint_positions[i] = current_pos.copy()
        joint_axes_world[i] = np.einsum(
            'bij,j->bi', current_rot, axes[i])

        # Apply joint motion
        delta_angles = joint_angles_batch[:, i] - ref_angles[i]

        if joint_types[i] == 'prismatic':
            current_pos = current_pos + np.einsum(
                'bij,j,b->bi', current_rot, axes[i], delta_angles)
        else:
            joint_rots = _axis_angle_to_matrix_batched(axes[i], delta_angles)
            current_rot = np.einsum('bij,bjk->bik', current_rot, joint_rots)

    # Apply end-effector offset
    ee_pos = current_pos + np.einsum('bij,j->bi', current_rot, ee_offset_pos)
    ee_rot = np.einsum('bij,jk->bik', current_rot, ee_offset_rot)

    # Compute full Jacobian columns: (batch, 6, n_joints)
    J_full = np.zeros((batch_size, 6, n_joints))

    for i in range(n_joints):
        z_i = joint_axes_world[i]  # (batch, 3)
        p_i = joint_positions[i]    # (batch, 3)

        if joint_types[i] in ('revolute', 'continuous'):
            # J_v = z × (p_ee - p_i)
            r = ee_pos - p_i  # (batch, 3)
            j_v = np.cross(z_i, r)  # (batch, 3)
            j_w = z_i               # (batch, 3)
        else:
            j_v = z_i
            j_w = np.zeros_like(z_i)

        J_full[:, :3, i] = j_v
        J_full[:, 3:, i] = j_w

    # Fold mimic joint contributions into parent joints
    # Build mapping from full joint index to non-mimic index
    full_to_opt = {}
    for opt_idx, full_idx in enumerate(non_mimic_indices):
        full_to_opt[int(full_idx)] = opt_idx

    J_opt = np.zeros((batch_size, 6, n_opt))
    for i in range(n_joints):
        parent_idx = int(mimic_parent_indices[i])
        if parent_idx < 0:
            opt_idx = full_to_opt[i]
            J_opt[:, :, opt_idx] += J_full[:, :, i]
        else:
            if parent_idx in full_to_opt:
                opt_idx = full_to_opt[parent_idx]
                J_opt[:, :, opt_idx] += mimic_multipliers[i] * J_full[:, :, i]

    return J_opt, ee_pos, ee_rot


def forward_kinematics_ee_batched_numpy(joint_angles_batch, fk_params):
    """Compute end-effector pose for a batch of joint angles using NumPy.

    Optimized version that processes all batch elements simultaneously.

    Parameters
    ----------
    joint_angles_batch : numpy.ndarray
        Joint angles (batch, n_joints).
    fk_params : dict
        FK parameters from extract_fk_parameters().

    Returns
    -------
    positions : numpy.ndarray
        End-effector positions (batch, 3).
    rotations : numpy.ndarray
        End-effector rotations (batch, 3, 3).
    """
    batch_size = joint_angles_batch.shape[0]
    n_joints = fk_params['n_joints']

    translations = fk_params['link_translations']  # (n_joints, 3)
    local_rotations = fk_params['link_rotations']  # (n_joints, 3, 3)
    axes = fk_params['joint_axes']  # (n_joints, 3)
    base_pos = fk_params['base_position']  # (3,)
    base_rot = fk_params['base_rotation']  # (3, 3)
    ref_angles = fk_params['ref_angles']  # (n_joints,)
    joint_types = fk_params['joint_types']
    ee_offset_pos = fk_params['ee_offset_position']  # (3,)
    ee_offset_rot = fk_params['ee_offset_rotation']  # (3, 3)

    # Handle mimic joints
    mimic_parent_indices = fk_params.get('mimic_parent_indices')
    if mimic_parent_indices is not None:
        mimic_multipliers = fk_params['mimic_multipliers']
        mimic_offsets = fk_params['mimic_offsets']

        effective_angles = joint_angles_batch.copy()
        for i in range(n_joints):
            parent_idx = mimic_parent_indices[i]
            if parent_idx >= 0:
                effective_angles[:, i] = (
                    joint_angles_batch[:, parent_idx] * mimic_multipliers[i]
                    + mimic_offsets[i]
                )
        joint_angles_batch = effective_angles

    # Initialize batch transforms
    current_pos = np.tile(base_pos, (batch_size, 1))  # (batch, 3)
    current_rot = np.tile(base_rot, (batch_size, 1, 1))  # (batch, 3, 3)

    for i in range(n_joints):
        # Apply link static transform
        # pos = pos + rot @ translation
        current_pos = current_pos + np.einsum(
            'bij,j->bi', current_rot, translations[i]
        )
        # rot = rot @ local_rotation
        current_rot = np.einsum(
            'bij,jk->bik', current_rot, local_rotations[i]
        )

        # Apply joint motion
        delta_angles = joint_angles_batch[:, i] - ref_angles[i]  # (batch,)

        if joint_types[i] == 'prismatic':
            # Translate along axis
            current_pos = current_pos + np.einsum(
                'bij,j,b->bi', current_rot, axes[i], delta_angles
            )
        else:
            # Revolute: rotate around axis
            joint_rots = _axis_angle_to_matrix_batched(
                axes[i], delta_angles
            )  # (batch, 3, 3)
            current_rot = np.einsum('bij,bjk->bik', current_rot, joint_rots)

    # Apply end-effector offset
    current_pos = current_pos + np.einsum('bij,j->bi', current_rot, ee_offset_pos)
    current_rot = np.einsum('bij,jk->bik', current_rot, ee_offset_rot)

    return current_pos, current_rot


def _compute_gradient_batched_numpy(loss_fn_batched, angles_batch, eps=1e-7):
    """Compute gradients for a batch of angles using vectorized finite differences.

    Parameters
    ----------
    loss_fn_batched : callable
        Loss function that takes (batch, n_joints) and returns (batch,).
    angles_batch : numpy.ndarray
        Joint angles (batch, n_joints).
    eps : float
        Finite difference step size.

    Returns
    -------
    gradients : numpy.ndarray
        Gradients (batch, n_joints).
    """
    batch_size, n_joints = angles_batch.shape

    # Create perturbed angles for all parameters at once
    # Shape: (batch, n_joints, 2) - plus and minus perturbations
    angles_plus = np.tile(angles_batch[:, :, None], (1, 1, n_joints))  # (batch, n_joints, n_joints)
    angles_minus = angles_plus.copy()

    # Add perturbations along diagonal (perturbing each joint independently)
    for j in range(n_joints):
        angles_plus[:, j, j] += eps
        angles_minus[:, j, j] -= eps

    # Reshape for batch evaluation: (batch * n_joints, n_joints)
    angles_plus_flat = angles_plus.reshape(-1, n_joints)
    angles_minus_flat = angles_minus.reshape(-1, n_joints)

    # Evaluate loss for all perturbed inputs
    loss_plus = loss_fn_batched(angles_plus_flat).reshape(batch_size, n_joints)
    loss_minus = loss_fn_batched(angles_minus_flat).reshape(batch_size, n_joints)

    # Central difference
    gradients = (loss_plus - loss_minus) / (2 * eps)

    return gradients


def create_jax_fk_function(fk_params):
    """Create a pure JAX FK function from extracted parameters.

    This function is provided for backward compatibility with code that
    uses JAX-specific FK functions. For new code, prefer using
    forward_kinematics_ee with get_backend('jax') instead.

    The returned function is fully compatible with JAX transformations
    (jit, vmap, grad) for high-performance batch IK solving.

    Parameters
    ----------
    fk_params : dict
        FK parameters from extract_fk_parameters.

    Returns
    -------
    callable
        Pure JAX FK function: fk_fn(angles) -> (pos, rot)
        - angles: Joint angles (n_joints,)
        - pos: End effector position (3,)
        - rot: End effector rotation matrix (3, 3)

    Examples
    --------
    >>> from skrobot.kinematics.differentiable import (
    ...     extract_fk_parameters, create_jax_fk_function
    ... )
    >>> fk_params = extract_fk_parameters(robot, link_list, move_target)
    >>> fk_fn = create_jax_fk_function(fk_params)
    >>> pos, rot = fk_fn(joint_angles)
    """
    try:
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for create_jax_fk_function. "
            "Install it with: pip install jax jaxlib"
        )

    # Convert parameters to JAX arrays
    link_translations = jnp.array(fk_params['link_translations'])
    link_rotations = jnp.array(fk_params['link_rotations'])
    joint_axes = jnp.array(fk_params['joint_axes'])
    base_position = jnp.array(fk_params['base_position'])
    base_rotation = jnp.array(fk_params['base_rotation'])
    ref_angles = jnp.array(fk_params['ref_angles'])
    ee_offset_position = jnp.array(fk_params['ee_offset_position'])
    ee_offset_rotation = jnp.array(fk_params['ee_offset_rotation'])
    n_joints = fk_params['n_joints']

    def rotation_matrix_axis_angle(axis, theta):
        """Compute rotation matrix from axis-angle (Rodrigues formula)."""
        # Normalize axis
        axis = axis / jnp.sqrt(jnp.dot(axis, axis) + 1e-10)

        # Skew-symmetric matrix
        K = jnp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        # Rodrigues formula: R = I + sin(theta)*K + (1-cos(theta))*K^2
        I = jnp.eye(3)
        return I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)

    def fk_fn(angles):
        """Pure JAX forward kinematics.

        Parameters
        ----------
        angles : jax.numpy.ndarray
            Joint angles (n_joints,).

        Returns
        -------
        tuple
            (position, rotation) - End effector pose
        """
        # Start from base
        current_pos = base_position
        current_rot = base_rotation

        # Chain through each joint
        for i in range(n_joints):
            # Static link transform
            link_trans = link_translations[i]
            link_rot = link_rotations[i]

            # Apply link static transform
            current_pos = current_pos + current_rot @ link_trans
            current_rot = current_rot @ link_rot

            # Joint rotation as delta from reference angle
            delta_angle = angles[i] - ref_angles[i]
            joint_rot = rotation_matrix_axis_angle(joint_axes[i], delta_angle)
            current_rot = current_rot @ joint_rot

        # Apply end effector offset
        current_pos = current_pos + current_rot @ ee_offset_position
        current_rot = current_rot @ ee_offset_rotation

        return current_pos, current_rot

    return fk_fn


def compute_manipulability(backend, joint_angles, fk_params, position_only: bool = True):
    """Compute Yoshikawa manipulability measure.

    Parameters
    ----------
    backend : object
        Backend to use for computation.
    joint_angles : array
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters().
    position_only : bool
        If True, compute manipulability for position only.

    Returns
    -------
    manipulability : float
        Manipulability measure (sqrt of det(J @ J.T)).
    """
    if position_only:
        J = compute_jacobian(backend, joint_angles, fk_params)
    else:
        J = compute_full_jacobian(backend, joint_angles, fk_params)

    JJT = backend.matmul(J, backend.transpose(J))
    det = backend.det(JJT)
    return backend.sqrt(backend.maximum(det, backend.array(0.0)))


def solve_ik_scipy(
    robot_model,
    target_coords,
    link_list,
    move_target,
    rotation_mask=True,
    position_mask=True,
    max_iterations=100,
    pos_threshold=0.001,
    rot_threshold=0.017,
):
    """Solve IK using scipy optimization.

    This is a NumPy/SciPy-based IK solver that works without JAX.
    For batch IK solving with autodiff, use create_batch_ik_solver instead.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    target_coords : Coordinates
        Target pose.
    link_list : list
        List of links in the kinematic chain.
    move_target : CascadedCoords
        End effector coordinates.
    rotation_mask : bool
        Whether to constrain rotation.
    position_mask : bool
        Whether to constrain translation.
    max_iterations : int
        Maximum iterations.
    pos_threshold : float
        Position threshold.
    rot_threshold : float
        Rotation threshold in radians.

    Returns
    -------
    numpy.ndarray or False
        Joint angles if successful, False otherwise.
    """
    from scipy.optimize import minimize

    # Get joint limits
    lower_bounds = []
    upper_bounds = []
    for link in link_list:
        joint = link.joint
        if hasattr(joint, 'min_angle') and joint.min_angle is not None:
            lower_bounds.append(joint.min_angle)
        else:
            lower_bounds.append(-np.pi)
        if hasattr(joint, 'max_angle') and joint.max_angle is not None:
            upper_bounds.append(joint.max_angle)
        else:
            upper_bounds.append(np.pi)

    bounds = list(zip(lower_bounds, upper_bounds))

    # Get target pose
    target_pos = target_coords.worldpos()
    target_rot = target_coords.worldrot()

    # Save current angles
    original_angles = np.array([link.joint.joint_angle() for link in link_list])

    def objective(angles):
        """Compute pose error."""
        # Set joint angles
        for i, link in enumerate(link_list):
            link.joint.joint_angle(angles[i])

        # Get current pose
        current_pos = move_target.worldpos()
        current_rot = move_target.worldrot()

        # Position error
        pos_err = np.sum((current_pos - target_pos) ** 2)

        # Rotation error
        if rotation_mask:
            rot_err = np.sum((current_rot - target_rot) ** 2)
        else:
            rot_err = 0.0

        return pos_err + 0.1 * rot_err

    # Optimize
    result = minimize(
        objective,
        original_angles,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations}
    )

    # Check if converged
    for i, link in enumerate(link_list):
        link.joint.joint_angle(result.x[i])

    final_pos = move_target.worldpos()

    pos_err = np.linalg.norm(final_pos - target_pos)

    # Restore original angles
    for i, link in enumerate(link_list):
        link.joint.joint_angle(original_angles[i])

    if pos_err < pos_threshold:
        return result.x
    else:
        return False


def filter_targets_by_reachability(
    target_positions,
    reachability_map,
    return_mask=False,
):
    """Filter target positions by reachability.

    This is a utility function that can be used with any IK solver
    to pre-filter unreachable targets.

    Parameters
    ----------
    target_positions : np.ndarray
        Target positions with shape (N, 3) or (N, 6/7) where first 3
        columns are position.
    reachability_map : ReachabilityMap
        Computed reachability map.
    return_mask : bool
        If True, also return the boolean mask.

    Returns
    -------
    np.ndarray
        Filtered target positions (M, ...) where M <= N.
    np.ndarray (optional)
        Boolean mask of shape (N,) if return_mask=True.

    Examples
    --------
    >>> # Filter targets before batch IK
    >>> filtered, mask = filter_targets_by_reachability(
    ...     targets, rmap, return_mask=True
    ... )
    >>> solutions = batch_ik_solve(filtered)
    >>> # Reconstruct full results
    >>> full_solutions = np.full((len(targets), n_joints), np.nan)
    >>> full_solutions[mask] = solutions
    """
    target_positions = np.asarray(target_positions)
    positions = target_positions[:, :3]

    scores = reachability_map.get_reachability_at_positions(positions)
    mask = scores > 0

    if return_mask:
        return target_positions[mask], mask
    return target_positions[mask]


def create_reachability_aware_solver(
    robot_model,
    link_list,
    move_target,
    reachability_map,
    backend_name='jax',
):
    """Create a batch IK solver with built-in reachability filtering.

    This wraps create_batch_ik_solver with automatic reachability
    pre-filtering for efficiency.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    link_list : list
        List of links in the kinematic chain.
    move_target : CascadedCoords
        End effector coordinates.
    reachability_map : ReachabilityMap
        Pre-computed reachability map.
    backend_name : str
        Backend to use. Default is 'jax'.

    Returns
    -------
    callable
        Batch IK solver with signature:
        solve(target_positions, target_rotations, filter_unreachable=True, ...)
        -> (solutions, success, errors, original_indices)

    Examples
    --------
    >>> solver = create_reachability_aware_solver(
    ...     robot, link_list, move_target, rmap
    ... )
    >>> solutions, success, errors, indices = solver(
    ...     positions, rotations, filter_unreachable=True
    ... )
    >>> # indices maps back to original target array
    """
    base_solver = create_batch_ik_solver(
        robot_model, link_list, move_target, backend_name
    )

    def solve_with_reachability(
        target_positions,
        target_rotations,
        filter_unreachable=True,
        **kwargs
    ):
        """Solve batch IK with optional reachability filtering.

        Parameters
        ----------
        target_positions : array-like
            Target positions (N, 3).
        target_rotations : array-like
            Target rotations (N, 3, 3).
        filter_unreachable : bool
            If True, skip unreachable targets. Default is True.
        **kwargs
            Additional arguments passed to the base solver.

        Returns
        -------
        tuple
            (solutions, success_flags, errors, original_indices)
            where original_indices maps results back to input array.
        """
        target_positions = np.asarray(target_positions)
        target_rotations = np.asarray(target_rotations)
        n_targets = len(target_positions)
        original_indices = np.arange(n_targets)

        if filter_unreachable:
            scores = reachability_map.get_reachability_at_positions(
                target_positions
            )
            reachable_mask = scores > 0

            if not np.all(reachable_mask):
                target_positions = target_positions[reachable_mask]
                target_rotations = target_rotations[reachable_mask]
                original_indices = original_indices[reachable_mask]

                if len(target_positions) == 0:
                    # All unreachable
                    backend = base_solver.backend
                    empty = backend.array([]).reshape(0, base_solver.n_joints)
                    return empty, np.array([]), np.array([]), original_indices

        solutions, success, errors = base_solver(
            target_positions, target_rotations, **kwargs
        )

        return solutions, success, errors, original_indices

    # Attach properties from base solver
    solve_with_reachability.n_joints = base_solver.n_joints
    solve_with_reachability.joint_limits_lower = base_solver.joint_limits_lower
    solve_with_reachability.joint_limits_upper = base_solver.joint_limits_upper
    solve_with_reachability.fk_fn = base_solver.fk_fn
    solve_with_reachability.fk_params = base_solver.fk_params
    solve_with_reachability.backend = base_solver.backend
    solve_with_reachability.reachability_map = reachability_map

    return solve_with_reachability
