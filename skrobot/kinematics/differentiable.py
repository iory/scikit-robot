"""Backend-agnostic differentiable kinematics.

This module provides forward kinematics and Jacobian computation that works
with any differentiable backend (NumPy, JAX, PyTorch).
"""

import numpy as np


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
    if mimic_parent_indices is not None:
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

    def loss_fn(q):
        pos, rot = forward_kinematics_ee(backend, q, fk_params)
        pos_error = backend.sum((pos - target_position) ** 2)
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


def create_batch_ik_solver(robot_model, link_list, move_target, backend_name='jax'):
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

    Returns
    -------
    callable
        Batch IK solver function with signature:
        solve(target_positions, target_rotations, initial_angles=None,
              max_iterations=100, learning_rate=0.1) -> (solutions, success, errors)

    Examples
    --------
    >>> solver = create_batch_ik_solver(robot, link_list, move_target)
    >>> solutions, success, errors = solver(target_positions, target_rotations)
    """
    import numpy as np

    from skrobot.backend import get_backend

    backend = get_backend(backend_name)

    # Extract FK parameters
    fk_params = extract_fk_parameters(robot_model, link_list, move_target)

    n_joints = fk_params['n_joints']

    # Identify non-mimic joints (these are the actual optimization variables)
    mimic_parent_indices = fk_params.get('mimic_parent_indices', np.array([-1] * n_joints))
    mimic_multipliers = fk_params.get('mimic_multipliers', np.ones(n_joints))
    mimic_offsets = fk_params.get('mimic_offsets', np.zeros(n_joints))

    # Build list of non-mimic joint indices
    non_mimic_indices = [i for i in range(n_joints) if mimic_parent_indices[i] < 0]
    len(non_mimic_indices)

    # Get joint limits for non-mimic joints only
    joint_limits_lower = backend.array(fk_params['joint_limits_lower'][non_mimic_indices])
    joint_limits_upper = backend.array(fk_params['joint_limits_upper'][non_mimic_indices])

    # JIT cache for different iteration counts
    _jit_cache = {}

    # Precompute arrays for expanding opt angles to full angles
    _non_mimic_indices_arr = backend.array(np.array(non_mimic_indices, dtype=np.int32))
    _mimic_parent_indices_arr = backend.array(mimic_parent_indices.astype(np.int32))
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

    def _create_solver_fn(max_iterations, learning_rate, pos_weight, rot_weight, pos_threshold):
        """Create a JIT-compiled solver for given parameters."""

        def loss_fn(opt_angles, target_pos, target_rot):
            """Compute weighted pose error."""
            # Expand to full angles (including mimic joints)
            full_angles = _expand_to_full_angles(opt_angles)
            pos, rot = forward_kinematics_ee(backend, full_angles, fk_params)
            pos_err = backend.sum((pos - target_pos) ** 2)
            rot_err = backend.sum((rot - target_rot) ** 2)
            return pos_weight * pos_err + rot_weight * rot_err

        loss_and_grad = backend.value_and_grad(loss_fn)

        def solve_single(init_opt_angles, target_pos, target_rot):
            """Solve IK for a single target using scan for efficient JIT."""

            def body_fn(carry, _):
                """Single iteration step."""
                opt_angles = carry
                loss, grad = loss_and_grad(opt_angles, target_pos, target_rot)
                new_opt_angles = opt_angles - learning_rate * grad
                new_opt_angles = backend.clip(new_opt_angles, joint_limits_lower, joint_limits_upper)
                return new_opt_angles, loss

            # Use scan for efficient JIT compilation
            final_opt_angles, _ = backend.scan(body_fn, init_opt_angles, None, length=max_iterations)

            # Expand to full angles for final error check
            final_full_angles = _expand_to_full_angles(final_opt_angles)
            final_pos, _ = forward_kinematics_ee(backend, final_full_angles, fk_params)
            pos_err = backend.sqrt(backend.sum((final_pos - target_pos) ** 2))
            success = pos_err < pos_threshold

            return final_full_angles, success, pos_err

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
              pos_threshold=0.001):
        """Solve batch IK.

        Parameters
        ----------
        target_positions : array-like
            Target positions (N, 3).
        target_rotations : array-like
            Target rotation matrices (N, 3, 3).
        initial_angles : array-like, optional
            Initial joint angles (N, n_joints) or (n_joints,).
        max_iterations : int
            Maximum gradient descent iterations.
        learning_rate : float
            Gradient descent step size.
        pos_weight : float
            Position error weight.
        rot_weight : float
            Rotation error weight.
        pos_threshold : float
            Position error threshold for success.

        Returns
        -------
        tuple
            (solutions, success_flags, position_errors)
        """
        target_positions = backend.array(target_positions)
        target_rotations = backend.array(target_rotations)
        n_targets = target_positions.shape[0]

        if initial_angles is None:
            # Use middle of joint limits as initial guess (for non-mimic joints only)
            init = (joint_limits_lower + joint_limits_upper) / 2
            initial_opt_angles = backend.stack([init] * n_targets)
        else:
            initial_angles = backend.array(initial_angles)
            if len(initial_angles.shape) == 1:
                initial_angles = backend.stack([initial_angles] * n_targets)
            # Extract non-mimic joint angles
            initial_opt_angles = initial_angles[:, non_mimic_indices]

        # Get or create JIT-compiled solver for these parameters
        cache_key = (max_iterations, learning_rate, pos_weight, rot_weight, pos_threshold)
        if cache_key not in _jit_cache:
            _jit_cache[cache_key] = _create_solver_fn(
                max_iterations, learning_rate, pos_weight, rot_weight, pos_threshold
            )

        solver_fn = _jit_cache[cache_key]

        return solver_fn(initial_opt_angles, target_positions, target_rotations)

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
    rotation_axis=True,
    translation_axis=True,
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
    rotation_axis : bool
        Whether to constrain rotation.
    translation_axis : bool
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
        if rotation_axis:
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
