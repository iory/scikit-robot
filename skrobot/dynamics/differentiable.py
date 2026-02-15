"""Differentiable Gravity Dynamics for Sensorless Control.

This module provides differentiable gravity torque computation with backend
abstraction. Supports both JAX (fast autodiff) and NumPy (numerical fallback).

The key insight is that gravity torque is the gradient of potential energy:

    τ_gravity = ∂U/∂q

where U(q) = Σ_i m_i * g^T * p_i(q) is the gravitational potential energy.

By using automatic differentiation, we can:
1. Compute gravity torques efficiently
2. Optimize mass/CoM parameters to match real robot behavior
3. Enable sensorless gravity compensation using only position errors

Physical Model:
    For position-controlled servos, the measured torque can be estimated from
    servo deviation (position error):

        τ_measured = K * (q_cmd - q_actual)

    where K is the joint stiffness matrix (diagonal).

    By comparing τ_measured with τ_gravity, we can identify parameters and
    implement gravity compensation.

Usage:
    >>> from skrobot.dynamics.jax_dynamics import build_gravity_fn
    >>> gravity_fn = build_gravity_fn(robot_model)
    >>> tau = gravity_fn(joint_angles, masses, coms)
"""

import numpy as np

from skrobot.backend import get_backend
from skrobot.backend import rodrigues_rotation as _rodrigues_rotation


def extract_dynamics_parameters(robot_model, link_list=None):
    """Extract dynamics parameters from RobotModel for JAX computation.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include. If None, uses all movable joints.

    Returns
    -------
    params : dict
        Dictionary containing:
        - n_joints: int, number of joints
        - joint_axes: (n_joints, 3) rotation axes
        - link_translations: (n_joints, 3) static translations
        - link_rotations: (n_joints, 3, 3) static rotations
        - default_masses: (n_joints,) default link masses
        - default_coms: (n_joints, 3) default center of mass positions
        - joint_limits_lower: (n_joints,) lower limits
        - joint_limits_upper: (n_joints,) upper limits
        - base_position: (3,) base position
        - base_rotation: (3, 3) base rotation
    """
    # Determine link list
    if link_list is None:
        link_list = []
        for link in robot_model.link_list:
            if hasattr(link, 'joint') and link.joint is not None:
                if link.joint.joint_type != 'fixed':
                    link_list.append(link)

    n_joints = len(link_list)

    joint_axes = []
    link_translations = []
    link_rotations = []
    default_masses = []
    default_coms = []
    joint_limits_lower = []
    joint_limits_upper = []

    for link in link_list:
        joint = link.joint

        # Joint axis
        joint_axes.append(np.array(joint.axis, dtype=np.float64))

        # Static transform
        if hasattr(joint, 'default_coords') and joint.default_coords is not None:
            link_translations.append(
                joint.default_coords.translation.astype(np.float64)
            )
            link_rotations.append(
                joint.default_coords.rotation.astype(np.float64)
            )
        else:
            if hasattr(link, '_translation') and link._translation is not None:
                link_translations.append(link._translation.astype(np.float64))
            else:
                link_translations.append(np.zeros(3, dtype=np.float64))
            if hasattr(link, '_rotation') and link._rotation is not None:
                link_rotations.append(link._rotation.astype(np.float64))
            else:
                link_rotations.append(np.eye(3, dtype=np.float64))

        # Mass properties
        mass = getattr(link, 'mass', None) or 0.1  # Default 100g
        com = getattr(link, 'center_of_mass', None)
        if com is None:
            com = np.zeros(3)
        default_masses.append(float(mass))
        default_coms.append(np.array(com, dtype=np.float64))

        # Joint limits
        min_angle = getattr(joint, 'min_angle', None)
        max_angle = getattr(joint, 'max_angle', None)
        if min_angle is None or not np.isfinite(min_angle):
            min_angle = -np.pi
        if max_angle is None or not np.isfinite(max_angle):
            max_angle = np.pi
        joint_limits_lower.append(float(min_angle))
        joint_limits_upper.append(float(max_angle))

    # Base transform
    first_link = link_list[0]
    if first_link.parent is not None:
        base_position = first_link.parent.worldpos().astype(np.float64)
        base_rotation = first_link.parent.worldrot().astype(np.float64)
    else:
        base_position = np.zeros(3, dtype=np.float64)
        base_rotation = np.eye(3, dtype=np.float64)

    return {
        'n_joints': n_joints,
        'joint_axes': np.array(joint_axes, dtype=np.float64),
        'link_translations': np.array(link_translations, dtype=np.float64),
        'link_rotations': np.array(link_rotations, dtype=np.float64),
        'default_masses': np.array(default_masses, dtype=np.float64),
        'default_coms': np.array(default_coms, dtype=np.float64),
        'joint_limits_lower': np.array(joint_limits_lower, dtype=np.float64),
        'joint_limits_upper': np.array(joint_limits_upper, dtype=np.float64),
        'base_position': base_position,
        'base_rotation': base_rotation,
    }


def build_gravity_fn(robot_model, link_list=None, backend=None):
    """Build differentiable gravity torque function with autodiff support.

    This function creates a differentiable gravity torque computation
    that can be used for:
    - Parameter optimization (masses, CoMs)
    - Real-time gravity compensation
    - Sensorless force estimation

    The gravity torque is computed as:
        τ = ∂U/∂q

    where U(q) = Σ_i m_i * g^T * p_i(q) is the gravitational potential energy.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.

    Returns
    -------
    gravity_fn : Callable
        Function with signature:
        gravity_fn(q, masses, coms, gravity) -> tau

        Parameters:
        - q: (n_joints,) joint angles
        - masses: (n_joints,) link masses in kg
        - coms: (n_joints, 3) center of mass in link frame
        - gravity: (3,) gravity vector (default: [0, 0, -9.81])

        Returns:
        - tau: (n_joints,) gravity torques in N*m

    Examples
    --------
    >>> from skrobot.models import Panda
    >>> from skrobot.dynamics.jax_dynamics import build_gravity_fn
    >>> robot = Panda()
    >>> gravity_fn = build_gravity_fn(robot)
    >>> import numpy as np
    >>> q = np.zeros(7)
    >>> masses = np.ones(7) * 0.5  # 500g per link
    >>> coms = np.zeros((7, 3))
    >>> tau = gravity_fn(q, masses, coms)
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    # Extract parameters
    params = extract_dynamics_parameters(robot_model, link_list)

    # Convert to backend arrays (these become traced constants)
    joint_axes = backend.array(params['joint_axes'])
    link_translations = backend.array(params['link_translations'])
    link_rotations = backend.array(params['link_rotations'])
    base_position = backend.array(params['base_position'])
    base_rotation = backend.array(params['base_rotation'])
    n_joints = params['n_joints']

    def rodrigues_rotation(axis, angle):
        """Rodrigues' rotation formula: axis-angle -> rotation matrix."""
        return _rodrigues_rotation(backend, axis, angle)

    def potential_energy(q, masses, coms, gravity):
        """Compute gravitational potential energy.

        U(q) = -Σ_i m_i * g^T * p_i(q)

        The negative sign is because gravity points downward,
        and we want U to increase with height.

        Parameters
        ----------
        q : array, shape (n_joints,)
            Joint angles.
        masses : array, shape (n_joints,)
            Link masses in kg.
        coms : array, shape (n_joints, 3)
            Center of mass positions in link frame.
        gravity : array, shape (3,)
            Gravity vector in world frame.

        Returns
        -------
        U : float
            Total potential energy in Joules.
        """
        pos = base_position
        rot = base_rotation
        total_energy = 0.0

        for i in range(n_joints):
            # Apply static transform
            pos = pos + rot @ link_translations[i]
            rot = rot @ link_rotations[i]

            # Apply joint rotation
            joint_rot = rodrigues_rotation(joint_axes[i], q[i])
            rot = rot @ joint_rot

            # COM position in world frame
            com_world = rot @ coms[i] + pos

            # Potential energy: U = -m * g^T * h
            # Negative because gravity points down
            total_energy = total_energy - masses[i] * backend.dot(gravity, com_world)

        return total_energy

    def gravity_torque(q, masses, coms, gravity=None):
        """Compute gravity torque via automatic differentiation.

        τ = ∂U/∂q

        This is the torque required to hold the robot against gravity.
        It's the "weight" that each joint feels.

        Parameters
        ----------
        q : array, shape (n_joints,)
            Joint angles in radians.
        masses : array, shape (n_joints,)
            Link masses in kg.
        coms : array, shape (n_joints, 3)
            Center of mass in link local frame (meters).
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81]

        Returns
        -------
        tau : array, shape (n_joints,)
            Gravity torque at each joint (N*m).
        """
        if gravity is None:
            gravity = backend.array([0.0, 0.0, -9.81])

        # Use backend autodiff to compute gradient of potential energy
        # τ = ∂U/∂q
        grad_fn = backend.gradient(
            lambda q_: potential_energy(q_, masses, coms, gravity)
        )
        return grad_fn(q)

    # Return compiled function for performance if supported
    return backend.compile(gravity_torque)


def build_gravity_fn_with_stiffness(robot_model, link_list=None, backend=None):
    """Build gravity function that also returns measured torque from stiffness.

    This is useful for calibration where we compare:
        τ_measured = K * (q_cmd - q_actual)
        τ_gravity = ∂U/∂q

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    link_list : list, optional
        Links to include.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.

    Returns
    -------
    compute_torques : Callable
        Function with signature:
        compute_torques(q_cmd, q_actual, stiffness, masses, coms, gravity)
        -> (tau_measured, tau_gravity)

    Examples
    --------
    >>> compute_torques = build_gravity_fn_with_stiffness(robot)
    >>> tau_meas, tau_grav = compute_torques(q_cmd, q_actual, K, masses, coms)
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    gravity_fn = build_gravity_fn(robot_model, link_list, backend=backend)

    def compute_torques(q_cmd, q_actual, stiffness, masses, coms, gravity=None):
        """Compute both measured and gravity torques.

        Parameters
        ----------
        q_cmd : array, shape (n_joints,)
            Commanded joint angles.
        q_actual : array, shape (n_joints,)
            Actual (measured) joint angles.
        stiffness : array, shape (n_joints,)
            Joint stiffness values (N*m/rad).
        masses : array, shape (n_joints,)
            Link masses.
        coms : array, shape (n_joints, 3)
            Center of mass positions.
        gravity : array, shape (3,), optional
            Gravity vector.

        Returns
        -------
        tau_measured : array
            Torque estimated from servo deviation.
        tau_gravity : array
            Theoretical gravity torque from model.
        """
        if gravity is None:
            gravity = backend.array([0.0, 0.0, -9.81])

        # Measured torque from servo deviation
        # τ_measured = K * (q_cmd - q_actual)
        tau_measured = stiffness * (q_cmd - q_actual)

        # Theoretical gravity torque
        tau_gravity = gravity_fn(q_actual, masses, coms, gravity)

        return tau_measured, tau_gravity

    return backend.compile(compute_torques)


def build_jacobian_gravity_fn(robot_model, link_list=None, backend=None):
    """Build function that computes both gravity torque and its Jacobian.

    The Jacobian ∂τ/∂(masses, coms) is useful for optimization.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model.
    link_list : list, optional
        Links to include.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.

    Returns
    -------
    gravity_and_jac : Callable
        Function returning (tau, d_tau_d_masses, d_tau_d_coms)
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    gravity_fn = build_gravity_fn(robot_model, link_list, backend=backend)

    def gravity_and_jacobian(q, masses, coms, gravity=None):
        """Compute gravity torque and its Jacobian w.r.t. parameters.

        Returns
        -------
        tau : (n_joints,)
            Gravity torque.
        d_tau_d_masses : (n_joints, n_joints)
            Jacobian of tau w.r.t. masses.
        d_tau_d_coms : (n_joints, n_joints, 3)
            Jacobian of tau w.r.t. CoM positions.
        """
        if gravity is None:
            gravity = backend.array([0.0, 0.0, -9.81])

        # Compute torque
        tau = gravity_fn(q, masses, coms, gravity)

        # Jacobian w.r.t. masses
        jac_fn_masses = backend.jacobian(
            lambda m: gravity_fn(q, m, coms, gravity)
        )
        jac_masses = jac_fn_masses(masses)

        # Jacobian w.r.t. CoMs
        jac_fn_coms = backend.jacobian(
            lambda c: gravity_fn(q, masses, c, gravity)
        )
        jac_coms = jac_fn_coms(coms)

        return tau, jac_masses, jac_coms

    return backend.compile(gravity_and_jacobian)


def estimate_external_torque(
    q_cmd,
    q_actual,
    stiffness,
    masses,
    coms,
    gravity_fn,
    gravity=None,
    backend=None
):
    """Estimate external torque (e.g., from human touch).

    The external torque is the difference between measured torque
    and expected gravity torque:

        τ_external = τ_measured - τ_gravity
                   = K * (q_cmd - q_actual) - ∂U/∂q

    If τ_external ≈ 0, the robot is in static equilibrium.
    If τ_external ≠ 0, something (like a human) is pushing the robot.

    Parameters
    ----------
    q_cmd : array
        Commanded joint angles.
    q_actual : array
        Actual joint angles.
    stiffness : array
        Joint stiffness (N*m/rad).
    masses : array
        Link masses.
    coms : array
        Center of mass positions.
    gravity_fn : Callable
        Gravity torque function from build_gravity_fn().
    gravity : array, optional
        Gravity vector.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.

    Returns
    -------
    tau_external : array
        Estimated external torque.

    Examples
    --------
    >>> tau_ext = estimate_external_torque(
    ...     q_cmd, q_actual, K, masses, coms, gravity_fn
    ... )
    >>> # If tau_ext[i] > 0, joint i is being pushed in positive direction
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    if gravity is None:
        gravity = backend.array([0.0, 0.0, -9.81])

    # Measured torque from servo deviation
    tau_measured = stiffness * (q_cmd - q_actual)

    # Expected gravity torque
    tau_gravity = gravity_fn(q_actual, masses, coms, gravity)

    # External torque = measured - expected
    tau_external = tau_measured - tau_gravity

    return tau_external


def extract_inverse_dynamics_parameters(robot_model, link_list=None,
                                         include_all_mass_links=True):
    """Extract parameters for inverse dynamics computation.

    This function extracts all kinematic and dynamic parameters needed
    for differentiable inverse dynamics computation. It extends
    extract_dynamics_parameters with inertia tensors and proper reference
    angle handling similar to kinematics/differentiable.py.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links defining the kinematic chain (joints to compute
        torques for). If None, uses all movable joints.
    include_all_mass_links : bool, optional
        If True (default), includes all links with mass in the robot
        (including end-effector, gripper, etc.) when computing torques.
        This matches the behavior of robot_model.inverse_dynamics().
        If False, only considers links in link_list.

    Returns
    -------
    params : dict
        Dictionary containing:
        - n_joints: int, number of joints
        - joint_axes: (n_joints, 3) rotation axes
        - joint_types: list of str, joint types ('revolute' or 'prismatic')
        - link_translations: (n_joints, 3) static translations
        - link_rotations: (n_joints, 3, 3) static rotations
        - link_masses: (n_joints,) link masses in kg (for chain links)
        - link_coms: (n_joints, 3) center of mass positions in link frame
        - link_inertias: (n_joints, 3, 3) inertia tensors about CoM
        - joint_limits_lower: (n_joints,) lower limits
        - joint_limits_upper: (n_joints,) upper limits
        - ref_angles: (n_joints,) reference angles
        - base_position: (3,) base position
        - base_rotation: (3, 3) base rotation
        - all_mass_links: list of all links with mass (if include_all_mass_links)
        - chain_link_names: list of names of links in the kinematic chain

    Examples
    --------
    >>> from skrobot.dynamics import extract_inverse_dynamics_parameters
    >>> params = extract_inverse_dynamics_parameters(robot)
    >>> print(params['n_joints'])
    7
    """
    # Determine link list
    if link_list is None:
        link_list = []
        for link in robot_model.link_list:
            if hasattr(link, 'joint') and link.joint is not None:
                if link.joint.joint_type != 'fixed':
                    link_list.append(link)

    # Save original order for later reordering of output
    original_link_list = list(link_list)
    original_link_names = [link.name for link in original_link_list]

    # Topologically sort link_list so that parents always come before children
    # This ensures that in the FK computation, parent transforms are available
    link_set = set(link.name for link in link_list)

    def get_depth(link):
        """Compute depth: number of ancestors in link_set."""
        depth = 0
        current = link.parent
        while current is not None:
            if current.name in link_set:
                depth += 1
            current = current.parent
        return depth

    # Sort by depth (parents first)
    link_list = sorted(link_list, key=get_depth)

    # Compute permutations between sorted and original order
    # sorted_to_original[sorted_idx] = original_idx (for reordering output)
    # original_to_sorted[original_idx] = sorted_idx (for reordering input)
    sorted_link_names = [link.name for link in link_list]
    sorted_to_original = np.array([
        original_link_names.index(name) for name in sorted_link_names
    ], dtype=np.int32)
    original_to_sorted = np.array([
        sorted_link_names.index(name) for name in original_link_names
    ], dtype=np.int32)

    n_joints = len(link_list)

    # Save original joint angles
    original_angles = [link.joint.joint_angle() for link in link_list]

    # Determine reference angles for each joint.
    # Use 0 if valid, otherwise use the closest limit.
    ref_angles = []
    joint_limits_lower = []
    joint_limits_upper = []

    for link in link_list:
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

        # Choose reference angle
        if min_angle <= 0 <= max_angle:
            ref_angle = 0.0
        elif 0 < min_angle:
            ref_angle = min_angle
        else:  # max_angle < 0
            ref_angle = max_angle

        ref_angles.append(ref_angle)

    ref_angles = np.array(ref_angles, dtype=np.float64)

    # Set joints to their reference angles
    for i, link in enumerate(link_list):
        link.joint.joint_angle(ref_angles[i])

    # Get base link world transform
    first_link = link_list[0]
    if first_link.parent is not None:
        base_position = first_link.parent.worldpos().copy().astype(np.float64)
        base_rotation = first_link.parent.worldrot().copy().astype(np.float64)
    else:
        base_position = np.zeros(3, dtype=np.float64)
        base_rotation = np.eye(3, dtype=np.float64)

    # Build link name to index mapping for parent lookup
    link_name_to_idx = {link.name: i for i, link in enumerate(link_list)}

    # Extract transforms and properties for each link
    link_translations = []
    link_rotations = []
    parent_indices = []  # -1 means parent is base, otherwise index in link_list
    joint_axes = []
    joint_types = []
    link_masses = []
    link_coms = []
    link_inertias = []

    for i, link in enumerate(link_list):
        joint = link.joint

        # Find the actual parent link in the kinematic chain
        # Traverse up the parent chain until we find a link in link_list or reach base
        parent_idx = -1  # Default: parent is base
        current = link.parent
        while current is not None:
            if current.name in link_name_to_idx:
                parent_idx = link_name_to_idx[current.name]
                break
            current = current.parent

        parent_indices.append(parent_idx)

        # Get parent coordinates for relative transform computation
        if parent_idx >= 0:
            from_coords = link_list[parent_idx].worldcoords()
        elif first_link.parent is not None:
            from_coords = first_link.parent.worldcoords()
        else:
            from_coords = None

        # Compute relative transform at reference configuration
        link_coords = link.worldcoords()

        if from_coords is not None:
            rel_coords = from_coords.inverse_transformation().transform(link_coords)
            trans = rel_coords.worldpos().copy().astype(np.float64)
            rot = rel_coords.worldrot().copy().astype(np.float64)
        else:
            trans = link_coords.worldpos().copy().astype(np.float64)
            rot = link_coords.worldrot().copy().astype(np.float64)

        link_translations.append(trans)
        link_rotations.append(rot)

        # Joint info
        joint_axes.append(np.array(joint.axis, dtype=np.float64))
        joint_types.append(joint.joint_type)

        # Mass properties
        mass = getattr(link, 'mass', None)
        if mass is None or mass <= 0:
            mass = 0.1  # Default 100g
        link_masses.append(float(mass))

        # Center of mass (centroid in link frame)
        com = getattr(link, 'centroid', None)
        if com is None:
            com = getattr(link, 'center_of_mass', None)
        if com is None:
            com = np.zeros(3)
        link_coms.append(np.array(com, dtype=np.float64))

        # Inertia tensor
        inertia = getattr(link, 'inertia_tensor', None)
        if inertia is None:
            inertia = np.eye(3) * 0.001  # Small default inertia
        link_inertias.append(np.array(inertia, dtype=np.float64))

    # Restore original angles
    for i, link in enumerate(link_list):
        link.joint.joint_angle(original_angles[i])

    # Collect all links with mass if requested
    all_mass_links = []
    all_mass_link_masses = []
    all_mass_link_coms = []
    all_mass_link_inertias = []
    chain_link_names = [link.name for link in link_list]

    if include_all_mass_links:
        for link in robot_model.link_list:
            if hasattr(link, 'mass') and link.mass is not None and link.mass > 0:
                all_mass_links.append(link)
                all_mass_link_masses.append(float(link.mass))

                # Center of mass
                com = getattr(link, 'centroid', None)
                if com is None:
                    com = getattr(link, 'center_of_mass', None)
                if com is None:
                    com = np.zeros(3)
                all_mass_link_coms.append(np.array(com, dtype=np.float64))

                # Inertia tensor
                inertia = getattr(link, 'inertia_tensor', None)
                if inertia is None:
                    inertia = np.eye(3) * 0.001
                all_mass_link_inertias.append(np.array(inertia, dtype=np.float64))

    # Build descendant matrix for kinematic tree structure
    # descendant_matrix[i, j] = 1 if joint j is a descendant of joint i
    # (i.e., joint i is on the path from root to joint j)
    descendant_matrix = np.zeros((n_joints, n_joints), dtype=np.float64)

    # For each joint j, find all ancestor joints
    for j, link_j in enumerate(link_list):
        # Traverse from link_j back to root, marking ancestors
        current = link_j
        while current is not None:
            # Check if current link corresponds to any joint in our list
            for i, link_i in enumerate(link_list):
                if current == link_i:
                    # Joint i is an ancestor of (or equal to) joint j
                    descendant_matrix[i, j] = 1.0
            current = current.parent

    return {
        'n_joints': n_joints,
        'joint_axes': np.array(joint_axes, dtype=np.float64),
        'joint_types': joint_types,
        'link_translations': np.array(link_translations, dtype=np.float64),
        'link_rotations': np.array(link_rotations, dtype=np.float64),
        'parent_indices': np.array(parent_indices, dtype=np.int32),
        'link_masses': np.array(link_masses, dtype=np.float64),
        'link_coms': np.array(link_coms, dtype=np.float64),
        'link_inertias': np.array(link_inertias, dtype=np.float64),
        'joint_limits_lower': np.array(joint_limits_lower, dtype=np.float64),
        'joint_limits_upper': np.array(joint_limits_upper, dtype=np.float64),
        'ref_angles': ref_angles,
        'base_position': base_position,
        'base_rotation': base_rotation,
        'all_mass_links': all_mass_links,
        'all_mass_link_masses': np.array(all_mass_link_masses, dtype=np.float64) if all_mass_link_masses else None,
        'all_mass_link_coms': np.array(all_mass_link_coms, dtype=np.float64) if all_mass_link_coms else None,
        'all_mass_link_inertias': (
            np.array(all_mass_link_inertias, dtype=np.float64)
            if all_mass_link_inertias else None),
        'chain_link_names': chain_link_names,
        'descendant_matrix': descendant_matrix,
        'sorted_to_original': sorted_to_original,
        'original_to_sorted': original_to_sorted,
    }


def build_inverse_dynamics_fn(robot_model, link_list=None, backend=None,
                               include_all_mass_links=True):
    """Build differentiable inverse dynamics function.

    This function creates a differentiable inverse dynamics computation.
    For static case (zero velocities and accelerations), it computes gravity
    torques using an efficient direct formula. For dynamic case, it uses
    the recursive Newton-Euler algorithm.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links defining the kinematic chain. If None, uses all
        movable joints.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.
    include_all_mass_links : bool, optional
        If True (default), includes all links with mass in the robot
        (including end-effector, gripper, etc.) when computing torques.
        This matches the behavior of robot_model.inverse_dynamics().
        If False, only considers links in link_list.

    Returns
    -------
    inverse_dynamics_fn : Callable
        Function with signature:
        inverse_dynamics_fn(q, qd, qdd, gravity, external_forces, external_moments)
        -> tau

        Parameters:
        - q: (n_joints,) joint angles
        - qd: (n_joints,) joint velocities (optional, zeros if None)
        - qdd: (n_joints,) joint accelerations (optional, zeros if None)
        - gravity: (3,) gravity vector (default: [0, 0, -9.81])
        - external_forces: (n_joints, 3) external forces at each link (optional)
        - external_moments: (n_joints, 3) external moments at each link (optional)

        Returns:
        - tau: (n_joints,) joint torques in N*m

    Examples
    --------
    >>> from skrobot.models import Panda
    >>> from skrobot.dynamics import build_inverse_dynamics_fn
    >>> robot = Panda()
    >>> id_fn = build_inverse_dynamics_fn(robot)
    >>> import numpy as np
    >>> q = np.zeros(7)
    >>> tau = id_fn(q)  # Static gravity torques
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    # Determine link list
    if link_list is None:
        link_list = []
        for link in robot_model.link_list:
            if hasattr(link, 'joint') and link.joint is not None:
                if link.joint.joint_type != 'fixed':
                    link_list.append(link)

    # Extract parameters
    params = extract_inverse_dynamics_parameters(
        robot_model, link_list, include_all_mass_links=include_all_mass_links
    )

    # Convert to backend arrays (these become traced constants)
    joint_axes = backend.array(params['joint_axes'])
    link_translations = backend.array(params['link_translations'])
    link_rotations = backend.array(params['link_rotations'])
    parent_indices = params['parent_indices']  # Keep as numpy for indexing
    link_masses = backend.array(params['link_masses'])
    link_coms = backend.array(params['link_coms'])
    link_inertias = backend.array(params['link_inertias'])
    base_position = backend.array(params['base_position'])
    base_rotation = backend.array(params['base_rotation'])
    ref_angles = backend.array(params['ref_angles'])
    descendant_matrix = backend.array(params['descendant_matrix'])
    sorted_to_original = params['sorted_to_original']  # Permutation for output reordering
    original_to_sorted = params['original_to_sorted']  # Permutation for input reordering
    n_joints = params['n_joints']
    joint_types = params['joint_types']
    chain_link_names = params['chain_link_names']

    # Process extra mass links (not in chain but contribute to torques)
    extra_mass_data = []
    if include_all_mass_links and params['all_mass_links']:
        for k, extra_link in enumerate(params['all_mass_links']):
            # Skip if this link is in the chain
            if extra_link.name in chain_link_names:
                continue

            # Find the attachment link (closest ancestor in chain)
            # by traversing parents until we find a chain link
            attachment_idx = None
            current = extra_link
            attachment_link = None

            while current is not None:
                if current.name in chain_link_names:
                    attachment_idx = chain_link_names.index(current.name)
                    attachment_link = current
                    break
                current = current.parent

            if attachment_idx is None:
                # This link is not connected to the chain, skip it
                continue

            # Compute fixed transform from attachment link to extra link
            # at current (reference) configuration
            attach_pos = attachment_link.worldpos()
            attach_rot = attachment_link.worldrot()
            extra_pos = extra_link.worldpos()
            extra_rot = extra_link.worldrot()

            # Relative transform: extra = attach * relative
            # relative_pos = attach_rot.T @ (extra_pos - attach_pos)
            # relative_rot = attach_rot.T @ extra_rot
            relative_pos = attach_rot.T @ (extra_pos - attach_pos)
            relative_rot = attach_rot.T @ extra_rot

            extra_mass = params['all_mass_link_masses'][k]
            extra_com = params['all_mass_link_coms'][k]

            extra_mass_data.append({
                'attachment_idx': attachment_idx,
                'relative_pos': backend.array(relative_pos),
                'relative_rot': backend.array(relative_rot),
                'mass': extra_mass,
                'com': backend.array(extra_com),
            })

    n_extra = len(extra_mass_data)

    # Pre-convert extra mass data to arrays for efficiency
    if n_extra > 0:
        extra_attachment_indices = [d['attachment_idx'] for d in extra_mass_data]
        extra_relative_pos = backend.stack([d['relative_pos'] for d in extra_mass_data])
        extra_relative_rot = backend.stack([d['relative_rot'] for d in extra_mass_data])
        extra_masses = backend.array([d['mass'] for d in extra_mass_data])
        extra_coms = backend.stack([d['com'] for d in extra_mass_data])

    def rodrigues_rotation(axis, angle):
        """Rodrigues' rotation formula: axis-angle -> rotation matrix."""
        return _rodrigues_rotation(backend, axis, angle)

    def compute_fk(q):
        """Compute forward kinematics for all joints.

        Returns positions and rotations of each link frame.
        Position is the origin of the link frame (joint location).
        Rotation is the orientation of the link frame (after joint rotation).

        Supports branching kinematic trees via parent_indices.
        """
        positions = [None] * n_joints
        rotations = [None] * n_joints

        for i in range(n_joints):
            parent_idx = parent_indices[i]

            if parent_idx < 0:
                # Parent is base
                parent_pos = base_position
                parent_rot = base_rotation
            else:
                # Parent is another link in the chain
                parent_pos = positions[parent_idx]
                parent_rot = rotations[parent_idx]

            # Apply static transform from parent to this joint
            pos = parent_pos + parent_rot @ link_translations[i]
            rot = parent_rot @ link_rotations[i]

            # Save position before joint rotation (joint axis passes through here)
            joint_pos = pos

            # Apply joint motion as delta from reference angle
            delta_angle = q[i] - ref_angles[i]

            if joint_types[i] == 'prismatic':
                # Prismatic joint: translate along axis
                axis_world = rot @ joint_axes[i]
                pos = pos + axis_world * delta_angle
            else:
                # Revolute joint: rotate about axis
                joint_rot = rodrigues_rotation(joint_axes[i], delta_angle)
                rot = rot @ joint_rot

            # Store joint position and link rotation
            positions[i] = joint_pos
            rotations[i] = rot

        return positions, rotations

    def compute_fk_with_velocity(q, qd, qdd):
        """Compute FK with link velocities and accelerations.

        Returns positions, rotations, linear velocities, angular velocities,
        linear accelerations, and angular accelerations for each link.
        """
        positions = [None] * n_joints
        rotations = [None] * n_joints
        lin_vels = [None] * n_joints
        ang_vels = [None] * n_joints
        lin_accs = [None] * n_joints
        ang_accs = [None] * n_joints

        for i in range(n_joints):
            parent_idx = parent_indices[i]

            if parent_idx < 0:
                # Parent is base (assumed fixed)
                parent_pos = base_position
                parent_rot = base_rotation
                parent_lin_vel = backend.zeros(3)
                parent_ang_vel = backend.zeros(3)
                parent_lin_acc = backend.zeros(3)
                parent_ang_acc = backend.zeros(3)
            else:
                parent_pos = positions[parent_idx]
                parent_rot = rotations[parent_idx]
                parent_lin_vel = lin_vels[parent_idx]
                parent_ang_vel = ang_vels[parent_idx]
                parent_lin_acc = lin_accs[parent_idx]
                parent_ang_acc = ang_accs[parent_idx]

            # Apply static transform
            pos = parent_pos + parent_rot @ link_translations[i]
            rot = parent_rot @ link_rotations[i]

            # Save joint position
            joint_pos = pos

            # Apply joint motion
            delta_angle = q[i] - ref_angles[i]
            axis_local = joint_axes[i]

            if joint_types[i] == 'prismatic':
                # Prismatic joint
                axis_world = rot @ axis_local
                pos = pos + axis_world * delta_angle

                # Velocity and acceleration for prismatic joint
                lin_vel = parent_lin_vel + backend.cross(parent_ang_vel, pos - parent_pos)
                lin_vel = lin_vel + axis_world * qd[i]

                ang_vel = parent_ang_vel

                lin_acc = parent_lin_acc + backend.cross(parent_ang_acc, pos - parent_pos)
                lin_acc = lin_acc + backend.cross(
                    parent_ang_vel, backend.cross(parent_ang_vel, pos - parent_pos))
                lin_acc = lin_acc + 2 * backend.cross(parent_ang_vel, axis_world * qd[i])
                lin_acc = lin_acc + axis_world * qdd[i]

                ang_acc = parent_ang_acc
            else:
                # Revolute joint
                joint_rot = rodrigues_rotation(axis_local, delta_angle)
                rot = rot @ joint_rot

                # Joint axis in world frame (after static transform, before joint rotation)
                axis_world = (parent_rot @ link_rotations[i]) @ axis_local

                # Velocity for revolute joint
                lin_vel = parent_lin_vel + backend.cross(parent_ang_vel, pos - parent_pos)
                ang_vel = parent_ang_vel + axis_world * qd[i]

                # Acceleration for revolute joint
                lin_acc = parent_lin_acc + backend.cross(parent_ang_acc, pos - parent_pos)
                lin_acc = lin_acc + backend.cross(
                    parent_ang_vel, backend.cross(parent_ang_vel, pos - parent_pos))

                ang_acc = parent_ang_acc + axis_world * qdd[i]
                ang_acc = ang_acc + backend.cross(parent_ang_vel, axis_world * qd[i])

            positions[i] = joint_pos
            rotations[i] = rot
            lin_vels[i] = lin_vel
            ang_vels[i] = ang_vel
            lin_accs[i] = lin_acc
            ang_accs[i] = ang_acc

        return positions, rotations, lin_vels, ang_vels, lin_accs, ang_accs

    def inverse_dynamics_fn(q, qd=None, qdd=None, gravity=None,
                            external_forces=None, external_moments=None,
                            point_forces=None, point_force_link_indices=None,
                            point_force_local_positions=None):
        """Compute inverse dynamics.

        For static case (qd=0, qdd=0), computes gravity torques directly.
        For dynamic case, uses recursive Newton-Euler algorithm.

        Parameters
        ----------
        q : array, shape (n_joints,)
            Joint angles in radians.
        qd : array, shape (n_joints,), optional
            Joint velocities. Default: zeros.
        qdd : array, shape (n_joints,), optional
            Joint accelerations. Default: zeros.
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81].
        external_forces : array, shape (n_joints, 3), optional
            External forces at each link CoM.
        external_moments : array, shape (n_joints, 3), optional
            External moments at each link.
        point_forces : array, shape (n_points, 3), optional
            External forces applied at specific points (for target_coords).
        point_force_link_indices : array, shape (n_points,), optional
            Index of the link each point force is attached to.
        point_force_local_positions : array, shape (n_points, 3), optional
            Local position of each point force in link frame.

        Returns
        -------
        tau : array, shape (n_joints,)
            Joint torques (N*m) or forces (N) for prismatic joints.
        """
        if gravity is None:
            gravity = backend.array([0.0, 0.0, -9.81])

        # Reorder input from original to sorted order
        q_sorted = q[original_to_sorted]

        # Reorder external forces/moments if provided
        external_forces_sorted = None
        if external_forces is not None:
            external_forces_sorted = external_forces[original_to_sorted]
        external_moments_sorted = None
        if external_moments is not None:
            external_moments_sorted = external_moments[original_to_sorted]

        # Remap point force link indices from original to sorted order
        point_force_link_indices_sorted = None
        if point_force_link_indices is not None:
            point_force_link_indices_sorted = original_to_sorted[point_force_link_indices]

        # Check if dynamic case (has velocities or accelerations)
        is_dynamic = (qd is not None and qdd is not None)

        if is_dynamic:
            # Reorder velocities and accelerations
            qd_sorted = qd[original_to_sorted]
            qdd_sorted = qdd[original_to_sorted]
            # Dynamic case: use Newton-Euler with velocities and accelerations
            positions, rotations, lin_vels, ang_vels, lin_accs, ang_accs = \
                compute_fk_with_velocity(q_sorted, qd_sorted, qdd_sorted)
        else:
            # Static case: only need positions and rotations
            positions, rotations = compute_fk(q_sorted)

        # Initialize torques
        tau_list = []

        # Compute torque for each joint
        for i in range(n_joints):
            joint_pos = positions[i]
            rot_i = rotations[i]

            # Joint axis in world frame
            axis_world = rot_i @ joint_axes[i]

            # Sum torque contributions from descendant links of joint i
            # Using descendant_matrix to handle branching kinematic trees
            # Multiply by mask for JAX compatibility (instead of if statements)
            total_torque = 0.0
            is_prismatic_i = joint_types[i] == 'prismatic'

            for j in range(n_joints):
                # Get mask for this link (1 if descendant, 0 otherwise)
                mask_j = descendant_matrix[i, j]

                rot_j = rotations[j]

                # CoM position in world frame
                com_world = positions[j] + rot_j @ link_coms[j]

                # Vector from joint i to CoM of link j
                r = com_world - joint_pos

                # Force on link j
                F_total = link_masses[j] * gravity

                # Add external force if provided (at CoM)
                if external_forces_sorted is not None:
                    F_total = F_total + external_forces_sorted[j]

                if is_dynamic:
                    # Add inertial forces for dynamic case
                    # Linear acceleration at CoM
                    r_com_local = link_coms[j]
                    com_acc = lin_accs[j] + backend.cross(ang_accs[j], rot_j @ r_com_local)
                    com_acc = com_acc + backend.cross(
                        ang_vels[j], backend.cross(ang_vels[j], rot_j @ r_com_local))

                    # Inertial force: F = m * a
                    F_inertial = link_masses[j] * com_acc
                    F_total = F_total - F_inertial  # Subtract because d'Alembert's principle

                # Torque = r × F, projected onto joint axis
                torque_vec = backend.cross(r, F_total)

                if is_prismatic_i:
                    total_torque = total_torque + mask_j * backend.dot(F_total, axis_world)
                else:
                    total_torque = total_torque + mask_j * backend.dot(torque_vec, axis_world)

                # Add inertial moment for dynamic case
                if is_dynamic:
                    # Moment about CoM due to angular motion
                    I_local = link_inertias[j]
                    I_world = rot_j @ I_local @ rot_j.T

                    # Angular momentum rate: I * alpha + omega x (I * omega)
                    M_inertial = I_world @ ang_accs[j]
                    M_inertial = M_inertial + backend.cross(
                        ang_vels[j], I_world @ ang_vels[j])

                    # Add external moment at link j
                    if external_moments_sorted is not None:
                        moment_world = rot_j @ external_moments_sorted[j]
                        M_total = moment_world - M_inertial
                    else:
                        M_total = -M_inertial

                    if not is_prismatic_i:
                        total_torque = total_torque + mask_j * backend.dot(M_total, axis_world)

            # Add contributions from extra mass links
            if n_extra > 0:
                for k in range(n_extra):
                    attach_idx = extra_attachment_indices[k]

                    # Get mask (1 if attachment is a descendant, 0 otherwise)
                    mask_k = descendant_matrix[i, attach_idx]

                    # Compute extra link position in world frame
                    attach_pos = positions[attach_idx]
                    attach_rot = rotations[attach_idx]

                    extra_pos_world = attach_pos + attach_rot @ extra_relative_pos[k]
                    extra_rot_world = attach_rot @ extra_relative_rot[k]

                    # CoM of extra link in world frame
                    extra_com_world = extra_pos_world + extra_rot_world @ extra_coms[k]

                    # Vector from joint i to extra CoM
                    r = extra_com_world - joint_pos

                    # Gravity force
                    F_gravity = extra_masses[k] * gravity

                    # Torque contribution (masked)
                    torque_vec = backend.cross(r, F_gravity)

                    if is_prismatic_i:
                        total_torque = total_torque + mask_k * backend.dot(F_gravity, axis_world)
                    else:
                        total_torque = total_torque + mask_k * backend.dot(torque_vec, axis_world)

            # Add external moment contribution if provided (static case)
            if external_moments_sorted is not None and not is_dynamic:
                for j in range(n_joints):
                    mask_j = descendant_matrix[i, j]
                    rot_j = rotations[j]
                    moment_world = rot_j @ external_moments_sorted[j]
                    total_torque = total_torque + mask_j * backend.dot(moment_world, axis_world)

            # Add point force contributions (for target_coords)
            if point_forces is not None and point_force_link_indices_sorted is not None:
                n_point_forces = len(point_force_link_indices_sorted)
                for pf_idx in range(n_point_forces):
                    pf_link_idx = point_force_link_indices_sorted[pf_idx]

                    # Get mask (1 if the link is a descendant of joint i)
                    mask_pf = descendant_matrix[i, pf_link_idx]

                    # Compute point position in world frame
                    link_pos = positions[pf_link_idx]
                    link_rot = rotations[pf_link_idx]

                    if point_force_local_positions is not None:
                        point_world = link_pos + link_rot @ point_force_local_positions[pf_idx]
                    else:
                        point_world = link_pos

                    # Vector from joint i to force application point
                    r_pf = point_world - joint_pos

                    # Force at this point
                    F_pf = point_forces[pf_idx]

                    # Torque contribution
                    torque_vec_pf = backend.cross(r_pf, F_pf)

                    if is_prismatic_i:
                        total_torque = total_torque + mask_pf * backend.dot(F_pf, axis_world)
                    else:
                        total_torque = total_torque + mask_pf * backend.dot(torque_vec_pf, axis_world)

            tau_list.append(total_torque)

        # Build tau array from list (in sorted order)
        tau_sorted = backend.stack(tau_list)

        # Reorder output from sorted to original order
        tau = tau_sorted[sorted_to_original]

        return tau

    # Return compiled function for performance
    return backend.compile(inverse_dynamics_fn)


def build_torque_vector_fn(robot_model, link_list=None, backend=None,
                            include_all_mass_links=True):
    """Build differentiable torque vector function for static analysis.

    This is a simplified version of inverse dynamics for static cases
    (zero velocities and accelerations), which is commonly used for
    gravity compensation.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include. If None, uses all movable joints.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.
    include_all_mass_links : bool, optional
        If True (default), includes all links with mass in the robot
        (including end-effector, gripper, etc.) when computing torques.
        This matches the behavior of robot_model.inverse_dynamics().
        If False, only considers links in link_list.

    Returns
    -------
    torque_vector_fn : Callable
        Function with signature:
        torque_vector_fn(q, gravity, external_forces, external_moments) -> tau

    Examples
    --------
    >>> from skrobot.models import Panda
    >>> from skrobot.dynamics import build_torque_vector_fn
    >>> robot = Panda()
    >>> torque_fn = build_torque_vector_fn(robot)
    >>> import numpy as np
    >>> q = np.zeros(7)
    >>> tau = torque_fn(q)  # Static gravity torques
    """
    id_fn = build_inverse_dynamics_fn(
        robot_model, link_list, backend,
        include_all_mass_links=include_all_mass_links
    )

    def torque_vector_fn(q, gravity=None, external_forces=None, external_moments=None,
                          point_forces=None, point_force_link_indices=None,
                          point_force_local_positions=None):
        """Compute static torque vector.

        Parameters
        ----------
        q : array, shape (n_joints,)
            Joint angles.
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81].
        external_forces : array, shape (n_joints, 3), optional
            External forces at each link CoM.
        external_moments : array, shape (n_joints, 3), optional
            External moments at each link.
        point_forces : array, shape (n_points, 3), optional
            External forces applied at specific points.
        point_force_link_indices : array, shape (n_points,), optional
            Index of the link each point force is attached to.
        point_force_local_positions : array, shape (n_points, 3), optional
            Local position of each point force in link frame.

        Returns
        -------
        tau : array, shape (n_joints,)
            Joint torques.
        """
        return id_fn(q, qd=None, qdd=None, gravity=gravity,
                     external_forces=external_forces,
                     external_moments=external_moments,
                     point_forces=point_forces,
                     point_force_link_indices=point_force_link_indices,
                     point_force_local_positions=point_force_local_positions)

    return torque_vector_fn


def preprocess_external_forces(robot, force_list=None, moment_list=None,
                                target_coords=None, link_list=None):
    """Convert high-level force specification to low-level arrays.

    This function handles the conversion from user-friendly force/moment
    specification (using coordinates) to the array format required by
    the differentiable inverse dynamics function.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance.
    force_list : list of np.ndarray, optional
        External forces [N] applied at target_coords or link CoMs.
    moment_list : list of np.ndarray, optional
        External moments [Nm] applied at target_coords or link CoMs.
    target_coords : list of Coordinates, optional
        Coordinate frames where forces/moments are applied.
        If None, forces/moments are applied at link CoMs.
    link_list : list of Link, optional
        List of links to consider. If None, uses robot.joint_list.

    Returns
    -------
    external_forces : np.ndarray or None
        Shape (n_joints, 3). Forces at link CoMs.
    external_moments : np.ndarray or None
        Shape (n_joints, 3). Moments at link origins.
    point_forces : np.ndarray or None
        Shape (n_points, 3). Forces at arbitrary points.
    point_force_link_indices : np.ndarray or None
        Shape (n_points,). Link index for each point force.
    point_force_local_positions : np.ndarray or None
        Shape (n_points, 3). Local position in link frame.

    Examples
    --------
    >>> # Forces at link CoMs (no target_coords)
    >>> ext_f, ext_m, pt_f, pt_idx, pt_pos = preprocess_external_forces(
    ...     robot, force_list=[np.array([0, 0, -10])])
    >>>
    >>> # Force at end effector (with target_coords)
    >>> ext_f, ext_m, pt_f, pt_idx, pt_pos = preprocess_external_forces(
    ...     robot,
    ...     force_list=[np.array([10, 0, 0])],
    ...     target_coords=[robot.rarm.end_coords])
    """
    if link_list is None:
        # Build link list from joint list
        link_list = []
        for link in robot.link_list:
            if hasattr(link, 'joint') and link.joint is not None:
                if link.joint.joint_type != 'fixed':
                    link_list.append(link)

    n_joints = len(link_list)

    # Build link name to index mapping
    link_name_to_idx = {}
    for idx, link in enumerate(link_list):
        link_name_to_idx[link.name] = idx

    # Case 1: No target_coords - forces/moments applied at link CoMs
    external_forces = None
    external_moments = None

    if target_coords is None:
        if force_list is not None:
            external_forces = np.zeros((n_joints, 3))
            for i, force in enumerate(force_list):
                if i < n_joints:
                    external_forces[i] = force

        if moment_list is not None:
            external_moments = np.zeros((n_joints, 3))
            for i, moment in enumerate(moment_list):
                if i < n_joints:
                    external_moments[i] = moment

        return external_forces, external_moments, None, None, None

    # Case 2: With target_coords - convert to point forces
    point_forces = None
    point_force_link_indices = None
    point_force_local_positions = None

    if force_list is not None:
        valid_point_forces = []
        valid_link_indices = []
        valid_local_positions = []

        for force, coords in zip(force_list, target_coords):
            # Traverse up parent chain to find a link in link_list
            current = coords.parent if hasattr(coords, 'parent') else None
            found_link = None

            while current is not None:
                if current.name in link_name_to_idx:
                    found_link = current
                    break
                current = current.parent if hasattr(current, 'parent') else None

            if found_link is not None:
                link_idx = link_name_to_idx[found_link.name]

                # Compute local position relative to link frame
                point_world = coords.worldpos()
                link_pos = found_link.worldpos()
                link_rot = found_link.worldrot()
                local_pos = link_rot.T @ (point_world - link_pos)

                valid_point_forces.append(force)
                valid_link_indices.append(link_idx)
                valid_local_positions.append(local_pos)

        if len(valid_point_forces) > 0:
            point_forces = np.array(valid_point_forces)
            point_force_link_indices = np.array(valid_link_indices, dtype=np.int32)
            point_force_local_positions = np.array(valid_local_positions)

    # Handle moments at target_coords (convert to external_moments on parent link)
    if moment_list is not None and target_coords is not None:
        external_moments = np.zeros((n_joints, 3))
        for moment, coords in zip(moment_list, target_coords):
            current = coords.parent if hasattr(coords, 'parent') else None
            while current is not None:
                if current.name in link_name_to_idx:
                    link_idx = link_name_to_idx[current.name]
                    external_moments[link_idx] += moment
                    break
                current = current.parent if hasattr(current, 'parent') else None

    return external_forces, external_moments, point_forces, point_force_link_indices, point_force_local_positions


def preprocess_velocities(av_prev, av, av_next, dt):
    """Compute joint velocities and accelerations from position sequence.

    Uses finite differences to estimate velocities and accelerations
    from a sequence of joint angle vectors.

    Parameters
    ----------
    av_prev : np.ndarray or None
        Previous joint angle vector [rad].
    av : np.ndarray
        Current joint angle vector [rad].
    av_next : np.ndarray or None
        Next joint angle vector [rad].
    dt : float
        Time step between samples [s].

    Returns
    -------
    qd : np.ndarray or None
        Joint velocities [rad/s]. None if cannot be computed.
    qdd : np.ndarray or None
        Joint accelerations [rad/s^2]. None if cannot be computed.

    Notes
    -----
    Velocity computation:
        - If both av_prev and av_next available: central difference
          qd = (av_next - av_prev) / (2 * dt)
        - If only av_prev: backward difference
          qd = (av - av_prev) / dt
        - If only av_next: forward difference
          qd = (av_next - av) / dt

    Acceleration computation:
        - Requires both av_prev and av_next
          qdd = (av_next - 2*av + av_prev) / dt^2

    Examples
    --------
    >>> qd, qdd = preprocess_velocities(av_prev, av, av_next, dt=0.01)
    """
    if av_prev is None and av_next is None:
        return None, None

    qd = None
    qdd = None

    if av_prev is not None and av_next is not None:
        # Central difference for velocity
        qd = (av_next - av_prev) / (2 * dt)
        # Second order central difference for acceleration
        qdd = (av_next - 2 * av + av_prev) / (dt * dt)
    elif av_prev is not None:
        # Backward difference
        qd = (av - av_prev) / dt
        qdd = np.zeros_like(av)  # Cannot compute acceleration
    elif av_next is not None:
        # Forward difference
        qd = (av_next - av) / dt
        qdd = np.zeros_like(av)  # Cannot compute acceleration

    return qd, qdd


def build_optimized_gravity_torque_fn(robot_model, link_list=None,
                                       use_float32=False):
    """Build highly optimized gravity torque function using JAX.

    This function creates an optimized version of gravity torque computation
    specifically for JAX. It uses lax.scan instead of Python loops,
    supports float32 for faster computation, and is highly vectorized.

    This is the fastest option for batched gravity torque computation
    when using JAX.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include. If None, uses all movable joints.
    use_float32 : bool, optional
        If True, use float32 for faster computation. Default: False.

    Returns
    -------
    gravity_torque_fn : Callable
        JIT-compiled function with signature:
        gravity_torque_fn(q, gravity) -> tau

        Parameters:
        - q: (n_joints,) or (batch, n_joints) joint angles
        - gravity: (3,) gravity vector (default: [0, 0, -9.81])

        Returns:
        - tau: (n_joints,) or (batch, n_joints) gravity torques

    Examples
    --------
    >>> from skrobot.models import Kuka
    >>> from skrobot.dynamics import build_optimized_gravity_torque_fn
    >>> robot = Kuka()
    >>> gravity_fn = build_optimized_gravity_torque_fn(robot)
    >>> import jax.numpy as jnp
    >>> q = jnp.zeros(7)
    >>> tau = gravity_fn(q)  # Single configuration
    >>> # Batched computation (very fast with vmap)
    >>> import jax
    >>> q_batch = jnp.zeros((1000, 7))
    >>> tau_batch = jax.vmap(gravity_fn)(q_batch)
    """
    try:
        import jax
        from jax import lax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for build_optimized_gravity_torque_fn. "
            "Install with: pip install jax jaxlib"
        )

    # Extract parameters
    params = extract_inverse_dynamics_parameters(
        robot_model, link_list, include_all_mass_links=True
    )

    dtype = jnp.float32 if use_float32 else jnp.float64

    # Convert to JAX arrays with specified dtype
    n_joints = params['n_joints']
    joint_axes = jnp.array(params['joint_axes'], dtype=dtype)
    link_translations = jnp.array(params['link_translations'], dtype=dtype)
    link_rotations = jnp.array(params['link_rotations'], dtype=dtype)
    parent_indices = jnp.array(params['parent_indices'], dtype=jnp.int32)
    link_masses = jnp.array(params['link_masses'], dtype=dtype)
    link_coms = jnp.array(params['link_coms'], dtype=dtype)
    base_position = jnp.array(params['base_position'], dtype=dtype)
    base_rotation = jnp.array(params['base_rotation'], dtype=dtype)
    ref_angles = jnp.array(params['ref_angles'], dtype=dtype)
    descendant_matrix = jnp.array(params['descendant_matrix'], dtype=dtype)
    sorted_to_original = jnp.array(params['sorted_to_original'], dtype=jnp.int32)
    original_to_sorted = jnp.array(params['original_to_sorted'], dtype=jnp.int32)

    # Check joint types (all revolute for this optimized version)
    joint_types = params['joint_types']
    all_revolute = all(jt == 'revolute' for jt in joint_types)
    if not all_revolute:
        raise ValueError(
            "build_optimized_gravity_torque_fn only supports revolute joints. "
            "Use build_inverse_dynamics_fn for prismatic joints."
        )

    # Process extra mass links
    extra_mass_data = []
    chain_link_names = params['chain_link_names']
    if params['all_mass_links']:
        for k, extra_link in enumerate(params['all_mass_links']):
            if extra_link.name in chain_link_names:
                continue

            attachment_idx = None
            current = extra_link
            attachment_link = None

            while current is not None:
                if current.name in chain_link_names:
                    attachment_idx = chain_link_names.index(current.name)
                    attachment_link = current
                    break
                current = current.parent

            if attachment_idx is None:
                continue

            attach_pos = attachment_link.worldpos()
            attach_rot = attachment_link.worldrot()
            extra_pos = extra_link.worldpos()
            extra_rot = extra_link.worldrot()

            relative_pos = attach_rot.T @ (extra_pos - attach_pos)
            relative_rot = attach_rot.T @ extra_rot

            extra_mass = params['all_mass_link_masses'][k]
            extra_com = params['all_mass_link_coms'][k]

            extra_mass_data.append({
                'attachment_idx': attachment_idx,
                'relative_pos': relative_pos,
                'relative_rot': relative_rot,
                'mass': extra_mass,
                'com': extra_com,
            })

    n_extra = len(extra_mass_data)
    has_extra_mass = n_extra > 0

    if has_extra_mass:
        extra_attachment_indices = jnp.array(
            [d['attachment_idx'] for d in extra_mass_data], dtype=jnp.int32)
        extra_relative_pos = jnp.array(
            [d['relative_pos'] for d in extra_mass_data], dtype=dtype)
        extra_relative_rot = jnp.array(
            [d['relative_rot'] for d in extra_mass_data], dtype=dtype)
        extra_masses = jnp.array(
            [d['mass'] for d in extra_mass_data], dtype=dtype)
        extra_coms = jnp.array(
            [d['com'] for d in extra_mass_data], dtype=dtype)

    def rodrigues_rotation_optimized(axis, angle):
        """Optimized Rodrigues' rotation formula."""
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        c1 = 1.0 - c

        x, y, z = axis[0], axis[1], axis[2]

        return jnp.array([
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1]
        ], dtype=dtype)

    def gravity_torque_fn(q, gravity=None):
        """Compute gravity torques using optimized JAX computation.

        Parameters
        ----------
        q : array, shape (n_joints,)
            Joint angles in radians.
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81].

        Returns
        -------
        tau : array, shape (n_joints,)
            Gravity torques at each joint (N*m).
        """
        if gravity is None:
            gravity = jnp.array([0.0, 0.0, -9.81], dtype=dtype)
        else:
            gravity = jnp.asarray(gravity, dtype=dtype)

        # Reorder input
        q_sorted = q[original_to_sorted]

        # Forward kinematics using lax.scan
        def fk_step(carry, i):
            positions, rotations = carry

            parent_idx = parent_indices[i]

            # Use where for conditional (JAX-friendly)
            parent_pos = lax.cond(
                parent_idx < 0,
                lambda: base_position,
                lambda: positions[parent_idx]
            )
            parent_rot = lax.cond(
                parent_idx < 0,
                lambda: base_rotation,
                lambda: rotations[parent_idx]
            )

            # Apply static transform
            pos = parent_pos + parent_rot @ link_translations[i]
            rot = parent_rot @ link_rotations[i]

            # Apply joint rotation
            delta_angle = q_sorted[i] - ref_angles[i]
            joint_rot = rodrigues_rotation_optimized(joint_axes[i], delta_angle)
            rot = rot @ joint_rot

            # Update storage
            positions = positions.at[i].set(pos)
            rotations = rotations.at[i].set(rot)

            return (positions, rotations), None

        # Initialize storage
        init_positions = jnp.zeros((n_joints, 3), dtype=dtype)
        init_rotations = jnp.zeros((n_joints, 3, 3), dtype=dtype)

        # Run FK
        (positions, rotations), _ = lax.scan(
            fk_step,
            (init_positions, init_rotations),
            jnp.arange(n_joints)
        )

        # Compute CoM positions in world frame for all links
        # com_world[j] = positions[j] + rotations[j] @ link_coms[j]
        com_worlds = positions + jnp.einsum('ijk,ik->ij', rotations, link_coms)

        # Gravity forces for all links
        # F_gravity[j] = mass[j] * gravity
        forces = link_masses[:, None] * gravity[None, :]  # (n_joints, 3)

        # Compute torques for all joints using vectorized operations
        def compute_torque(i):
            joint_pos = positions[i]
            rot_i = rotations[i]
            axis_world = rot_i @ joint_axes[i]

            # Vector from joint i to all CoMs
            r_vectors = com_worlds - joint_pos  # (n_joints, 3)

            # Torque vectors = r × F
            torque_vecs = jnp.cross(r_vectors, forces)  # (n_joints, 3)

            # Project onto axis and mask by descendant matrix
            projections = jnp.sum(torque_vecs * axis_world, axis=1)  # (n_joints,)
            masked_projections = projections * descendant_matrix[i]

            total_torque = jnp.sum(masked_projections)

            # Add extra mass contributions
            if has_extra_mass:
                def add_extra_torque(carry, k):
                    attach_idx = extra_attachment_indices[k]
                    mask_k = descendant_matrix[i, attach_idx]

                    attach_pos = positions[attach_idx]
                    attach_rot = rotations[attach_idx]

                    extra_pos_world = attach_pos + attach_rot @ extra_relative_pos[k]
                    extra_rot_world = attach_rot @ extra_relative_rot[k]
                    extra_com_world = extra_pos_world + extra_rot_world @ extra_coms[k]

                    r = extra_com_world - joint_pos
                    F_gravity = extra_masses[k] * gravity
                    torque_vec = jnp.cross(r, F_gravity)
                    contrib = mask_k * jnp.dot(torque_vec, axis_world)

                    return carry + contrib, None

                extra_torque, _ = lax.scan(
                    add_extra_torque,
                    jnp.array(0.0, dtype=dtype),
                    jnp.arange(n_extra)
                )
                total_torque = total_torque + extra_torque

            return total_torque

        # Compute all torques using vmap
        tau_sorted = jax.vmap(compute_torque)(jnp.arange(n_joints))

        # Reorder output
        tau = tau_sorted[sorted_to_original]

        return tau

    # JIT compile the function
    return jax.jit(gravity_torque_fn)


def build_gravity_torque_fn_vectorized(robot_model, link_list=None,
                                        use_float32=False):
    """Build fully vectorized gravity torque function using JAX.

    This version avoids lax.scan entirely by pre-computing the kinematic
    structure and using matrix operations. It's designed for maximum
    throughput with batched computation.

    Limitations:
    - Only supports serial kinematic chains (no branching)
    - Only supports revolute joints
    - No external forces/moments

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include. If None, uses all movable joints.
    use_float32 : bool, optional
        If True, use float32 for faster computation. Default: False.

    Returns
    -------
    gravity_torque_fn : Callable
        JIT-compiled function with signature:
        gravity_torque_fn(q, gravity) -> tau
    """
    try:
        import jax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for build_gravity_torque_fn_vectorized. "
            "Install with: pip install jax jaxlib"
        )

    # Extract parameters
    params = extract_inverse_dynamics_parameters(
        robot_model, link_list, include_all_mass_links=True
    )

    dtype = jnp.float32 if use_float32 else jnp.float64

    n_joints = params['n_joints']
    joint_axes = jnp.array(params['joint_axes'], dtype=dtype)
    link_translations = jnp.array(params['link_translations'], dtype=dtype)
    link_rotations = jnp.array(params['link_rotations'], dtype=dtype)
    parent_indices = params['parent_indices']
    link_masses = jnp.array(params['link_masses'], dtype=dtype)
    link_coms = jnp.array(params['link_coms'], dtype=dtype)
    base_position = jnp.array(params['base_position'], dtype=dtype)
    base_rotation = jnp.array(params['base_rotation'], dtype=dtype)
    ref_angles = jnp.array(params['ref_angles'], dtype=dtype)
    sorted_to_original = jnp.array(params['sorted_to_original'], dtype=jnp.int32)
    original_to_sorted = jnp.array(params['original_to_sorted'], dtype=jnp.int32)

    # Check for serial chain (all parents are previous link or base)
    is_serial = all(
        parent_indices[i] == i - 1 or parent_indices[i] == -1
        for i in range(n_joints)
    )
    if not is_serial:
        raise ValueError(
            "build_gravity_torque_fn_vectorized only supports serial chains. "
            "Use build_optimized_gravity_torque_fn for branching kinematic trees."
        )

    # Check all revolute
    if not all(jt == 'revolute' for jt in params['joint_types']):
        raise ValueError(
            "build_gravity_torque_fn_vectorized only supports revolute joints."
        )

    # Pre-compute cumulative transforms at reference configuration
    # This allows us to compute FK more efficiently

    def rodrigues_batch(axes, angles):
        """Batch Rodrigues rotation for multiple joints."""
        c = jnp.cos(angles)[:, None, None]
        s = jnp.sin(angles)[:, None, None]
        c1 = 1.0 - c

        # Skew-symmetric matrices from axes
        x = axes[:, 0:1, None]
        y = axes[:, 1:2, None]
        z = axes[:, 2:3, None]

        # Cross product matrix K
        K = jnp.concatenate([
            jnp.concatenate([jnp.zeros_like(x), -z, y], axis=2),
            jnp.concatenate([z, jnp.zeros_like(y), -x], axis=2),
            jnp.concatenate([-y, x, jnp.zeros_like(z)], axis=2),
        ], axis=1)

        # K^2
        K2 = jnp.matmul(K, K)

        # R = I + sin(theta)*K + (1-cos(theta))*K^2
        I = jnp.eye(3, dtype=dtype)[None, :, :]
        R = I + s * K + c1 * K2

        return R

    def gravity_torque_fn(q, gravity=None):
        """Compute gravity torques using fully vectorized JAX computation."""
        if gravity is None:
            gravity = jnp.array([0.0, 0.0, -9.81], dtype=dtype)
        else:
            gravity = jnp.asarray(gravity, dtype=dtype)

        q_sorted = q[original_to_sorted]
        delta_angles = q_sorted - ref_angles

        # Compute joint rotations
        joint_rots = rodrigues_batch(joint_axes, delta_angles)

        # Forward pass to compute cumulative transforms
        # For serial chain:
        #   joint_pos_i = p_{i-1} + R_{i-1} @ static_p_i  (position where joint i is located)
        #   rot_before_joint_i = R_{i-1} @ static_R_i  (rotation before applying joint i)
        #   rot_after_joint_i = rot_before_joint_i @ joint_R_i  (rotation after joint i)
        #
        # The joint axis is expressed in rot_before_joint coordinates

        def fk_serial(carry, i):
            pos, rot = carry
            # Position of joint i (before joint rotation)
            joint_pos = pos + rot @ link_translations[i]
            # Rotation after static transform but before joint rotation
            rot_before = rot @ link_rotations[i]
            # Rotation after joint rotation (used for next iteration and CoM)
            rot_after = rot_before @ joint_rots[i]
            return (joint_pos, rot_after), (joint_pos, rot_before, rot_after)

        _, (joint_positions, rotations_before, rotations_after) = jax.lax.scan(
            fk_serial,
            (base_position, base_rotation),
            jnp.arange(n_joints)
        )

        # Compute CoM positions using rotation after joint
        com_worlds = joint_positions + jnp.einsum('ijk,ik->ij', rotations_after, link_coms)

        # Gravity forces
        forces = link_masses[:, None] * gravity[None, :]

        # Joint axes in world frame (using rotation BEFORE joint rotation)
        axes_world = jnp.einsum('ijk,ik->ij', rotations_before, joint_axes)

        # For serial chain, joint i affects all links j >= i
        # Torque at joint i = sum_{j>=i} (r_{ij} × F_j) · axis_i
        # where r_{ij} = com_j - joint_position_i

        def compute_torque(i):
            joint_pos = joint_positions[i]
            axis_world = axes_world[i]

            # Vectors from joint i to all CoMs (only j >= i matter)
            r_vectors = com_worlds - joint_pos

            # Cross products
            torque_vecs = jnp.cross(r_vectors, forces)

            # Project onto axis
            projections = jnp.sum(torque_vecs * axis_world, axis=1)

            # Mask for j >= i (serial chain)
            mask = jnp.arange(n_joints) >= i

            return jnp.sum(projections * mask)

        tau_sorted = jax.vmap(compute_torque)(jnp.arange(n_joints))
        tau = tau_sorted[sorted_to_original]

        return tau

    return jax.jit(gravity_torque_fn)


def build_rnea_gravity_fn(robot_model, link_list=None, use_float32=False):
    """Build O(n) gravity torque function using RNEA backward pass.

    This function implements the Recursive Newton-Euler Algorithm (RNEA)
    for computing gravity torques. Unlike the O(n²) naive implementation,
    RNEA uses a backward pass to propagate forces from tip to base,
    achieving O(n) complexity.

    This is the fastest JAX implementation for single-call gravity
    torque computation.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include. If None, uses all movable joints.
    use_float32 : bool, optional
        If True, use float32 for faster computation. Default: False.

    Returns
    -------
    gravity_torque_fn : Callable
        JIT-compiled function with signature:
        gravity_torque_fn(q, gravity) -> tau

    Notes
    -----
    Algorithm complexity:
    - Forward pass (FK): O(n)
    - Backward pass (force propagation): O(n)
    - Total: O(n)

    Compared to the naive O(n²) implementation, this is n times faster
    for large kinematic chains.

    Examples
    --------
    >>> from skrobot.models import Kuka
    >>> from skrobot.dynamics import build_rnea_gravity_fn
    >>> robot = Kuka()
    >>> gravity_fn = build_rnea_gravity_fn(robot)
    >>> import jax.numpy as jnp
    >>> q = jnp.zeros(7)
    >>> tau = gravity_fn(q)
    """
    try:
        import jax
        from jax import lax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError(
            "JAX is required for build_rnea_gravity_fn. "
            "Install with: pip install jax jaxlib"
        )

    # Extract parameters
    params = extract_inverse_dynamics_parameters(
        robot_model, link_list, include_all_mass_links=True
    )

    dtype = jnp.float32 if use_float32 else jnp.float64

    n_joints = params['n_joints']
    joint_axes = jnp.array(params['joint_axes'], dtype=dtype)
    link_translations = jnp.array(params['link_translations'], dtype=dtype)
    link_rotations = jnp.array(params['link_rotations'], dtype=dtype)
    parent_indices = jnp.array(params['parent_indices'], dtype=jnp.int32)
    link_masses = jnp.array(params['link_masses'], dtype=dtype)
    link_coms = jnp.array(params['link_coms'], dtype=dtype)
    base_position = jnp.array(params['base_position'], dtype=dtype)
    base_rotation = jnp.array(params['base_rotation'], dtype=dtype)
    ref_angles = jnp.array(params['ref_angles'], dtype=dtype)
    sorted_to_original = jnp.array(params['sorted_to_original'], dtype=jnp.int32)
    original_to_sorted = jnp.array(params['original_to_sorted'], dtype=jnp.int32)

    # Check all revolute
    joint_types = params['joint_types']
    if not all(jt == 'revolute' for jt in joint_types):
        raise ValueError(
            "build_rnea_gravity_fn currently only supports revolute joints."
        )

    # Build child list for each joint (for RNEA backward pass)
    # children[i] = list of indices j where parent_indices[j] == i
    children_list = [[] for _ in range(n_joints)]
    for j in range(n_joints):
        p = parent_indices[j]
        if p >= 0:
            children_list[p].append(j)

    # Convert to padded array for JAX (max children per joint)
    max_children = max(len(c) for c in children_list) if children_list else 1
    max_children = max(max_children, 1)  # At least 1
    children_array = jnp.full((n_joints, max_children), -1, dtype=jnp.int32)
    n_children = jnp.zeros(n_joints, dtype=jnp.int32)
    for i, ch in enumerate(children_list):
        for k, c in enumerate(ch):
            children_array = children_array.at[i, k].set(c)
        n_children = n_children.at[i].set(len(ch))

    # Process extra mass links (attached to chain links)
    chain_link_names = params['chain_link_names']
    extra_mass_per_link = [[] for _ in range(n_joints)]

    if params['all_mass_links']:
        for k, extra_link in enumerate(params['all_mass_links']):
            if extra_link.name in chain_link_names:
                continue

            attachment_idx = None
            current = extra_link
            attachment_link = None

            while current is not None:
                if current.name in chain_link_names:
                    attachment_idx = chain_link_names.index(current.name)
                    attachment_link = current
                    break
                current = current.parent

            if attachment_idx is None:
                continue

            attach_pos = attachment_link.worldpos()
            attach_rot = attachment_link.worldrot()
            extra_pos = extra_link.worldpos()
            extra_rot = extra_link.worldrot()

            relative_pos = attach_rot.T @ (extra_pos - attach_pos)
            relative_rot = attach_rot.T @ extra_rot

            extra_mass = params['all_mass_link_masses'][k]
            extra_com = params['all_mass_link_coms'][k]

            extra_mass_per_link[attachment_idx].append({
                'relative_pos': relative_pos,
                'relative_rot': relative_rot,
                'mass': extra_mass,
                'com': extra_com,
            })

    # Convert extra mass data to arrays
    max_extra = max(len(e) for e in extra_mass_per_link) if extra_mass_per_link else 0
    max_extra = max(max_extra, 1)
    n_extra_per_link = jnp.zeros(n_joints, dtype=jnp.int32)
    extra_rel_pos = jnp.zeros((n_joints, max_extra, 3), dtype=dtype)
    extra_rel_rot = jnp.zeros((n_joints, max_extra, 3, 3), dtype=dtype)
    extra_masses_arr = jnp.zeros((n_joints, max_extra), dtype=dtype)
    extra_coms_arr = jnp.zeros((n_joints, max_extra, 3), dtype=dtype)

    for i, extras in enumerate(extra_mass_per_link):
        n_extra_per_link = n_extra_per_link.at[i].set(len(extras))
        for k, e in enumerate(extras):
            extra_rel_pos = extra_rel_pos.at[i, k].set(jnp.array(e['relative_pos'], dtype=dtype))
            extra_rel_rot = extra_rel_rot.at[i, k].set(jnp.array(e['relative_rot'], dtype=dtype))
            extra_masses_arr = extra_masses_arr.at[i, k].set(e['mass'])
            extra_coms_arr = extra_coms_arr.at[i, k].set(jnp.array(e['com'], dtype=dtype))

    def rodrigues_rotation(axis, angle):
        """Rodrigues' rotation formula."""
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        c1 = 1.0 - c
        x, y, z = axis[0], axis[1], axis[2]
        return jnp.array([
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1]
        ], dtype=dtype)

    def gravity_torque_fn(q, gravity=None):
        """Compute gravity torques using RNEA O(n) algorithm.

        Parameters
        ----------
        q : array, shape (n_joints,)
            Joint angles in radians.
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81].

        Returns
        -------
        tau : array, shape (n_joints,)
            Gravity torques at each joint (N*m).
        """
        if gravity is None:
            gravity = jnp.array([0.0, 0.0, -9.81], dtype=dtype)
        else:
            gravity = jnp.asarray(gravity, dtype=dtype)

        q_sorted = q[original_to_sorted]

        # ===== Forward pass: compute FK =====
        def fk_step(carry, i):
            positions, rotations = carry

            parent_idx = parent_indices[i]
            parent_pos = jnp.where(
                parent_idx < 0,
                base_position,
                positions[parent_idx]
            )
            parent_rot = jnp.where(
                parent_idx < 0,
                base_rotation,
                rotations[parent_idx]
            )

            # Static transform
            pos = parent_pos + parent_rot @ link_translations[i]
            rot = parent_rot @ link_rotations[i]

            # Joint rotation
            delta_angle = q_sorted[i] - ref_angles[i]
            joint_rot = rodrigues_rotation(joint_axes[i], delta_angle)
            rot_after = rot @ joint_rot

            positions = positions.at[i].set(pos)
            rotations = rotations.at[i].set(rot_after)

            return (positions, rotations), (pos, rot, rot_after)

        init_pos = jnp.zeros((n_joints, 3), dtype=dtype)
        init_rot = jnp.zeros((n_joints, 3, 3), dtype=dtype)

        (positions, rotations), (joint_positions, rots_before, rots_after) = lax.scan(
            fk_step,
            (init_pos, init_rot),
            jnp.arange(n_joints)
        )

        # Compute CoM positions in world frame
        com_worlds = joint_positions + jnp.einsum('ijk,ik->ij', rots_after, link_coms)

        # Compute axis directions (using rotation before joint)
        axes_world = jnp.einsum('ijk,ik->ij', rots_before, joint_axes)

        # ===== Backward pass: RNEA force propagation =====
        # Process joints from tip to base (reverse order)
        # For each joint i:
        #   f_i = m_i * g + sum of forces from children
        #   tau_i = (r_com_i × f_gravity_i + sum of (r_child × f_child)) · axis_i

        def backward_step(carry, rev_i):
            # rev_i goes from n_joints-1 to 0
            i = n_joints - 1 - rev_i
            forces, moments, taus = carry

            # Gravity force on this link
            f_gravity = link_masses[i] * gravity

            # CoM position relative to joint
            r_com = com_worlds[i] - joint_positions[i]

            # Moment from this link's gravity
            moment_self = jnp.cross(r_com, f_gravity)

            # Add extra mass contributions for this link
            def add_extra_mass(carry_em, k):
                f_acc, m_acc = carry_em
                valid = k < n_extra_per_link[i]

                # Extra link position
                extra_pos_world = joint_positions[i] + rots_after[i] @ extra_rel_pos[i, k]
                extra_rot_world = rots_after[i] @ extra_rel_rot[i, k]
                extra_com_world = extra_pos_world + extra_rot_world @ extra_coms_arr[i, k]

                # Force and moment
                f_extra = extra_masses_arr[i, k] * gravity
                r_extra = extra_com_world - joint_positions[i]
                m_extra = jnp.cross(r_extra, f_extra)

                # Only add if valid
                f_acc = f_acc + jnp.where(valid, f_extra, jnp.zeros(3, dtype=dtype))
                m_acc = m_acc + jnp.where(valid, m_extra, jnp.zeros(3, dtype=dtype))

                return (f_acc, m_acc), None

            (f_extra_total, m_extra_total), _ = lax.scan(
                add_extra_mass,
                (jnp.zeros(3, dtype=dtype), jnp.zeros(3, dtype=dtype)),
                jnp.arange(max_extra)
            )

            # Sum forces and moments from children
            def sum_children(carry_ch, k):
                f_acc, m_acc = carry_ch
                child_idx = children_array[i, k]
                valid = (k < n_children[i]) & (child_idx >= 0)

                # Force from child
                f_child = forces[child_idx]

                # Moment: child's moment + r_child × f_child
                # r_child = joint_positions[child] - joint_positions[i]
                r_child = joint_positions[child_idx] - joint_positions[i]
                m_child = moments[child_idx] + jnp.cross(r_child, f_child)

                f_acc = f_acc + jnp.where(valid, f_child, jnp.zeros(3, dtype=dtype))
                m_acc = m_acc + jnp.where(valid, m_child, jnp.zeros(3, dtype=dtype))

                return (f_acc, m_acc), None

            (f_children, m_children), _ = lax.scan(
                sum_children,
                (jnp.zeros(3, dtype=dtype), jnp.zeros(3, dtype=dtype)),
                jnp.arange(max_children)
            )

            # Total force at this joint (propagated to parent)
            f_total = f_gravity + f_extra_total + f_children

            # Total moment at this joint
            m_total = moment_self + m_extra_total + m_children

            # Torque at joint i = moment projected onto axis
            tau_i = jnp.dot(m_total, axes_world[i])

            # Update storage
            forces = forces.at[i].set(f_total)
            moments = moments.at[i].set(m_total)
            taus = taus.at[i].set(tau_i)

            return (forces, moments, taus), None

        init_forces = jnp.zeros((n_joints, 3), dtype=dtype)
        init_moments = jnp.zeros((n_joints, 3), dtype=dtype)
        init_taus = jnp.zeros(n_joints, dtype=dtype)

        (forces_final, moments_final, taus_sorted), _ = lax.scan(
            backward_step,
            (init_forces, init_moments, init_taus),
            jnp.arange(n_joints)
        )

        # Reorder output
        tau = taus_sorted[sorted_to_original]

        return tau

    return jax.jit(gravity_torque_fn)


def build_rnea_serial_gravity_fn(robot_model, link_list=None, use_float32=False):
    """Build ultra-fast O(n) gravity torque function for serial chains.

    This implementation avoids dynamic indexing and complex control flow,
    using a simple sequential scan that JAX can optimize very efficiently.

    For serial chains, this is the fastest single-call JAX implementation.

    Parameters
    ----------
    robot_model : RobotModel
        scikit-robot RobotModel instance.
    link_list : list, optional
        List of links to include. If None, uses all movable joints.
    use_float32 : bool, optional
        If True, use float32. Default: False.

    Returns
    -------
    gravity_torque_fn : Callable
        JIT-compiled function: gravity_torque_fn(q, gravity) -> tau
    """
    try:
        import jax
        from jax import lax
        import jax.numpy as jnp
    except ImportError:
        raise ImportError("JAX is required.")

    params = extract_inverse_dynamics_parameters(
        robot_model, link_list, include_all_mass_links=False
    )

    dtype = jnp.float32 if use_float32 else jnp.float64
    n_joints = params['n_joints']

    # Check serial chain
    parent_indices = params['parent_indices']
    is_serial = all(
        parent_indices[i] == i - 1 or parent_indices[i] == -1
        for i in range(n_joints)
    )
    if not is_serial:
        raise ValueError("Only serial chains supported.")

    if not all(jt == 'revolute' for jt in params['joint_types']):
        raise ValueError("Only revolute joints supported.")

    joint_axes = jnp.array(params['joint_axes'], dtype=dtype)
    link_translations = jnp.array(params['link_translations'], dtype=dtype)
    link_rotations = jnp.array(params['link_rotations'], dtype=dtype)
    link_masses = jnp.array(params['link_masses'], dtype=dtype)
    link_coms = jnp.array(params['link_coms'], dtype=dtype)
    base_position = jnp.array(params['base_position'], dtype=dtype)
    base_rotation = jnp.array(params['base_rotation'], dtype=dtype)
    ref_angles = jnp.array(params['ref_angles'], dtype=dtype)
    sorted_to_original = jnp.array(params['sorted_to_original'], dtype=jnp.int32)
    original_to_sorted = jnp.array(params['original_to_sorted'], dtype=jnp.int32)

    def rodrigues(axis, angle):
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        c1 = 1.0 - c
        x, y, z = axis[0], axis[1], axis[2]
        return jnp.array([
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1]
        ], dtype=dtype)

    def gravity_torque_fn(q, gravity=None):
        if gravity is None:
            gravity = jnp.array([0.0, 0.0, -9.81], dtype=dtype)
        else:
            gravity = jnp.asarray(gravity, dtype=dtype)

        q_sorted = q[original_to_sorted]

        # Forward pass: compute FK
        def fk_step(carry, i):
            pos, rot = carry
            pos = pos + rot @ link_translations[i]
            rot = rot @ link_rotations[i]
            delta = q_sorted[i] - ref_angles[i]
            rot_after = rot @ rodrigues(joint_axes[i], delta)
            return (pos, rot_after), (pos, rot, rot_after)

        _, (joint_pos, rot_before, rot_after) = lax.scan(
            fk_step, (base_position, base_rotation), jnp.arange(n_joints)
        )

        # Compute all forces and moments at once
        com_pos = joint_pos + jnp.einsum('ijk,ik->ij', rot_after, link_coms)
        forces = link_masses[:, None] * gravity  # (n, 3)

        # Vector from each joint to its link's CoM
        r_com = com_pos - joint_pos  # (n, 3)

        # Moment from each link's own gravity
        moments_self = jnp.cross(r_com, forces)  # (n, 3)

        # Axes in world frame
        axes_world = jnp.einsum('ijk,ik->ij', rot_before, joint_axes)  # (n, 3)

        # Backward pass: accumulate from tip to base
        # For serial chain: torque[i] = moment_self[i] · axis[i]
        #                            + sum_{j>i} ((r_j - joint_pos[i]) × f_j) · axis[i]
        #
        # Rewritten using cumsum from the back:
        # total_force_from_i_onwards = cumsum(forces, reverse)
        # total_moment_from_i_onwards = cumsum(moments + cross(r, f), reverse)

        # Compute cumulative force from tip to base
        forces_rev = forces[::-1]
        cumsum_forces_rev = jnp.cumsum(forces_rev, axis=0)
        cumsum_forces = cumsum_forces_rev[::-1]  # cumsum_forces[i] = sum of forces[i:]

        # For moment contribution from child links to joint i:
        # sum_{j>i} (joint_pos[j] - joint_pos[i]) × f[j]
        # = sum_{j>i} joint_pos[j] × f[j] - joint_pos[i] × sum_{j>i} f[j]
        #
        # Let's compute: sum_{j>=i} joint_pos[j] × f[j]
        pos_cross_f = jnp.cross(joint_pos, forces)  # (n, 3)
        pos_cross_f_rev = pos_cross_f[::-1]
        cumsum_pos_cross_f_rev = jnp.cumsum(pos_cross_f_rev, axis=0)
        _ = cumsum_pos_cross_f_rev[::-1]  # (n, 3) reserved for future optimization

        # For joint i, the moment from links j > i:
        # sum_{j>i} (joint_pos[j] - joint_pos[i]) × f[j]
        # = (cumsum_pos_cross_f[i+1] if i < n-1 else 0) - joint_pos[i] × (cumsum_forces[i+1] if i < n-1 else 0)
        #
        # But we also need moment from link i itself: r_com[i] × f[i]

        # Simpler approach: compute total moment at each joint
        def compute_torque(i):
            # Moment from this link
            m_self = moments_self[i]

            # Moment from all child links (j > i)
            # sum_{j>i} (com_pos[j] - joint_pos[i]) × f[j]
            # Using cumsum trick is complex, let's just use the direct formula
            # but vectorized over j

            # For child links j in [i+1, n), compute contribution
            # This is still O(n) per joint = O(n²) total
            # Let's try a different approach

            # Total force from children (reserved for future optimization)
            _ = cumsum_forces[i] - forces[i]  # sum of forces[i+1:]

            # We need: sum_{j>i} (joint_pos[j] - joint_pos[i]) × f[j]
            # = sum_{j>i} joint_pos[j] × f[j] - joint_pos[i] × sum_{j>i} f[j]
            # = (cumsum_pos_cross_f - pos_cross_f)[i] from i+1 onwards
            # Actually, let me compute this differently

            # Moment from children about joint i
            # For each child j > i: (com_pos[j] - joint_pos[i]) × f[j]
            # = com_pos[j] × f[j] - joint_pos[i] × f[j]

            com_cross_f = jnp.cross(com_pos, forces)  # (n, 3)

            # Sum from i+1 to n-1
            # Use mask
            mask = jnp.arange(n_joints) > i
            m_children = jnp.sum(
                mask[:, None] * (com_cross_f - jnp.cross(joint_pos[i], forces)),
                axis=0
            )

            m_total = m_self + m_children
            return jnp.dot(m_total, axes_world[i])

        # Still O(n²) due to the sum over children
        # Let's use vmap but that's also O(n²)

        # Actually, let's just use cumsum properly
        # Define: M_i = sum_{j>=i} com_cross_f[j]
        com_cross_f = jnp.cross(com_pos, forces)
        com_cross_f_rev = com_cross_f[::-1]
        cumsum_com_cross_f = jnp.cumsum(com_cross_f_rev, axis=0)[::-1]

        # Also: sum_{j>i} joint_pos[i] × f[j] = joint_pos[i] × (cumsum_forces[i] - forces[i])

        # Torque at joint i:
        # tau[i] = (cumsum_com_cross_f[i] - joint_pos[i] × (cumsum_forces[i])) · axes_world[i]
        # Wait, that's not quite right either...

        # Let me reconsider. The torque at joint i is:
        # tau[i] = sum_{j>=i} (com_pos[j] - joint_pos[i]) × (m[j] * g) · axis[i]
        #        = sum_{j>=i} (com_pos[j] × f[j] - joint_pos[i] × f[j]) · axis[i]
        #        = (sum_{j>=i} com_pos[j] × f[j] - joint_pos[i] × sum_{j>=i} f[j]) · axis[i]
        #        = (cumsum_com_cross_f[i] - joint_pos[i] × cumsum_forces[i]) · axis[i]

        # But we need to include the contribution from the CoM, not joint pos
        # Hmm, let me re-derive this.

        # Actually, the moment arm should be from joint_pos[i] to com_pos[j]:
        # r = com_pos[j] - joint_pos[i]
        # moment = r × f = com_pos[j] × f - joint_pos[i] × f

        # So:
        # tau[i] = sum_{j>=i} (com_pos[j] × f[j] - joint_pos[i] × f[j]) · axis[i]
        #        = (cumsum_com_cross_f[i] - joint_pos[i] × cumsum_forces[i]) · axis[i]

        joint_cross_cumforce = jnp.cross(joint_pos, cumsum_forces)  # (n, 3)
        total_moment = cumsum_com_cross_f - joint_cross_cumforce  # (n, 3)

        # Project onto axes
        taus_sorted = jnp.sum(total_moment * axes_world, axis=1)

        return taus_sorted[sorted_to_original]

    return jax.jit(gravity_torque_fn)
