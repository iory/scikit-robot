"""Gravity torque computation for robotics.

This module provides functions to compute gravity-induced torques
on robot joints. The gravity torque is the gradient of gravitational
potential energy with respect to joint angles:

    tau = grad(U(q))

where U(q) = sum_i(m_i * g^T * p_i(q)) is the potential energy.

Supports both NumPy and JAX backends for automatic differentiation.
"""

import numpy as np

from skrobot.backend import get_backend
from skrobot.backend import rodrigues_rotation as _rodrigues_rotation


def compute_gravity_torque(
    robot_model,
    joint_angles=None,
    gravity=None,
    link_list=None
):
    """Compute gravity torque using NumPy.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model with mass properties.
    joint_angles : array-like, optional
        Joint angles. If None, uses current angles.
    gravity : array-like, optional
        Gravity vector in world frame. Default: [0, 0, -9.81]
    link_list : list, optional
        Links to consider. If None, uses all movable joints.

    Returns
    -------
    torques : np.ndarray, shape (n_joints,)
        Gravity-induced torque at each joint in N*m.

    Examples
    --------
    >>> from skrobot.dynamics import compute_gravity_torque
    >>> tau = compute_gravity_torque(robot)
    >>> print(f"Gravity torques: {tau}")
    """
    if gravity is None:
        gravity = np.array([0.0, 0.0, -9.81])
    else:
        gravity = np.asarray(gravity)

    # Determine link list
    if link_list is None:
        link_list = []
        for link in robot_model.link_list:
            if hasattr(link, 'joint') and link.joint is not None:
                if link.joint.joint_type != 'fixed':
                    link_list.append(link)

    n_joints = len(link_list)

    # Store current angles and optionally set new ones
    original_angles = [link.joint.joint_angle() for link in link_list]
    if joint_angles is not None:
        for i, link in enumerate(link_list):
            link.joint.joint_angle(float(joint_angles[i]))

    # Compute torques
    torques = np.zeros(n_joints)

    for i in range(n_joints):
        link_i = link_list[i]
        joint_i = link_i.joint

        # Joint position and axis in world frame
        joint_pos = link_i.worldpos()
        joint_axis = link_i.worldrot() @ np.array(joint_i.axis)

        total_torque = 0.0

        # Sum contributions from all links j >= i
        for j in range(i, n_joints):
            link_j = link_list[j]

            # Get mass and COM
            mass = getattr(link_j, 'mass', 0.0) or 0.0
            if mass <= 0:
                continue

            com_local = getattr(link_j, 'center_of_mass', None)
            if com_local is None:
                com_local = np.zeros(3)
            else:
                com_local = np.asarray(com_local)

            # COM in world frame
            com_world = link_j.worldrot() @ com_local + link_j.worldpos()

            # Moment arm from joint i to COM j
            r = com_world - joint_pos

            # Gravity force
            F_gravity = mass * gravity

            # Torque = r x F, projected onto joint axis
            torque_vec = np.cross(r, F_gravity)
            total_torque += np.dot(torque_vec, joint_axis)

        torques[i] = total_torque

    # Restore original angles
    for i, link in enumerate(link_list):
        link.joint.joint_angle(original_angles[i])

    return torques


def build_gravity_torque_function(params, backend=None):
    """Build gravity torque function with autodiff support.

    Parameters
    ----------
    params : dict
        Kinematic parameters from extract_kinematic_parameters().
        Must include link_masses and link_coms.
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.

    Returns
    -------
    gravity_torque_fn : Callable
        (joint_angles, gravity) -> torques
        Pure function compatible with backend's autodiff.

    Examples
    --------
    >>> from skrobot.kinematics import extract_kinematic_parameters
    >>> from skrobot.dynamics import build_gravity_torque_function
    >>> params = extract_kinematic_parameters(robot)
    >>> gravity_fn = build_gravity_torque_function(params)
    >>> q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    >>> g = np.array([0.0, 0.0, -9.81])
    >>> tau = gravity_fn(q, g)
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    # Extract parameters
    joint_axes = backend.array(params['joint_axes'])
    link_translations = backend.array(params['link_translations'])
    link_rotations = backend.array(params['link_rotations'])
    link_masses = backend.array(params['link_masses'])
    link_coms = backend.array(params['link_coms'])
    base_position = backend.array(params['base_position'])
    base_rotation = backend.array(params['base_rotation'])
    n_joints = params['n_joints']

    def rodrigues_rotation(axis, angle):
        """Rodrigues' rotation formula."""
        return _rodrigues_rotation(backend, axis, angle)

    def compute_link_transforms(joint_angles):
        """Compute transformation matrices for all links."""
        positions = []
        rotations = []
        pos = base_position
        rot = base_rotation

        for i in range(n_joints):
            pos = pos + rot @ link_translations[i]
            rot = rot @ link_rotations[i]
            joint_rot = rodrigues_rotation(joint_axes[i], joint_angles[i])
            rot = rot @ joint_rot
            positions.append(pos)
            rotations.append(rot)

        return backend.stack(positions), backend.stack(rotations)

    def gravity_torque_fn(joint_angles, gravity=None):
        """Compute gravity torque at each joint.

        Parameters
        ----------
        joint_angles : array, shape (n_joints,)
            Current joint angles.
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81]

        Returns
        -------
        torques : array, shape (n_joints,)
            Gravity-induced torque at each joint.
        """
        if gravity is None:
            gravity = backend.array([0.0, 0.0, -9.81])

        positions, rotations = compute_link_transforms(joint_angles)

        torques = backend.zeros(n_joints)

        for i in range(n_joints):
            joint_pos = positions[i]
            joint_rot = rotations[i]
            joint_axis_world = joint_rot @ joint_axes[i]

            total_torque = 0.0

            for j in range(i, n_joints):
                # COM in world frame
                com_local = link_coms[j]
                com_world = rotations[j] @ com_local + positions[j]

                # Moment arm
                r = com_world - joint_pos

                # Gravity force
                F_gravity = link_masses[j] * gravity

                # Torque contribution
                torque_vec = backend.cross(r, F_gravity)
                total_torque = total_torque + backend.dot(torque_vec, joint_axis_world)

            # Update torques array
            if hasattr(torques, 'at'):
                # JAX-style immutable array
                torques = torques.at[i].set(total_torque)
            else:
                # NumPy-style mutable array
                torques[i] = total_torque

        return torques

    # Apply JIT if backend supports it
    if backend.supports_jit:
        import jax
        return jax.jit(gravity_torque_fn)
    return gravity_torque_fn


def build_potential_energy_function(params, backend=None):
    """Build potential energy function with autodiff support.

    The gravity torque can be computed as the gradient of potential energy:
        tau = grad(U(q))

    This function returns U(q), which is useful for energy-based analysis.

    Parameters
    ----------
    params : dict
        Kinematic parameters from extract_kinematic_parameters().
    backend : DifferentiableBackend, optional
        Backend to use. If None, uses JAX if available, else NumPy.

    Returns
    -------
    potential_energy_fn : Callable
        (joint_angles, gravity) -> scalar energy
        Pure function compatible with backend's autodiff.

    Examples
    --------
    >>> U_fn = build_potential_energy_function(params)
    >>> U = U_fn(q, gravity)
    >>> # Gravity torque via autodiff:
    >>> from skrobot.backend import get_backend
    >>> backend = get_backend('jax')
    >>> tau = backend.gradient(U_fn)(q, gravity)
    """
    if backend is None:
        try:
            backend = get_backend('jax')
        except (ImportError, KeyError):
            backend = get_backend('numpy')

    joint_axes = backend.array(params['joint_axes'])
    link_translations = backend.array(params['link_translations'])
    link_rotations = backend.array(params['link_rotations'])
    link_masses = backend.array(params['link_masses'])
    link_coms = backend.array(params['link_coms'])
    base_position = backend.array(params['base_position'])
    base_rotation = backend.array(params['base_rotation'])
    n_joints = params['n_joints']

    def rodrigues_rotation(axis, angle):
        """Rodrigues' rotation formula."""
        return _rodrigues_rotation(backend, axis, angle)

    def potential_energy_fn(joint_angles, gravity=None):
        """Compute gravitational potential energy.

        U(q) = sum_i(m_i * g^T * p_i(q))

        Parameters
        ----------
        joint_angles : array, shape (n_joints,)
            Joint angles.
        gravity : array, shape (3,), optional
            Gravity vector. Default: [0, 0, -9.81]

        Returns
        -------
        U : float
            Potential energy in Joules.
        """
        if gravity is None:
            gravity = backend.array([0.0, 0.0, -9.81])

        pos = base_position
        rot = base_rotation
        total_energy = 0.0

        for i in range(n_joints):
            pos = pos + rot @ link_translations[i]
            rot = rot @ link_rotations[i]
            joint_rot = rodrigues_rotation(joint_axes[i], joint_angles[i])
            rot = rot @ joint_rot

            # COM in world frame
            com_world = rot @ link_coms[i] + pos

            # Potential energy contribution: U = -m * g^T * h
            # (negative because gravity points down)
            total_energy = total_energy - link_masses[i] * backend.dot(
                gravity, com_world)

        return total_energy

    # Apply JIT if backend supports it
    if backend.supports_jit:
        import jax
        return jax.jit(potential_energy_fn)
    return potential_energy_fn
