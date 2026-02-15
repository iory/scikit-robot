"""Dynamics module for scikit-robot.

This module provides dynamics computations including:
- Gravity torque computation
- Potential energy calculation
- Inverse dynamics (Newton-Euler algorithm)
- Differentiable dynamics with backend abstraction

All functions support both NumPy and JAX backends for
automatic differentiation and JIT compilation.
"""

from skrobot.dynamics.differentiable import build_gravity_fn
from skrobot.dynamics.differentiable import build_gravity_fn_with_stiffness
from skrobot.dynamics.differentiable import build_gravity_torque_fn_vectorized
from skrobot.dynamics.differentiable import build_inverse_dynamics_fn
from skrobot.dynamics.differentiable import build_jacobian_gravity_fn
from skrobot.dynamics.differentiable import build_optimized_gravity_torque_fn
from skrobot.dynamics.differentiable import build_rnea_gravity_fn
from skrobot.dynamics.differentiable import build_rnea_serial_gravity_fn
from skrobot.dynamics.differentiable import build_torque_vector_fn
from skrobot.dynamics.differentiable import estimate_external_torque
from skrobot.dynamics.differentiable import extract_dynamics_parameters
from skrobot.dynamics.differentiable import extract_inverse_dynamics_parameters
from skrobot.dynamics.differentiable import preprocess_external_forces
from skrobot.dynamics.differentiable import preprocess_velocities
from skrobot.dynamics.gravity import build_gravity_torque_function
from skrobot.dynamics.gravity import build_potential_energy_function
from skrobot.dynamics.gravity import compute_gravity_torque


__all__ = [
    # From gravity.py
    'compute_gravity_torque',
    'build_gravity_torque_function',
    'build_potential_energy_function',
    # From differentiable.py
    'build_gravity_fn',
    'build_gravity_fn_with_stiffness',
    'build_jacobian_gravity_fn',
    'build_inverse_dynamics_fn',
    'build_torque_vector_fn',
    'estimate_external_torque',
    'extract_dynamics_parameters',
    'extract_inverse_dynamics_parameters',
    # Optimized gravity torque functions
    'build_optimized_gravity_torque_fn',
    'build_gravity_torque_fn_vectorized',
    'build_rnea_gravity_fn',
    'build_rnea_serial_gravity_fn',
    # Preprocessing helpers
    'preprocess_external_forces',
    'preprocess_velocities',
]
