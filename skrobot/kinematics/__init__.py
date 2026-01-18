"""Kinematics module for scikit-robot.

This module provides kinematics algorithms including:
- Forward kinematics (FK)
- Inverse kinematics (IK)
- Jacobian computation
- Differentiable kinematics with backend abstraction
- Reachability map computation

The algorithms support multiple backends (NumPy, JAX) for
different performance and autodiff requirements.
"""

# Differentiable kinematics (backend-agnostic)
from skrobot.kinematics.differentiable import create_batch_ik_solver
from skrobot.kinematics.differentiable import extract_fk_parameters
from skrobot.kinematics.differentiable import forward_kinematics
from skrobot.kinematics.differentiable import forward_kinematics_ee

# Reachability map
from skrobot.kinematics.reachability_map import ReachabilityMap


__all__ = [
    'extract_fk_parameters',
    'forward_kinematics',
    'forward_kinematics_ee',
    'create_batch_ik_solver',
    'ReachabilityMap',
]
