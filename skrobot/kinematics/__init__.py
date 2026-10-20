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

# Mobile-base placement for several end-effector targets
from skrobot.kinematics.base_placement import BasePlacementResult
from skrobot.kinematics.base_placement import find_base_placement
from skrobot.kinematics.base_placement import set_base_pose
from skrobot.kinematics.base_placement import vectorized_target

# Differentiable kinematics (backend-agnostic)
from skrobot.kinematics.differentiable import create_batch_ik_solver
from skrobot.kinematics.differentiable import create_dynamic_limit_mask
from skrobot.kinematics.differentiable import extract_fk_parameters
from skrobot.kinematics.differentiable import forward_kinematics
from skrobot.kinematics.differentiable import forward_kinematics_ee

# Inverse reachability map: base poses that reach a hand pose
from skrobot.kinematics.inverse_reachability_map import BaseCandidates
from skrobot.kinematics.inverse_reachability_map import InverseReachabilityMap

# Loop-closure solving for cut kinematic loops
from skrobot.kinematics.loop_closure import LoopClosureSolver

# Reachability map
from skrobot.kinematics.reachability_map import ReachabilityMap


__all__ = [
    'extract_fk_parameters',
    'forward_kinematics',
    'forward_kinematics_ee',
    'create_batch_ik_solver',
    'create_dynamic_limit_mask',
    'LoopClosureSolver',
    'ReachabilityMap',
    'BasePlacementResult',
    'find_base_placement',
    'set_base_pose',
    'vectorized_target',
    'BaseCandidates',
    'InverseReachabilityMap',
]
