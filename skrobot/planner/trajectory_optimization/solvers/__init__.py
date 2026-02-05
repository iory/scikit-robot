"""Trajectory optimization solvers.

Available solvers:
- 'jaxls': JAX-based nonlinear least squares (recommended)
- 'gradient_descent': Simple gradient descent (fast but less robust)
- 'augmented_lagrangian': Augmented Lagrangian for hard constraints
- 'scipy': SciPy-based optimization (no JAX dependency)
"""

from skrobot.planner.trajectory_optimization.solvers.base import BaseSolver


def create_solver(solver_type='augmented_lagrangian', **kwargs):
    """Create a trajectory optimization solver.

    Parameters
    ----------
    solver_type : str
        Solver type: 'jaxls', 'gradient_descent', 'augmented_lagrangian',
        or 'scipy'.
    **kwargs
        Solver-specific options.

    Returns
    -------
    BaseSolver
        Solver instance.
    """
    if solver_type == 'jaxls':
        from skrobot.planner.trajectory_optimization.solvers.jaxls_solver import JaxlsSolver
        return JaxlsSolver(**kwargs)
    elif solver_type == 'gradient_descent':
        from skrobot.planner.trajectory_optimization.solvers.gradient_descent import GradientDescentSolver
        return GradientDescentSolver(**kwargs)
    elif solver_type == 'augmented_lagrangian':
        from skrobot.planner.trajectory_optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver
        return AugmentedLagrangianSolver(**kwargs)
    elif solver_type == 'scipy':
        from skrobot.planner.trajectory_optimization.solvers.scipy_solver import ScipySolver
        return ScipySolver(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


__all__ = [
    'BaseSolver',
    'create_solver',
]


# Lazy import for direct access
def __getattr__(name):
    if name == 'AugmentedLagrangianSolver':
        from skrobot.planner.trajectory_optimization.solvers.augmented_lagrangian import AugmentedLagrangianSolver
        return AugmentedLagrangianSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
