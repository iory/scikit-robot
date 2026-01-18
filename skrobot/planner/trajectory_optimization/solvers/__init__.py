"""Trajectory optimization solvers.

Available solvers:
- 'jaxls': JAX-based nonlinear least squares (recommended)
- 'gradient_descent': Simple gradient descent (fast but less robust)
- 'scipy': SciPy-based optimization (no JAX dependency)
"""

from skrobot.planner.trajectory_optimization.solvers.base import BaseSolver


def create_solver(solver_type='jaxls', **kwargs):
    """Create a trajectory optimization solver.

    Parameters
    ----------
    solver_type : str
        Solver type: 'jaxls', 'gradient_descent', or 'scipy'.
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
    elif solver_type == 'scipy':
        from skrobot.planner.trajectory_optimization.solvers.scipy_solver import ScipySolver
        return ScipySolver(**kwargs)
    else:
        raise ValueError(f"Unknown solver type: {solver_type}")


__all__ = [
    'BaseSolver',
    'create_solver',
]
