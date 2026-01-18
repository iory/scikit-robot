"""Backend-agnostic trajectory optimization.

This module provides a unified interface for trajectory optimization
that can work with different solvers (jaxls, scipy, gradient descent).

Architecture:
- Residuals: Pure functions computing optimization residuals
- Problem: Describes optimization problem (costs, constraints, variables)
- Solver: Backend-specific solver implementation

Usage:
    from skrobot.planner.trajectory_optimization import (
        TrajectoryProblem,
        create_solver,
    )

    problem = TrajectoryProblem(robot, link_list, ...)
    problem.add_smoothness_cost(weight=1.0)
    problem.add_collision_cost(obstacles, weight=100.0)

    solver = create_solver('jaxls')  # or 'gradient_descent', 'scipy'
    result = solver.solve(problem, initial_trajectory)
"""

from skrobot.planner.trajectory_optimization.problem import TrajectoryProblem
from skrobot.planner.trajectory_optimization.solvers import create_solver


__all__ = [
    'TrajectoryProblem',
    'create_solver',
]
