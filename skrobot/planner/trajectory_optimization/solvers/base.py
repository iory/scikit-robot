"""Base solver interface for trajectory optimization."""

from abc import ABC
from abc import abstractmethod

import numpy as np


class SolverResult:
    """Result of trajectory optimization.

    Attributes
    ----------
    trajectory : ndarray
        Optimized trajectory (n_waypoints, n_joints).
    success : bool
        Whether optimization succeeded.
    cost : float
        Final cost value.
    iterations : int
        Number of iterations.
    message : str
        Status message.
    info : dict
        Additional solver-specific information.
    """

    def __init__(
        self,
        trajectory,
        success=True,
        cost=0.0,
        iterations=0,
        message='',
        info=None,
    ):
        self.trajectory = np.asarray(trajectory)
        self.success = success
        self.cost = cost
        self.iterations = iterations
        self.message = message
        self.info = info or {}


class BaseSolver(ABC):
    """Abstract base class for trajectory optimization solvers.

    Subclasses must implement the `solve` method.
    """

    def __init__(self, verbose=False):
        """Initialize solver.

        Parameters
        ----------
        verbose : bool
            Print optimization progress.
        """
        self.verbose = verbose

    @abstractmethod
    def solve(
        self,
        problem,
        initial_trajectory,
        **kwargs,
    ):
        """Solve the trajectory optimization problem.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory guess (n_waypoints, n_joints).
        **kwargs
            Solver-specific options.

        Returns
        -------
        SolverResult
            Optimization result.
        """
        pass

    def _validate_trajectory(self, trajectory, problem):
        """Validate trajectory shape.

        Parameters
        ----------
        trajectory : ndarray
            Trajectory to validate.
        problem : TrajectoryProblem
            Problem definition.

        Raises
        ------
        ValueError
            If trajectory shape is incorrect.
        """
        trajectory = np.asarray(trajectory)
        expected_shape = (problem.n_waypoints, problem.n_joints)

        if trajectory.shape != expected_shape:
            raise ValueError(
                f"Trajectory shape {trajectory.shape} does not match "
                f"expected shape {expected_shape}"
            )

        return trajectory
