"""SciPy SLSQP solver for trajectory optimization."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from skrobot.planner.trajectory_optimization.solvers.base import BaseSolver
from skrobot.planner.trajectory_optimization.solvers.base import SolverResult


class ScipySolver(BaseSolver):
    """SciPy SLSQP-based trajectory optimizer.

    This solver uses Sequential Least Squares Programming (SLSQP) to
    optimize trajectories with collision avoidance constraints.

    Supports two collision checking modes:
    1. SDF-based: Pass SweptSphereSdfCollisionChecker to solve()
    2. Sphere-based: Use problem.add_collision_cost() with world_obstacles
    """

    def __init__(
        self,
        max_iterations=100,
        ftol=1e-4,
        safety_margin=0.05,
        verbose=False,
    ):
        """Initialize scipy solver.

        Parameters
        ----------
        max_iterations : int
            Maximum number of SLSQP iterations.
        ftol : float
            Precision goal for the objective function.
        safety_margin : float
            Safety margin for collision checking.
        verbose : bool
            Print optimization progress.
        """
        super().__init__(verbose=verbose)
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.safety_margin = safety_margin

    def solve(
        self,
        problem,
        initial_trajectory,
        collision_checker=None,
        with_base=False,
        weights=None,
        **kwargs,
    ):
        """Solve trajectory optimization using SLSQP.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory (n_waypoints, n_joints).
        collision_checker : SweptSphereSdfCollisionChecker, optional
            Collision checker for SDF-based collision checking.
            If None and problem has collision cost, uses sphere-based collision.
        with_base : bool
            Whether trajectory includes base pose (x, y, theta).
        weights : ndarray, optional
            Joint movement weights. If None, auto-determined.
        **kwargs
            Additional options:
            - joint_list: List of joints for collision checking.

        Returns
        -------
        SolverResult
            Optimization result.
        """
        initial_trajectory = self._validate_trajectory(
            initial_trajectory, problem
        )
        n_wp = problem.n_waypoints
        n_dof = problem.n_joints

        # Determine weights
        if weights is None:
            weights = np.ones(n_dof)
            if with_base and n_dof > 3:
                # Base should be difficult to move
                weights[-3:] = 3.0
        weights = tuple(weights.tolist())

        # Build smoothness cost matrix
        A = _construct_smoothcost_fullmat(n_wp, n_dof, weights)

        def objective(x):
            """Smoothness objective function."""
            f = (0.5 * A.dot(x).dot(x)) / n_wp
            grad = A.dot(x) / n_wp
            return f, grad

        # Terminal constraints (fixed start/end)
        av_start = initial_trajectory[0]
        av_end = initial_trajectory[-1]

        def eq_constraint(x):
            """Equality constraint for fixed endpoints."""
            Q = x.reshape(n_wp, n_dof)
            f = np.hstack((av_start - Q[0], av_end - Q[-1]))
            grad = np.zeros((n_dof * 2, n_dof * n_wp))
            grad[:n_dof, :n_dof] = -np.eye(n_dof)
            grad[-n_dof:, -n_dof:] = -np.eye(n_dof)
            return f, grad

        # Build constraints list
        eq_scipy, eq_jac_scipy = _scipinize(eq_constraint)
        eq_dict = {'type': 'eq', 'fun': eq_scipy, 'jac': eq_jac_scipy}
        constraints = [eq_dict]

        # Collision inequality constraint
        if collision_checker is not None:
            # SDF-based collision checking (original method)
            joint_list_for_collision = kwargs.get(
                'joint_list', problem.joint_list
            )

            def ineq_constraint(x):
                av_seq = x.reshape(n_wp, n_dof)
                sd_vals, sd_val_jac = collision_checker.compute_batch_sd_vals(
                    joint_list_for_collision, av_seq,
                    with_base=with_base, with_jacobian=True
                )
                return sd_vals - self.safety_margin, sd_val_jac

            ineq_scipy, ineq_jac_scipy = _scipinize(ineq_constraint)
            ineq_dict = {'type': 'ineq', 'fun': ineq_scipy, 'jac': ineq_jac_scipy}
            constraints.append(ineq_dict)

        elif problem.collision_spheres is not None and problem.world_obstacles:
            # Sphere-based collision checking using new collision module
            ineq_constraint_fn = self._build_sphere_collision_constraint(
                problem, n_wp, n_dof
            )
            ineq_scipy, ineq_jac_scipy = _scipinize(ineq_constraint_fn)
            ineq_dict = {'type': 'ineq', 'fun': ineq_scipy, 'jac': ineq_jac_scipy}
            constraints.append(ineq_dict)

        obj_scipy, obj_jac_scipy = _scipinize(objective)

        # Joint limit bounds
        bounds = []
        for _ in range(n_wp):
            for j in range(n_dof):
                lower = problem.joint_limits_lower[j]
                upper = problem.joint_limits_upper[j]
                bounds.append((lower, upper))

        # Run optimization
        x_init = initial_trajectory.reshape(-1)
        options = {
            'maxiter': self.max_iterations,
            'ftol': self.ftol,
            'disp': self.verbose,
        }

        result = minimize(
            obj_scipy, x_init,
            method='SLSQP',
            jac=obj_jac_scipy,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        trajectory = result.x.reshape(n_wp, n_dof)

        return SolverResult(
            trajectory=trajectory,
            success=result.success,
            cost=result.fun,
            iterations=result.nit,
            message=result.message if hasattr(result, 'message') else '',
            info={'scipy_result': result},
        )

    def _build_sphere_collision_constraint(self, problem, n_wp, n_dof):
        """Build sphere-based collision constraint function.

        Uses numerical differentiation for Jacobian computation.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition with collision spheres and obstacles.
        n_wp : int
            Number of waypoints.
        n_dof : int
            Number of DOF.

        Returns
        -------
        callable
            Constraint function returning (values, jacobian).
        """
        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_sphere_obstacle_distances
        from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data

        # Parse obstacles
        sphere_obs = [o for o in problem.world_obstacles if o['type'] == 'sphere']
        if not sphere_obs:
            def dummy_constraint(x):
                return np.array([1.0]), np.zeros((1, n_wp * n_dof))
            return dummy_constraint

        obs_centers = np.array([o['center'] for o in sphere_obs])
        obs_radii = np.array([o['radius'] for o in sphere_obs])

        # Build FK functions using shared module
        fk_data = prepare_fk_data(problem, np)
        sphere_radii = fk_data['sphere_radii']
        n_joints = fk_data['n_joints']
        _, get_sphere_positions = build_fk_functions(fk_data, np)

        safety_margin = self.safety_margin

        def compute_min_distances(x):
            """Compute minimum signed distances for each waypoint."""
            Q = x.reshape(n_wp, n_dof)
            min_dists = []

            for t in range(n_wp):
                angles = Q[t, :n_joints]
                sphere_pos = get_sphere_positions(angles)

                # Compute distances to all obstacles
                signed_dists = compute_sphere_obstacle_distances(
                    sphere_pos, sphere_radii, obs_centers, obs_radii, np
                )
                min_dist = np.min(signed_dists)
                min_dists.append(min_dist)

            return np.array(min_dists)

        def ineq_constraint(x):
            """Inequality constraint: signed_distance - safety_margin >= 0."""
            min_dists = compute_min_distances(x)

            # Numerical Jacobian
            eps = 1e-6
            n_constraints = n_wp
            jac = np.zeros((n_constraints, n_wp * n_dof))

            for i in range(n_wp * n_dof):
                x_plus = x.copy()
                x_plus[i] += eps
                dists_plus = compute_min_distances(x_plus)
                jac[:, i] = (dists_plus - min_dists) / eps

            return min_dists - safety_margin, jac

        return ineq_constraint


def _scipinize(fun):
    """Convert function returning (f, jac) to scipy format.

    Parameters
    ----------
    fun : callable
        Function returning (value, jacobian).

    Returns
    -------
    f_scipy : callable
        Function returning value.
    jac_scipy : callable
        Function returning jacobian.
    """
    cache = {}

    def compute(x):
        key = tuple(x.tolist())
        if key not in cache:
            cache[key] = fun(x)
            # Keep cache small
            if len(cache) > 100:
                oldest = next(iter(cache))
                del cache[oldest]
        return cache[key]

    def f_scipy(x):
        return compute(x)[0]

    def jac_scipy(x):
        return compute(x)[1]

    return f_scipy, jac_scipy


@lru_cache(maxsize=100)
def _construct_smoothcost_fullmat(n_wp, n_dof, weights):
    """Compute smoothness cost matrix.

    This implements eq. (17) of the IJRR CHOMP paper.
    Uses squared acceleration cost.

    Parameters
    ----------
    n_wp : int
        Number of waypoints.
    n_dof : int
        Number of DOFs.
    weights : tuple
        Per-joint weights.

    Returns
    -------
    ndarray
        Smoothness cost matrix (n_wp * n_dof, n_wp * n_dof).
    """
    # Acceleration cost block
    acc_block = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])

    A_ = np.zeros((n_wp, n_wp))
    for i in range(1, n_wp - 1):
        A_[i - 1:i + 2, i - 1:i + 2] += acc_block

    w_mat = np.diag(weights)
    A = np.kron(A_, w_mat ** 2)
    return A
