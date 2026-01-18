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

    This is the backend-agnostic version of the original sqp_plan_trajectory.
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
            If None, uses sphere-based collision from problem.
        with_base : bool
            Whether trajectory includes base pose (x, y, theta).
        weights : ndarray, optional
            Joint movement weights. If None, auto-determined.
        **kwargs
            Additional options.

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

        # Collision inequality constraint
        if collision_checker is not None:
            # Use SDF-based collision checker (original method)
            # Use the joint_list from kwargs if provided, otherwise from problem
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
        else:
            # Use sphere-based collision from problem
            ineq_constraint = self._build_sphere_collision_constraint(problem)

        # Convert to scipy format
        eq_scipy, eq_jac_scipy = _scipinize(eq_constraint)
        eq_dict = {'type': 'eq', 'fun': eq_scipy, 'jac': eq_jac_scipy}

        if ineq_constraint is not None:
            ineq_scipy, ineq_jac_scipy = _scipinize(ineq_constraint)
            ineq_dict = {'type': 'ineq', 'fun': ineq_scipy, 'jac': ineq_jac_scipy}
            constraints = [eq_dict, ineq_dict]
        else:
            constraints = [eq_dict]

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

    def _build_sphere_collision_constraint(self, problem):
        """Build collision constraint using sphere approximations.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition with collision spheres.

        Returns
        -------
        callable or None
            Constraint function returning (values, jacobian).
        """
        if not problem.world_obstacles and len(problem.self_collision_pairs) == 0:
            return None

        if problem.collision_spheres is None:
            return None

        # Get FK parameters and collision info
        fk_params = problem.fk_params
        spheres = problem.collision_spheres
        sphere_radii = spheres['radii']
        sphere_offsets = spheres['offsets']
        link_indices = problem.collision_link_to_chain_idx
        offset_pos = problem.collision_link_offsets_pos
        offset_rot = problem.collision_link_offsets_rot

        n_wp = problem.n_waypoints
        n_dof = problem.n_joints
        n_spheres = len(sphere_radii)

        def ineq_constraint(x):
            """Sphere-based collision constraint."""
            av_seq = x.reshape(n_wp, n_dof)
            all_residuals = []

            # Compute FK for each waypoint
            for t in range(n_wp):
                q = av_seq[t]

                # Compute link positions via FK
                link_positions, link_rotations = _compute_fk_chain(
                    q, fk_params
                )

                # Compute collision link positions with offsets
                coll_link_positions = np.zeros((len(link_indices), 3))
                for i, (idx, off_pos, off_rot) in enumerate(
                    zip(link_indices, offset_pos, offset_rot)
                ):
                    parent_pos = link_positions[idx]
                    parent_rot = link_rotations[idx]
                    coll_link_positions[i] = parent_pos + parent_rot @ off_pos

                # Compute sphere positions
                sphere_positions = np.zeros((n_spheres, 3))
                sphere_link_indices = spheres['link_indices']
                for s in range(n_spheres):
                    link_idx = sphere_link_indices[s]
                    sphere_positions[s] = (
                        coll_link_positions[link_idx] + sphere_offsets[s]
                    )

                # World collision
                for obs in problem.world_obstacles:
                    obs_center = np.array(obs['center'])
                    obs_radius = obs['radius']
                    activation_dist = self.safety_margin

                    for s in range(n_spheres):
                        dist = np.linalg.norm(
                            sphere_positions[s] - obs_center
                        )
                        signed_dist = dist - sphere_radii[s] - obs_radius
                        all_residuals.append(signed_dist - activation_dist)

                # Self collision
                if len(problem.self_collision_pairs) > 0:
                    pairs_i, pairs_j = problem.self_collision_pairs
                    for i, j in zip(pairs_i, pairs_j):
                        dist = np.linalg.norm(
                            sphere_positions[i] - sphere_positions[j]
                        )
                        signed_dist = dist - sphere_radii[i] - sphere_radii[j]
                        all_residuals.append(signed_dist - 0.02)

            # Return residuals (>= 0 means no collision)
            residuals = np.array(all_residuals)
            # Numerical jacobian (for simplicity)
            jac = _numerical_jacobian(
                lambda x: ineq_constraint(x)[0],
                x, residuals.shape[0]
            )
            return residuals, jac

        return ineq_constraint


def _compute_fk_chain(q, fk_params):
    """Compute forward kinematics for kinematic chain.

    Parameters
    ----------
    q : ndarray
        Joint angles (n_joints,).
    fk_params : dict
        FK parameters from extract_fk_parameters.

    Returns
    -------
    positions : ndarray
        Link positions (n_links, 3).
    rotations : ndarray
        Link rotations (n_links, 3, 3).
    """
    n_joints = len(q)
    positions = np.zeros((n_joints, 3))
    rotations = np.zeros((n_joints, 3, 3))

    axes = fk_params.get('axes', np.array([[0, 0, 1]] * n_joints))
    parent_transforms = fk_params.get('parent_transforms', None)

    # Simple chain FK
    current_pos = np.zeros(3)
    current_rot = np.eye(3)

    for i in range(n_joints):
        # Apply joint rotation
        angle = q[i]
        axis = axes[i] if i < len(axes) else np.array([0, 0, 1])
        joint_rot = _axis_angle_to_rotation(axis, angle)

        if parent_transforms is not None and i < len(parent_transforms):
            # Apply parent transform
            parent_pos = parent_transforms[i][:3, 3]
            parent_rot = parent_transforms[i][:3, :3]
            current_pos = current_pos + current_rot @ parent_pos
            current_rot = current_rot @ parent_rot

        current_rot = current_rot @ joint_rot
        positions[i] = current_pos
        rotations[i] = current_rot

    return positions, rotations


def _axis_angle_to_rotation(axis, angle):
    """Convert axis-angle to rotation matrix."""
    axis = np.asarray(axis)
    axis = axis / (np.linalg.norm(axis) + 1e-10)
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1 - c
    x, y, z = axis

    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def _numerical_jacobian(f, x, m, eps=1e-7):
    """Compute numerical Jacobian.

    Parameters
    ----------
    f : callable
        Function returning (m,) array.
    x : ndarray
        Point to evaluate at (n,).
    m : int
        Output dimension.
    eps : float
        Finite difference step.

    Returns
    -------
    ndarray
        Jacobian (m, n).
    """
    n = len(x)
    jac = np.zeros((m, n))
    f0 = f(x)
    for i in range(n):
        x_plus = x.copy()
        x_plus[i] += eps
        f_plus = f(x_plus)
        jac[:, i] = (f_plus - f0) / eps
    return jac


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
