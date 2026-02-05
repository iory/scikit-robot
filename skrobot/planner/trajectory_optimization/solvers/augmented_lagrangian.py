"""Augmented Lagrangian solver for trajectory optimization.

This solver handles hard constraints (collision, joint limits) using
the Augmented Lagrangian method, which provides better constraint
satisfaction than pure penalty methods.

The Augmented Lagrangian for inequality constraints g(x) >= 0 is:
    L(x, λ, ρ) = f(x) - Σ λ_i * g_i(x) + (ρ/2) * Σ max(0, -g_i(x))²

where:
    f(x) = objective (soft costs like smoothness, pose tracking)
    g(x) = constraint functions (collision distance, joint limits)
    λ = Lagrange multipliers
    ρ = penalty parameter

The algorithm alternates between:
    1. Inner loop: minimize L(x, λ, ρ) w.r.t. x using gradient descent
    2. Outer loop: update λ and increase ρ if needed
"""

import os
import platform


# Ensure CPU backend on Mac before JAX imports
if platform.system() == 'Darwin':
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np

from skrobot.planner.trajectory_optimization.solvers.base import BaseSolver
from skrobot.planner.trajectory_optimization.solvers.base import SolverResult


class AugmentedLagrangianSolver(BaseSolver):
    """Augmented Lagrangian solver for trajectory optimization.

    This solver separates soft costs (objective) from hard constraints
    (collision, joint limits) and uses the AL method to ensure constraint
    satisfaction while minimizing the objective.

    Parameters
    ----------
    max_outer_iterations : int
        Maximum outer loop iterations (multiplier updates).
    max_inner_iterations : int
        Maximum inner loop iterations (gradient descent per outer iter).
    learning_rate : float
        Gradient descent step size for inner loop.
    initial_penalty : float
        Initial penalty parameter ρ.
    penalty_multiplier : float
        Factor to increase ρ when constraints not satisfied.
    max_penalty : float
        Maximum penalty parameter.
    constraint_tolerance : float
        Tolerance for constraint satisfaction.
    verbose : bool
        Print optimization progress.

    Examples
    --------
    >>> from skrobot.planner.trajectory_optimization import TrajectoryProblem
    >>> from skrobot.planner.trajectory_optimization.solvers import (
    ...     AugmentedLagrangianSolver)
    >>> problem = TrajectoryProblem(robot, link_list, n_waypoints=20)
    >>> problem.add_smoothness_cost(weight=1.0)
    >>> problem.add_collision_cost(collision_links, obstacles, weight=100.0)
    >>> solver = AugmentedLagrangianSolver()
    >>> result = solver.solve(problem, initial_trajectory)
    """

    def __init__(
        self,
        max_outer_iterations=20,
        max_inner_iterations=100,
        max_iterations=None,
        learning_rate=0.01,
        initial_penalty=1.0,
        penalty_multiplier=2.0,
        max_penalty=1e6,
        constraint_tolerance=1e-4,
        verbose=False,
    ):
        """Initialize Augmented Lagrangian solver."""
        super().__init__(verbose=verbose)
        self.max_outer_iterations = max_outer_iterations
        # max_iterations is an alias for max_inner_iterations (for compatibility)
        if max_iterations is not None:
            self.max_inner_iterations = max_iterations
        else:
            self.max_inner_iterations = max_inner_iterations
        self.learning_rate = learning_rate
        self.initial_penalty = initial_penalty
        self.penalty_multiplier = penalty_multiplier
        self.max_penalty = max_penalty
        self.constraint_tolerance = constraint_tolerance
        self._jit_cache = {}

    @property
    def max_iterations(self):
        """Alias for max_inner_iterations (for compatibility)."""
        return self.max_inner_iterations

    @max_iterations.setter
    def max_iterations(self, value):
        """Set max_inner_iterations (for compatibility)."""
        self.max_inner_iterations = value

    def _get_problem_structure_key(self, problem):
        """Generate cache key from problem structure."""
        from skrobot.planner.trajectory_optimization.solvers.solver_utils import get_problem_structure_key
        return get_problem_structure_key(problem)

    def _get_problem_value_hash(self, problem):
        """Generate hash of problem values."""
        from skrobot.planner.trajectory_optimization.solvers.solver_utils import get_problem_value_hash
        return get_problem_value_hash(problem)

    def solve(
        self,
        problem,
        initial_trajectory,
        **kwargs,
    ):
        """Solve trajectory optimization using Augmented Lagrangian method.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory (n_waypoints, n_joints).
        **kwargs
            Additional options:
            - max_outer_iterations: Override default outer iterations
            - max_inner_iterations: Override default inner iterations
            - learning_rate: Override default learning rate

        Returns
        -------
        SolverResult
            Optimization result with additional info:
            - 'constraint_violation': Final max constraint violation
            - 'outer_iterations': Number of outer iterations used
            - 'final_penalty': Final penalty parameter value
        """
        import jax
        import jax.numpy as jnp

        initial_trajectory = self._validate_trajectory(
            initial_trajectory, problem
        )

        max_outer = kwargs.get('max_outer_iterations', self.max_outer_iterations)
        max_inner = kwargs.get('max_inner_iterations', self.max_inner_iterations)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)

        # Build objective and constraint functions using two-level caching:
        # 1. Structure key for cache slot identity
        # 2. Value hash to detect when functions need rebuilding
        structure_key = self._get_problem_structure_key(problem)
        value_hash = self._get_problem_value_hash(problem)

        need_rebuild = False
        if structure_key not in self._jit_cache:
            need_rebuild = True
        elif self._jit_cache[structure_key].get('value_hash') != value_hash:
            need_rebuild = True

        if need_rebuild:
            self._jit_cache[structure_key] = {
                'value_hash': value_hash,
                'functions': self._build_functions(problem),
            }

        objective_fn, constraint_fn, n_constraints = \
            self._jit_cache[structure_key]['functions']

        # Initialize (use float32 for consistency with JAX JIT)
        trajectory = jnp.array(initial_trajectory, dtype=jnp.float32)
        lower = jnp.array(problem.joint_limits_lower, dtype=jnp.float32)
        upper = jnp.array(problem.joint_limits_upper, dtype=jnp.float32)

        # Initialize Lagrange multipliers (for inequality: λ >= 0)
        # For g(x) >= 0 constraints, we have n_constraints multipliers
        lambdas = jnp.zeros(n_constraints, dtype=jnp.float32)
        rho = jnp.float32(self.initial_penalty)

        # Build augmented Lagrangian and its gradient
        def augmented_lagrangian(traj, lam, penalty):
            obj = objective_fn(traj)
            g = constraint_fn(traj)  # g >= 0 is satisfied

            # For inequality g >= 0:
            # AL term = (ρ/2) * ||max(0, λ/ρ - g)||²
            # This is equivalent to the standard formulation
            violation = jnp.maximum(0.0, lam / penalty - g)
            al_penalty = (penalty / 2.0) * jnp.sum(violation ** 2)

            return obj + al_penalty

        al_grad = jax.grad(augmented_lagrangian)

        # Adam optimizer parameters
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        # Collect waypoint constraints as JAX arrays (float32 for consistency)
        wp_constraints = [
            (idx, jnp.array(angles, dtype=jnp.float32))
            for idx, angles in problem.waypoint_constraints
        ]

        # JIT compile inner loop with Adam optimizer
        @jax.jit
        def inner_loop(traj, lam, penalty, lr, n_iters):
            traj_shape = traj.shape

            def body_fn(i, state):
                t, m, v, best_t, best_cost = state
                grad = al_grad(t, lam, penalty)

                # Gradient clipping
                grad_norm = jnp.sqrt(jnp.sum(grad ** 2) + 1e-10)
                grad = jnp.where(
                    grad_norm > 100.0,
                    grad * (100.0 / grad_norm),
                    grad
                )

                # Adam update
                m_new = beta1 * m + (1 - beta1) * grad
                v_new = beta2 * v + (1 - beta2) * (grad ** 2)

                # Bias correction
                step = i + 1
                m_hat = m_new / (1 - beta1 ** step)
                v_hat = v_new / (1 - beta2 ** step)

                # Update trajectory
                new_t = t - lr * m_hat / (jnp.sqrt(v_hat) + eps)

                # Clip to joint limits
                new_t = jnp.clip(new_t, lower, upper)

                # Fix endpoints
                if problem.fixed_start:
                    new_t = new_t.at[0].set(traj[0])
                if problem.fixed_end:
                    new_t = new_t.at[-1].set(traj[-1])

                # Fix intermediate waypoints
                for wp_idx, wp_angles in wp_constraints:
                    new_t = new_t.at[wp_idx].set(wp_angles)

                # Track best
                new_cost = augmented_lagrangian(new_t, lam, penalty)
                is_better = new_cost < best_cost
                new_best_t = jnp.where(is_better, new_t, best_t)
                new_best_cost = jnp.where(is_better, new_cost, best_cost)

                return (new_t, m_new, v_new, new_best_t, new_best_cost)

            init_cost = augmented_lagrangian(traj, lam, penalty)
            m_init = jnp.zeros(traj_shape, dtype=jnp.float32)
            v_init = jnp.zeros(traj_shape, dtype=jnp.float32)
            _, _, _, best_traj, best_cost = jax.lax.fori_loop(
                0, n_iters, body_fn,
                (traj, m_init, v_init, traj, init_cost)
            )
            return best_traj, best_cost

        # Outer loop: update multipliers
        total_inner_iters = 0
        prev_max_violation = float('inf')

        # Convert learning_rate to float32 for consistency
        lr_f32 = jnp.float32(learning_rate)

        for outer_iter in range(max_outer):
            # Inner loop: minimize AL
            trajectory, al_cost = inner_loop(
                trajectory, lambdas, rho, lr_f32, max_inner
            )
            total_inner_iters += max_inner

            # Compute constraint violations
            g = constraint_fn(trajectory)
            violations = jnp.maximum(0.0, -g)  # violation = max(0, -g)
            max_violation = float(jnp.max(violations))

            if self.verbose:
                obj_val = float(objective_fn(trajectory))
                print(f"Outer iter {outer_iter + 1}: "
                      f"obj={obj_val:.6f}, "
                      f"max_violation={max_violation:.6f}, "
                      f"ρ={rho:.1f}")

            # Check convergence
            if max_violation < self.constraint_tolerance:
                if self.verbose:
                    print("Converged: constraints satisfied")
                break

            # Update multipliers: λ = max(0, λ - ρ * g)
            # For g >= 0 constraints
            lambdas = jnp.maximum(jnp.float32(0.0), lambdas - rho * g)

            # Increase penalty if insufficient progress
            if max_violation > 0.25 * prev_max_violation:
                rho = jnp.float32(min(float(rho) * self.penalty_multiplier, self.max_penalty))

            prev_max_violation = max_violation

        # Final evaluation
        final_cost = float(objective_fn(trajectory))
        g_final = constraint_fn(trajectory)
        final_violation = float(jnp.max(jnp.maximum(0.0, -g_final)))

        success = final_violation < self.constraint_tolerance

        return SolverResult(
            trajectory=np.array(trajectory),
            success=success,
            cost=final_cost,
            iterations=total_inner_iters,
            message=f'AL completed: violation={final_violation:.2e}',
            info={
                'constraint_violation': final_violation,
                'outer_iterations': outer_iter + 1,
                'final_penalty': rho,
            }
        )

    def _build_functions(self, problem):
        """Build objective and constraint functions.

        Separates residuals into:
        - Soft costs (objective): smoothness, acceleration, pose tracking, etc.
        - Hard constraints: collision >= 0, joint limits

        Returns
        -------
        tuple
            (objective_fn, constraint_fn, n_constraints)
        """
        import jax
        import jax.numpy as jnp

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals
        from skrobot.planner.trajectory_optimization.fk_utils import compute_self_collision_distances
        from skrobot.planner.trajectory_optimization.fk_utils import compute_sphere_obstacle_distances
        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log
        from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data

        dt = problem.dt

        # Check for collision/FK needs
        has_collision = problem.collision_spheres is not None
        has_cartesian = any(s.name == 'cartesian_path' for s in problem.residuals)
        has_ee_waypoints = len(problem.ee_waypoint_costs) > 0

        get_sphere_positions = None
        get_ee_pose = None
        sphere_radii = None

        if has_collision or has_cartesian or has_ee_waypoints:
            fk_data = prepare_fk_data(problem, jnp)
            if has_collision:
                sphere_radii = fk_data['sphere_radii']
            _, get_sphere_positions_fn, _, get_ee_pose_fn = \
                build_fk_functions(fk_data, jnp)
            get_sphere_positions = get_sphere_positions_fn
            get_ee_pose = get_ee_pose_fn

        # Separate soft costs and hard constraints based on 'kind' attribute
        # kind='soft' -> soft costs (in objective)
        # kind='geq' -> hard inequality constraints g(x) >= 0
        soft_costs = []
        hard_constraints = []

        for spec in problem.residuals:
            if spec.kind == 'geq':
                hard_constraints.append(spec)
            else:
                soft_costs.append(spec)

        # Count constraints
        n_constraints = 0
        constraint_specs = []

        for spec in hard_constraints:
            if spec.name == 'world_collision' and has_collision:
                n_spheres = len(sphere_radii)
                n_obstacles = len(spec.params['obstacles'])
                # One constraint per sphere-obstacle pair per waypoint
                n_waypoints = problem.n_waypoints
                n_coll = n_spheres * n_obstacles * n_waypoints
                constraint_specs.append(('world_collision', spec, n_coll))
                n_constraints += n_coll

            elif spec.name == 'self_collision' and has_collision:
                pairs_i, pairs_j = spec.params['pair_indices']
                n_pairs = len(pairs_i)
                n_self = n_pairs * problem.n_waypoints
                constraint_specs.append(('self_collision', spec, n_self))
                n_constraints += n_self

            elif spec.name == 'joint_limits':
                # 2 constraints per joint per waypoint (lower and upper)
                n_jl = 2 * problem.n_joints * problem.n_waypoints
                constraint_specs.append(('joint_limits', spec, n_jl))
                n_constraints += n_jl

            elif spec.name == 'joint_velocity_limit':
                # 2 constraints per joint per step (upper and lower velocity)
                n_steps = problem.n_waypoints - 1
                n_vel = 2 * problem.n_joints * n_steps
                constraint_specs.append(('joint_velocity_limit', spec, n_vel))
                n_constraints += n_vel

        # If no hard constraints, add a dummy
        if n_constraints == 0:
            n_constraints = 1

        # Build objective function (soft costs only)
        def objective_fn(trajectory):
            total_cost = 0.0

            for spec in soft_costs:
                name = spec.name
                weight = spec.weight
                params = spec.params

                if name == 'smoothness':
                    diff = trajectory[1:] - trajectory[:-1]
                    total_cost = total_cost + weight * jnp.sum(diff ** 2)

                elif name == 'acceleration':
                    acc = (trajectory[2:] - 2 * trajectory[1:-1]
                           + trajectory[:-2]) / (dt ** 2)
                    total_cost = total_cost + weight * jnp.sum(acc ** 2)

                elif name == 'jerk':
                    acc = (trajectory[2:] - 2 * trajectory[1:-1]
                           + trajectory[:-2]) / (dt ** 2)
                    jerk = (acc[1:] - acc[:-1]) / dt
                    total_cost = total_cost + weight * jnp.sum(jerk ** 2)

                elif name == 'posture':
                    nominal = jnp.array(params['nominal_angles'])
                    diff = trajectory - nominal[None, :]
                    total_cost = total_cost + weight * jnp.sum(diff ** 2)

                elif name == 'world_collision' and has_collision:
                    obstacles = params['obstacles']
                    activation = params['activation_distance']

                    sphere_obs = [o for o in obstacles if o['type'] == 'sphere']
                    if sphere_obs:
                        obs_centers = jnp.stack(
                            [jnp.array(o['center']) for o in sphere_obs]
                        )
                        obs_radii = jnp.array([o['radius'] for o in sphere_obs])

                        def coll_cost_single(angles):
                            sphere_pos = get_sphere_positions(angles)
                            signed_dists = compute_sphere_obstacle_distances(
                                sphere_pos, sphere_radii,
                                obs_centers, obs_radii, jnp
                            )
                            residuals = compute_collision_residuals(
                                signed_dists, activation, jnp
                            )
                            return jnp.sum(residuals ** 2)

                        coll_costs = jax.vmap(coll_cost_single)(trajectory)
                        total_cost = total_cost + weight * jnp.sum(coll_costs)

                elif name == 'self_collision' and has_collision:
                    pair_indices = params['pair_indices']
                    activation = params['activation_distance']
                    pairs_i, pairs_j = pair_indices

                    if len(pairs_i) > 0:
                        pairs_i_arr = jnp.array(pairs_i)
                        pairs_j_arr = jnp.array(pairs_j)

                        def self_coll_cost_single(angles):
                            sphere_pos = get_sphere_positions(angles)
                            signed_dists = compute_self_collision_distances(
                                sphere_pos, sphere_radii,
                                pairs_i_arr, pairs_j_arr, jnp
                            )
                            residuals = compute_collision_residuals(
                                signed_dists, activation, jnp
                            )
                            return jnp.sum(residuals ** 2)

                        self_coll_costs = jax.vmap(
                            self_coll_cost_single
                        )(trajectory)
                        total_cost = (total_cost
                                      + weight * jnp.sum(self_coll_costs))

                elif name == 'cartesian_path' and has_cartesian:
                    target_pos = jnp.array(params['target_positions'])
                    target_rots = params.get('target_rotations')
                    rot_w = params.get('rotation_weight', 1.0)

                    if target_rots is not None:
                        target_rots = jnp.array(target_rots)

                        def cart_cost_single(args):
                            angles, t_pos, t_rot = args
                            ee_pos, ee_rot = get_ee_pose(angles)
                            pose_err = pose_error_log(
                                ee_pos, ee_rot, t_pos, t_rot)
                            pos_err = jnp.sum(pose_err[:3] ** 2)
                            rot_err = jnp.sum(pose_err[3:] ** 2)
                            return pos_err + rot_w * rot_err

                        cart_costs = jax.vmap(cart_cost_single)(
                            (trajectory, target_pos, target_rots)
                        )
                    else:
                        def cart_cost_pos_only(args):
                            angles, t_pos = args
                            ee_pos, _ = get_ee_pose(angles)
                            return jnp.sum((ee_pos - t_pos) ** 2)

                        cart_costs = jax.vmap(cart_cost_pos_only)(
                            (trajectory, target_pos)
                        )
                    total_cost = total_cost + weight * jnp.sum(cart_costs)

                elif name == 'joint_velocity_limit':
                    v_max = jnp.array(params['max_velocities'])
                    vel_dt = params['dt']
                    v_limit = v_max * vel_dt
                    dq = trajectory[1:] - trajectory[:-1]
                    violation = jnp.maximum(jnp.abs(dq) - v_limit, 0.0)
                    total_cost = total_cost + weight * jnp.sum(violation ** 2)

            # EE waypoint costs
            if has_ee_waypoints:
                for c in problem.ee_waypoint_costs:
                    wp_idx = c['waypoint_index']
                    t_pos = jnp.array(c['target_position'])
                    t_rot = jnp.array(c['target_rotation'])
                    pw = c['position_weight']
                    rw = c['rotation_weight']
                    angles = trajectory[wp_idx]
                    ee_pos, ee_rot = get_ee_pose(angles)
                    pose_err = pose_error_log(ee_pos, ee_rot, t_pos, t_rot)
                    pos_err = jnp.sum(pose_err[:3] ** 2)
                    rot_err = jnp.sum(pose_err[3:] ** 2)
                    total_cost = total_cost + pw * pos_err + rw * rot_err

            return total_cost

        # Build constraint function: g(x) >= 0 means satisfied
        def constraint_fn(trajectory):
            all_constraints = []

            for name, spec, _ in constraint_specs:
                params = spec.params

                if name == 'world_collision':
                    obstacles = params['obstacles']
                    activation = params['activation_distance']

                    sphere_obs = [o for o in obstacles if o['type'] == 'sphere']
                    if sphere_obs:
                        obs_centers = jnp.stack(
                            [jnp.array(o['center']) for o in sphere_obs]
                        )
                        obs_radii = jnp.array([o['radius'] for o in sphere_obs])

                        def coll_dist_single(angles):
                            sphere_pos = get_sphere_positions(angles)
                            signed_dists = compute_sphere_obstacle_distances(
                                sphere_pos, sphere_radii,
                                obs_centers, obs_radii, jnp
                            )
                            # Return signed distance (>= 0 is no collision)
                            return signed_dists.flatten()

                        # (n_waypoints, n_spheres * n_obstacles)
                        dists = jax.vmap(coll_dist_single)(trajectory)
                        # Constraint: dist - activation >= 0
                        # g >= 0 when dist >= activation (safety margin)
                        g = dists.flatten() - activation
                        all_constraints.append(g)

                elif name == 'self_collision':
                    pairs_i, pairs_j = params['pair_indices']
                    activation = params['activation_distance']

                    if len(pairs_i) > 0:
                        pairs_i_arr = jnp.array(pairs_i)
                        pairs_j_arr = jnp.array(pairs_j)

                        def self_coll_dist_single(angles):
                            sphere_pos = get_sphere_positions(angles)
                            signed_dists = compute_self_collision_distances(
                                sphere_pos, sphere_radii,
                                pairs_i_arr, pairs_j_arr, jnp
                            )
                            return signed_dists

                        dists = jax.vmap(self_coll_dist_single)(trajectory)
                        # Constraint: dist - activation >= 0
                        # g >= 0 when dist >= activation (safety margin)
                        g = dists.flatten() - activation
                        all_constraints.append(g)

                elif name == 'joint_limits':
                    lower = jnp.array(params['lower'])
                    upper = jnp.array(params['upper'])

                    # g1 = q - lower >= 0 (not below lower limit)
                    # g2 = upper - q >= 0 (not above upper limit)
                    g_lower = (trajectory - lower[None, :]).flatten()
                    g_upper = (upper[None, :] - trajectory).flatten()
                    all_constraints.append(g_lower)
                    all_constraints.append(g_upper)

                elif name == 'joint_velocity_limit':
                    v_max = jnp.array(params['max_velocities'])
                    vel_dt = params['dt']
                    v_limit = v_max * vel_dt

                    dq = trajectory[1:] - trajectory[:-1]
                    # g1 = v_limit - dq >= 0 (velocity not too positive)
                    # g2 = v_limit + dq >= 0 (velocity not too negative)
                    g_upper_vel = (v_limit[None, :] - dq).flatten()
                    g_lower_vel = (v_limit[None, :] + dq).flatten()
                    all_constraints.append(g_upper_vel)
                    all_constraints.append(g_lower_vel)

            if all_constraints:
                return jnp.concatenate(all_constraints)
            else:
                # No constraints: return dummy satisfied constraint
                return jnp.array([1.0])

        return jax.jit(objective_fn), jax.jit(constraint_fn), n_constraints
