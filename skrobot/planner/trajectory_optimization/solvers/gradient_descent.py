"""Gradient descent based trajectory optimization solver.

This is a simple but fast solver using gradient descent with JAX autodiff.
Less robust than jaxls but faster per iteration.
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


class GradientDescentSolver(BaseSolver):
    """Simple gradient descent solver.

    Fast but can get stuck in local minima for complex problems.
    Recommended for simple problems or when speed is critical.

    JIT compilation is cached based on problem structure, so solving
    the same type of problem multiple times will be much faster after
    the first call.
    """

    def __init__(
        self,
        max_iterations=300,
        learning_rate=0.001,
        max_grad_norm=100.0,
        verbose=False,
    ):
        """Initialize gradient descent solver.

        Parameters
        ----------
        max_iterations : int
            Maximum optimization iterations.
        learning_rate : float
            Gradient descent step size.
        max_grad_norm : float
            Maximum gradient norm for clipping.
        verbose : bool
            Print optimization progress.
        """
        super().__init__(verbose=verbose)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        # Cache for JIT-compiled solve functions
        self._jit_cache = {}

    def _get_problem_cache_key(self, problem):
        """Generate cache key from problem structure."""
        residual_names = tuple(sorted(r.name for r in problem.residuals))
        return (
            problem.n_waypoints,
            problem.n_joints,
            residual_names,
            problem.fixed_start,
            problem.fixed_end,
            problem.collision_spheres is not None,
        )

    def solve(
        self,
        problem,
        initial_trajectory,
        **kwargs,
    ):
        """Solve trajectory optimization using gradient descent.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory (n_waypoints, n_joints).
        **kwargs
            Additional options.

        Returns
        -------
        SolverResult
            Optimization result.
        """
        import jax.numpy as jnp

        initial_trajectory = self._validate_trajectory(
            initial_trajectory, problem
        )

        max_iterations = kwargs.get('max_iterations', self.max_iterations)
        learning_rate = kwargs.get('learning_rate', self.learning_rate)

        # Check cache for JIT-compiled solver
        cache_key = self._get_problem_cache_key(problem)
        if cache_key not in self._jit_cache:
            # Build and cache the JIT-compiled solve function
            self._jit_cache[cache_key] = self._build_jit_solver(problem)

        jit_solve = self._jit_cache[cache_key]

        # Convert to JAX arrays
        trajectory = jnp.array(initial_trajectory)
        lower = jnp.array(problem.joint_limits_lower)
        upper = jnp.array(problem.joint_limits_upper)

        # Run JIT-compiled solver
        trajectory, final_cost = jit_solve(
            trajectory, lower, upper, learning_rate, max_iterations
        )

        return SolverResult(
            trajectory=np.array(trajectory),
            success=True,
            cost=float(final_cost),
            iterations=max_iterations,
            message='Gradient descent completed',
        )

    def _build_jit_solver(self, problem):
        """Build JIT-compiled solver function for given problem structure."""
        import jax
        import jax.numpy as jnp

        # Build cost function
        cost_fn = self._build_cost_function(problem)
        cost_and_grad = jax.value_and_grad(cost_fn)

        max_grad_norm = self.max_grad_norm
        fixed_start = problem.fixed_start
        fixed_end = problem.fixed_end

        def solve_fn(trajectory, lower, upper, learning_rate, max_iterations):
            start = trajectory[0]
            end = trajectory[-1]

            def body_fn(i, state):
                traj, prev_cost = state
                cost, grad = cost_and_grad(traj)

                # Gradient clipping
                grad_norm = jnp.sqrt(jnp.sum(grad ** 2) + 1e-10)
                grad = jax.lax.cond(
                    grad_norm > max_grad_norm,
                    lambda g: g * (max_grad_norm / grad_norm),
                    lambda g: g,
                    grad
                )

                # Update
                should_update = (cost > 1e-6) & (cost <= prev_cost * 1.1)
                new_traj = jax.lax.cond(
                    should_update,
                    lambda t: t - learning_rate * grad,
                    lambda t: t,
                    traj
                )

                # Clip to joint limits
                new_traj = jnp.clip(new_traj, lower, upper)

                # Fix endpoints
                if fixed_start:
                    new_traj = new_traj.at[0].set(start)
                if fixed_end:
                    new_traj = new_traj.at[-1].set(end)

                return (new_traj, cost)

            initial_cost = cost_fn(trajectory)
            trajectory, final_cost = jax.lax.fori_loop(
                0, max_iterations, body_fn, (trajectory, initial_cost + 1.0)
            )
            return trajectory, final_cost

        # JIT compile with static argnums for max_iterations
        return jax.jit(solve_fn, static_argnums=(4,))

    def _build_cost_function(self, problem):
        """Build cost function from problem definition.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.

        Returns
        -------
        callable
            Cost function: cost(trajectory) -> scalar.
        """
        import jax
        import jax.numpy as jnp

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals
        from skrobot.planner.trajectory_optimization.fk_utils import compute_self_collision_distances
        from skrobot.planner.trajectory_optimization.fk_utils import compute_sphere_obstacle_distances
        from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data

        dt = problem.dt

        # Collision data
        has_collision = problem.collision_spheres is not None

        get_sphere_positions = None
        sphere_radii = None

        if has_collision:
            fk_data = prepare_fk_data(problem, jnp)
            sphere_radii = fk_data['sphere_radii']
            _, get_sphere_positions = build_fk_functions(fk_data, jnp)

        # Parse residual specs
        costs_config = []
        for spec in problem.residuals:
            costs_config.append({
                'name': spec.name,
                'weight': spec.weight,
                'params': spec.params,
            })

        def cost_fn(trajectory):
            total_cost = 0.0

            for config in costs_config:
                name = config['name']
                weight = config['weight']
                params = config['params']

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

            return total_cost

        return cost_fn
