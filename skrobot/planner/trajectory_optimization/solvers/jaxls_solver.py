"""JAXls-based trajectory optimization solver.

This solver uses jaxls (JAX Least Squares) for robust nonlinear
least squares optimization with constraints.
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


class JaxlsSolver(BaseSolver):
    """JAXls-based trajectory optimization solver.

    Uses Levenberg-Marquardt algorithm for robust optimization.
    Supports both soft costs and hard constraints via augmented Lagrangian.
    """

    def __init__(
        self,
        max_iterations=100,
        verbose=False,
    ):
        """Initialize JAXls solver.

        Parameters
        ----------
        max_iterations : int
            Maximum optimization iterations.
        verbose : bool
            Print optimization progress.
        """
        super().__init__(verbose=verbose)
        self.max_iterations = max_iterations
        self._cached_problem = None
        self._cached_traj_var = None
        self._cache_key = None

    def _make_cache_key(self, problem, initial_trajectory):
        """Create a cache key based on problem structure.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory.

        Returns
        -------
        tuple
            Cache key tuple.
        """
        residual_names = tuple(r.name for r in problem.residuals)
        residual_weights = tuple(r.weight for r in problem.residuals)

        key = (
            problem.n_waypoints,
            problem.n_joints,
            residual_names,
            residual_weights,
            problem.fixed_start,
            problem.fixed_end,
            tuple(problem.joint_limits_lower.tolist()),
            tuple(problem.joint_limits_upper.tolist()),
            tuple(initial_trajectory[0].tolist()),
            tuple(initial_trajectory[-1].tolist()),
        )
        return key

    def solve(
        self,
        problem,
        initial_trajectory,
        **kwargs,
    ):
        """Solve trajectory optimization using jaxls.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.
        initial_trajectory : ndarray
            Initial trajectory (n_waypoints, n_joints).
        **kwargs
            Additional options:
            - max_iterations: Override default max iterations.

        Returns
        -------
        SolverResult
            Optimization result.
        """
        import jax.numpy as jnp
        import jaxls

        initial_trajectory = self._validate_trajectory(
            initial_trajectory, problem
        )

        max_iterations = kwargs.get('max_iterations', self.max_iterations)
        T = problem.n_waypoints
        n_joints = problem.n_joints

        cache_key = self._make_cache_key(problem, initial_trajectory)

        if self._cache_key == cache_key and self._cached_problem is not None:
            ls_problem = self._cached_problem
            TrajectoryVar = self._cached_traj_var
            traj_vars = TrajectoryVar(jnp.arange(T))
        else:
            default_cfg = jnp.zeros(n_joints)

            class TrajectoryVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cfg,
            ):
                pass

            traj_vars = TrajectoryVar(jnp.arange(T))

            # Prepare FK data for collision checking
            from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data
            fk_data = prepare_fk_data(problem, jnp)

            costs = []

            for residual_spec in problem.residuals:
                if residual_spec.name == 'smoothness':
                    costs.append(self._make_smoothness_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'acceleration':
                    costs.append(self._make_acceleration_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'world_collision':
                    costs.append(self._make_world_collision_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
                    ))
                elif residual_spec.name == 'self_collision':
                    costs.append(self._make_self_collision_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
                    ))

            if problem.fixed_start:
                start_cfg = jnp.array(initial_trajectory[0])

                @jaxls.Cost.factory(kind='constraint_eq_zero', name='start_constraint')
                def start_constraint(vals, var):
                    return (vals[var] - start_cfg).flatten()

                costs.append(start_constraint(TrajectoryVar(jnp.array([0]))))

            if problem.fixed_end:
                end_cfg = jnp.array(initial_trajectory[-1])

                @jaxls.Cost.factory(kind='constraint_eq_zero', name='end_constraint')
                def end_constraint(vals, var):
                    return (vals[var] - end_cfg).flatten()

                costs.append(end_constraint(TrajectoryVar(jnp.array([T - 1]))))

            lower = jnp.array(problem.joint_limits_lower)
            upper = jnp.array(problem.joint_limits_upper)

            @jaxls.Cost.factory(kind='constraint_geq_zero', name='joint_limits')
            def joint_limit_cost(vals, var):
                q = vals[var]
                lower_margin = q - lower
                upper_margin = upper - q
                return jnp.concatenate([lower_margin, upper_margin]).flatten()

            costs.append(joint_limit_cost(traj_vars))

            ls_problem = jaxls.LeastSquaresProblem(
                costs=costs,
                variables=[traj_vars],
            ).analyze()

            self._cached_problem = ls_problem
            self._cached_traj_var = TrajectoryVar
            self._cache_key = cache_key

        init_vals = jaxls.VarValues.make((
            traj_vars.with_value(jnp.array(initial_trajectory)),
        ))

        solution = ls_problem.solve(
            initial_vals=init_vals,
            verbose=self.verbose,
        )

        result_traj = np.array(solution[traj_vars])

        return SolverResult(
            trajectory=result_traj,
            success=True,
            iterations=max_iterations,
            message='Optimization completed',
        )

    def _make_smoothness_cost(self, problem, TrajectoryVar, spec):
        """Create smoothness cost."""
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='smoothness')
        def smoothness_cost(vals, curr_var, prev_var):
            q_curr = vals[curr_var]
            q_prev = vals[prev_var]
            return weight * (q_curr - q_prev).flatten()

        return smoothness_cost(
            TrajectoryVar(jnp.arange(1, T)),
            TrajectoryVar(jnp.arange(0, T - 1)),
        )

    def _make_acceleration_cost(self, problem, TrajectoryVar, spec):
        """Create acceleration cost."""
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='acceleration')
        def acceleration_cost(vals, curr_var, next_var, prev_var):
            q_curr = vals[curr_var]
            q_next = vals[next_var]
            q_prev = vals[prev_var]
            acc = (q_next - 2 * q_curr + q_prev) / (dt ** 2)
            return weight * acc.flatten()

        return acceleration_cost(
            TrajectoryVar(jnp.arange(1, T - 1)),
            TrajectoryVar(jnp.arange(2, T)),
            TrajectoryVar(jnp.arange(0, T - 2)),
        )

    def _make_world_collision_cost(self, problem, TrajectoryVar, fk_data, spec):
        """Create world collision avoidance cost."""
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals
        from skrobot.planner.trajectory_optimization.fk_utils import compute_sphere_obstacle_distances

        T = problem.n_waypoints
        obstacles = spec.params['obstacles']
        activation_dist = spec.params['activation_distance']
        weight = jnp.sqrt(spec.weight)

        # Parse obstacles
        sphere_obs = [
            obs for obs in obstacles if obs['type'] == 'sphere'
        ]

        if not sphere_obs:
            # No obstacles, return dummy cost
            @jaxls.Cost.factory(name='world_collision_dummy')
            def dummy_cost(vals, var):
                return jnp.array([0.0])

            return dummy_cost(TrajectoryVar(jnp.array([0])))

        obs_centers = jnp.stack([jnp.array(o['center']) for o in sphere_obs])
        obs_radii = jnp.array([o['radius'] for o in sphere_obs])
        sphere_radii = fk_data['sphere_radii']

        _, get_sphere_positions = build_fk_functions(fk_data, jnp)

        @jaxls.Cost.factory(name='world_collision')
        def world_collision_cost(vals, var):
            angles = vals[var]
            sphere_pos = get_sphere_positions(angles)

            # Use helper functions for distance computation
            signed_dists = compute_sphere_obstacle_distances(
                sphere_pos, sphere_radii, obs_centers, obs_radii, jnp
            )
            residuals = compute_collision_residuals(
                signed_dists, activation_dist, jnp
            )
            return (weight * residuals).flatten()

        return world_collision_cost(TrajectoryVar(jnp.arange(T)))

    def _make_self_collision_cost(self, problem, TrajectoryVar, fk_data, spec):
        """Create self-collision avoidance cost."""
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import compute_collision_residuals
        from skrobot.planner.trajectory_optimization.fk_utils import compute_self_collision_distances

        T = problem.n_waypoints
        pair_indices = spec.params['pair_indices']
        activation_dist = spec.params['activation_distance']
        weight = jnp.sqrt(spec.weight)

        pairs_i, pairs_j = pair_indices

        if len(pairs_i) == 0:
            @jaxls.Cost.factory(name='self_collision_dummy')
            def dummy_cost(vals, var):
                return jnp.array([0.0])

            return dummy_cost(TrajectoryVar(jnp.array([0])))

        pairs_i = jnp.array(pairs_i)
        pairs_j = jnp.array(pairs_j)
        sphere_radii = fk_data['sphere_radii']

        _, get_sphere_positions = build_fk_functions(fk_data, jnp)

        @jaxls.Cost.factory(name='self_collision')
        def self_collision_cost(vals, var):
            angles = vals[var]
            sphere_pos = get_sphere_positions(angles)

            # Use helper functions for distance computation
            signed_dists = compute_self_collision_distances(
                sphere_pos, sphere_radii, pairs_i, pairs_j, jnp
            )
            residuals = compute_collision_residuals(
                signed_dists, activation_dist, jnp
            )
            return (weight * residuals).flatten()

        return self_collision_cost(TrajectoryVar(jnp.arange(T)))
