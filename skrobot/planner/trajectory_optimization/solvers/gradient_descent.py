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

        dt = problem.dt

        # Build FK components
        fk_params = problem.fk_params
        link_trans = jnp.array(fk_params['link_translations'])
        link_rots = jnp.array(fk_params['link_rotations'])
        joint_axes = jnp.array(fk_params['joint_axes'])
        base_pos = jnp.array(fk_params['base_position'])
        base_rot = jnp.array(fk_params['base_rotation'])
        n_joints = fk_params['n_joints']

        # Collision data
        has_collision = problem.collision_spheres is not None

        if has_collision:
            coll_link_idx = jnp.array(problem.collision_link_to_chain_idx)
            coll_offsets_pos = jnp.array(problem.collision_link_offsets_pos)
            coll_offsets_rot = jnp.array(problem.collision_link_offsets_rot)
            sphere_centers = jnp.array(problem.collision_spheres['sphere_centers_local'])
            sphere_radii = jnp.array(problem.collision_spheres['sphere_radii'])
            sphere_link_indices = jnp.array(problem.collision_spheres['link_indices'])

        # Parse residual specs
        costs_config = []
        for spec in problem.residuals:
            costs_config.append({
                'name': spec.name,
                'weight': spec.weight,
                'params': spec.params,
            })

        def rotation_matrix_axis_angle(axis, theta):
            axis = axis / jnp.sqrt(jnp.dot(axis, axis) + 1e-10)
            K = jnp.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            I = jnp.eye(3)
            return I + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)

        def get_link_transforms(angles):
            positions = []
            rotations = []
            current_pos = base_pos
            current_rot = base_rot

            for i in range(n_joints):
                current_pos = current_pos + current_rot @ link_trans[i]
                current_rot = current_rot @ link_rots[i]
                joint_rot = rotation_matrix_axis_angle(joint_axes[i], angles[i])
                current_rot = current_rot @ joint_rot
                positions.append(current_pos)
                rotations.append(current_rot)

            return jnp.stack(positions), jnp.stack(rotations)

        def get_sphere_positions(angles):
            link_positions, link_rotations = get_link_transforms(angles)

            chain_idx = coll_link_idx[sphere_link_indices]
            sphere_link_pos = link_positions[chain_idx]
            sphere_link_rot = link_rotations[chain_idx]

            offsets_pos = coll_offsets_pos[sphere_link_indices]
            offsets_rot = coll_offsets_rot[sphere_link_indices]

            local = jnp.einsum('ijk,ik->ij', offsets_rot, sphere_centers) + offsets_pos
            world = sphere_link_pos + jnp.einsum('ijk,ik->ij', sphere_link_rot, local)
            return world

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
                    acc = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / (dt ** 2)
                    total_cost = total_cost + weight * jnp.sum(acc ** 2)

                elif name == 'jerk':
                    acc = (trajectory[2:] - 2 * trajectory[1:-1] + trajectory[:-2]) / (dt ** 2)
                    jerk = (acc[1:] - acc[:-1]) / dt
                    total_cost = total_cost + weight * jnp.sum(jerk ** 2)

                elif name == 'world_collision' and has_collision:
                    obstacles = params['obstacles']
                    activation = params['activation_distance']

                    sphere_obs = [o for o in obstacles if o['type'] == 'sphere']
                    if sphere_obs:
                        obs_centers = jnp.stack([jnp.array(o['center']) for o in sphere_obs])
                        obs_radii = jnp.array([o['radius'] for o in sphere_obs])

                        def coll_cost_single(angles):
                            sphere_pos = get_sphere_positions(angles)
                            diff = sphere_pos[:, None, :] - obs_centers[None, :, :]
                            dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)
                            signed = dists - sphere_radii[:, None] - obs_radii[None, :]
                            costs = jnp.maximum(0.0, activation - signed)
                            return jnp.sum(costs ** 2)

                        coll_costs = jax.vmap(coll_cost_single)(trajectory)
                        total_cost = total_cost + weight * jnp.sum(coll_costs)

                elif name == 'self_collision' and has_collision:
                    pair_indices = params['pair_indices']
                    activation = params['activation_distance']
                    pairs_i, pairs_j = pair_indices

                    if len(pairs_i) > 0:
                        pairs_i = jnp.array(pairs_i)
                        pairs_j = jnp.array(pairs_j)

                        def self_coll_cost_single(angles):
                            sphere_pos = get_sphere_positions(angles)
                            pos_i = sphere_pos[pairs_i]
                            pos_j = sphere_pos[pairs_j]
                            rad_i = sphere_radii[pairs_i]
                            rad_j = sphere_radii[pairs_j]

                            diff = pos_i - pos_j
                            dists = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-10)
                            signed = dists - rad_i - rad_j
                            costs = jnp.maximum(0.0, activation - signed)
                            return jnp.sum(costs ** 2)

                        self_coll_costs = jax.vmap(self_coll_cost_single)(trajectory)
                        total_cost = total_cost + weight * jnp.sum(self_coll_costs)

            return total_cost

        return cost_fn
