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

    Dynamic values (constraint targets, Cartesian path targets) are passed
    as frozen ``jaxls.Var`` objects (``tangent_dim=0``) so that the JIT-
    compiled problem can be reused when only these values change.
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
        self._cached_constraint_param_var = None
        self._cached_cartesian_pos_param_var = None
        self._cached_cartesian_rot_param_var = None
        self._cached_ee_wp_pos_param_var = None
        self._cached_ee_wp_rot_param_var = None
        self._cached_constraint_ids = None
        self._cached_has_cartesian = False
        self._cached_has_cart_rot = False
        self._cached_has_ee_waypoints = False
        self._cache_key = None

    def _make_cache_key(self, problem):
        """Create a structure-only cache key.

        The key captures the problem *structure* (number of waypoints,
        residual names/weights, constraint layout) but NOT the dynamic
        values (constraint targets, Cartesian targets, initial trajectory).
        This allows the compiled problem to be reused when only values
        change.

        Parameters
        ----------
        problem : TrajectoryProblem
            Problem definition.

        Returns
        -------
        tuple
            Cache key tuple.
        """
        residual_names = tuple(r.name for r in problem.residuals)
        residual_weights = tuple(r.weight for r in problem.residuals)

        wp_constraint_indices = tuple(
            idx for idx, _ in problem.waypoint_constraints
        )

        has_cart_rot = any(
            r.name == 'cartesian_path'
            and r.params.get('target_rotations') is not None
            for r in problem.residuals
        )

        ee_wp_key = tuple(
            (c['waypoint_index'], c['position_weight'], c['rotation_weight'])
            for c in problem.ee_waypoint_costs
        )

        # Include obstacle positions in cache key
        # (obstacles change position, so compiled problem must be invalidated)
        obstacle_key = tuple()
        for r in problem.residuals:
            if r.name == 'world_collision':
                obstacles = r.params.get('obstacles', [])
                obstacle_key = tuple(
                    (tuple(o['center']), o['radius'])
                    for o in obstacles
                )
                break

        key = (
            problem.n_waypoints,
            problem.n_joints,
            residual_names,
            residual_weights,
            problem.fixed_start,
            problem.fixed_end,
            tuple(problem.joint_limits_lower.tolist()),
            tuple(problem.joint_limits_upper.tolist()),
            wp_constraint_indices,
            has_cart_rot,
            ee_wp_key,
            obstacle_key,  # Include obstacle positions
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

        cache_key = self._make_cache_key(problem)

        if self._cache_key == cache_key and self._cached_problem is not None:
            ls_problem = self._cached_problem
            TrajectoryVar = self._cached_traj_var
            ConstraintParamVar = self._cached_constraint_param_var
            CartesianPosParamVar = self._cached_cartesian_pos_param_var
            CartesianRotParamVar = self._cached_cartesian_rot_param_var
            EEWpPosParamVar = self._cached_ee_wp_pos_param_var
            EEWpRotParamVar = self._cached_ee_wp_rot_param_var
            constraint_ids = self._cached_constraint_ids
            has_cartesian = self._cached_has_cartesian
            has_cart_rot = self._cached_has_cart_rot
            has_ee_waypoints = self._cached_has_ee_waypoints
        else:
            default_cfg = jnp.zeros(n_joints)
            default_pos = jnp.zeros(3)
            default_rot = jnp.zeros(9)

            class TrajectoryVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cfg,
            ):
                pass

            # Frozen param vars: tangent_dim=0 means the solver never
            # updates them; retract_fn returns the original value.
            class ConstraintParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_cfg,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class CartesianPosParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_pos,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class CartesianRotParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class EEWpPosParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_pos,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            class EEWpRotParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass

            traj_vars = TrajectoryVar(jnp.arange(T))

            # Prepare FK data
            from skrobot.planner.trajectory_optimization.fk_utils import prepare_fk_data
            fk_data = prepare_fk_data(problem, jnp)

            costs = []

            has_cartesian = False
            has_cart_rot = False
            has_ee_waypoints = len(problem.ee_waypoint_costs) > 0

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
                elif residual_spec.name == 'posture':
                    costs.append(self._make_posture_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'cartesian_path':
                    has_cartesian = True
                    has_cart_rot = (
                        residual_spec.params.get('target_rotations')
                        is not None
                    )
                    costs.append(self._make_cartesian_path_cost(
                        problem, TrajectoryVar,
                        CartesianPosParamVar, CartesianRotParamVar,
                        fk_data, residual_spec,
                    ))
                elif residual_spec.name == 'joint_velocity_limit':
                    costs.append(self._make_joint_velocity_limit(
                        problem, TrajectoryVar, residual_spec
                    ))

            # --- EE waypoint costs ---
            if has_ee_waypoints:
                costs.append(self._make_ee_waypoint_costs(
                    problem, TrajectoryVar,
                    EEWpPosParamVar, EEWpRotParamVar,
                    fk_data,
                ))

            # --- Constraint targets as frozen ParamVars ---
            constraint_ids = {}
            next_ct_id = 0

            if problem.fixed_start:
                constraint_ids['start'] = next_ct_id

                @jaxls.Cost.factory(
                    kind='constraint_eq_zero',
                    name='start_constraint',
                )
                def start_constraint(vals, var, param):
                    return (vals[var] - vals[param]).flatten()

                costs.append(start_constraint(
                    TrajectoryVar(jnp.array([0])),
                    ConstraintParamVar(jnp.array([next_ct_id])),
                ))
                next_ct_id += 1

            if problem.fixed_end:
                constraint_ids['end'] = next_ct_id

                @jaxls.Cost.factory(
                    kind='constraint_eq_zero',
                    name='end_constraint',
                )
                def end_constraint(vals, var, param):
                    return (vals[var] - vals[param]).flatten()

                costs.append(end_constraint(
                    TrajectoryVar(jnp.array([T - 1])),
                    ConstraintParamVar(jnp.array([next_ct_id])),
                ))
                next_ct_id += 1

            wp_ct_ids = {}
            for wp_idx, _wp_angles in problem.waypoint_constraints:
                wp_ct_ids[wp_idx] = next_ct_id

                @jaxls.Cost.factory(
                    kind='constraint_eq_zero',
                    name='waypoint_constraint',
                )
                def waypoint_constraint(vals, var, param):
                    return (vals[var] - vals[param]).flatten()

                costs.append(waypoint_constraint(
                    TrajectoryVar(jnp.array([wp_idx])),
                    ConstraintParamVar(jnp.array([next_ct_id])),
                ))
                next_ct_id += 1

            constraint_ids['waypoints'] = wp_ct_ids
            constraint_ids['n_params'] = next_ct_id

            # Joint limits
            lower = jnp.array(problem.joint_limits_lower)
            upper = jnp.array(problem.joint_limits_upper)

            @jaxls.Cost.factory(
                kind='constraint_geq_zero', name='joint_limits',
            )
            def joint_limit_cost(vals, var):
                q = vals[var]
                lower_margin = q - lower
                upper_margin = upper - q
                return jnp.concatenate(
                    [lower_margin, upper_margin]
                ).flatten()

            costs.append(joint_limit_cost(traj_vars))

            # Build variable list (ParamVars included but frozen)
            all_variables = [traj_vars]
            if next_ct_id > 0:
                all_variables.append(
                    ConstraintParamVar(jnp.arange(next_ct_id))
                )
            if has_cartesian:
                all_variables.append(
                    CartesianPosParamVar(jnp.arange(T))
                )
                if has_cart_rot:
                    all_variables.append(
                        CartesianRotParamVar(jnp.arange(T))
                    )
            if has_ee_waypoints:
                n_ee_wps = len(problem.ee_waypoint_costs)
                all_variables.append(
                    EEWpPosParamVar(jnp.arange(n_ee_wps))
                )
                all_variables.append(
                    EEWpRotParamVar(jnp.arange(n_ee_wps))
                )

            ls_problem = jaxls.LeastSquaresProblem(
                costs=costs,
                variables=all_variables,
            ).analyze()

            self._cached_problem = ls_problem
            self._cached_traj_var = TrajectoryVar
            self._cached_constraint_param_var = ConstraintParamVar
            self._cached_cartesian_pos_param_var = CartesianPosParamVar
            self._cached_cartesian_rot_param_var = CartesianRotParamVar
            self._cached_ee_wp_pos_param_var = EEWpPosParamVar
            self._cached_ee_wp_rot_param_var = EEWpRotParamVar
            self._cached_constraint_ids = constraint_ids
            self._cached_has_cartesian = has_cartesian
            self._cached_has_cart_rot = has_cart_rot
            self._cached_has_ee_waypoints = has_ee_waypoints
            self._cache_key = cache_key

        # --- Build init_vals with current dynamic values ---
        traj_vars = TrajectoryVar(jnp.arange(T))
        init_pairs = [
            traj_vars.with_value(jnp.array(initial_trajectory)),
        ]

        # Constraint param values
        n_ct = constraint_ids['n_params']
        if n_ct > 0:
            ct_values = np.zeros((n_ct, n_joints))
            if 'start' in constraint_ids:
                ct_values[constraint_ids['start']] = initial_trajectory[0]
            if 'end' in constraint_ids:
                ct_values[constraint_ids['end']] = initial_trajectory[-1]
            for wp_idx, wp_angles in problem.waypoint_constraints:
                ct_id = constraint_ids['waypoints'][wp_idx]
                ct_values[ct_id] = wp_angles
            init_pairs.append(
                ConstraintParamVar(jnp.arange(n_ct)).with_value(
                    jnp.array(ct_values)
                )
            )

        # EE waypoint param values
        if has_ee_waypoints:
            ee_wps = problem.ee_waypoint_costs
            n_ee_wps = len(ee_wps)
            ee_pos_values = np.stack(
                [c['target_position'] for c in ee_wps])
            ee_rot_values = np.stack(
                [c['target_rotation'].flatten() for c in ee_wps])
            init_pairs.append(
                EEWpPosParamVar(jnp.arange(n_ee_wps)).with_value(
                    jnp.array(ee_pos_values)
                )
            )
            init_pairs.append(
                EEWpRotParamVar(jnp.arange(n_ee_wps)).with_value(
                    jnp.array(ee_rot_values)
                )
            )

        # Cartesian param values
        if has_cartesian:
            for r in problem.residuals:
                if r.name == 'cartesian_path':
                    init_pairs.append(
                        CartesianPosParamVar(jnp.arange(T)).with_value(
                            jnp.array(r.params['target_positions'])
                        )
                    )
                    if has_cart_rot:
                        rot_flat = np.array(
                            r.params['target_rotations']
                        ).reshape(T, 9)
                        init_pairs.append(
                            CartesianRotParamVar(jnp.arange(T)).with_value(
                                jnp.array(rot_flat)
                            )
                        )
                    break

        init_vals = jaxls.VarValues.make(tuple(init_pairs))

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

    def _make_posture_cost(self, problem, TrajectoryVar, spec):
        """Create posture regularization cost.

        Penalizes deviation from nominal joint angles at each waypoint.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        nominal = jnp.array(spec.params['nominal_angles'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='posture')
        def posture_cost(vals, var):
            q = vals[var]
            diff = q - nominal
            return (weight * diff).flatten()

        return posture_cost(TrajectoryVar(jnp.arange(T)))

    def _make_ee_waypoint_costs(
        self, problem, TrajectoryVar,
        EEWpPosParamVar, EEWpRotParamVar,
        fk_data,
    ):
        """Create end-effector waypoint tracking costs.

        Constrains end-effector pose at specific trajectory waypoints
        without fixing joint angles, leaving the optimizer free to find
        natural joint configurations.  Targets are stored in frozen
        ParamVar objects for JIT cache reuse.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log

        _, _, _, get_ee_pose = build_fk_functions(fk_data, jnp)

        ee_wps = problem.ee_waypoint_costs
        n_ee_wps = len(ee_wps)
        indices = jnp.array([c['waypoint_index'] for c in ee_wps])

        # Use uniform weight (from first EE waypoint); weights are
        # part of the cache key so changing them rebuilds the problem.
        pos_weight = jnp.sqrt(ee_wps[0]['position_weight'])
        rot_weight = jnp.sqrt(ee_wps[0]['rotation_weight'])

        @jaxls.Cost.factory(name='ee_waypoint')
        def ee_waypoint_cost(vals, var, pos_param, rot_param):
            angles = vals[var]
            ee_pos, ee_rot = get_ee_pose(angles)
            target_pos = vals[pos_param]
            target_rot = vals[rot_param].reshape(3, 3)
            # Use SE(3) logarithmic map for pose error
            pose_err = pose_error_log(ee_pos, ee_rot, target_pos, target_rot)
            # pose_err is (6,): [tx, ty, tz, rx, ry, rz]
            pos_err = pos_weight * pose_err[:3]
            rot_err = rot_weight * pose_err[3:]
            return jnp.concatenate([pos_err, rot_err]).flatten()

        return ee_waypoint_cost(
            TrajectoryVar(indices),
            EEWpPosParamVar(jnp.arange(n_ee_wps)),
            EEWpRotParamVar(jnp.arange(n_ee_wps)),
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

        _, get_sphere_positions, _, _ = build_fk_functions(fk_data, jnp)

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

        _, get_sphere_positions, _, _ = build_fk_functions(fk_data, jnp)

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

    def _make_cartesian_path_cost(
        self, problem, TrajectoryVar,
        CartesianPosParamVar, CartesianRotParamVar,
        fk_data, spec,
    ):
        """Create Cartesian path tracking cost with parametric targets.

        Targets are read from frozen ParamVar objects so that the
        compiled problem can be reused when targets change.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import build_fk_functions
        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log

        T = problem.n_waypoints
        rotation_weight = spec.params.get('rotation_weight', 1.0)
        pos_weight = jnp.sqrt(spec.weight)

        _, _, _, get_ee_pose = build_fk_functions(fk_data, jnp)

        has_rot = spec.params.get('target_rotations') is not None

        if has_rot:
            rot_weight = jnp.sqrt(spec.weight * rotation_weight)

            @jaxls.Cost.factory(name='cartesian_path')
            def cartesian_path_cost(vals, var, pos_param, rot_param):
                angles = vals[var]
                ee_pos, ee_rot = get_ee_pose(angles)
                target_pos = vals[pos_param]
                target_rot = vals[rot_param].reshape(3, 3)
                # Use SE(3) logarithmic map for pose error
                pose_err = pose_error_log(ee_pos, ee_rot, target_pos, target_rot)
                pos_err = pos_weight * pose_err[:3]
                rot_err = rot_weight * pose_err[3:]
                return jnp.concatenate([pos_err, rot_err]).flatten()

            return cartesian_path_cost(
                TrajectoryVar(jnp.arange(T)),
                CartesianPosParamVar(jnp.arange(T)),
                CartesianRotParamVar(jnp.arange(T)),
            )
        else:
            @jaxls.Cost.factory(name='cartesian_path')
            def cartesian_path_cost(vals, var, pos_param):
                angles = vals[var]
                ee_pos, _ = get_ee_pose(angles)
                target_pos = vals[pos_param]
                return (pos_weight * (ee_pos - target_pos)).flatten()

            return cartesian_path_cost(
                TrajectoryVar(jnp.arange(T)),
                CartesianPosParamVar(jnp.arange(T)),
            )

    def _make_joint_velocity_limit(self, problem, TrajectoryVar, spec):
        """Create joint velocity limit constraint.

        Enforces ``|q[t+1] - q[t]| / dt <= v_max`` for each joint,
        expressed as two ``geq_zero`` inequalities per step:

            v_max * dt - (q_next - q_prev)  >= 0
            v_max * dt + (q_next - q_prev)  >= 0
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        max_velocities = jnp.array(spec.params['max_velocities'])
        v_limit = max_velocities * dt  # max allowable delta per step

        @jaxls.Cost.factory(
            kind='constraint_geq_zero',
            name='joint_velocity_limit',
        )
        def velocity_limit(vals, curr_var, prev_var):
            dq = vals[curr_var] - vals[prev_var]
            upper_margin = v_limit - dq   # v_limit - dq >= 0
            lower_margin = v_limit + dq   # v_limit + dq >= 0
            return jnp.concatenate([upper_margin, lower_margin]).flatten()

        return velocity_limit(
            TrajectoryVar(jnp.arange(1, T)),
            TrajectoryVar(jnp.arange(0, T - 1)),
        )
