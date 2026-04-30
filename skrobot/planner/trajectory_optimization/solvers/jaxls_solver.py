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


_JAXLS_INSTALL_HINT = (
    "jaxls is required for JaxlsSolver but is not importable.\n"
    "It is not on PyPI, so 'pip install jaxls' / 'uv pip install jaxls' "
    "will not work.\n"
    "Install it from source instead:\n"
    "    pip install \"git+https://github.com/brentyi/jaxls.git\""
)


def _require_jaxls():
    """Import jaxls or raise ImportError with the correct install hint."""
    try:
        import jaxls  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "{}\nOriginal error: {}".format(_JAXLS_INSTALL_HINT, e)) from e


def _root_link_world_pose(problem):
    """Return the world pose of the robot's root link as numpy arrays."""
    root = problem.robot_model.root_link.worldcoords()
    return root.worldpos().astype(np.float64), \
        root.worldrot().astype(np.float64)


def _chain_parent_relative_to_root(fk_params, root_pos, root_rot):
    """Compute (rel_pos, rel_rot): chain parent pose in root_link frame."""
    natural_pos = np.asarray(fk_params['base_position'], dtype=np.float64)
    natural_rot = np.asarray(fk_params['base_rotation'], dtype=np.float64)
    root_T = np.eye(4)
    root_T[:3, :3] = root_rot
    root_T[:3, 3] = root_pos
    inv = np.linalg.inv(root_T)
    rel_pos = inv[:3, :3] @ natural_pos + inv[:3, 3]
    rel_rot = inv[:3, :3] @ natural_rot
    return rel_pos, rel_rot


def _build_root_relative_chain_fks(problem, fk_data, jnp_module):
    """Per-chain link FK that accepts root_link world pose as base.

    Each callable composes the chain's natural parent transform
    (captured at extraction time, relative to root_link) with the
    floating-base pose at call time, then runs the chain's serial FK.
    """
    from skrobot.planner.trajectory_optimization.fk_utils import build_chain_link_transforms_with_base

    root_pos_np, root_rot_np = _root_link_world_pose(problem)
    callables = []
    for fkp in problem.fk_params_per_chain:
        jdata = {
            'link_translations': jnp_module.array(fkp['link_translations']),
            'link_rotations':    jnp_module.array(fkp['link_rotations']),
            'joint_axes':        jnp_module.array(fkp['joint_axes']),
            'n_joints':          fkp['n_joints'],
            'ref_angles':        jnp_module.array(fkp['ref_angles']),
        }
        rel_pos_np, rel_rot_np = _chain_parent_relative_to_root(
            fkp, root_pos_np, root_rot_np)
        rel_pos = jnp_module.asarray(rel_pos_np)
        rel_rot = jnp_module.asarray(rel_rot_np)
        inner = build_chain_link_transforms_with_base(jdata, jnp_module)

        def _make(inner=inner, rel_pos=rel_pos, rel_rot=rel_rot):
            def chain_fk(angles, base_pos, base_rot):
                parent_pos = base_pos + base_rot @ rel_pos
                parent_rot = base_rot @ rel_rot
                return inner(angles, parent_pos, parent_rot)
            return chain_fk
        callables.append(_make())
    return callables


def _build_root_relative_chain_ee_fks(problem, jnp_module):
    """Per-chain EE pose FK that accepts root_link world pose as base."""
    from skrobot.planner.trajectory_optimization.fk_utils import build_chain_ee_pose_with_base

    root_pos_np, root_rot_np = _root_link_world_pose(problem)
    callables = []
    for fkp in problem.fk_params_per_chain:
        jdata = {
            'link_translations': jnp_module.array(fkp['link_translations']),
            'link_rotations':    jnp_module.array(fkp['link_rotations']),
            'joint_axes':        jnp_module.array(fkp['joint_axes']),
            'n_joints':          fkp['n_joints'],
            'ref_angles':        jnp_module.array(fkp['ref_angles']),
            'ee_offset_position': jnp_module.array(
                fkp['ee_offset_position']),
            'ee_offset_rotation': jnp_module.array(
                fkp['ee_offset_rotation']),
        }
        rel_pos_np, rel_rot_np = _chain_parent_relative_to_root(
            fkp, root_pos_np, root_rot_np)
        rel_pos = jnp_module.asarray(rel_pos_np)
        rel_rot = jnp_module.asarray(rel_rot_np)
        inner = build_chain_ee_pose_with_base(jdata, jnp_module)

        def _make(inner=inner, rel_pos=rel_pos, rel_rot=rel_rot):
            def ee_fk(angles, base_pos, base_rot):
                parent_pos = base_pos + base_rot @ rel_pos
                parent_rot = base_rot @ rel_rot
                return inner(angles, parent_pos, parent_rot)
            return ee_fk
        callables.append(_make())
    return callables


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
        # Fail fast with an actionable hint if jaxls is missing — it is
        # not on PyPI, so the usual ``pip install jaxls`` will not work.
        _require_jaxls()
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
        n_total_dof = getattr(problem, 'n_total_dof', n_joints)
        n_base_dof = getattr(problem, 'n_base_dof', 0)

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
            default_cfg = jnp.zeros(n_total_dof)
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
                elif residual_spec.name == 'five_point_velocity':
                    costs.append(self._make_five_point_velocity_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'five_point_acceleration':
                    costs.append(self._make_five_point_acceleration_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'five_point_jerk':
                    costs.append(self._make_five_point_jerk_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'acceleration_limit':
                    costs.append(self._make_acceleration_limit_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'jerk_limit':
                    costs.append(self._make_jerk_limit_cost(
                        problem, TrajectoryVar, residual_spec
                    ))
                elif residual_spec.name == 'com':
                    costs.append(self._make_com_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
                    ))
                elif residual_spec.name == 'multi_ee_waypoint':
                    costs.append(self._make_multi_ee_waypoint_cost(
                        problem, TrajectoryVar, fk_data, residual_spec
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

            # Joint limits (extend with -inf/+inf for base DoF so the
            # base translation/rotation are unconstrained).
            if n_base_dof > 0:
                lower_full = np.concatenate([
                    problem.joint_limits_lower,
                    np.full(n_base_dof, -1e6, dtype=np.float64),
                ])
                upper_full = np.concatenate([
                    problem.joint_limits_upper,
                    np.full(n_base_dof, +1e6, dtype=np.float64),
                ])
            else:
                lower_full = problem.joint_limits_lower
                upper_full = problem.joint_limits_upper
            lower = jnp.array(lower_full)
            upper = jnp.array(upper_full)

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

            # CoG cost ParamVars (one per add_com_cost call). The cost
            # factory stashes them on ``problem._com_param_vars``.
            for entry in getattr(problem, '_com_param_vars', []):
                vc = entry['var_class']
                n = entry['targets_sel'].shape[0]
                all_variables.append(vc(jnp.arange(n)))
            # Multi-EE waypoint cost ParamVars.
            for entry in getattr(problem, '_multi_ee_param_vars', []):
                pos_vc = entry['pos_var_class']
                all_variables.append(pos_vc(jnp.arange(T)))
                if entry['rot_var_class'] is not None:
                    all_variables.append(
                        entry['rot_var_class'](jnp.arange(T)))

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
            ct_values = np.zeros((n_ct, n_total_dof))
            if 'start' in constraint_ids:
                ct_values[constraint_ids['start']] = initial_trajectory[0]
            if 'end' in constraint_ids:
                ct_values[constraint_ids['end']] = initial_trajectory[-1]
            for wp_idx, wp_angles in problem.waypoint_constraints:
                ct_id = constraint_ids['waypoints'][wp_idx]
                wp_full = np.zeros(n_total_dof)
                wp_full[:len(wp_angles)] = wp_angles
                ct_values[ct_id] = wp_full
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

        # CoG ParamVar values: write the per-waypoint targets stashed
        # by ``_make_com_cost`` so the closure-frozen ParamVars carry
        # the right numbers when the JIT plan executes.
        for entry in getattr(problem, '_com_param_vars', []):
            vc = entry['var_class']
            n = entry['targets_sel'].shape[0]
            init_pairs.append(
                vc(jnp.arange(n)).with_value(entry['targets_sel'])
            )
        # Multi-EE waypoint ParamVar values.
        for entry in getattr(problem, '_multi_ee_param_vars', []):
            pos_vc = entry['pos_var_class']
            init_pairs.append(
                pos_vc(jnp.arange(T)).with_value(entry['target_pos'])
            )
            if entry['rot_var_class'] is not None:
                init_pairs.append(
                    entry['rot_var_class'](jnp.arange(T)).with_value(
                        entry['target_rot'])
                )

        init_vals = jaxls.VarValues.make(tuple(init_pairs))

        solution = ls_problem.solve(
            initial_vals=init_vals,
            verbose=self.verbose,
            termination=jaxls.TerminationConfig(
                max_iterations=int(max_iterations),
            ),
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

    def _make_five_point_velocity_cost(self, problem, TrajectoryVar, spec):
        """Create velocity cost using 5-point stencil.

        Computes velocity with O(h^4) accuracy:
            v = (-q[t+2] + 8*q[t+1] - 8*q[t-1] + q[t-2]) / (12*dt)

        Penalizes velocities that exceed the velocity limits.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        velocity_limits = jnp.array(spec.params['velocity_limits'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='five_point_velocity')
        def five_point_velocity_cost(
            vals, var_tp2, var_tp1, var_tm1, var_tm2
        ):
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]

            velocity = (-q_tp2 + 8 * q_tp1 - 8 * q_tm1 + q_tm2) / (12 * dt)
            # Penalize only when |velocity| > limit
            residual = jnp.maximum(0.0, jnp.abs(velocity) - velocity_limits)
            return (weight * residual).flatten()

        # Apply to waypoints [2, T-2] (need 2 points on each side)
        return five_point_velocity_cost(
            TrajectoryVar(jnp.arange(4, T)),      # t+2
            TrajectoryVar(jnp.arange(3, T - 1)),  # t+1
            TrajectoryVar(jnp.arange(1, T - 3)),  # t-1
            TrajectoryVar(jnp.arange(0, T - 4)),  # t-2
        )

    def _make_five_point_acceleration_cost(self, problem, TrajectoryVar, spec):
        """Create acceleration cost using 5-point stencil.

        Computes acceleration with O(h^4) accuracy:
            a = (-q[t+2] + 16*q[t+1] - 30*q[t] + 16*q[t-1] - q[t-2]) / (12*dt^2)
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='five_point_acceleration')
        def five_point_acceleration_cost(
            vals, var_t, var_tp2, var_tp1, var_tm1, var_tm2
        ):
            q_t = vals[var_t]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]

            acceleration = (
                -q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2
            ) / (12 * dt ** 2)
            return (weight * jnp.abs(acceleration)).flatten()

        # Apply to waypoints [2, T-2]
        return five_point_acceleration_cost(
            TrajectoryVar(jnp.arange(2, T - 2)),  # t
            TrajectoryVar(jnp.arange(4, T)),      # t+2
            TrajectoryVar(jnp.arange(3, T - 1)),  # t+1
            TrajectoryVar(jnp.arange(1, T - 3)),  # t-1
            TrajectoryVar(jnp.arange(0, T - 4)),  # t-2
        )

    def _make_five_point_jerk_cost(self, problem, TrajectoryVar, spec):
        """Create jerk cost using 7-point stencil.

        Computes jerk with O(h^4) accuracy:
            j = (-q[t+3] + 8*q[t+2] - 13*q[t+1] + 13*q[t-1] - 8*q[t-2] + q[t-3])
                / (8*dt^3)
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='five_point_jerk')
        def five_point_jerk_cost(
            vals, var_tp3, var_tp2, var_tp1, var_tm1, var_tm2, var_tm3
        ):
            q_tp3 = vals[var_tp3]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]
            q_tm3 = vals[var_tm3]

            jerk = (
                -q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3
            ) / (8 * dt ** 3)
            return (weight * jnp.abs(jerk)).flatten()

        # Apply to waypoints [3, T-3]
        return five_point_jerk_cost(
            TrajectoryVar(jnp.arange(6, T)),      # t+3
            TrajectoryVar(jnp.arange(5, T - 1)),  # t+2
            TrajectoryVar(jnp.arange(4, T - 2)),  # t+1
            TrajectoryVar(jnp.arange(2, T - 4)),  # t-1
            TrajectoryVar(jnp.arange(1, T - 5)),  # t-2
            TrajectoryVar(jnp.arange(0, T - 6)),  # t-3
        )

    def _make_acceleration_limit_cost(self, problem, TrajectoryVar, spec):
        """Create acceleration limit cost using 5-point stencil.

        Penalizes accelerations that exceed the specified limit.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        acceleration_limit = jnp.array(spec.params['acceleration_limit'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='acceleration_limit')
        def acceleration_limit_cost(
            vals, var_t, var_tp2, var_tp1, var_tm1, var_tm2
        ):
            q_t = vals[var_t]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]

            acceleration = (
                -q_tp2 + 16 * q_tp1 - 30 * q_t + 16 * q_tm1 - q_tm2
            ) / (12 * dt ** 2)
            # Penalize only when |acceleration| > limit
            residual = jnp.maximum(
                0.0, jnp.abs(acceleration) - acceleration_limit
            )
            return (weight * residual).flatten()

        # Apply to waypoints [2, T-2]
        return acceleration_limit_cost(
            TrajectoryVar(jnp.arange(2, T - 2)),  # t
            TrajectoryVar(jnp.arange(4, T)),      # t+2
            TrajectoryVar(jnp.arange(3, T - 1)),  # t+1
            TrajectoryVar(jnp.arange(1, T - 3)),  # t-1
            TrajectoryVar(jnp.arange(0, T - 4)),  # t-2
        )

    def _make_jerk_limit_cost(self, problem, TrajectoryVar, spec):
        """Create jerk limit cost using 7-point stencil.

        Penalizes jerks that exceed the specified limit.
        """
        import jax.numpy as jnp
        import jaxls

        T = problem.n_waypoints
        dt = spec.params['dt']
        jerk_limit = jnp.array(spec.params['jerk_limit'])
        weight = jnp.sqrt(spec.weight)

        @jaxls.Cost.factory(name='jerk_limit')
        def jerk_limit_cost(
            vals, var_tp3, var_tp2, var_tp1, var_tm1, var_tm2, var_tm3
        ):
            q_tp3 = vals[var_tp3]
            q_tp2 = vals[var_tp2]
            q_tp1 = vals[var_tp1]
            q_tm1 = vals[var_tm1]
            q_tm2 = vals[var_tm2]
            q_tm3 = vals[var_tm3]

            jerk = (
                -q_tp3 + 8 * q_tp2 - 13 * q_tp1 + 13 * q_tm1 - 8 * q_tm2 + q_tm3
            ) / (8 * dt ** 3)
            # Penalize only when |jerk| > limit
            residual = jnp.maximum(0.0, jnp.abs(jerk) - jerk_limit)
            return (weight * residual).flatten()

        # Apply to waypoints [3, T-3]
        return jerk_limit_cost(
            TrajectoryVar(jnp.arange(6, T)),      # t+3
            TrajectoryVar(jnp.arange(5, T - 1)),  # t+2
            TrajectoryVar(jnp.arange(4, T - 2)),  # t+1
            TrajectoryVar(jnp.arange(2, T - 4)),  # t-1
            TrajectoryVar(jnp.arange(1, T - 5)),  # t-2
            TrajectoryVar(jnp.arange(0, T - 6)),  # t-3
        )

    def _make_com_cost(self, problem, TrajectoryVar, fk_data, spec):
        """Per-waypoint centre-of-gravity tracking cost.

        Splits the per-waypoint augmented variable
        ``aug = [chain0_q | chain1_q | ... | base_xyz | base_rpy]``
        and routes:
          * each chain's joint slice through its own FK to recover
            chain-link world poses (mass contribution from the chain),
          * the base portion through the unified centroid forward
            from :mod:`skrobot.dynamics` so fixed-to-base links also
            pick up the right rigid transform.

        Per-waypoint targets are wired through a frozen
        ``ComTargetParamVar`` (``tangent_dim=0``) so the optimisation
        treats them as constants while still benefiting from the
        cached JIT plan.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.coordinates.math import normalize_mask
        from skrobot.dynamics import build_world_centroid_fn

        targets = np.asarray(spec.params['target_positions'],
                             dtype=np.float64)
        wp_indices = np.asarray(spec.params['waypoint_indices'],
                                dtype=np.int64)
        axis_mask = np.asarray(normalize_mask(spec.params['translation_axis']))
        sel_np = np.where(axis_mask == 1)[0].astype(np.int64)
        n_axes = int(sel_np.size)
        sel = jnp.asarray(sel_np)
        weight = jnp.sqrt(spec.weight)

        n_base_dof = getattr(problem, 'n_base_dof', 0)
        chain_offsets = []
        offset = 0
        for chain in problem.link_lists:
            chain_offsets.append((offset, offset + len(chain)))
            offset += len(chain)

        # Build per-chain link FK that takes the root_link world pose
        # as its base argument. The chain's natural parent (e.g.
        # ``torso_lift_link`` for Fetch's right arm) is recovered by
        # composing with its relative-to-root transform, so a single
        # ``base_pos`` / ``base_rot`` (the floating-base pose) drives
        # both the chain FK and the fixed-link contribution
        # consistently.
        get_lt_per_chain = _build_root_relative_chain_fks(
            problem, fk_data, jnp)
        root_pos_np, root_rot_np = _root_link_world_pose(problem)
        compute_centroid_mc = build_world_centroid_fn(
            problem.centroid_data, get_lt_per_chain, backend=jnp,
            base_pos_default=root_pos_np,
            base_rot_default=root_rot_np,
        )

        base_pos_default = jnp.asarray(root_pos_np)
        base_rot_default = jnp.asarray(root_rot_np)
        n_joints_total = problem.n_joints

        def _euler_xyz_to_matrix(rx, ry, rz):
            cx, sx = jnp.cos(rx), jnp.sin(rx)
            cy, sy = jnp.cos(ry), jnp.sin(ry)
            cz, sz = jnp.cos(rz), jnp.sin(rz)
            Rx = jnp.array([[1, 0, 0],
                            [0, cx, -sx],
                            [0, sx, cx]])
            Ry = jnp.array([[cy, 0, sy],
                            [0, 1, 0],
                            [-sy, 0, cy]])
            Rz = jnp.array([[cz, -sz, 0],
                            [sz, cz, 0],
                            [0, 0, 1]])
            return Rx @ Ry @ Rz

        def compute_centroid(angles_aug):
            angles_per_chain = [
                angles_aug[a:b] for (a, b) in chain_offsets
            ]
            if n_base_dof == 0:
                return compute_centroid_mc(
                    angles_per_chain,
                    base_pos=base_pos_default,
                    base_rot=base_rot_default,
                )
            base_section = angles_aug[n_joints_total:]
            if n_base_dof == 6:
                bx, by, bz, rx, ry, rz = (
                    base_section[0], base_section[1], base_section[2],
                    base_section[3], base_section[4], base_section[5])
                base_pos = base_pos_default + jnp.array([bx, by, bz])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    rx, ry, rz)
            else:  # planar 3 DoF (x, y, yaw)
                bx, by, ryaw = (
                    base_section[0], base_section[1], base_section[2])
                base_pos = base_pos_default + jnp.array([bx, by, 0.0])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    0.0, 0.0, ryaw)
            return compute_centroid_mc(
                angles_per_chain, base_pos=base_pos, base_rot=base_rot,
            )

        # Targets per waypoint, only the selected axes.
        if targets.ndim == 2:
            targets_sel = targets[:, sel_np]
        else:
            # Single target broadcast to every waypoint.
            targets_sel = np.broadcast_to(
                targets[sel_np], (len(wp_indices), n_axes),
            )

        default_target = jnp.zeros(n_axes)

        class ComTargetParamVar(
            jaxls.Var[jnp.ndarray],
            default_factory=lambda: default_target,
            retract_fn=lambda x, delta: x,
            tangent_dim=0,
        ):
            pass

        @jaxls.Cost.factory(name='com')
        def com_cost(vals, joint_var, target_var):
            angles = vals[joint_var]
            cog = compute_centroid(angles)
            target = vals[target_var]
            err = (cog[sel] - target) * weight
            return err.flatten()

        # Stash the param var class + values so the outer ``solve()``
        # method can populate them in init_pairs alongside the joint
        # trajectory.
        if not hasattr(problem, '_com_param_vars'):
            problem._com_param_vars = []
        problem._com_param_vars.append({
            'var_class': ComTargetParamVar,
            'targets_sel': jnp.asarray(targets_sel, dtype=jnp.float64),
        })

        return com_cost(
            TrajectoryVar(jnp.asarray(wp_indices)),
            ComTargetParamVar(jnp.arange(len(wp_indices))),
        )

    def _make_multi_ee_waypoint_cost(self, problem, TrajectoryVar,
                                      fk_data, spec):
        """Multi-EE per-waypoint pose tracking with floating base.

        At every waypoint, drive every chain's end-effector to its
        corresponding world-frame target. This is the missing piece
        that lets a single trajectory-optimisation solve handle a
        multi-stance gait (foot pose constraints + base 6-DoF + CoM)
        in one shot.
        """
        import jax.numpy as jnp
        import jaxls

        from skrobot.planner.trajectory_optimization.fk_utils import pose_error_log

        T = problem.n_waypoints
        n_joints_total = problem.n_joints
        n_base_dof = getattr(problem, 'n_base_dof', 0)
        n_chains = len(problem.link_lists)

        chain_offsets = []
        offset = 0
        for chain in problem.link_lists:
            chain_offsets.append((offset, offset + len(chain)))
            offset += len(chain)

        target_pos_per_chain = spec.params['target_positions_per_chain']
        target_rot_per_chain = spec.params['target_rotations_per_chain']
        pos_w = jnp.sqrt(spec.params['position_weight'])
        rot_w = jnp.sqrt(spec.params['rotation_weight']) \
            if target_rot_per_chain is not None else None

        # Per-chain EE FK that takes the root_link world pose as its
        # base argument; the chain's natural parent transform relative
        # to root is composed inside.
        get_ee_pose_per_chain = _build_root_relative_chain_ee_fks(
            problem, jnp)
        root_pos_np, root_rot_np = _root_link_world_pose(problem)
        base_pos_default = jnp.asarray(root_pos_np)
        base_rot_default = jnp.asarray(root_rot_np)

        def _euler_xyz_to_matrix(rx, ry, rz):
            cx, sx = jnp.cos(rx), jnp.sin(rx)
            cy, sy = jnp.cos(ry), jnp.sin(ry)
            cz, sz = jnp.cos(rz), jnp.sin(rz)
            Rx = jnp.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
            Ry = jnp.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            Rz = jnp.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
            return Rx @ Ry @ Rz

        def _split_aug(angles_aug):
            angles_per_chain = [
                angles_aug[a:b] for (a, b) in chain_offsets
            ]
            if n_base_dof == 0:
                return angles_per_chain, base_pos_default, base_rot_default
            base_section = angles_aug[n_joints_total:]
            if n_base_dof == 6:
                base_pos = base_pos_default + base_section[:3]
                rx, ry, rz = base_section[3], base_section[4], base_section[5]
                base_rot = base_rot_default @ _euler_xyz_to_matrix(rx, ry, rz)
            else:
                base_pos = base_pos_default + jnp.array(
                    [base_section[0], base_section[1], 0.0])
                base_rot = base_rot_default @ _euler_xyz_to_matrix(
                    0.0, 0.0, base_section[2])
            return angles_per_chain, base_pos, base_rot

        # Stack per-chain targets to (T, n_chains, 3) and (T, n_chains, 9).
        T_pos = np.stack(target_pos_per_chain, axis=1)  # (T, n_chains, 3)
        if target_rot_per_chain is not None:
            T_rot = np.stack(
                [r.reshape(T, 9) for r in target_rot_per_chain], axis=1,
            )
        else:
            T_rot = None

        default_pos = jnp.zeros((n_chains, 3))
        default_rot = jnp.zeros((n_chains, 9))

        class MEEPosParamVar(
            jaxls.Var[jnp.ndarray],
            default_factory=lambda: default_pos,
            retract_fn=lambda x, delta: x,
            tangent_dim=0,
        ):
            pass

        if T_rot is not None:
            class MEERotParamVar(
                jaxls.Var[jnp.ndarray],
                default_factory=lambda: default_rot,
                retract_fn=lambda x, delta: x,
                tangent_dim=0,
            ):
                pass
        else:
            MEERotParamVar = None

        if T_rot is not None:
            @jaxls.Cost.factory(name='multi_ee_waypoint')
            def cost_pose(vals, var, pos_param, rot_param):
                aug = vals[var]
                apc, bpos, brot = _split_aug(aug)
                tgt_pos = vals[pos_param]    # (n_chains, 3)
                tgt_rot = vals[rot_param]    # (n_chains, 9)
                errs = []
                for ci in range(n_chains):
                    ee_pos, ee_rot = get_ee_pose_per_chain[ci](
                        apc[ci], bpos, brot)
                    target_rot = tgt_rot[ci].reshape(3, 3)
                    pose_err = pose_error_log(
                        ee_pos, ee_rot, tgt_pos[ci], target_rot)
                    errs.append(pos_w * pose_err[:3])
                    errs.append(rot_w * pose_err[3:])
                return jnp.concatenate(errs).flatten()

            cost_node = cost_pose(
                TrajectoryVar(jnp.arange(T)),
                MEEPosParamVar(jnp.arange(T)),
                MEERotParamVar(jnp.arange(T)),
            )
        else:
            @jaxls.Cost.factory(name='multi_ee_waypoint')
            def cost_pos_only(vals, var, pos_param):
                aug = vals[var]
                apc, bpos, brot = _split_aug(aug)
                tgt_pos = vals[pos_param]    # (n_chains, 3)
                errs = []
                for ci in range(n_chains):
                    ee_pos, _ = get_ee_pose_per_chain[ci](
                        apc[ci], bpos, brot)
                    errs.append(pos_w * (ee_pos - tgt_pos[ci]))
                return jnp.concatenate(errs).flatten()

            cost_node = cost_pos_only(
                TrajectoryVar(jnp.arange(T)),
                MEEPosParamVar(jnp.arange(T)),
            )

        # Stash so solve() can populate values.
        if not hasattr(problem, '_multi_ee_param_vars'):
            problem._multi_ee_param_vars = []
        problem._multi_ee_param_vars.append({
            'pos_var_class': MEEPosParamVar,
            'rot_var_class': MEERotParamVar,
            'target_pos': jnp.asarray(T_pos, dtype=jnp.float64),
            'target_rot': (jnp.asarray(T_rot, dtype=jnp.float64)
                            if T_rot is not None else None),
        })

        return cost_node
