"""Backend-agnostic trajectory optimization problem definition.

This module provides a TrajectoryProblem class that describes the
optimization problem without coupling to any specific solver.
"""

from typing import List

import numpy as np

from skrobot.planner.trajectory_optimization.residuals import ResidualSpec


class TrajectoryProblem:
    """Trajectory optimization problem definition.

    This class collects all the information needed to solve a trajectory
    optimization problem, without being tied to any specific solver.

    Attributes
    ----------
    robot_model : RobotModel
        Robot model for kinematics.
    link_list : list
        Links in the kinematic chain.
    n_waypoints : int
        Number of waypoints in trajectory.
    n_joints : int
        Number of joints.
    dt : float
        Time step between waypoints.
    initial_trajectory : ndarray
        Initial trajectory guess (n_waypoints, n_joints).
    residuals : list
        List of ResidualSpec objects defining the problem.
    """

    def __init__(
        self,
        robot_model,
        link_list,
        n_waypoints,
        dt=0.1,
        move_target=None,
    ):
        """Initialize trajectory problem.

        Parameters
        ----------
        robot_model : RobotModel
            Robot model.
        link_list : list
            Links in the kinematic chain.
        n_waypoints : int
            Number of waypoints.
        dt : float
            Time step between waypoints.
        move_target : CascadedCoords, optional
            End-effector coordinates for pose tracking.
        """
        self.robot_model = robot_model
        self.link_list = link_list
        self.n_waypoints = n_waypoints
        self.dt = dt
        self.move_target = move_target

        self.joint_list = [link.joint for link in link_list]
        self.n_joints = len(self.joint_list)

        # Extract joint limits
        self.joint_limits_lower = np.array([
            j.min_angle if j.min_angle is not None else -np.pi
            for j in self.joint_list
        ])
        self.joint_limits_upper = np.array([
            j.max_angle if j.max_angle is not None else np.pi
            for j in self.joint_list
        ])

        # Residual specifications
        self.residuals: List[ResidualSpec] = []

        # Collision parameters (populated by add_collision_cost)
        self.collision_link_list = None
        self.collision_spheres = None
        self.world_obstacles = []
        self.self_collision_pairs = []

        # FK parameters (lazily computed)
        self._fk_params = None

        # Fixed waypoints
        self.fixed_start = True
        self.fixed_end = True

        # Intermediate waypoint constraints: list of (index, angles)
        self.waypoint_constraints = []

        # End-effector waypoint costs: list of dicts
        self.ee_waypoint_costs = []

    @property
    def fk_params(self):
        """Get FK parameters (lazily computed)."""
        if self._fk_params is None:
            from skrobot.kinematics.differentiable import extract_fk_parameters
            self._fk_params = extract_fk_parameters(
                self.robot_model, self.link_list,
                self.move_target or self.robot_model
            )
        return self._fk_params

    def add_smoothness_cost(self, weight=1.0):
        """Add smoothness cost (minimize velocity between waypoints).

        Parameters
        ----------
        weight : float
            Cost weight.
        """
        self.residuals.append(ResidualSpec(
            name='smoothness',
            residual_fn='smoothness',
            params={'weight': weight},
            kind='soft',
            weight=weight,
        ))

    def add_acceleration_cost(self, weight=1.0):
        """Add acceleration minimization cost.

        Parameters
        ----------
        weight : float
            Cost weight.
        """
        self.residuals.append(ResidualSpec(
            name='acceleration',
            residual_fn='acceleration',
            params={'weight': weight, 'dt': self.dt},
            kind='soft',
            weight=weight,
        ))

    def add_posture_cost(self, nominal_angles, weight=0.1):
        """Add posture regularization cost.

        Penalizes deviation from a nominal set of joint angles.
        This encourages the robot to stay close to a comfortable
        pose, avoiding unnecessary large joint movements and
        producing more natural-looking trajectories.

        Parameters
        ----------
        nominal_angles : array-like
            Target nominal joint angles (n_joints,).
        weight : float
            Cost weight.
        """
        nominal_angles = np.array(nominal_angles)
        self.residuals.append(ResidualSpec(
            name='posture',
            residual_fn='posture',
            params={
                'nominal_angles': nominal_angles,
            },
            kind='soft',
            weight=weight,
        ))

    def add_jerk_cost(self, weight=0.1):
        """Add jerk minimization cost.

        Parameters
        ----------
        weight : float
            Cost weight.
        """
        self.residuals.append(ResidualSpec(
            name='jerk',
            residual_fn='jerk',
            params={'weight': weight, 'dt': self.dt},
            kind='soft',
            weight=weight,
        ))

    def add_smooth_trajectory_costs(
        self,
        weight=1.0,
        use_high_precision=True,
        velocity_weight_scale=1.0,
        acceleration_weight_scale=0.5,
        jerk_weight_scale=0.1,
    ):
        """Add costs for generating smooth trajectories.

        This is a convenience method that automatically selects the
        appropriate smoothness costs based on the number of waypoints.
        When high precision is enabled and sufficient waypoints are
        available, it uses 5-point/7-point stencils for more accurate
        derivative computation.

        Parameters
        ----------
        weight : float
            Base weight for smoothness costs.
        use_high_precision : bool
            If True, use 5-point/7-point stencils when possible.
            If False, always use simple finite differences.
        velocity_weight_scale : float
            Scale factor for velocity cost relative to base weight.
        acceleration_weight_scale : float
            Scale factor for acceleration cost relative to base weight.
        jerk_weight_scale : float
            Scale factor for jerk cost relative to base weight.

        Notes
        -----
        The method selects costs based on waypoint count:

        - n_waypoints >= 7 and high_precision:
            Uses 5-point velocity, 5-point acceleration, 7-point jerk
        - n_waypoints >= 5 and high_precision:
            Uses 5-point velocity, 5-point acceleration
        - Otherwise:
            Uses simple smoothness (velocity) and 3-point acceleration
        """
        velocity_w = weight * velocity_weight_scale
        acceleration_w = weight * acceleration_weight_scale
        jerk_w = weight * jerk_weight_scale

        if use_high_precision and self.n_waypoints >= 7:
            self.add_five_point_velocity_cost(weight=velocity_w)
            self.add_five_point_acceleration_cost(weight=acceleration_w)
            self.add_five_point_jerk_cost(weight=jerk_w)
        elif use_high_precision and self.n_waypoints >= 5:
            self.add_five_point_velocity_cost(weight=velocity_w)
            self.add_five_point_acceleration_cost(weight=acceleration_w)
        else:
            self.add_smoothness_cost(weight=velocity_w)
            self.add_acceleration_cost(weight=acceleration_w)

    def add_five_point_velocity_cost(self, weight=1.0, velocity_limits=None):
        """Add velocity cost using 5-point stencil for higher accuracy.

        The 5-point stencil computes velocity with O(h^4) accuracy:
            v = (-q[t+2] + 8*q[t+1] - 8*q[t-1] + q[t-2]) / (12*dt)

        This method requires at least 5 waypoints and applies to
        waypoints [2, n_waypoints-2].

        Parameters
        ----------
        weight : float
            Cost weight.
        velocity_limits : array-like, optional
            Maximum velocity for each joint. If None, uses joint velocity
            limits from the robot model.
        """
        if self.n_waypoints < 5:
            raise ValueError(
                "5-point stencil requires at least 5 waypoints, "
                f"got {self.n_waypoints}"
            )
        if velocity_limits is None:
            velocity_limits = np.array([
                j.max_joint_velocity for j in self.joint_list
            ])
        else:
            velocity_limits = np.array(velocity_limits)
        self.residuals.append(ResidualSpec(
            name='five_point_velocity',
            residual_fn='five_point_velocity',
            params={
                'dt': self.dt,
                'velocity_limits': velocity_limits,
            },
            kind='soft',
            weight=weight,
        ))

    def add_five_point_acceleration_cost(self, weight=1.0):
        """Add acceleration cost using 5-point stencil for higher accuracy.

        The 5-point stencil computes acceleration with O(h^4) accuracy:
            a = (-q[t+2] + 16*q[t+1] - 30*q[t] + 16*q[t-1] - q[t-2]) / (12*dt^2)

        This method requires at least 5 waypoints and applies to
        waypoints [2, n_waypoints-2].

        Parameters
        ----------
        weight : float
            Cost weight.
        """
        if self.n_waypoints < 5:
            raise ValueError(
                "5-point stencil requires at least 5 waypoints, "
                f"got {self.n_waypoints}"
            )
        self.residuals.append(ResidualSpec(
            name='five_point_acceleration',
            residual_fn='five_point_acceleration',
            params={'dt': self.dt},
            kind='soft',
            weight=weight,
        ))

    def add_five_point_jerk_cost(self, weight=0.1):
        """Add jerk cost using 7-point stencil for higher accuracy.

        The 7-point stencil computes jerk with O(h^4) accuracy:
            j = (-q[t+3] + 8*q[t+2] - 13*q[t+1] + 13*q[t-1] - 8*q[t-2] + q[t-3])
                / (8*dt^3)

        This method requires at least 7 waypoints and applies to
        waypoints [3, n_waypoints-3].

        Parameters
        ----------
        weight : float
            Cost weight.
        """
        if self.n_waypoints < 7:
            raise ValueError(
                "7-point stencil for jerk requires at least 7 waypoints, "
                f"got {self.n_waypoints}"
            )
        self.residuals.append(ResidualSpec(
            name='five_point_jerk',
            residual_fn='five_point_jerk',
            params={'dt': self.dt},
            kind='soft',
            weight=weight,
        ))

    def add_acceleration_limit(self, acceleration_limit, weight=1.0):
        """Add acceleration limit constraint using 5-point stencil.

        Penalizes accelerations that exceed the specified limit.

        Parameters
        ----------
        acceleration_limit : float or array-like
            Maximum acceleration for each joint. If scalar, applies to
            all joints.
        weight : float
            Cost weight.
        """
        if self.n_waypoints < 5:
            raise ValueError(
                "5-point stencil requires at least 5 waypoints, "
                f"got {self.n_waypoints}"
            )
        if np.isscalar(acceleration_limit):
            acceleration_limit = np.full(self.n_joints, acceleration_limit)
        else:
            acceleration_limit = np.array(acceleration_limit)
        self.residuals.append(ResidualSpec(
            name='acceleration_limit',
            residual_fn='acceleration_limit',
            params={
                'dt': self.dt,
                'acceleration_limit': acceleration_limit,
            },
            kind='soft',
            weight=weight,
        ))

    def add_jerk_limit(self, jerk_limit, weight=0.1):
        """Add jerk limit constraint using 7-point stencil.

        Penalizes jerks that exceed the specified limit.

        Parameters
        ----------
        jerk_limit : float or array-like
            Maximum jerk for each joint. If scalar, applies to all joints.
        weight : float
            Cost weight.
        """
        if self.n_waypoints < 7:
            raise ValueError(
                "7-point stencil for jerk requires at least 7 waypoints, "
                f"got {self.n_waypoints}"
            )
        if np.isscalar(jerk_limit):
            jerk_limit = np.full(self.n_joints, jerk_limit)
        else:
            jerk_limit = np.array(jerk_limit)
        self.residuals.append(ResidualSpec(
            name='jerk_limit',
            residual_fn='jerk_limit',
            params={
                'dt': self.dt,
                'jerk_limit': jerk_limit,
            },
            kind='soft',
            weight=weight,
        ))

    def add_joint_limit_constraint(self):
        """Add joint limit constraints."""
        self.residuals.append(ResidualSpec(
            name='joint_limits',
            residual_fn='joint_limits',
            params={
                'lower': self.joint_limits_lower,
                'upper': self.joint_limits_upper,
            },
            kind='geq',
        ))

    def add_collision_cost(
        self,
        collision_link_list,
        world_obstacles,
        weight=100.0,
        activation_distance=0.05,
        as_constraint=True,
    ):
        """Add world collision avoidance cost.

        Parameters
        ----------
        collision_link_list : list
            Links to check for collisions.
        world_obstacles : list
            List of obstacle dicts with 'type', 'center', 'radius'.
        weight : float
            Cost weight (only used when as_constraint=False).
        activation_distance : float
            Distance below which collision cost activates.
        as_constraint : bool
            If True (default), treat as hard constraint for Augmented Lagrangian
            solver (collision distance >= 0). If False, treat as soft cost.
        """
        self.collision_link_list = collision_link_list
        self.world_obstacles = world_obstacles

        # Extract collision spheres
        from skrobot.planner.trajectory_optimization.collision import extract_collision_spheres
        self.collision_spheres = extract_collision_spheres(
            self.robot_model, collision_link_list, n_spheres_per_link=3
        )

        # Compute collision link offsets
        self._compute_collision_link_offsets()

        # Use 'geq' for hard constraint (Augmented Lagrangian)
        # Use 'soft' for soft cost (gradient descent, etc.)
        kind = 'geq' if as_constraint else 'soft'

        self.residuals.append(ResidualSpec(
            name='world_collision',
            residual_fn='world_collision',
            params={
                'obstacles': world_obstacles,
                'activation_distance': activation_distance,
            },
            kind=kind,
            weight=weight,
        ))

    def add_self_collision_cost(
        self,
        weight=100.0,
        activation_distance=0.02,
        as_constraint=True,
    ):
        """Add self-collision avoidance cost.

        Parameters
        ----------
        weight : float
            Cost weight (only used when as_constraint=False).
        activation_distance : float
            Distance below which collision cost activates.
        as_constraint : bool
            If True (default), treat as hard constraint for Augmented Lagrangian
            solver (collision distance >= 0). If False, treat as soft cost.
        """
        if self.collision_link_list is None:
            raise ValueError(
                "Must call add_collision_cost first to set collision_link_list"
            )

        # Create self-collision pairs
        from skrobot.planner.trajectory_optimization.collision import create_self_collision_pairs
        link_pairs = create_self_collision_pairs(
            self.collision_link_list, ignore_adjacent=True
        )

        # Build sphere pair indices
        collision_link_indices = self.collision_spheres['link_indices']
        n_spheres = len(collision_link_indices)
        pairs_i = []
        pairs_j = []

        for link_i, link_j in link_pairs:
            for si in range(n_spheres):
                if collision_link_indices[si] != link_i:
                    continue
                for sj in range(n_spheres):
                    if collision_link_indices[sj] != link_j:
                        continue
                    pairs_i.append(si)
                    pairs_j.append(sj)

        self.self_collision_pairs = (np.array(pairs_i), np.array(pairs_j))

        # Use 'geq' for hard constraint (Augmented Lagrangian)
        # Use 'soft' for soft cost (gradient descent, etc.)
        kind = 'geq' if as_constraint else 'soft'

        self.residuals.append(ResidualSpec(
            name='self_collision',
            residual_fn='self_collision',
            params={
                'pair_indices': self.self_collision_pairs,
                'activation_distance': activation_distance,
            },
            kind=kind,
            weight=weight,
        ))

    def _compute_collision_link_offsets(self):
        """Compute offsets from kinematic chain links to collision links."""
        link_to_idx = {link: idx for idx, link in enumerate(self.link_list)}
        self.collision_link_to_chain_idx = []
        self.collision_link_offsets_pos = []
        self.collision_link_offsets_rot = []

        for link in self.collision_link_list:
            if link in link_to_idx:
                self.collision_link_to_chain_idx.append(link_to_idx[link])
                self.collision_link_offsets_pos.append(np.zeros(3))
                self.collision_link_offsets_rot.append(np.eye(3))
            else:
                # Find parent in kinematic chain
                parent = link.parent_link
                while parent is not None and parent not in link_to_idx:
                    parent = parent.parent_link

                if parent is not None:
                    self.collision_link_to_chain_idx.append(link_to_idx[parent])
                    parent_coords = parent.worldcoords()
                    link_coords = link.worldcoords()
                    rel_pos = parent_coords.inverse_transform_vector(
                        link_coords.worldpos()
                    )
                    rel_rot = parent_coords.worldrot().T @ link_coords.worldrot()
                    self.collision_link_offsets_pos.append(rel_pos)
                    self.collision_link_offsets_rot.append(rel_rot)
                else:
                    self.collision_link_to_chain_idx.append(0)
                    self.collision_link_offsets_pos.append(np.zeros(3))
                    self.collision_link_offsets_rot.append(np.eye(3))

        self.collision_link_to_chain_idx = np.array(self.collision_link_to_chain_idx)
        self.collision_link_offsets_pos = np.array(self.collision_link_offsets_pos)
        self.collision_link_offsets_rot = np.array(self.collision_link_offsets_rot)

    def add_pose_cost(
        self,
        target_positions,
        target_rotations,
        position_weight=10.0,
        rotation_weight=1.0,
    ):
        """Add end-effector pose tracking cost.

        Parameters
        ----------
        target_positions : ndarray
            Target positions (n_waypoints, 3).
        target_rotations : ndarray
            Target rotation matrices (n_waypoints, 3, 3).
        position_weight : float
            Position tracking weight.
        rotation_weight : float
            Rotation tracking weight.
        """
        self.residuals.append(ResidualSpec(
            name='pose',
            residual_fn='pose',
            params={
                'target_positions': target_positions,
                'target_rotations': target_rotations,
                'position_weight': position_weight,
                'rotation_weight': rotation_weight,
            },
            kind='soft',
            weight=1.0,  # Weights are in params
        ))

    def add_joint_velocity_limit(self, scale=1.0):
        """Add joint velocity limit constraint.

        Constrains ``|q[t+1] - q[t]| / dt <= max_joint_velocity * scale``
        for every consecutive pair of waypoints and every joint.

        Parameters
        ----------
        scale : float
            Fraction of maximum joint velocity to allow (0, 1].
            For example, 0.8 uses 80 % of each joint's velocity limit.
        """
        max_velocities = np.array([
            j.max_joint_velocity for j in self.joint_list
        ])
        self.residuals.append(ResidualSpec(
            name='joint_velocity_limit',
            residual_fn='joint_velocity_limit',
            params={
                'max_velocities': max_velocities * scale,
                'dt': self.dt,
            },
            kind='geq',
            weight=1.0,
        ))

    def add_cartesian_path_cost(
        self,
        target_positions,
        target_rotations=None,
        weight=10.0,
        rotation_weight=1.0,
    ):
        """Add end-effector pose tracking cost for Cartesian path.

        Penalizes deviation of the end-effector pose from the target
        poses at each trajectory waypoint, encouraging the end-effector
        to follow a straight line in Cartesian space with smooth rotation.

        Parameters
        ----------
        target_positions : ndarray
            Target EE positions (n_waypoints, 3).
        target_rotations : ndarray, optional
            Target EE rotation matrices (n_waypoints, 3, 3).
            If None, only position is tracked.
        weight : float
            Position tracking weight.
        rotation_weight : float
            Rotation tracking weight relative to position weight.
        """
        self.residuals.append(ResidualSpec(
            name='cartesian_path',
            residual_fn='cartesian_path',
            params={
                'target_positions': target_positions,
                'target_rotations': target_rotations,
                'rotation_weight': rotation_weight,
            },
            kind='soft',
            weight=weight,
        ))

    def set_fixed_endpoints(self, start=True, end=True):
        """Set whether to fix start and end waypoints.

        Parameters
        ----------
        start : bool
            Fix start waypoint.
        end : bool
            Fix end waypoint.
        """
        self.fixed_start = start
        self.fixed_end = end

    def add_waypoint_constraint(self, waypoint_index, joint_angles):
        """Pin a specific trajectory waypoint to given joint angles.

        Parameters
        ----------
        waypoint_index : int
            Index in the trajectory to pin.
        joint_angles : array-like
            Joint angles to enforce at this index.
        """
        self.waypoint_constraints.append(
            (waypoint_index, np.array(joint_angles))
        )

    def add_ee_waypoint_cost(
        self,
        waypoint_index,
        target_position,
        target_rotation,
        position_weight=100.0,
        rotation_weight=10.0,
    ):
        """Constrain end-effector pose at a specific trajectory waypoint.

        Unlike ``add_waypoint_constraint`` which fixes all joint angles,
        this only constrains the end-effector pose, leaving the optimizer
        free to choose joint configurations. Combined with posture
        regularization, this produces more natural robot motions.

        Parameters
        ----------
        waypoint_index : int
            Index in the trajectory to constrain.
        target_position : array-like
            Target EE position (3,).
        target_rotation : array-like
            Target EE rotation matrix (3, 3).
        position_weight : float
            Position tracking weight.
        rotation_weight : float
            Rotation tracking weight.
        """
        self.ee_waypoint_costs.append({
            'waypoint_index': waypoint_index,
            'target_position': np.array(target_position),
            'target_rotation': np.array(target_rotation),
            'position_weight': position_weight,
            'rotation_weight': rotation_weight,
        })

    def to_dict(self):
        """Export problem to dictionary for serialization."""
        return {
            'n_waypoints': self.n_waypoints,
            'n_joints': self.n_joints,
            'dt': self.dt,
            'joint_limits_lower': self.joint_limits_lower.tolist(),
            'joint_limits_upper': self.joint_limits_upper.tolist(),
            'residuals': [
                {
                    'name': r.name,
                    'kind': r.kind,
                    'weight': r.weight,
                }
                for r in self.residuals
            ],
            'fixed_start': self.fixed_start,
            'fixed_end': self.fixed_end,
        }
