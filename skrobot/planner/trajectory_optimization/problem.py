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
    ):
        """Add world collision avoidance cost.

        Parameters
        ----------
        collision_link_list : list
            Links to check for collisions.
        world_obstacles : list
            List of obstacle dicts with 'type', 'center', 'radius'.
        weight : float
            Cost weight.
        activation_distance : float
            Distance below which collision cost activates.
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

        self.residuals.append(ResidualSpec(
            name='world_collision',
            residual_fn='world_collision',
            params={
                'obstacles': world_obstacles,
                'activation_distance': activation_distance,
            },
            kind='soft',
            weight=weight,
        ))

    def add_self_collision_cost(
        self,
        weight=100.0,
        activation_distance=0.02,
    ):
        """Add self-collision avoidance cost.

        Parameters
        ----------
        weight : float
            Cost weight.
        activation_distance : float
            Distance below which collision cost activates.
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

        self.residuals.append(ResidualSpec(
            name='self_collision',
            residual_fn='self_collision',
            params={
                'pair_indices': self.self_collision_pairs,
                'activation_distance': activation_distance,
            },
            kind='soft',
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
