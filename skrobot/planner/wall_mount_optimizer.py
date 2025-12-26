"""Wall-mounted robot base optimization.

This module provides optimization for determining the optimal base configuration
of a wall-mounted robot with vacuum suction and support protrusions.
"""

import numpy as np
import scipy.optimize

from skrobot.coordinates import Coordinates
from skrobot.planner.constraint_ik import FaceTarget


class WallMountedRobotModel:
    """Model of a robot mounted on a wall via suction cup and protrusions.

    Parameters
    ----------
    robot : RobotModel
        Robot model instance (e.g., Panda)
    wall : FaceTarget
        Wall surface definition
    base_link : Link, optional
        Base link of the robot. If None, uses robot.root_link
    end_coords : Coordinates, optional
        End-effector coordinates. If None, uses robot's default

    Attributes
    ----------
    base_position_local : np.ndarray
        Position of robot base on wall in local (x, y) coordinates
    suction_radius : float
        Radius of suction cup in meters
    protrusion_positions : np.ndarray
        4x2 array of protrusion positions relative to suction center
    """

    def __init__(self, robot, wall, base_link=None, end_coords=None):
        self.robot = robot
        self.wall = wall
        self.base_link = base_link if base_link else robot.root_link
        self.end_coords = end_coords

        # Default values
        self.base_position_local = np.array([0.0, 0.0])
        self.suction_radius = 0.05  # 5cm default
        # Initial square configuration
        self.protrusion_positions = np.array([
            [-0.1, -0.1],
            [0.1, -0.1],
            [0.1, 0.1],
            [-0.1, 0.1]
        ])

        # Store original robot pose
        self._original_base_coords = self.base_link.copy_worldcoords()

    def set_base_position(self, local_xy):
        """Set robot base position on wall.

        Parameters
        ----------
        local_xy : np.ndarray
            2D position on wall in local coordinates (relative to wall center)
        """
        self.base_position_local = np.array(local_xy)
        world_pos = self._local_to_world(local_xy)

        # Compute rotation to align robot base with wall normal
        # Robot's original z-axis should point along wall normal
        rot = self._compute_wall_aligned_rotation()

        new_coords = Coordinates(pos=world_pos, rot=rot)
        self.base_link.newcoords(new_coords)

    def _local_to_world(self, local_xy):
        """Convert local wall coordinates to world coordinates."""
        return (self.wall.center
                + local_xy[0] * self.wall.x_axis
                + local_xy[1] * self.wall.y_axis)

    def _compute_wall_aligned_rotation(self):
        """Compute rotation matrix to align robot base with wall."""
        # Wall normal points away from wall
        # Robot should be mounted with its arm extending away from wall
        # So robot's z-axis (arm direction) should align with wall normal

        # Build rotation matrix: robot z points along wall normal (away from wall)
        z_axis = self.wall.normal  # Away from the wall
        # Choose x_axis from wall's local axes
        x_axis = self.wall.x_axis
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        rot = np.column_stack([x_axis, y_axis, z_axis])
        return rot

    def get_suction_position_world(self):
        """Get suction cup position in world coordinates."""
        return self._local_to_world(self.base_position_local)

    def get_protrusion_positions_world(self):
        """Get protrusion positions in world coordinates.

        Returns
        -------
        positions : np.ndarray
            4x3 array of protrusion positions in world coordinates
        """
        positions = np.zeros((4, 3))
        for i, local_pos in enumerate(self.protrusion_positions):
            # Protrusion position relative to wall center
            wall_local = self.base_position_local + local_pos
            positions[i] = self._local_to_world(wall_local)
        return positions

    def reset_robot_pose(self):
        """Reset robot to original pose."""
        self.base_link.newcoords(self._original_base_coords)


def compute_gravity_moment(robot, joint_list, suction_position,
                           wall_normal, gravity=np.array([0, 0, -9.81]),
                           payload_mass=0.5, robot_mass=None):
    """Compute gravity-induced moment about the suction cup.

    Parameters
    ----------
    robot : RobotModel
        Robot model in current configuration
    joint_list : list
        List of joints to consider for mass
    suction_position : np.ndarray
        Suction cup position in world coordinates
    wall_normal : np.ndarray
        Wall normal vector (pointing away from wall)
    gravity : np.ndarray
        Gravity vector
    payload_mass : float
        End-effector payload mass in kg
    robot_mass : float, optional
        Total robot mass in kg. If None, uses mass from URDF or default (18kg for Panda).

    Returns
    -------
    moment : np.ndarray
        Moment vector about suction cup (3D)
    total_force : np.ndarray
        Total gravity force (3D)
    com : np.ndarray
        Center of mass position (3D)
    """
    # Panda link masses from official specs (kg)
    # These are approximate values based on Franka documentation
    PANDA_LINK_MASSES = {
        'panda_link0': 0.629,
        'panda_link1': 4.970,
        'panda_link2': 0.646,
        'panda_link3': 3.228,
        'panda_link4': 3.587,
        'panda_link5': 1.225,
        'panda_link6': 1.666,
        'panda_link7': 0.735,
        'panda_hand': 0.730,
        'panda_leftfinger': 0.015,
        'panda_rightfinger': 0.015,
    }

    # Collect all links and their masses
    total_mass = 0.0
    com_sum = np.zeros(3)

    for link in robot.link_list:
        # Try to get mass from URDF, otherwise use known values for Panda
        mass = getattr(link, 'mass', 0)
        if mass <= 0.01:  # If mass is not set properly (placeholder value)
            mass = PANDA_LINK_MASSES.get(link.name, 0)

        if mass > 0:
            # Get centroid in world coordinates
            centroid = getattr(link, 'centroid', None)
            if centroid is not None and np.linalg.norm(centroid) > 1e-6:
                # Transform centroid from local to world coordinates
                pos = link.worldpos() + link.worldrot().dot(centroid)
            else:
                # Use link position as approximation
                pos = link.worldpos()

            total_mass += mass
            com_sum += mass * pos

    # Add payload at end-effector
    if payload_mass > 0 and hasattr(robot, 'rarm_end_coords'):
        ee_pos = robot.rarm_end_coords.worldpos()
        total_mass += payload_mass
        com_sum += payload_mass * ee_pos

    # Override with user-specified mass if provided
    if robot_mass is not None:
        if total_mass > 0:
            # Scale COM to match specified mass
            com = com_sum / total_mass
        else:
            com = robot.rarm_end_coords.worldpos() if hasattr(robot, 'rarm_end_coords') else suction_position
        total_mass = robot_mass + payload_mass
    elif total_mass > 0:
        com = com_sum / total_mass
    else:
        # Fallback: estimate mass from Panda specs (~18kg)
        total_mass = 18.0 + payload_mass
        com = robot.rarm_end_coords.worldpos() if hasattr(robot, 'rarm_end_coords') else suction_position

    # Compute gravity force and moment
    gravity_force = total_mass * gravity
    r = com - suction_position
    moment = np.cross(r, gravity_force)

    return moment, gravity_force, com


def solve_protrusion_forces(protrusion_positions, suction_position,
                            moment, gravity_force, wall_normal,
                            suction_force_magnitude,
                            friction_coeff=0.6):
    """Solve for protrusion forces that balance gravity moment.

    For a wall-mounted robot:
    - Gravity acts downward (typically z-direction)
    - Suction pulls robot toward wall (normal direction)
    - Protrusions push against wall (normal direction)
    - Gravity is resisted by friction at suction cup and protrusions
    - Moment is balanced by differential normal forces on protrusions

    The protrusions can only resist moment components that are perpendicular
    to the wall normal. The moment component parallel to the wall normal
    (roll about the normal) must be resisted by friction at contact points.

    Parameters
    ----------
    protrusion_positions : np.ndarray
        4x3 array of protrusion positions in world coordinates
    suction_position : np.ndarray
        Suction cup position in world coordinates
    moment : np.ndarray
        Gravity moment about suction cup
    gravity_force : np.ndarray
        Total gravity force
    wall_normal : np.ndarray
        Wall normal (pointing away from wall)
    suction_force_magnitude : float
        Magnitude of suction force (positive value)
    friction_coeff : float
        Friction coefficient at protrusions and suction cup

    Returns
    -------
    forces : np.ndarray
        4x3 array of protrusion forces (normal direction only)
    feasible : bool
        True if solution is feasible (all constraints satisfied)
    info : dict
        Additional information about the solution
    """
    from scipy.optimize import lsq_linear

    r_vecs = protrusion_positions - suction_position  # 4x3

    # Protrusions can only push against the wall (normal direction)
    normal_dir = wall_normal

    # Decompose moment into:
    # 1. Component parallel to wall normal (roll) - cannot be balanced by protrusions
    # 2. Component perpendicular to wall normal (pitch/yaw) - can be balanced
    moment_parallel = np.dot(moment, wall_normal) * wall_normal
    moment_perpendicular = moment - moment_parallel

    # For moment balance about the suction point:
    # Σ r_i × (f_i * normal_dir) = -moment_perpendicular
    #
    # Note: Protrusion forces (in normal direction) create moments perpendicular
    # to the normal, so they can only balance moment_perpendicular

    # Build moment matrix
    M = np.zeros((3, 4))
    for i in range(4):
        M[:, i] = np.cross(r_vecs[i], normal_dir)

    # Target moment (only the perpendicular component)
    target_moment = -moment_perpendicular

    # Solve for protrusion forces (normal component only)
    # We want to minimize |M @ f - target_moment|^2
    # subject to f >= 0 (protrusions can only push)
    result = lsq_linear(M, target_moment, bounds=(0, np.inf))

    forces_magnitude = result.x
    achieved_moment = M @ forces_magnitude
    moment_residual = np.linalg.norm(achieved_moment - target_moment)

    # Compute 3D force vectors
    forces = np.outer(forces_magnitude, normal_dir)  # 4x3

    # Check friction constraints
    # The tangential force (gravity component parallel to wall) must be
    # supported by friction at all contact points
    gravity_tangent = gravity_force - np.dot(gravity_force, wall_normal) * wall_normal
    gravity_tangent_mag = np.linalg.norm(gravity_tangent)

    # Total normal force available for friction
    total_normal_force = suction_force_magnitude + np.sum(forces_magnitude)

    # Maximum tangential force from friction
    max_friction_force = friction_coeff * total_normal_force

    # The roll moment (moment_parallel) creates a tangential force distribution
    # that must also be resisted by friction. This is accounted for in the
    # total tangential load.
    moment_parallel_mag = np.linalg.norm(moment_parallel)

    # Friction must support both gravity and roll moment
    # Roll moment at distance r creates force M/r
    # Approximate using average protrusion distance
    avg_prot_dist = np.mean([np.linalg.norm(r) for r in r_vecs])
    if avg_prot_dist > 0.01:
        roll_tangent_force = moment_parallel_mag / avg_prot_dist
    else:
        roll_tangent_force = 0

    total_tangent_force = gravity_tangent_mag + roll_tangent_force
    friction_ok = total_tangent_force <= max_friction_force

    # Also check that suction force can hold the robot against the wall
    # Sum of protrusion forces (pushing away) vs suction force (pulling in)
    total_protrusion_force = np.sum(forces_magnitude)
    normal_force_margin = suction_force_magnitude - total_protrusion_force

    # Feasibility criteria:
    # 1. Moment residual should be small (moment balanced)
    # 2. Friction constraint satisfied
    # 3. Normal forces non-negative and reasonable
    # 4. Suction can hold against protrusion reaction
    max_reasonable_force = 1000.0  # 1000N per protrusion is reasonable limit
    feasible = (
        moment_residual < 10.0 and  # Allow some residual
        friction_ok and
        np.all(forces_magnitude >= -1e-6) and
        np.all(forces_magnitude < max_reasonable_force) and
        normal_force_margin > -50.0  # Some margin for numerical issues
    )

    info = {
        'forces_magnitude': forces_magnitude,
        'moment_residual': moment_residual,
        'moment_parallel_mag': moment_parallel_mag,
        'moment_perpendicular_mag': np.linalg.norm(moment_perpendicular),
        'friction_ok': friction_ok,
        'gravity_tangent_mag': gravity_tangent_mag,
        'roll_tangent_force': roll_tangent_force,
        'total_tangent_force': total_tangent_force,
        'max_friction_force': max_friction_force,
        'total_normal_force': total_normal_force,
        'normal_force_margin': normal_force_margin,
        'result': result
    }

    return forces, feasible, info


def check_ik_feasibility(robot, base_position_local, wall, target_poses,
                         link_list=None, end_coords=None):
    """Check if all target poses are reachable from the given base position.

    Parameters
    ----------
    robot : RobotModel
        Robot model
    base_position_local : np.ndarray
        Base position on wall in local coordinates
    wall : FaceTarget
        Wall surface
    target_poses : list
        List of target pose dicts with 'coords', 'translation_axis', 'rotation_axis'
    link_list : list, optional
        Link list for IK
    end_coords : Coordinates, optional
        End-effector coordinates

    Returns
    -------
    feasible : bool
        True if all poses are reachable
    results : list
        IK results for each pose
    """
    if end_coords is None:
        end_coords = robot.rarm_end_coords
    if link_list is None:
        link_list = [
            robot.panda_link1,
            robot.panda_link2,
            robot.panda_link3,
            robot.panda_link4,
            robot.panda_link5,
            robot.panda_link6,
            robot.panda_link7,
        ]

    # Create wall-mounted model and set base position
    model = WallMountedRobotModel(robot, wall, end_coords=end_coords)
    model.set_base_position(base_position_local)

    results = []
    all_feasible = True

    for target in target_poses:
        coords = target['coords']
        trans_axis = target.get('translation_axis', True)
        rot_axis = target.get('rotation_axis', True)

        # Try IK
        result = robot.inverse_kinematics(
            coords,
            move_target=end_coords,
            link_list=link_list,
            rotation_axis=rot_axis,
            translation_axis=trans_axis,
            stop=50,
            thre=0.01,
            rthre=np.deg2rad(3.0)
        )

        success = result is not False and result is not None
        results.append({
            'success': success,
            'target': coords,
            'joint_angles': robot.angle_vector().copy() if success else None
        })

        if not success:
            all_feasible = False

    model.reset_robot_pose()
    return all_feasible, results


def compute_quadrilateral_area(points):
    """Compute area of quadrilateral given 4 2D points.

    Uses shoelace formula for arbitrary quadrilateral.

    Parameters
    ----------
    points : np.ndarray
        4x2 array of 2D points

    Returns
    -------
    area : float
        Area of quadrilateral
    """
    # Order points in convex hull order first
    ordered = order_points_convex(points)
    # Use shoelace formula
    n = len(ordered)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += ordered[i, 0] * ordered[j, 1]
        area -= ordered[j, 0] * ordered[i, 1]
    return abs(area) / 2.0


def order_points_convex(points):
    """Order 4 points in counter-clockwise convex hull order.

    Parameters
    ----------
    points : np.ndarray
        4x2 array of 2D points

    Returns
    -------
    ordered : np.ndarray
        4x2 array of points in counter-clockwise order
    """
    # Find centroid
    centroid = np.mean(points, axis=0)

    # Sort by angle from centroid
    angles = np.arctan2(points[:, 1] - centroid[1],
                        points[:, 0] - centroid[0])
    order = np.argsort(angles)
    return points[order]


def is_convex_quadrilateral(points):
    """Check if 4 points form a convex quadrilateral.

    Parameters
    ----------
    points : np.ndarray
        4x2 array of 2D points

    Returns
    -------
    convex : bool
        True if points form a convex quadrilateral
    """
    # First order points by angle from centroid
    ordered = order_points_convex(points)
    n = len(ordered)
    sign = None
    for i in range(n):
        p0 = ordered[i]
        p1 = ordered[(i + 1) % n]
        p2 = ordered[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def optimize_wall_mount_base(robot, wall, target_poses,
                             vacuum_pressure=40000,
                             friction_coeff=0.6,
                             gravity=np.array([0, 0, -9.81]),
                             payload_mass=0.5,
                             robot_mass=None,
                             initial_base_position=None,
                             initial_suction_radius=0.15,
                             initial_protrusion_size=0.30,
                             optimize_suction_radius=True,
                             min_suction_radius=0.12,
                             max_suction_radius=0.30,
                             min_protrusion_distance=0.08,
                             min_protrusion_area=0.04,
                             symmetric_protrusions=False,
                             max_iterations=200,
                             verbose=False,
                             callback=None):
    """Optimize wall-mounted robot base configuration.

    Parameters
    ----------
    robot : RobotModel
        Robot model (e.g., Panda)
    wall : FaceTarget
        Wall surface definition
    target_poses : list
        List of 4 target pose dicts with 'coords', 'translation_axis', 'rotation_axis'
    vacuum_pressure : float
        Vacuum pressure in Pa (default: 40000 = -40kPa)
    friction_coeff : float
        Friction coefficient at protrusions
    gravity : np.ndarray
        Gravity vector
    payload_mass : float
        End-effector payload mass in kg
    robot_mass : float, optional
        Total robot mass in kg. If None, uses default Panda mass (~18kg).
    initial_base_position : np.ndarray, optional
        Initial base position guess (2D local coords)
    initial_suction_radius : float
        Initial suction cup radius
    initial_protrusion_size : float
        Initial size of square protrusion pattern
    optimize_suction_radius : bool
        If True, include suction radius in optimization
    min_suction_radius : float
        Minimum suction cup radius
    max_suction_radius : float
        Maximum suction cup radius
    min_protrusion_distance : float
        Minimum distance between protrusions
    min_protrusion_area : float
        Minimum area of protrusion quadrilateral in m^2 (default: 0.005 = 50 cm^2)
    symmetric_protrusions : bool
        If True, constrain protrusions to form a symmetric rectangle.
        Only 2 parameters (half_width_x, half_width_y) are optimized instead of 8.
    max_iterations : int
        Maximum optimization iterations
    verbose : bool
        Print optimization progress
    callback : callable, optional
        Callback function called after each iteration. Signature:
        callback(iteration, base_pos, suction_radius, protrusion_positions, area, info)
        where info is a dict with additional optimization state.

    Returns
    -------
    result : dict
        Optimization result containing:
        - 'success': bool
        - 'base_position': base position on wall (2D)
        - 'suction_radius': optimized suction radius
        - 'protrusion_positions': 4x2 protrusion positions
        - 'base_area': area of protrusion quadrilateral
        - 'max_protrusion_force': maximum force on any protrusion
        - 'suction_force': suction force
    """
    if len(target_poses) != 4:
        raise ValueError("Exactly 4 target poses required")

    # Get robot components
    if hasattr(robot, 'rarm_end_coords'):
        end_coords = robot.rarm_end_coords
    else:
        raise ValueError("Robot must have rarm_end_coords")

    link_list = [
        robot.panda_link1,
        robot.panda_link2,
        robot.panda_link3,
        robot.panda_link4,
        robot.panda_link5,
        robot.panda_link6,
        robot.panda_link7,
    ]
    joint_list = robot.joint_list_from_link_list(link_list, ignore_fixed_joint=True)

    # Initial guess
    if initial_base_position is None:
        # Compute initial base position based on target poses
        # Project targets to wall's local coordinates
        target_positions = np.array([t['coords'].worldpos() for t in target_poses])
        target_centroid = np.mean(target_positions, axis=0)

        # Project centroid to wall coordinate system
        # The base should be positioned so the robot can reach the targets
        offset = target_centroid - wall.center
        initial_x = np.dot(offset, wall.x_axis)
        initial_y = np.dot(offset, wall.y_axis)
        initial_base_position = np.array([initial_x, initial_y])

    # Build initial x vector
    s = initial_protrusion_size / 2

    if symmetric_protrusions:
        # Symmetric mode: x = [base_x, base_y, suction_radius (opt), half_w_x, half_w_y]
        # Protrusions at: (-hw_x, -hw_y), (+hw_x, -hw_y), (+hw_x, +hw_y), (-hw_x, +hw_y)
        if optimize_suction_radius:
            x0 = np.concatenate([
                initial_base_position,
                [initial_suction_radius],
                [s, s]  # half_width_x, half_width_y
            ])
            n_base = 3  # base_x, base_y, suction_radius
        else:
            x0 = np.concatenate([
                initial_base_position,
                [s, s]
            ])
            n_base = 2
            fixed_suction_radius = initial_suction_radius

        # Bounds
        base_bounds = [
            (-wall.x_length / 2, wall.x_length / 2),
            (-wall.y_length / 2, wall.y_length / 2),
        ]
        if optimize_suction_radius:
            base_bounds.append((min_suction_radius, max_suction_radius))

        # Symmetric protrusion bounds (half-widths must be positive)
        min_half_width = np.sqrt(min_protrusion_area) / 2
        max_prot_dist = 0.3
        prot_bounds = [(min_half_width, max_prot_dist)] * 2

        bounds = base_bounds + prot_bounds
        n_prot_params = 2
    else:
        # Full mode: x = [base_x, base_y, suction_radius (opt),
        #                 p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y]
        initial_protrusions = np.array([
            [-s, -s],
            [s, -s],
            [s, s],
            [-s, s]
        ])

        if optimize_suction_radius:
            x0 = np.concatenate([
                initial_base_position,
                [initial_suction_radius],
                initial_protrusions.flatten()
            ])
            n_base = 3  # base_x, base_y, suction_radius
        else:
            x0 = np.concatenate([
                initial_base_position,
                initial_protrusions.flatten()
            ])
            n_base = 2  # base_x, base_y
            fixed_suction_radius = initial_suction_radius

        # Bounds
        base_bounds = [
            (-wall.x_length / 2, wall.x_length / 2),
            (-wall.y_length / 2, wall.y_length / 2),
        ]
        if optimize_suction_radius:
            base_bounds.append((min_suction_radius, max_suction_radius))

        # Protrusion bounds (relative to base, within reasonable range)
        max_prot_dist = 0.3  # Max 30cm from center
        prot_bounds = [(-max_prot_dist, max_prot_dist)] * 8

        bounds = base_bounds + prot_bounds
        n_prot_params = 8

    # Create wall-mounted model
    model = WallMountedRobotModel(robot, wall, end_coords=end_coords)

    def unpack_x(x):
        """Unpack optimization variables."""
        base_pos = x[:2]
        if optimize_suction_radius:
            suction_r = x[2]
            prot_start = 3
        else:
            suction_r = fixed_suction_radius
            prot_start = 2

        if symmetric_protrusions:
            # Convert half-widths to 4 symmetric positions
            hw_x, hw_y = x[prot_start], x[prot_start + 1]
            prot_pos = np.array([
                [-hw_x, -hw_y],
                [+hw_x, -hw_y],
                [+hw_x, +hw_y],
                [-hw_x, +hw_y]
            ])
        else:
            prot_pos = x[prot_start:].reshape(4, 2)
        return base_pos, suction_r, prot_pos

    def objective(x):
        """Minimize protrusion quadrilateral area."""
        base_pos, suction_r, prot_pos = unpack_x(x)
        area = compute_quadrilateral_area(prot_pos)
        return area

    def objective_grad(x):
        """Gradient of objective (numerical)."""
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = objective(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            grad[i] = (objective(x_plus) - f0) / eps
        return grad

    # Constraint functions
    def convexity_constraint(x):
        """Protrusions must form convex quadrilateral."""
        base_pos, suction_r, prot_pos = unpack_x(x)
        # Return positive if convex, measure how convex it is
        n = len(prot_pos)
        min_cross = float('inf')
        for i in range(n):
            p0 = prot_pos[i]
            p1 = prot_pos[(i + 1) % n]
            p2 = prot_pos[(i + 2) % n]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            min_cross = min(min_cross, cross)
        return min_cross  # Positive means convex (counter-clockwise order)

    def min_distance_constraint(x):
        """Minimum distance between protrusions."""
        base_pos, suction_r, prot_pos = unpack_x(x)
        min_dist = float('inf')
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.linalg.norm(prot_pos[i] - prot_pos[j])
                min_dist = min(min_dist, dist)
            # Also check distance from suction cup
            dist_to_suction = np.linalg.norm(prot_pos[i])
            min_dist = min(min_dist, dist_to_suction - suction_r)
        return min_dist - min_protrusion_distance

    def feasibility_constraint(x):
        """IK and force feasibility."""
        base_pos, suction_r, prot_pos = unpack_x(x)

        # Set base position
        model.set_base_position(base_pos)
        model.suction_radius = suction_r
        model.protrusion_positions = prot_pos

        # Check IK for all poses
        ik_success_count = 0
        force_feasibility_score = 0.0

        for target in target_poses:
            coords = target['coords']
            trans_axis = target.get('translation_axis', True)
            rot_axis = target.get('rotation_axis', True)

            result = robot.inverse_kinematics(
                coords,
                move_target=end_coords,
                link_list=link_list,
                rotation_axis=rot_axis,
                translation_axis=trans_axis,
                stop=30,
                thre=0.03,
                rthre=np.deg2rad(10.0)
            )

            if result is not False and result is not None:
                ik_success_count += 1

                # Check force feasibility
                suction_pos = model.get_suction_position_world()
                prot_pos_world = model.get_protrusion_positions_world()

                moment, grav_force, com = compute_gravity_moment(
                    robot, joint_list, suction_pos, wall.normal,
                    gravity, payload_mass, robot_mass
                )

                suction_force_mag = vacuum_pressure * np.pi * suction_r**2

                forces, feasible, info = solve_protrusion_forces(
                    prot_pos_world, suction_pos, moment, grav_force,
                    wall.normal, suction_force_mag, friction_coeff
                )

                if feasible:
                    force_feasibility_score += 0.25

        model.reset_robot_pose()

        # Return positive if all 4 IK succeed and forces are feasible
        # ik_success_count ranges from 0 to 4
        # force_feasibility_score ranges from 0 to 1
        ik_ratio = ik_success_count / 4.0
        return ik_ratio + force_feasibility_score - 1.5  # Need > 1.5 to be positive

    def min_area_constraint(x):
        """Minimum area for protrusion quadrilateral."""
        base_pos, suction_r, prot_pos = unpack_x(x)
        area = compute_quadrilateral_area(prot_pos)
        return area - min_protrusion_area

    # Combine constraints
    constraints = [
        {'type': 'ineq', 'fun': convexity_constraint},
        {'type': 'ineq', 'fun': min_distance_constraint},
        {'type': 'ineq', 'fun': min_area_constraint},
        {'type': 'ineq', 'fun': feasibility_constraint},
    ]

    # Create callback wrapper
    iteration_count = [0]  # Use list to allow modification in nested function

    def optimization_callback(x):
        """Wrapper callback that parses x and calls user callback."""
        if callback is not None:
            base_pos, suction_r, prot_pos = unpack_x(x)
            area = compute_quadrilateral_area(prot_pos)
            suction_force = vacuum_pressure * np.pi * suction_r**2

            info = {
                'iteration': iteration_count[0],
                'objective': area,
                'suction_force': suction_force,
                'model': model,
                'robot': robot,
                'wall': wall,
                'target_poses': target_poses,
            }
            callback(iteration_count[0], base_pos, suction_r, prot_pos, area, info)
            iteration_count[0] += 1

    # Run optimization
    options = {'maxiter': max_iterations, 'disp': verbose, 'ftol': 1e-8, 'eps': 1e-6}

    result = scipy.optimize.minimize(
        objective, x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        callback=optimization_callback,
        options=options
    )

    # Unpack final solution
    base_pos, suction_r, prot_pos = unpack_x(result.x)
    area = compute_quadrilateral_area(prot_pos)
    suction_force = vacuum_pressure * np.pi * suction_r**2

    # Compute max protrusion force
    model.set_base_position(base_pos)
    max_prot_force = 0.0

    for target in target_poses:
        coords = target['coords']
        trans_axis = target.get('translation_axis', True)
        rot_axis = target.get('rotation_axis', True)

        robot.inverse_kinematics(
            coords,
            move_target=end_coords,
            link_list=link_list,
            rotation_axis=rot_axis,
            translation_axis=trans_axis,
            stop=30
        )

        suction_pos = model.get_suction_position_world()
        prot_pos_world = model.get_protrusion_positions_world()

        moment, grav_force, com = compute_gravity_moment(
            robot, joint_list, suction_pos, wall.normal,
            gravity, payload_mass, robot_mass
        )

        forces, feasible, info = solve_protrusion_forces(
            prot_pos_world, suction_pos, moment, grav_force,
            wall.normal, suction_force, friction_coeff
        )

        if 'forces_magnitude' in info:
            max_prot_force = max(max_prot_force, np.max(info['forces_magnitude']))

    model.reset_robot_pose()

    return {
        'success': result.success,
        'base_position': base_pos,
        'suction_radius': suction_r,
        'protrusion_positions': prot_pos,
        'base_area': area,
        'max_protrusion_force': max_prot_force,
        'suction_force': suction_force,
        'optimization_result': result
    }
