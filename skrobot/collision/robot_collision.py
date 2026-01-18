"""Robot collision checking with geometric primitives.

This module provides collision checking for robots using spheres and capsules
to approximate link geometries. It supports both NumPy and JAX backends for
differentiable collision avoidance in trajectory optimization.

Example
-------
>>> from skrobot.collision import RobotCollisionChecker
>>> from skrobot.models import PR2
>>> robot = PR2()
>>> checker = RobotCollisionChecker(robot)
>>> checker.add_link(robot.r_gripper_palm_link)  # Auto-generate from mesh
>>> distances = checker.compute_self_collision_distances()
"""

import numpy as np

from skrobot.collision.distance import collision_distance
from skrobot.collision.geometry import Capsule
from skrobot.collision.geometry import HalfSpace
from skrobot.collision.geometry import Sphere


class LinkCollisionGeometry:
    """Collision geometry attached to a robot link.

    Parameters
    ----------
    link : Link
        Robot link to attach geometry to.
    geometry : CollisionGeometry
        Collision geometry in link-local frame.
    """

    def __init__(self, link, geometry):
        self.link = link
        self.geometry = geometry

    def get_world_geometry(self, xp=np):
        """Get geometry transformed to world frame.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        CollisionGeometry
            Geometry in world frame.
        """
        pos = self.link.worldpos()
        rot = self.link.worldrot()
        return self.geometry.transform(pos, rot, xp)


class RobotCollisionChecker:
    """Collision checker for robot links using geometric primitives.

    This class manages collision geometries attached to robot links and
    provides methods to compute collision distances for:
    - Self-collision between robot links
    - Collision with world obstacles (primitives or mesh SDF)

    Supports both NumPy and JAX backends for differentiable collision.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model instance.

    Example
    -------
    >>> from skrobot.collision import RobotCollisionChecker
    >>> from skrobot.model.primitives import Box, Sphere
    >>> checker = RobotCollisionChecker(robot)
    >>> checker.add_link(robot.r_gripper_palm_link)  # Auto from mesh
    >>> checker.add_links([robot.r_forearm_link, robot.r_upper_arm_link])
    >>> # Use skrobot.model.primitives directly (auto-converted)
    >>> sphere = Sphere(radius=0.3)
    >>> sphere.translate([1, 0, 0])
    >>> checker.add_world_obstacle(sphere)
    >>> box = Box(extents=[0.5, 0.5, 0.5])
    >>> box.translate([0.8, 0, 0.8])
    >>> checker.add_world_obstacle(box)
    >>> checker.setup_self_collision_pairs()
    >>> min_dist = checker.compute_min_distance()
    """

    def __init__(self, robot_model):
        self.robot_model = robot_model
        self._link_geometries = []
        self._world_obstacles = []
        self._world_sdfs = []  # SDF functions for mesh obstacles
        self._self_collision_pairs = []

        # Visualization
        self._visual_spheres = []  # Visual sphere primitives for viewer
        self._visual_coords = []   # CascadedCoords for sphere positions
        self.color_normal_sphere = [250, 250, 10, 200]
        self.color_collision_sphere = [255, 0, 0, 200]

    def add_link(self, link, geometry_type='auto', n_spheres=None,
                 radius_scale=1.0, tol=0.1, aspect_threshold=1.5):
        """Add collision geometry for a link.

        Automatically generates collision geometry from the link's
        collision mesh. Can use spheres, capsule, or auto-select based
        on the mesh shape.

        Parameters
        ----------
        link : Link
            Robot link with collision_mesh attribute.
        geometry_type : str
            Type of geometry to use:
            - 'auto': Automatically select based on aspect ratio (default)
            - 'spheres': Use swept spheres (compatible with SweptSphereSdfCollisionChecker)
            - 'capsule': Use single capsule (more efficient for elongated shapes)
        n_spheres : int or None
            Number of spheres to use (only for 'spheres' type). If None,
            automatically determined based on tolerance.
        radius_scale : float
            Scale factor for computed radius.
        tol : float
            Tolerance for automatic sphere count determination.
        aspect_threshold : float
            Aspect ratio threshold for auto geometry selection.
            If length/diameter > threshold, use capsule. Default is 1.5.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("trimesh is required for mesh-based collision")

        mesh = getattr(link, 'collision_mesh', None)
        if mesh is None or not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            # Fallback: single sphere at origin
            self.add_link_sphere(link, radius=0.05)
            return

        try:
            # Compute capsule parameters from mesh
            p1, p2, radius = self._compute_capsule_from_mesh(mesh)
            radius = radius * radius_scale

            # Compute aspect ratio
            length = np.linalg.norm(p2 - p1)
            aspect_ratio = length / (2 * radius) if radius > 0 else 0

            # Auto-select geometry type
            if geometry_type == 'auto':
                if aspect_ratio > aspect_threshold:
                    geometry_type = 'capsule'
                else:
                    geometry_type = 'spheres'

            if geometry_type == 'capsule':
                # Use single capsule
                self.add_link_capsule(link, p1, p2, radius)
            else:
                # Use swept spheres (compatible with SweptSphereSdfCollisionChecker)
                from skrobot.planner.swept_sphere import compute_swept_sphere
                centers, radius = compute_swept_sphere(
                    mesh, n_sphere=n_spheres, tol=tol
                )
                radius = radius * radius_scale
                for center in centers:
                    self.add_link_sphere(link, center_local=center, radius=radius)

        except Exception:
            # Fallback: single sphere at mesh centroid
            centroid = mesh.centroid
            radius = np.max(mesh.bounding_box.extents) / 2 * radius_scale
            self.add_link_sphere(link, center_local=centroid, radius=radius)

    def _compute_capsule_from_mesh(self, mesh):
        """Compute capsule parameters (p1, p2, radius) from mesh.

        Uses PCA to find the principal axis and computes the bounding
        capsule along that axis.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Collision mesh.

        Returns
        -------
        p1 : ndarray (3,)
            First endpoint of capsule axis.
        p2 : ndarray (3,)
            Second endpoint of capsule axis.
        radius : float
            Capsule radius.
        """
        verts = mesh.vertices
        mean = np.mean(verts, axis=0)
        verts_centered = verts - mean

        # Use SVD for more stable PCA
        try:
            _, s, vh = np.linalg.svd(verts_centered, full_matrices=False)
            # Principal axis is the first right singular vector
            principal_axis = vh[0]
            # Normalize
            norm = np.linalg.norm(principal_axis)
            if norm < 1e-10:
                principal_axis = np.array([0.0, 0.0, 1.0])
            else:
                principal_axis = principal_axis / norm
        except Exception:
            # Fallback to Z-axis
            principal_axis = np.array([0.0, 0.0, 1.0])

        # Project vertices onto principal axis
        projections = verts_centered @ principal_axis

        # Compute radius (max distance from axis)
        proj_vecs = np.outer(projections, principal_axis)
        perp_vecs = verts_centered - proj_vecs
        distances = np.linalg.norm(perp_vecs, axis=1)
        radius = np.max(distances) * 1.01 if len(distances) > 0 else 0.01

        # Compute capsule endpoints
        h_min = np.min(projections) if len(projections) > 0 else 0
        h_max = np.max(projections) if len(projections) > 0 else 0

        p1 = mean + h_min * principal_axis
        p2 = mean + h_max * principal_axis

        # Ensure minimum capsule length
        if np.linalg.norm(p2 - p1) < 1e-6:
            p2 = p1 + np.array([0, 0, 0.01])

        return p1, p2, radius

    def add_links(self, links, geometry_type='auto', n_spheres=None,
                  radius_scale=1.0, aspect_threshold=1.5):
        """Add collision geometry for multiple links.

        Parameters
        ----------
        links : list of Link
            Robot links to add.
        geometry_type : str
            Type of geometry: 'auto', 'spheres', or 'capsule'.
        n_spheres : int or None
            Number of spheres per link (only for 'spheres' type).
        radius_scale : float
            Scale factor for computed radius.
        aspect_threshold : float
            Aspect ratio threshold for auto selection.
        """
        for link in links:
            self.add_link(
                link,
                geometry_type=geometry_type,
                n_spheres=n_spheres,
                radius_scale=radius_scale,
                aspect_threshold=aspect_threshold,
            )

    def add_link_sphere(self, link, center_local=None, radius=0.05):
        """Add a collision sphere to a link manually.

        Parameters
        ----------
        link : Link
            Robot link.
        center_local : array-like (3,), optional
            Sphere center in link-local frame. Defaults to origin.
        radius : float
            Sphere radius.
        """
        if center_local is None:
            center_local = np.zeros(3)
        else:
            center_local = np.asarray(center_local)

        geom = Sphere(center=center_local, radius=radius)
        self._link_geometries.append(LinkCollisionGeometry(link, geom))

    def add_link_capsule(self, link, p1_local, p2_local, radius=0.05):
        """Add a collision capsule to a link manually.

        Parameters
        ----------
        link : Link
            Robot link.
        p1_local : array-like (3,)
            First endpoint in link-local frame.
        p2_local : array-like (3,)
            Second endpoint in link-local frame.
        radius : float
            Capsule radius.
        """
        p1_local = np.asarray(p1_local)
        p2_local = np.asarray(p2_local)

        geom = Capsule(p1=p1_local, p2=p2_local, radius=radius)
        self._link_geometries.append(LinkCollisionGeometry(link, geom))

    def add_world_obstacle(self, obstacle, use_sdf=True):
        """Add a world obstacle for collision checking.

        Automatically detects the obstacle type and uses the best method:
        - Objects with .sdf attribute (when use_sdf=True): uses SDF
        - skrobot.model.primitives (Sphere, Box, Cylinder): analytical distance
        - skrobot.collision geometry: analytical distance
        - callable: treated as SDF function

        Parameters
        ----------
        obstacle : various
            Obstacle geometry in world frame. Accepts:
            - skrobot.model.primitives.Sphere, Box, Cylinder
            - skrobot.collision.Sphere, Capsule, Box, HalfSpace
            - Any object with .sdf callable attribute
            - callable SDF function: points (N, 3) -> distances (N,)
        use_sdf : bool
            If True (default), use SDF when available for more accurate
            collision checking. If False, prefer analytical primitives
            (faster, JAX-compatible).
        """
        # Check if it's a callable (SDF function)
        if callable(obstacle) and not hasattr(obstacle, 'worldpos'):
            self._world_sdfs.append(obstacle)
            return

        # Check if it has an SDF attribute first (e.g., with_sdf=True)
        # This ensures consistency with SweptSphereSdfCollisionChecker
        if use_sdf:
            sdf_func = getattr(obstacle, 'sdf', None)
            if sdf_func is not None and callable(sdf_func):
                self._world_sdfs.append(sdf_func)
                return

        # Check if it's a skrobot.model.primitives object
        # These have worldpos() and can be converted to collision geometry
        obstacle_class = type(obstacle).__name__

        if obstacle_class == 'Sphere' and hasattr(obstacle, 'worldpos'):
            # skrobot.model.primitives.Sphere
            radius = getattr(obstacle, 'radius', getattr(obstacle, '_radius', 0.05))
            center = obstacle.worldpos()
            geom = Sphere(center=center, radius=radius)
            self._world_obstacles.append(geom)
            return

        if obstacle_class == 'Box' and hasattr(obstacle, 'extents'):
            # skrobot.model.primitives.Box
            from skrobot.collision.geometry import Box as CollisionBox
            extents = obstacle.extents
            center = obstacle.worldpos()
            rot = obstacle.worldrot()
            geom = CollisionBox(
                center=center,
                half_extents=np.array(extents) / 2,
                rotation=rot
            )
            self._world_obstacles.append(geom)
            return

        if obstacle_class == 'Cylinder' and hasattr(obstacle, 'worldpos'):
            # skrobot.model.primitives.Cylinder -> approximate as Capsule
            radius = getattr(obstacle, 'radius', 0.05)
            height = getattr(obstacle, 'height', 0.1)
            center = obstacle.worldpos()
            rot = obstacle.worldrot()
            # Cylinder axis is Z in local frame
            axis = rot @ np.array([0, 0, 1])
            p1 = center - axis * height / 2
            p2 = center + axis * height / 2
            geom = Capsule(p1=p1, p2=p2, radius=radius)
            self._world_obstacles.append(geom)
            return

        # Otherwise, treat as CollisionGeometry for analytical distance
        self._world_obstacles.append(obstacle)

    def add_ground_plane(self, height=0.0):
        """Add a ground plane as world obstacle.

        Parameters
        ----------
        height : float
            Height of the ground plane.
        """
        ground = HalfSpace.ground_plane(height)
        self._world_obstacles.append(ground)

    def setup_self_collision_pairs(self, min_link_distance=2,
                                     ignore_pairs=None,
                                     use_urdf_adjacency=True):
        """Setup pairs of link geometries for self-collision checking.

        Parameters
        ----------
        min_link_distance : int
            Minimum kinematic chain distance for collision checking.
            Default is 2 (skip parent-child pairs).
        ignore_pairs : list of tuple, optional
            List of (link_name_a, link_name_b) pairs to ignore.
            Useful for known non-colliding pairs like torso vs arms.
        use_urdf_adjacency : bool
            If True, compute actual kinematic chain distance using
            parent-child relationships. If False, use insertion order.
        """
        self._self_collision_pairs = []
        n = len(self._link_geometries)

        # Build mapping from link to geometry indices
        link_to_indices = {}
        for i, lg in enumerate(self._link_geometries):
            link_name = lg.link.name
            if link_name not in link_to_indices:
                link_to_indices[link_name] = []
            link_to_indices[link_name].append(i)

        # Build ignore set
        ignore_set = set()
        if ignore_pairs:
            for name_a, name_b in ignore_pairs:
                ignore_set.add((name_a, name_b))
                ignore_set.add((name_b, name_a))

        # Compute kinematic chain distance between links
        def get_ancestors(link):
            """Get list of ancestor link names from link to root."""
            ancestors = []
            current = link
            while current is not None:
                ancestors.append(current.name)
                current = getattr(current, 'parent_link', None)
            return ancestors

        def kinematic_distance(link_a, link_b):
            """Compute minimum kinematic chain distance between two links."""
            if not use_urdf_adjacency:
                # Fallback to insertion order
                return abs(link_order.get(link_a.name, 0) -
                           link_order.get(link_b.name, 0))

            ancestors_a = get_ancestors(link_a)
            ancestors_b = get_ancestors(link_b)

            # Find common ancestor
            set_a = set(ancestors_a)
            for i, ancestor in enumerate(ancestors_b):
                if ancestor in set_a:
                    # Distance = steps from a to common + steps from b to common
                    dist_a = ancestors_a.index(ancestor)
                    dist_b = i
                    return dist_a + dist_b

            # No common ancestor (shouldn't happen for same robot)
            return float('inf')

        # Build link order for fallback
        link_order = {}
        for i, lg in enumerate(self._link_geometries):
            if lg.link.name not in link_order:
                link_order[lg.link.name] = len(link_order)

        # Cache link objects
        link_objects = {}
        for lg in self._link_geometries:
            if lg.link.name not in link_objects:
                link_objects[lg.link.name] = lg.link

        # Create pairs
        for i in range(n):
            for j in range(i + 1, n):
                link_i_name = self._link_geometries[i].link.name
                link_j_name = self._link_geometries[j].link.name

                # Skip if same link
                if link_i_name == link_j_name:
                    continue

                # Skip if in ignore set
                if (link_i_name, link_j_name) in ignore_set:
                    continue

                # Skip if links are too close in kinematic chain
                link_i = link_objects[link_i_name]
                link_j = link_objects[link_j_name]
                if kinematic_distance(link_i, link_j) < min_link_distance:
                    continue

                self._self_collision_pairs.append((i, j))

    def set_self_collision_pairs(self, pairs):
        """Manually set self-collision pairs.

        Parameters
        ----------
        pairs : list of tuple
            List of (i, j) index pairs into link_geometries.
        """
        self._self_collision_pairs = list(pairs)

    def compute_world_collision_distances(self, xp=np):
        """Compute distances from all link geometries to world obstacles.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        array
            Array of signed distances.
        """
        distances = []

        # Primitive obstacles
        for lg in self._link_geometries:
            world_geom = lg.get_world_geometry(xp)
            for obs in self._world_obstacles:
                dist = collision_distance(world_geom, obs, xp)
                distances.append(dist)

        # SDF obstacles (NumPy only for now)
        if self._world_sdfs and xp.__name__ == 'numpy':
            for lg in self._link_geometries:
                world_geom = lg.get_world_geometry(xp)
                if isinstance(world_geom, Sphere):
                    center = world_geom.center.reshape(1, 3)
                    for sdf_func in self._world_sdfs:
                        sdf_val = sdf_func(center)[0]
                        dist = sdf_val - world_geom.radius
                        distances.append(dist)
                elif isinstance(world_geom, Capsule):
                    # Sample points along capsule
                    pts = np.stack([world_geom.p1, world_geom.p2])
                    for sdf_func in self._world_sdfs:
                        sdf_vals = sdf_func(pts)
                        dist = np.min(sdf_vals) - world_geom.radius
                        distances.append(dist)

        if len(distances) == 0:
            return xp.array([])

        return xp.array(distances)

    def compute_self_collision_distances(self, xp=np):
        """Compute distances between self-collision pairs.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        array
            Array of signed distances for each collision pair.
        """
        if len(self._self_collision_pairs) == 0:
            return xp.array([])

        distances = []
        for i, j in self._self_collision_pairs:
            geom_i = self._link_geometries[i].get_world_geometry(xp)
            geom_j = self._link_geometries[j].get_world_geometry(xp)
            dist = collision_distance(geom_i, geom_j, xp)
            distances.append(dist)

        return xp.array(distances)

    def compute_all_distances(self, xp=np):
        """Compute all collision distances (world + self).

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        array
            Concatenated array of all collision distances.
        """
        world_dists = self.compute_world_collision_distances(xp)
        self_dists = self.compute_self_collision_distances(xp)

        if len(world_dists) == 0 and len(self_dists) == 0:
            return xp.array([])
        elif len(world_dists) == 0:
            return self_dists
        elif len(self_dists) == 0:
            return world_dists
        else:
            return xp.concatenate([world_dists, self_dists])

    def compute_min_distance(self, xp=np):
        """Compute minimum collision distance.

        Parameters
        ----------
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        float
            Minimum signed distance. Negative means collision.
        """
        all_dists = self.compute_all_distances(xp)
        if len(all_dists) == 0:
            return xp.inf
        return xp.min(all_dists)

    def is_collision_free(self, margin=0.0, xp=np):
        """Check if robot is collision-free.

        Parameters
        ----------
        margin : float
            Safety margin. Robot is collision-free if min_distance > margin.
        xp : module
            Array module (numpy or jax.numpy).

        Returns
        -------
        bool
            True if collision-free.
        """
        min_dist = self.compute_min_distance(xp)
        return min_dist > margin

    def collision_check(self, xp=np):
        """Check if any collision exists.

        Returns
        -------
        bool
            True if collision detected.
        """
        return not self.is_collision_free(margin=0.0, xp=xp)

    @property
    def n_feature(self):
        """Number of collision features (spheres/capsules)."""
        return len(self._link_geometries)

    @property
    def link_geometries(self):
        """List of LinkCollisionGeometry objects."""
        return self._link_geometries

    @property
    def world_obstacles(self):
        """List of world obstacle geometries."""
        return self._world_obstacles

    @property
    def self_collision_pairs(self):
        """List of (i, j) self-collision pairs."""
        return self._self_collision_pairs

    def get_collision_spheres_world(self):
        """Get all collision spheres in world frame.

        Useful for visualization.

        Returns
        -------
        list of tuple
            List of (center, radius) for each sphere.
        """
        spheres = []
        for lg in self._link_geometries:
            world_geom = lg.get_world_geometry()
            if isinstance(world_geom, Sphere):
                spheres.append((world_geom.center, world_geom.radius))
            elif isinstance(world_geom, Capsule):
                # Approximate capsule with spheres at endpoints
                spheres.append((world_geom.p1, world_geom.radius))
                spheres.append((world_geom.p2, world_geom.radius))
        return spheres

    def add_coll_spheres_to_viewer(self, viewer):
        """Add collision geometries to viewer.

        Creates visual primitives (spheres or capsules) that follow the robot
        links. Call update_color() to update colors based on collision state.

        Parameters
        ----------
        viewer : skrobot.viewers.TrimeshSceneViewer or similar
            Viewer to add geometries to.
        """
        from skrobot.coordinates import CascadedCoords
        from skrobot.model.primitives import Capsule as VisualCapsule
        from skrobot.model.primitives import Sphere as VisualSphere

        # Clear existing visual geometries
        self._visual_spheres = []
        self._visual_coords = []

        for lg in self._link_geometries:
            link = lg.link
            geom = lg.geometry

            if isinstance(geom, Sphere):
                # Create coords attached to link
                link_pos = link.copy_worldcoords()
                coll_coords = CascadedCoords(
                    pos=link_pos.worldpos(),
                    rot=link_pos.worldrot()
                )
                coll_coords.translate(geom.center)
                link.assoc(coll_coords)
                self._visual_coords.append(coll_coords)

                # Create visual sphere
                sp = VisualSphere(
                    radius=geom.radius,
                    pos=coll_coords.worldpos(),
                    color=self.color_normal_sphere
                )
                coll_coords.assoc(sp)
                self._visual_spheres.append(sp)
                viewer.add(sp)

            elif isinstance(geom, Capsule):
                # Compute capsule parameters in local frame
                p1 = geom.p1
                p2 = geom.p2
                center_local = (p1 + p2) / 2
                axis_local = p2 - p1
                height = np.linalg.norm(axis_local)

                # Compute rotation to align Z-axis with capsule axis
                rot_matrix = np.eye(3)
                if height > 1e-6:
                    axis_local_normalized = axis_local / height
                    z_axis = np.array([0.0, 0.0, 1.0])
                    # Rotation from Z to capsule axis
                    v = np.cross(z_axis, axis_local_normalized)
                    c = np.dot(z_axis, axis_local_normalized)
                    if np.linalg.norm(v) > 1e-6:
                        # Rodrigues' rotation formula
                        vx = np.array([
                            [0, -v[2], v[1]],
                            [v[2], 0, -v[0]],
                            [-v[1], v[0], 0]
                        ])
                        rot_matrix = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
                    elif c < 0:
                        # 180 degree rotation around X
                        rot_matrix = np.diag([1, -1, -1])

                # Create coords at capsule center with proper rotation
                link_pos = link.copy_worldcoords()
                # Combine link rotation with capsule local rotation
                combined_rot = link_pos.worldrot() @ rot_matrix
                coll_coords = CascadedCoords(
                    pos=link_pos.worldpos(),
                    rot=combined_rot
                )
                # Translate in original link frame
                world_center = link_pos.worldpos() + link_pos.worldrot() @ center_local
                coll_coords.newcoords(combined_rot, world_center)

                link.assoc(coll_coords)
                self._visual_coords.append(coll_coords)

                # Create visual capsule
                cap = VisualCapsule(
                    radius=geom.radius,
                    height=height,
                    pos=coll_coords.worldpos(),
                    rot=coll_coords.worldrot(),
                    face_colors=self.color_normal_sphere
                )
                coll_coords.assoc(cap)
                self._visual_spheres.append(cap)
                viewer.add(cap)

    def delete_coll_spheres_from_viewer(self, viewer):
        """Delete collision spheres from viewer.

        Parameters
        ----------
        viewer : skrobot.viewers.TrimeshSceneViewer or similar
            Viewer to remove spheres from.
        """
        for sp in self._visual_spheres:
            viewer.delete(sp)
        self._visual_spheres = []
        self._visual_coords = []

    def update_color(self):
        """Update collision geometry colors based on collision state.

        Geometries in collision are colored red, others are yellow.
        Call this after robot configuration changes to update visualization.

        Returns
        -------
        array
            Array of signed distances for each collision geometry.
        """
        if not self._visual_spheres:
            return np.array([])

        # Compute distances for each collision geometry
        distances = []
        geom_idx = 0

        for lg in self._link_geometries:
            world_geom = lg.get_world_geometry()

            # Compute min distance to all obstacles
            min_dist = float('inf')

            if isinstance(world_geom, Sphere):
                for obs in self._world_obstacles:
                    dist = collision_distance(world_geom, obs, np)
                    min_dist = min(min_dist, dist)
                for sdf_func in self._world_sdfs:
                    center = world_geom.center.reshape(1, 3)
                    sdf_val = sdf_func(center)[0]
                    dist = sdf_val - world_geom.radius
                    min_dist = min(min_dist, dist)

            elif isinstance(world_geom, Capsule):
                # For capsule, compute distance properly
                for obs in self._world_obstacles:
                    dist = collision_distance(world_geom, obs, np)
                    min_dist = min(min_dist, dist)
                # For SDF, sample along capsule axis
                for sdf_func in self._world_sdfs:
                    pts = np.stack([world_geom.p1, world_geom.p2])
                    sdf_vals = sdf_func(pts)
                    dist = np.min(sdf_vals) - world_geom.radius
                    min_dist = min(min_dist, dist)

            distances.append(min_dist)

            # Update color for visual geometry
            if geom_idx < len(self._visual_spheres):
                color = (self.color_collision_sphere if min_dist < 0
                         else self.color_normal_sphere)
                self._visual_spheres[geom_idx].set_color(color)
            geom_idx += 1

        return np.array(distances)
