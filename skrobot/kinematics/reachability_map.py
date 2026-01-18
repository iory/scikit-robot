"""Fast reachability map computation with backend abstraction.

This module provides efficient computation of robot workspace reachability
using batch processing and optional JIT compilation capabilities.
Supports multiple backends (JAX, NumPy) via the backend registry.

Typical usage:
    >>> from skrobot.kinematics.reachability_map import ReachabilityMap
    >>> robot = skrobot.models.PR2()
    >>> rmap = ReachabilityMap(robot, link_list, end_coords)
    >>> rmap.compute(n_samples=500000)
    >>> print(rmap.is_reachable([0.5, -0.3, 1.0]))

    # Using NumPy backend (no JAX dependency)
    >>> rmap = ReachabilityMap(robot, link_list, end_coords, backend='numpy')
"""

from typing import List
from typing import Optional
from typing import Tuple

import numpy as np


class ReachabilityMap:
    """Fast reachability map using batch FK computation.

    This class computes and stores workspace reachability information
    for a robot arm. It uses vectorized batch forward kinematics
    computation, with optional JIT compilation for faster execution.

    Parameters
    ----------
    robot_model : RobotModel
        Robot model instance.
    link_list : list
        List of links in the kinematic chain.
    end_coords : CascadedCoords
        End-effector coordinates.
    voxel_size : float
        Size of each voxel in meters. Default is 0.05 (5cm).
    backend : str, optional
        Backend to use ('jax' or 'numpy'). If None, automatically
        selects the best available backend (JAX if available).

    Attributes
    ----------
    reachability : ndarray
        3D array of reachability counts per voxel.
    manipulability : ndarray
        3D array of average manipulability per voxel.
    bounds : dict
        Workspace bounds {'x': (min, max), 'y': (min, max), 'z': (min, max)}.

    Examples
    --------
    >>> import skrobot
    >>> robot = skrobot.models.Panda()
    >>> link_list = [
    ...     robot.panda_link1, robot.panda_link2, robot.panda_link3,
    ...     robot.panda_link4, robot.panda_link5, robot.panda_link6,
    ...     robot.panda_link7
    ... ]
    >>> end_coords = skrobot.coordinates.CascadedCoords(
    ...     parent=robot.panda_hand, name='end_coords')
    >>> rmap = ReachabilityMap(robot, link_list, end_coords, voxel_size=0.05)
    >>> rmap.compute(n_samples=100000)
    >>> print(f"Reachable volume: {rmap.reachable_volume:.3f} m^3")

    # Using NumPy backend explicitly
    >>> rmap = ReachabilityMap(robot, link_list, end_coords, backend='numpy')
    """

    def __init__(
        self,
        robot_model,
        link_list: List,
        end_coords,
        voxel_size: float = 0.05,
        backend: Optional[str] = None,
    ):
        self.robot_model = robot_model
        self.link_list = link_list
        self.end_coords = end_coords
        self.voxel_size = voxel_size

        # Get backend instance
        from skrobot.backend import get_backend
        self._backend = get_backend(backend)
        self._backend_name = self._backend.name

        # Extract FK parameters
        from skrobot.kinematics.differentiable import extract_fk_parameters
        self._fk_params = extract_fk_parameters(robot_model, link_list, end_coords)

        # Results (populated by compute())
        self.reachability = None
        self.manipulability = None
        self.reachability_index = None
        self.bounds = None
        self._origin = None
        self._shape = None

        # Cache for compiled batch FK function
        self._fk_batch_compiled = None

    def _build_fk_function(self):
        """Build batch FK function using the selected backend.

        Returns
        -------
        callable
            Batch FK function that takes (N, n_joints) array of joint angles
            and returns (positions, approach_vectors, manipulabilities).
        """
        from skrobot.kinematics.differentiable import compute_jacobian_analytical
        from skrobot.kinematics.differentiable import compute_jacobian_analytical_batch

        backend = self._backend
        fk_params = self._fk_params

        if backend.name == 'jax':
            # JAX: use vmap + jit for efficient batch computation
            def fk_single(angles):
                """Forward kinematics for a single configuration."""
                J, ee_pos, ee_rot = compute_jacobian_analytical(
                    backend, angles, fk_params
                )
                approach_vector = ee_rot[:, 2]
                JJT = backend.matmul(J, backend.transpose(J))
                manipulability = backend.sqrt(
                    backend.maximum(backend.det(JJT), backend.array(1e-10))
                )
                return ee_pos, approach_vector, manipulability

            batched_fk = backend.vmap(fk_single)
            compiled_fk = backend.compile(batched_fk)
            return compiled_fk
        else:
            # NumPy: use vectorized batch computation
            def batch_fk_numpy(angles_batch):
                # Use batch-optimized analytical Jacobian
                jacobians, ee_positions, ee_rotations = \
                    compute_jacobian_analytical_batch(angles_batch, fk_params)

                # Extract approach vectors (Z-axis of end-effector)
                approach_vectors = ee_rotations[:, :, 2]

                # Compute manipulability = sqrt(det(J @ J.T))
                # jacobians: (batch_size, 3, n_joints)
                # JJT: (batch_size, 3, 3)
                JJT = np.einsum('bij,bkj->bik', jacobians, jacobians)
                det = np.linalg.det(JJT)
                manipulabilities = np.sqrt(np.maximum(det, 1e-10))

                return ee_positions, approach_vectors, manipulabilities

            return batch_fk_numpy

    def compute(
        self,
        n_samples: int = 500000,
        seed: int = 42,
        verbose: bool = True,
        sampling: str = "random",
        bins_per_joint: int = 10,
        orientation_bins: int = 50,
    ) -> "ReachabilityMap":
        """Compute reachability map by sampling joint space.

        Parameters
        ----------
        n_samples : int
            Number of joint configurations to sample (for random sampling).
            Ignored when sampling='grid'.
        seed : int
            Random seed for reproducibility (for random sampling).
        verbose : bool
            Print progress information.
        sampling : str
            Sampling method: 'random' or 'grid'.
            - 'random': Uniform random sampling in joint space.
            - 'grid': Systematic grid sampling (bins_per_joint^n_joints samples).
        bins_per_joint : int
            Number of bins per joint for grid sampling.
            Total samples = bins_per_joint^n_joints.
        orientation_bins : int
            Number of bins for orientation discretization on the sphere.
            Uses Fibonacci sphere sampling for uniform distribution.
            Reachability Index = (reachable orientations) / orientation_bins.
            Set to 0 to disable orientation tracking (position-only mode).

        Returns
        -------
        ReachabilityMap
            Self, for method chaining.
        """
        from itertools import product
        import time

        backend = self._backend
        self._orientation_bins = orientation_bins

        # Get joint limits
        joint_limits_lower = np.array(self._fk_params['joint_limits_lower'])
        joint_limits_upper = np.array(self._fk_params['joint_limits_upper'])
        n_joints = self._fk_params['n_joints']

        # Generate joint configurations based on sampling method
        if sampling == "grid":
            # Grid sampling: divide each joint into bins
            grid_points = []
            for i in range(n_joints):
                grid_points.append(
                    np.linspace(
                        joint_limits_lower[i],
                        joint_limits_upper[i],
                        bins_per_joint
                    )
                )
            angles = np.array(list(product(*grid_points)), dtype=np.float32)
            actual_samples = len(angles)
            if verbose:
                print(f"Computing reachability map (grid: {bins_per_joint} bins/joint, "
                      f"{actual_samples:,} samples)...")
        else:
            # Random sampling
            np.random.seed(seed)
            angles = np.random.uniform(
                joint_limits_lower, joint_limits_upper,
                (n_samples, n_joints)
            ).astype(np.float32)
            actual_samples = n_samples
            if verbose:
                print(f"Computing reachability map (random: {actual_samples:,} samples)...")
                print(f"  Backend: {backend.name}")

        if verbose and orientation_bins > 0:
            print(f"  Orientation bins: {orientation_bins}")

        t_start = time.time()

        # Build FK function if not cached
        if self._fk_batch_compiled is None:
            self._fk_batch_compiled = self._build_fk_function()

        # Convert angles to backend array
        angles_backend = backend.array(angles)

        # Compute FK
        if verbose:
            print("  Computing forward kinematics...")
        t1 = time.time()
        ee_positions, approach_vectors, manipulabilities = self._fk_batch_compiled(
            angles_backend
        )

        # Wait for computation to complete (JAX async execution)
        if backend.name == 'jax' and hasattr(ee_positions, 'block_until_ready'):
            ee_positions.block_until_ready()
        t_fk = time.time() - t1

        if verbose:
            print(f"    FK time: {t_fk:.3f}s ({actual_samples / t_fk:,.0f} FK/sec)")

        # Convert to numpy for post-processing
        ee_pos_np = backend.to_numpy(ee_positions)
        approach_np = backend.to_numpy(approach_vectors)
        manip_np = backend.to_numpy(manipulabilities)

        # Compute bounds
        x_min, x_max = ee_pos_np[:, 0].min(), ee_pos_np[:, 0].max()
        y_min, y_max = ee_pos_np[:, 1].min(), ee_pos_np[:, 1].max()
        z_min, z_max = ee_pos_np[:, 2].min(), ee_pos_np[:, 2].max()

        self.bounds = {
            'x': (float(x_min), float(x_max)),
            'y': (float(y_min), float(y_max)),
            'z': (float(z_min), float(z_max)),
        }
        self._origin = np.array([x_min, y_min, z_min])

        # Build voxel grid
        if verbose:
            print("  Building voxel grid...")

        nx = int(np.ceil((x_max - x_min) / self.voxel_size)) + 1
        ny = int(np.ceil((y_max - y_min) / self.voxel_size)) + 1
        nz = int(np.ceil((z_max - z_min) / self.voxel_size)) + 1
        self._shape = (nx, ny, nz)

        # Compute voxel indices
        ix = ((ee_pos_np[:, 0] - x_min) / self.voxel_size).astype(int)
        iy = ((ee_pos_np[:, 1] - y_min) / self.voxel_size).astype(int)
        iz = ((ee_pos_np[:, 2] - z_min) / self.voxel_size).astype(int)

        # Clip to valid range
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        iz = np.clip(iz, 0, nz - 1)

        # Accumulate sample count and manipulability
        self._sample_count = np.zeros((nx, ny, nz), dtype=np.int32)
        manipulability_sum = np.zeros((nx, ny, nz), dtype=np.float64)

        np.add.at(self._sample_count, (ix, iy, iz), 1)
        np.add.at(manipulability_sum, (ix, iy, iz), manip_np)

        # Orientation-aware reachability
        if orientation_bins > 0:
            if verbose:
                print("  Computing orientation-aware reachability...")

            # Generate Fibonacci sphere points for uniform orientation bins
            sphere_points = self._fibonacci_sphere(orientation_bins)

            # Map approach vectors to nearest orientation bin
            # Use float64 to avoid precision issues
            approach_f64 = approach_np.astype(np.float64)
            sphere_f64 = sphere_points.astype(np.float64)

            # Normalize approach vectors
            norms = np.linalg.norm(approach_f64, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)  # Avoid division by zero
            approach_norm = approach_f64 / norms

            # Find nearest bin for each sample using dot product
            # Shape: (n_samples, orientation_bins)
            with np.errstate(all='ignore'):
                dot_products = approach_norm @ sphere_f64.T
                # Handle any NaN values from numerical issues
                dot_products = np.nan_to_num(dot_products, nan=-2.0)
            orientation_indices = np.argmax(dot_products, axis=1)

            # Track unique orientations per voxel
            # Use a set for each voxel to count unique orientations
            orientation_sets = {}
            for i in range(len(ix)):
                voxel_key = (ix[i], iy[i], iz[i])
                if voxel_key not in orientation_sets:
                    orientation_sets[voxel_key] = set()
                orientation_sets[voxel_key].add(orientation_indices[i])

            # Compute reachability index: unique orientations / total bins
            self.reachability_index = np.zeros((nx, ny, nz), dtype=np.float32)
            for voxel_key, ori_set in orientation_sets.items():
                self.reachability_index[voxel_key] = len(ori_set) / orientation_bins

            # Reachability is now based on orientation diversity
            # (number of unique orientations reachable)
            self.reachability = np.zeros((nx, ny, nz), dtype=np.int32)
            for voxel_key, ori_set in orientation_sets.items():
                self.reachability[voxel_key] = len(ori_set)
        else:
            # Position-only mode (no orientation tracking)
            self.reachability = self._sample_count.copy()
            self.reachability_index = np.where(
                self.reachability > 0, 1.0, 0.0
            ).astype(np.float32)

        # Average manipulability
        with np.errstate(divide='ignore', invalid='ignore'):
            self.manipulability = np.where(
                self._sample_count > 0,
                manipulability_sum / self._sample_count,
                0
            ).astype(np.float32)

        t_total = time.time() - t_start

        if verbose:
            n_reachable = np.sum(self.reachability > 0)
            n_total = nx * ny * nz
            print(f"  Total time: {t_total:.3f}s")
            print(f"  Grid size: {nx} x {ny} x {nz} = {n_total:,} voxels")
            print(f"  Reachable: {n_reachable:,} voxels ({100 * n_reachable / n_total:.1f}%)")
            if orientation_bins > 0:
                avg_ri = self.reachability_index[self.reachability > 0].mean()
                print(f"  Avg Reachability Index: {avg_ri:.2%}")

        return self

    def _fibonacci_sphere(self, n_points: int) -> np.ndarray:
        """Generate uniformly distributed points on a sphere.

        Uses Fibonacci sphere algorithm for uniform distribution.

        Parameters
        ----------
        n_points : int
            Number of points to generate.

        Returns
        -------
        ndarray
            Points on unit sphere with shape (n_points, 3).
        """
        indices = np.arange(n_points, dtype=float)
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        y = 1 - (indices / (n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)

        theta = phi * indices

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        return np.stack([x, y, z], axis=1)

    def is_reachable(self, position: np.ndarray, threshold: int = 1) -> bool:
        """Check if a position is reachable.

        Parameters
        ----------
        position : array-like
            3D position [x, y, z].
        threshold : int
            Minimum reachability count to consider reachable.

        Returns
        -------
        bool
            True if position is reachable.
        """
        score = self.get_reachability(position)
        return score >= threshold

    def get_reachability(self, position: np.ndarray) -> int:
        """Get reachability score at a position.

        Parameters
        ----------
        position : array-like
            3D position [x, y, z].

        Returns
        -------
        int
            Number of configurations that can reach this position.
        """
        if self.reachability is None:
            raise RuntimeError("Must call compute() first")

        position = np.asarray(position)
        idx = self._position_to_index(position)

        if idx is None:
            return 0

        return int(self.reachability[idx])

    def get_manipulability(self, position: np.ndarray) -> float:
        """Get average manipulability at a position.

        Parameters
        ----------
        position : array-like
            3D position [x, y, z].

        Returns
        -------
        float
            Average manipulability at this position.
        """
        if self.manipulability is None:
            raise RuntimeError("Must call compute() first")

        position = np.asarray(position)
        idx = self._position_to_index(position)

        if idx is None:
            return 0.0

        return float(self.manipulability[idx])

    def _position_to_index(
        self,
        position: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """Convert position to voxel index."""
        if self._origin is None:
            return None

        idx = ((position - self._origin) / self.voxel_size).astype(int)

        if (idx < 0).any() or (idx >= self._shape).any():
            return None

        return tuple(idx)

    @property
    def backend(self) -> str:
        """Name of the backend being used.

        Returns
        -------
        str
            Backend name ('jax' or 'numpy').
        """
        return self._backend_name

    @property
    def reachable_volume(self) -> float:
        """Total reachable volume in cubic meters."""
        if self.reachability is None:
            return 0.0
        n_reachable = np.sum(self.reachability > 0)
        return n_reachable * (self.voxel_size ** 3)

    @property
    def n_reachable_voxels(self) -> int:
        """Number of reachable voxels."""
        if self.reachability is None:
            return 0
        return int(np.sum(self.reachability > 0))

    def get_reachable_points(
        self,
        min_score: int = 1,
        max_points: Optional[int] = None,
    ) -> np.ndarray:
        """Get positions of reachable voxels.

        Parameters
        ----------
        min_score : int
            Minimum reachability score to include.
        max_points : int, optional
            Maximum number of points to return.

        Returns
        -------
        ndarray
            Array of reachable positions (N, 3).
        """
        if self.reachability is None:
            raise RuntimeError("Must call compute() first")

        # Find reachable voxels
        indices = np.argwhere(self.reachability >= min_score)

        if max_points is not None and len(indices) > max_points:
            # Sample uniformly
            idx = np.random.choice(len(indices), max_points, replace=False)
            indices = indices[idx]

        # Convert to positions
        positions = self._origin + (indices + 0.5) * self.voxel_size

        return positions

    def find_nearest_reachable(
        self,
        position: np.ndarray,
        min_score: int = 1,
        max_distance: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Find the nearest reachable position to a given target.

        Parameters
        ----------
        position : array-like
            Target 3D position [x, y, z].
        min_score : int
            Minimum reachability score to consider.
        max_distance : float, optional
            Maximum distance to search. If None, searches all reachable points.

        Returns
        -------
        ndarray or None
            Nearest reachable position [x, y, z], or None if no reachable
            position found within max_distance.
        """
        if self.reachability is None:
            raise RuntimeError("Must call compute() first")

        position = np.asarray(position)

        # Check if already reachable
        if self.is_reachable(position, threshold=min_score):
            return position.copy()

        # Find all reachable voxels
        indices = np.argwhere(self.reachability >= min_score)
        if len(indices) == 0:
            return None

        # Convert to positions (voxel centers)
        reachable_positions = self._origin + (indices + 0.5) * self.voxel_size

        # Compute distances
        distances = np.linalg.norm(reachable_positions - position, axis=1)

        # Apply max_distance filter if specified
        if max_distance is not None:
            valid_mask = distances <= max_distance
            if not np.any(valid_mask):
                return None
            distances = distances[valid_mask]
            reachable_positions = reachable_positions[valid_mask]

        # Find nearest
        nearest_idx = np.argmin(distances)
        return reachable_positions[nearest_idx]

    def get_reachability_at_positions(
        self,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Get reachability scores for multiple positions efficiently.

        Parameters
        ----------
        positions : ndarray
            Array of 3D positions with shape (N, 3).

        Returns
        -------
        ndarray
            Array of reachability scores with shape (N,).
        """
        if self.reachability is None:
            raise RuntimeError("Must call compute() first")

        positions = np.asarray(positions)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        n_positions = len(positions)
        scores = np.zeros(n_positions, dtype=np.int32)

        # Compute voxel indices
        indices = ((positions - self._origin) / self.voxel_size).astype(int)

        # Check bounds and get scores
        for i in range(n_positions):
            idx = indices[i]
            if (idx >= 0).all() and (idx < self._shape).all():
                scores[i] = self.reachability[tuple(idx)]

        return scores

    def filter_reachable_targets(
        self,
        positions: np.ndarray,
        min_score: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Filter target positions to only reachable ones.

        Parameters
        ----------
        positions : ndarray
            Array of 3D target positions with shape (N, 3).
        min_score : int
            Minimum reachability score to consider reachable.

        Returns
        -------
        tuple
            (reachable_positions, reachable_indices) where reachable_positions
            is (M, 3) array and reachable_indices is (M,) array of original indices.
        """
        scores = self.get_reachability_at_positions(positions)
        reachable_mask = scores >= min_score
        reachable_indices = np.where(reachable_mask)[0]
        reachable_positions = positions[reachable_mask]
        return reachable_positions, reachable_indices

    def get_point_cloud(
        self,
        color_by: str = "reachability_index",
        max_points: int = 50000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get point cloud for visualization.

        Parameters
        ----------
        color_by : str
            Coloring mode:
            - "reachability_index": Ratio of reachable orientations (0-1).
              This metric measures orientation diversity.
            - "reachability": Raw count of reachable orientations.
            - "manipulability": Average manipulability measure.
        max_points : int
            Maximum number of points.

        Returns
        -------
        tuple
            (positions, colors) arrays.
        """
        if self.reachability is None:
            raise RuntimeError("Must call compute() first")

        # Find reachable voxels
        indices = np.argwhere(self.reachability > 0)

        if len(indices) > max_points:
            idx = np.random.choice(len(indices), max_points, replace=False)
            indices = indices[idx]

        # Get values for coloring
        if color_by == "manipulability":
            values = self.manipulability[
                indices[:, 0], indices[:, 1], indices[:, 2]
            ]
        elif color_by == "reachability_index":
            values = self.reachability_index[
                indices[:, 0], indices[:, 1], indices[:, 2]
            ]
        else:  # "reachability"
            values = self.reachability[
                indices[:, 0], indices[:, 1], indices[:, 2]
            ].astype(float)

        # Normalize to [0, 1]
        if values.max() > values.min():
            values = (values - values.min()) / (values.max() - values.min())
        else:
            values = np.ones_like(values)

        # Convert to positions
        positions = self._origin + (indices + 0.5) * self.voxel_size

        # Create colors using colormap
        # High reachability = Blue, Low reachability = Red
        # Rainbow gradient: Red -> Yellow -> Green -> Cyan -> Blue
        colors = self._value_to_color(values)

        return positions, colors

    def _value_to_color(self, values: np.ndarray) -> np.ndarray:
        """Convert normalized values to colors.

        Uses a rainbow colormap where:
        - Blue (high values) = high reachability/manipulability
        - Red (low values) = low reachability/manipulability

        Parameters
        ----------
        values : ndarray
            Normalized values in [0, 1].

        Returns
        -------
        ndarray
            RGBA colors with shape (N, 4).
        """
        n = len(values)
        colors = np.zeros((n, 4))

        # Colormap: high value = blue, low value = red
        t = values

        # Jet-like colormap: Red -> Orange -> Yellow -> Green -> Cyan -> Blue
        # Piecewise linear interpolation for smooth transitions
        # t=0: Red (1,0,0)
        # t=0.25: Orange/Yellow (1,0.5,0) -> (1,1,0)
        # t=0.5: Green (0,1,0)
        # t=0.75: Cyan (0,1,1)
        # t=1: Blue (0,0,1)

        # Red channel: 1 for t<0.375, then decreases to 0
        colors[:, 0] = np.clip(1.0 - (t - 0.375) * 4, 0, 1)
        colors[:, 0] = np.where(t < 0.375, 1.0, colors[:, 0])

        # Green channel: increases 0->1 for t in [0, 0.25], stays 1, decreases for t > 0.75
        colors[:, 1] = np.where(t < 0.25, t * 4, 1.0)
        colors[:, 1] = np.where(t > 0.75, 1.0 - (t - 0.75) * 4, colors[:, 1])

        # Blue channel: 0 for t<0.5, increases 0->1 for t in [0.5, 0.75], stays 1
        colors[:, 2] = np.where(t < 0.5, 0.0, (t - 0.5) * 4)
        colors[:, 2] = np.clip(colors[:, 2], 0, 1)

        colors[:, 3] = 0.8  # Alpha

        return colors

    def save(self, filepath: str):
        """Save reachability map to file.

        Parameters
        ----------
        filepath : str
            Path to save file (.npz).
        """
        if self.reachability is None:
            raise RuntimeError("Must call compute() first")

        save_dict = {
            'reachability': self.reachability,
            'manipulability': self.manipulability,
            'origin': self._origin,
            'voxel_size': self.voxel_size,
            'bounds_x': self.bounds['x'],
            'bounds_y': self.bounds['y'],
            'bounds_z': self.bounds['z'],
        }

        # Include reachability_index if computed
        if self.reachability_index is not None:
            save_dict['reachability_index'] = self.reachability_index

        np.savez_compressed(filepath, **save_dict)

    def load(self, filepath: str) -> "ReachabilityMap":
        """Load reachability map from file.

        Parameters
        ----------
        filepath : str
            Path to load file (.npz).

        Returns
        -------
        ReachabilityMap
            Self, for method chaining.
        """
        data = np.load(filepath)
        self.reachability = data['reachability']
        self.manipulability = data['manipulability']
        self._origin = data['origin']
        self.voxel_size = float(data['voxel_size'])
        self._shape = self.reachability.shape
        self.bounds = {
            'x': tuple(data['bounds_x']),
            'y': tuple(data['bounds_y']),
            'z': tuple(data['bounds_z']),
        }

        # Load reachability_index if available
        if 'reachability_index' in data:
            self.reachability_index = data['reachability_index']
        else:
            # Compute fallback for old files
            self.reachability_index = np.where(
                self.reachability > 0, 1.0, 0.0
            ).astype(np.float32)

        return self
