"""Inverse reachability map for planar mobile-base placement.

A reachability map answers "where can the hand go from this base"; its
inverse answers "where can the base stand so the hand reaches this pose".
Sampling the arm once gives end-effector poses ``T_be`` in the base frame,
and each one, inverted, is a base pose ``T_target * T_be^-1`` that puts
the hand exactly on a target. A planar base keeps the floor plane, so a
sample only serves a target with the same height and the same tilt
against gravity; the rest of the pose -- ``(x, y, yaw)`` -- follows.

Every target votes for the base cells its matching samples land in.
Cells that many targets vote for are base poses from which all of them
are within reach, up to the map's resolution, and make good starts for
:func:`skrobot.kinematics.find_base_placement`.
"""

import numpy as np

from skrobot.coordinates.math import normalize_angle
from skrobot.kinematics.base_placement import _chain_fk_params
from skrobot.kinematics.base_placement import _planar_coords
from skrobot.kinematics.base_placement import _resolve_link_list
from skrobot.kinematics.base_placement import _resolve_masks
from skrobot.kinematics.base_placement import _targets_to_pos_rot
from skrobot.kinematics.differentiable import forward_kinematics_ee_batched_numpy


class BaseCandidates(object):
    """Base poses an :class:`InverseReachabilityMap` proposes.

    Attributes
    ----------
    bases : numpy.ndarray
        ``(K, 3)`` ``(x, y, yaw)``, best first.
    coverage : numpy.ndarray
        ``(K,)`` how many targets voted for each base cell.
    votes : numpy.ndarray
        ``(K,)`` total votes each base cell received.
    seeds : numpy.ndarray
        ``(K, N, n_opt)`` per target, the arm configuration furthest from
        the joint limits among the samples that voted for the cell;
        ``NaN`` where the target cast no vote there.
    vote_bases : numpy.ndarray
        ``(n_votes, 3)`` every vote's base pose, for inspection.
    vote_targets : numpy.ndarray
        ``(n_votes,)`` which target cast each vote.
    base_height : float
        Height of the root link the bases were proposed for.
    """

    def __init__(self, bases, coverage, votes, seeds, vote_bases,
                 vote_targets, base_height=0.0):
        self.bases = bases
        self.base_height = base_height
        self.coverage = coverage
        self.votes = votes
        self.seeds = seeds
        self.vote_bases = vote_bases
        self.vote_targets = vote_targets

    def __len__(self):
        return len(self.bases)

    @property
    def poses(self):
        """list of Coordinates: the bases as root-link poses."""
        return [_planar_coords(*b, z=self.base_height) for b in self.bases]


class InverseReachabilityMap(object):
    """Base poses from which an arm reaches a given end-effector pose.

    Parameters
    ----------
    robot_model : RobotModel
        Robot. Its root link is the mobile base, assumed to stand on the
        ``z = 0`` plane as in :func:`find_base_placement`.
    move_target : CascadedCoords
        End effector.
    link_list : list of Link, optional
        Links whose joints are sampled. Defaults to the chain from the
        root to ``move_target``.
    joint_list : list of Joint, optional
        Joints to sample; alternative to ``link_list``.

    Examples
    --------
    >>> from skrobot.kinematics import InverseReachabilityMap
    >>> from skrobot.kinematics import find_base_placement
    >>> from skrobot.models import Fetch
    >>> robot = Fetch()
    >>> irm = InverseReachabilityMap(robot, robot.rarm.end_coords)
    >>> irm.compute(n_samples=200000)  # doctest: +SKIP
    >>> result = find_base_placement(
    ...     robot, targets, move_target=robot.rarm.end_coords,
    ...     irm=irm)  # doctest: +SKIP
    """

    def __init__(self, robot_model, move_target, link_list=None,
                 joint_list=None):
        self.robot_model = robot_model
        self.move_target = move_target
        self.link_list = _resolve_link_list(
            robot_model, move_target, link_list, joint_list)
        self.fk_params = _chain_fk_params(
            robot_model, self.link_list, move_target)
        mimic_parent = self.fk_params['mimic_parent_indices']
        self._non_mimic = np.array(
            [i for i in range(self.fk_params['n_joints'])
             if mimic_parent[i] < 0], dtype=np.int64)
        self.joint_names = [self.fk_params['joint_names'][i]
                            for i in self._non_mimic]

        # Populated by compute() or load(), sorted by height.
        self.positions = None
        self.rotations = None
        self.angles = None

    @property
    def n_samples(self):
        """int: Number of stored end-effector poses."""
        return 0 if self.positions is None else len(self.positions)

    def compute(self, n_samples=500000, seed=0, batch_size=50000):
        """Sample the arm and store the end-effector poses.

        Parameters
        ----------
        n_samples : int
            Joint configurations drawn uniformly within the limits.
        seed : int
            Random seed.
        batch_size : int
            Configurations per FK batch, to bound memory.

        Returns
        -------
        InverseReachabilityMap
            Self, for chaining.
        """
        rng = np.random.RandomState(seed)
        lower = self.fk_params['joint_limits_lower'][self._non_mimic]
        upper = self.fk_params['joint_limits_upper'][self._non_mimic]
        n_joints = self.fk_params['n_joints']
        positions, rotations, angles = [], [], []
        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            q = rng.uniform(lower, upper, (n, len(lower)))
            full = np.zeros((n, n_joints))
            full[:, self._non_mimic] = q
            pos, rot = forward_kinematics_ee_batched_numpy(
                full, self.fk_params)
            positions.append(pos.astype(np.float32))
            rotations.append(rot.astype(np.float32))
            angles.append(q.astype(np.float32))
        self._store(np.concatenate(positions), np.concatenate(rotations),
                    np.concatenate(angles))
        return self

    def _store(self, positions, rotations, angles):
        order = np.argsort(positions[:, 2], kind='stable')
        self.positions = positions[order]
        self.rotations = rotations[order]
        self.angles = angles[order]
        # Gravity in the hand frame: the part of a hand pose a planar base
        # cannot change.
        self._tilts = self.rotations[:, 2, :]
        # Joint-limit margin of each sample, to pick seeds that start
        # clear of the limits; uniform samples often sit near one.
        from skrobot.kinematics.base_placement import _joint_margins
        self._margins = _joint_margins(
            self.fk_params, self.angles[None].astype(np.float64))[0]

    def _matches(self, height, target_rot, mode, axis, z_tolerance,
                 tilt_tolerance):
        """Indices of samples a planar base can turn into the target.

        ``height`` is the target's height above the base. ``mode`` is
        ``'full'`` to match the whole orientation, ``'axis'`` to match
        only hand axis ``axis`` (the one a mask leaves the hand free to
        turn about), ``'none'`` for position only.
        """
        z = self.positions[:, 2]
        lo = np.searchsorted(z, height - z_tolerance, side='left')
        hi = np.searchsorted(z, height + z_tolerance, side='right')
        idx = np.arange(lo, hi)
        if not len(idx):
            return idx
        if mode == 'full':
            cos = self._tilts[idx] @ target_rot[2, :]
            idx = idx[cos >= np.cos(tilt_tolerance)]
        elif mode == 'axis':
            # Same elevation of that axis against gravity.
            elev = np.arcsin(np.clip(self._tilts[idx, axis], -1.0, 1.0))
            target_elev = np.arcsin(np.clip(target_rot[2, axis], -1.0, 1.0))
            idx = idx[np.abs(elev - target_elev) <= tilt_tolerance]
        return idx

    def _bases_for(self, idx, target_pos, target_rot, mode, axis,
                   yaw_samples, tilt_tolerance):
        """``(n, 3)`` base poses, and the sample each came from."""
        pos = self.positions[idx].astype(np.float64)
        rot = self.rotations[idx].astype(np.float64)
        free_yaw = mode == 'none'
        if mode == 'full':
            # yaw of target_rot @ rot^T; tilts already agree.
            rel = np.einsum('ij,njk->nik', target_rot,
                            np.transpose(rot, (0, 2, 1)))
            yaw = np.arctan2(rel[:, 1, 0], rel[:, 0, 0])
        elif mode == 'axis':
            # Turn the sample's axis onto the target's heading. A nearly
            # vertical axis has no heading, and every yaw serves.
            heading = target_rot[:2, axis]
            if np.linalg.norm(heading) < np.sin(tilt_tolerance):
                free_yaw = True
            else:
                yaw = np.arctan2(heading[1], heading[0]) \
                    - np.arctan2(rot[:, 1, axis], rot[:, 0, axis])
        if free_yaw:
            yaw = np.repeat(yaw_samples[None], len(idx), axis=0).ravel()
            pos = np.repeat(pos, len(yaw_samples), axis=0)
            sample = np.repeat(idx, len(yaw_samples))
        else:
            sample = idx
        c, s = np.cos(yaw), np.sin(yaw)
        x = target_pos[0] - (c * pos[:, 0] - s * pos[:, 1])
        y = target_pos[1] - (s * pos[:, 0] + c * pos[:, 1])
        return np.stack([x, y, normalize_angle(yaw)], axis=1), sample

    def candidate_bases(
            self, target_coords, n_candidates=10, xy_resolution=0.1,
            yaw_resolution=np.deg2rad(20.0), z_tolerance=0.03,
            tilt_tolerance=np.deg2rad(10.0), rotation_mask=True,
            max_votes_per_target=20000, min_separation=None,
            base_bounds=None, base_height=0.0, is_valid_base=None, seed=0):
        """Propose base poses that put every target within reach.

        Parameters
        ----------
        target_coords : Coordinates or list of Coordinates
            End-effector targets in the world frame.
        n_candidates : int
            Number of bases to return at most.
        xy_resolution : float
            Base cell size in meters.
        yaw_resolution : float
            Base cell size in yaw, radians.
        z_tolerance : float
            How far a sample's height may differ from a target's, meters.
        tilt_tolerance : float
            How far a sample's tilt may differ from a target's, radians.
        rotation_mask : bool, str, sequence or list
            End-effector rotation mask, one for all targets or one per
            target, as in :func:`find_base_placement`. A full mask
            matches the whole orientation; a mask leaving one hand axis
            free matches that axis's direction; anything else matches
            positions only and lets every yaw vote, which gives looser
            starts for the optimization to tighten. Position masks are
            not used here: every target votes as a full position.
        max_votes_per_target : int
            Votes each target casts at most, drawn at random, so that a
            target with many matches does not drown the others.
        min_separation : float, optional
            Distance in meters (yaw counted as ``yaw * xy_resolution /
            yaw_resolution``) the returned bases keep from each other.
            Defaults to ``2 * xy_resolution``.
        base_bounds : tuple, optional
            ``((x_min, x_max), (y_min, y_max))``; votes outside are
            ignored.
        base_height : float
            Height the root link stands at, as in
            :func:`find_base_placement`.
        is_valid_base : callable, optional
            ``f(Coordinates) -> bool`` rejecting cells where the base may
            not stand, e.g. inside furniture.
        seed : int
            Random seed for the vote subsampling.

        Returns
        -------
        BaseCandidates
            Best first: most targets covered, then most votes. Empty when
            no cell got a vote.
        """
        if self.positions is None:
            raise RuntimeError('Call compute() or load() first')
        rng = np.random.RandomState(seed)
        target_pos, target_rot = _targets_to_pos_rot(target_coords)
        n_targets = len(target_pos)
        masks = _resolve_masks(rotation_mask, n_targets)
        n_yaw = max(1, int(np.ceil(2 * np.pi / yaw_resolution)))
        yaw_samples = (np.arange(n_yaw) + 0.5) * (2 * np.pi / n_yaw) - np.pi
        if min_separation is None:
            min_separation = 2.0 * xy_resolution

        all_bases, all_samples, all_targets = [], [], []
        for t in range(n_targets):
            constrained = masks[t] != 0
            axis = None
            if np.all(constrained):
                mode = 'full'
            elif np.count_nonzero(constrained) == 2:
                mode = 'axis'
                axis = int(np.where(~constrained)[0][0])
            else:
                mode = 'none'
            idx = self._matches(
                target_pos[t, 2] - base_height, target_rot[t], mode, axis,
                z_tolerance, tilt_tolerance)
            cap = max_votes_per_target if mode != 'none' \
                else max(1, max_votes_per_target // n_yaw)
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            bases, sample = self._bases_for(
                idx, target_pos[t], target_rot[t], mode, axis, yaw_samples,
                tilt_tolerance)
            all_bases.append(bases)
            all_samples.append(sample)
            all_targets.append(np.full(len(sample), t))
        bases = np.concatenate(all_bases)
        samples = np.concatenate(all_samples)
        targets = np.concatenate(all_targets)
        if base_bounds is not None:
            keep = ((bases[:, 0] >= base_bounds[0][0])
                    & (bases[:, 0] <= base_bounds[0][1])
                    & (bases[:, 1] >= base_bounds[1][0])
                    & (bases[:, 1] <= base_bounds[1][1]))
            bases, samples, targets = bases[keep], samples[keep], \
                targets[keep]
        n_opt = len(self._non_mimic)
        if len(bases) == 0:
            return BaseCandidates(
                np.zeros((0, 3)), np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, n_targets, n_opt)), bases, targets, base_height)

        # A vote counts for its cell and the 26 around it, so a base
        # lying on a cell boundary is not split between two cells.
        ix = np.floor(bases[:, 0] / xy_resolution).astype(np.int64)
        iy = np.floor(bases[:, 1] / xy_resolution).astype(np.int64)
        iyaw = np.floor((bases[:, 2] + np.pi) / yaw_resolution).astype(
            np.int64) % n_yaw
        offsets = np.stack(np.meshgrid(
            [-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing='ij'),
            axis=-1).reshape(-1, 3)
        ex = (ix[:, None] + offsets[:, 0] - ix.min() + 1).ravel()
        ey = (iy[:, None] + offsets[:, 1] - iy.min() + 1).ravel()
        eyaw = ((iyaw[:, None] + offsets[:, 2]) % n_yaw).ravel()
        ny = int(iy.max() - iy.min()) + 3
        key = (ex * ny + ey) * n_yaw + eyaw
        exp_vote = np.repeat(np.arange(len(bases)), len(offsets))
        _, exp_cell, votes = np.unique(
            key, return_inverse=True, return_counts=True)
        exp_cell = exp_cell.ravel()
        n_cells = len(votes)
        covered = np.zeros((n_cells, n_targets), dtype=bool)
        covered[exp_cell, targets[exp_vote]] = True
        coverage = covered.sum(axis=1)
        order = np.lexsort((-votes, -coverage))

        # A cell's base is the mean of its votes, yaw on the circle.
        sum_xy = np.zeros((n_cells, 2))
        np.add.at(sum_xy, exp_cell, bases[exp_vote, :2])
        sum_cs = np.zeros((n_cells, 2))
        np.add.at(sum_cs, exp_cell, np.stack(
            [np.cos(bases[exp_vote, 2]), np.sin(bases[exp_vote, 2])], 1))
        cell_bases = np.concatenate([
            sum_xy / votes[:, None],
            np.arctan2(sum_cs[:, 1], sum_cs[:, 0])[:, None]], axis=1)

        yaw_scale = xy_resolution / yaw_resolution
        chosen = []
        for c in order:
            b = cell_bases[c]
            if any(np.hypot(*(b[:2] - cell_bases[o][:2]))
                   + abs(normalize_angle(b[2] - cell_bases[o][2])) * yaw_scale
                   < min_separation for o in chosen):
                continue
            if is_valid_base is not None:
                if not is_valid_base(_planar_coords(*b, z=base_height)):
                    continue
            chosen.append(c)
            if len(chosen) == n_candidates:
                break
        chosen = np.array(chosen, dtype=np.int64)

        seeds = np.full((len(chosen), n_targets, n_opt), np.nan)
        for k, c in enumerate(chosen):
            in_cell = exp_vote[exp_cell == c]
            # Per target, the vote whose arm configuration is furthest
            # from the joint limits.
            margin = self._margins[samples[in_cell]]
            in_cell = in_cell[np.argsort(-margin, kind='stable')]
            _, first = np.unique(targets[in_cell], return_index=True)
            seeds[k, targets[in_cell[first]]] = \
                self.angles[samples[in_cell[first]]]
        return BaseCandidates(
            cell_bases[chosen], coverage[chosen], votes[chosen], seeds,
            bases, targets, base_height)

    def save(self, filepath):
        """Save the samples to a ``.npz`` file."""
        np.savez_compressed(
            filepath, positions=self.positions, rotations=self.rotations,
            angles=self.angles, joint_names=np.array(self.joint_names))

    def load(self, filepath):
        """Load samples saved by :meth:`save` for the same chain.

        Returns
        -------
        InverseReachabilityMap
            Self, for chaining.
        """
        data = np.load(filepath)
        names = [str(n) for n in data['joint_names']]
        if names != self.joint_names:
            raise ValueError(
                '{} was built for joints {}, not {}'.format(
                    filepath, names, self.joint_names))
        self._store(data['positions'], data['rotations'], data['angles'])
        return self
