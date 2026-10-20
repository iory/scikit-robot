"""Contacts between a robot and a static environment over many postures.

The environment -- a fixture loaded from URDF, primitives, any mesh, concave
or not -- is sampled once into points on its surface, at most ``spacing``
apart, and the points go into a KD-tree. Each robot link is the convex hull
of its collision mesh, kept as half-spaces, so the signed distance of a
point to the link is the largest of its plane distances: exact inside the
link, where it is minus the depth, and a lower bound outside.

A posture's contacts are the environment points inside a link, or within
``margin`` of it. A link that crosses the environment's surface has a patch
of that surface inside it, so this finds every such overlap whose patch
holds a sample: anything deeper than about ``spacing``. A slot, a pocket
or a hole in the environment needs nothing special, since only its surface
is sampled. A robot link that is not convex is taken by its hull, which
errs toward reporting a contact. Missed is a link wholly inside an
environment part, which no motion reaches without crossing its surface
first.

The depth is that of the environment's surface inside the link, at its
deepest point. For a fingertip pushed into a block it is how far the block's
face reaches into the finger -- about the distance the finger has to move
to come free -- rather than how far the fingertip went below the face.
"""

import numpy as np


class CollisionReport(object):
    """Contacts of one :meth:`EnvironmentCollision.check` call.

    Each row is the deepest point of one (frame, robot link, environment
    link) triple closer than the margin.

    Attributes
    ----------
    robot_links : list of str
        Names of the robot links checked.
    environment_links : list of str
        Names of the environment parts.
    labels : list
        One label per frame, as given to ``check``, or the frame index.
    frame : numpy.ndarray (n,)
        Frame of each contact.
    robot_index : numpy.ndarray (n,)
        Index into ``robot_links``.
    environment_index : numpy.ndarray (n,)
        Index into ``environment_links``.
    distance : numpy.ndarray (n,)
        Signed distance of the deepest point, negative inside the link.
        Positive values are lower bounds on the clearance.
    point : numpy.ndarray (n, 3)
        That point on the environment's surface, in the world frame.
    margin : float
        Contacts are the triples with ``distance < margin``.
    """

    def __init__(self, robot_links, environment_links, labels, frame,
                 robot_index, environment_index, distance, point, margin):
        self.robot_links = list(robot_links)
        self.environment_links = list(environment_links)
        self.labels = list(labels)
        order = np.lexsort((distance, frame))
        self.frame = frame[order]
        self.robot_index = robot_index[order]
        self.environment_index = environment_index[order]
        self.distance = distance[order]
        self.point = point[order]
        self.margin = margin

    @property
    def n_frames(self):
        return len(self.labels)

    @property
    def min_distance(self):
        """Smallest distance per frame, ``inf`` where nothing came within
        the margin."""
        out = np.full(self.n_frames, np.inf)
        np.minimum.at(out, self.frame, self.distance)
        return out

    @property
    def in_collision(self):
        """Per frame, whether some environment point is inside a link."""
        return self.min_distance < 0

    @property
    def is_free(self):
        """True when no frame is in collision."""
        return not np.any(self.distance < 0)

    @property
    def max_depth(self):
        """Deepest penetration over all frames, 0 when there is none."""
        if len(self.distance) == 0:
            return 0.0
        return max(0.0, -float(np.min(self.distance)))

    def contacts(self, collisions_only=True):
        """The rows as dicts, deepest first.

        Parameters
        ----------
        collisions_only : bool
            Leave out the rows that are only within the margin.

        Returns
        -------
        list of dict
            ``frame``, ``label``, ``robot_link``, ``environment_link``,
            ``distance`` and ``depth`` (m), ``point`` (world, m).
        """
        rows = np.argsort(self.distance, kind='stable')
        if collisions_only:
            rows = rows[self.distance[rows] < 0]
        return [self._row(i) for i in rows]

    def _row(self, i):
        frame = int(self.frame[i])
        return {
            'frame': frame,
            'label': self.labels[frame],
            'robot_link': self.robot_links[self.robot_index[i]],
            'environment_link':
                self.environment_links[self.environment_index[i]],
            'distance': float(self.distance[i]),
            'depth': max(0.0, -float(self.distance[i])),
            'point': [float(v) for v in self.point[i]],
        }

    def pairs(self, collisions_only=True):
        """One summary per (robot link, environment link), deepest first.

        Parameters
        ----------
        collisions_only : bool
            Leave out the rows that are only within the margin.

        Returns
        -------
        list of dict
            ``robot_link``, ``environment_link``, ``n_frames`` touched,
            ``max_depth`` and the ``worst`` row (as in :meth:`contacts`),
            ``frames`` -- the frames as ``[first, last]`` runs -- and the
            distinct ``labels`` of those frames in order.
        """
        keep = self.distance < (0 if collisions_only else np.inf)
        out = []
        keys = self.robot_index * len(self.environment_links) \
            + self.environment_index
        for key in np.unique(keys[keep]):
            rows = np.flatnonzero(keep & (keys == key))
            worst = rows[np.argmin(self.distance[rows])]
            frames = np.unique(self.frame[rows])
            labels = []
            for f in frames:
                if self.labels[f] not in labels:
                    labels.append(self.labels[f])
            row = self._row(worst)
            out.append({
                'robot_link': row['robot_link'],
                'environment_link': row['environment_link'],
                'n_frames': len(frames),
                'max_depth': row['depth'],
                'worst': row,
                'frames': _runs(frames),
                'labels': labels,
            })
        out.sort(key=lambda p: -p['max_depth'])
        return out

    def to_dict(self):
        """JSON-ready summary: counts, the pairs, and every collision row."""
        return {
            'n_frames': self.n_frames,
            'n_frames_in_collision': int(np.count_nonzero(self.in_collision)),
            'max_depth': self.max_depth,
            'margin': self.margin,
            'pairs': self.pairs(),
            'contacts': self.contacts(),
        }

    def summary(self, max_pairs=20):
        """Human-readable text: one line per colliding pair."""
        pairs = self.pairs()
        lines = ['{} of {} frames in collision, deepest {:.1f} mm'.format(
            int(np.count_nonzero(self.in_collision)), self.n_frames,
            self.max_depth * 1000)]
        for p in pairs[:max_pairs]:
            w = p['worst']
            lines.append(
                '  {} x {}: {:.1f} mm at frame {} ({}), {} frames {}'.format(
                    p['robot_link'], p['environment_link'],
                    p['max_depth'] * 1000, w['frame'], w['label'],
                    p['n_frames'], _format_runs(p['frames'])))
        if len(pairs) > max_pairs:
            lines.append('  ... {} more pairs'.format(len(pairs) - max_pairs))
        return '\n'.join(lines)


def _runs(frames):
    """Sorted frame indices as ``[first, last]`` runs of consecutive ones."""
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) > 1)
    starts = np.concatenate([[frames[0]], frames[breaks + 1]])
    ends = np.concatenate([frames[breaks], [frames[-1]]])
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def _format_runs(runs, max_runs=4):
    text = ', '.join(str(a) if a == b else '{}-{}'.format(a, b)
                     for a, b in runs[:max_runs])
    if len(runs) > max_runs:
        text += ', ...'
    return text


def sample_surface(mesh, spacing):
    """Points on a mesh's surface, no surface point farther than about
    ``spacing`` from one of them.

    Each triangle is covered by rows parallel to its longest edge, at most
    ``spacing`` apart, each row a run of points at most ``spacing`` apart;
    the vertices and edges are on the rows, so corners and creases are
    always sampled. A long thin triangle gets few rows, unlike subdividing
    it.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to sample.
    spacing : float
        Largest gap between neighboring samples.

    Returns
    -------
    numpy.ndarray (n, 3)
        The samples, in the mesh's frame.
    """
    tris = np.asarray(mesh.triangles, dtype=np.float64)
    if len(tris) == 0:
        return np.zeros((0, 3))
    edges = np.stack([tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 1],
                      tris[:, 0] - tris[:, 2]], axis=1)
    lengths = np.linalg.norm(edges, axis=2)
    longest = np.argmax(lengths, axis=1)
    # Rotate each triangle so that a-b is its longest edge.
    idx = (longest[:, None] + np.arange(3)[None, :]) % 3
    tris = np.take_along_axis(tris, idx[:, :, None], axis=1)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    base = np.linalg.norm(b - a, axis=1)
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    height = np.where(base > 0, area2 / np.maximum(base, 1e-300), 0.0)
    n_rows = np.ceil(height / spacing).astype(np.int64)
    # One entry per row: its triangle and its height fraction t.
    tri_of_row = np.repeat(np.arange(len(tris)), n_rows + 1)
    first_row = np.cumsum(n_rows + 1) - (n_rows + 1)
    k = np.arange(len(tri_of_row)) - first_row[tri_of_row]
    t = np.where(n_rows[tri_of_row] > 0,
                 k / np.maximum(n_rows[tri_of_row], 1), 0.0)
    start = a[tri_of_row] + t[:, None] * (c - a)[tri_of_row]
    end = b[tri_of_row] + t[:, None] * (c - b)[tri_of_row]
    n_pts = np.ceil((1 - t) * base[tri_of_row] / spacing).astype(np.int64)
    # One entry per point: its row and its fraction s along the row.
    row_of_pt = np.repeat(np.arange(len(tri_of_row)), n_pts + 1)
    first_pt = np.cumsum(n_pts + 1) - (n_pts + 1)
    j = np.arange(len(row_of_pt)) - first_pt[row_of_pt]
    s = np.where(n_pts[row_of_pt] > 0,
                 j / np.maximum(n_pts[row_of_pt], 1), 0.0)
    return start[row_of_pt] + s[:, None] * (end - start)[row_of_pt]


def _frames_of_poses(pose, point, dist, link, pose_of_frame):
    """Rows per link pose -> rows per frame having that pose."""
    frames_by_pose = np.argsort(pose_of_frame, kind='stable')
    n_frames = np.bincount(pose_of_frame)
    first = np.cumsum(n_frames) - n_frames
    repeat = n_frames[pose]
    row = np.repeat(np.arange(len(pose)), repeat)
    offset = np.arange(len(row)) - np.repeat(np.cumsum(repeat) - repeat,
                                             repeat)
    frame = frames_by_pose[first[pose][row] + offset]
    return frame, np.full(len(row), link), point[row], dist[row]


def _environment_parts(environment):
    """``(name, world-frame mesh)`` for every part of ``environment``."""
    import trimesh

    from skrobot.model import Link
    from skrobot.model import RobotModel

    parts = []
    for item in environment:
        if isinstance(item, RobotModel):
            parts.extend(_environment_parts(item.link_list))
        elif isinstance(item, Link):
            mesh = item.collision_mesh
            if mesh is None or len(mesh.faces) == 0:
                continue
            name = item.name if item.name else 'part{}'.format(len(parts))
            parts.append((name, mesh.copy().apply_transform(
                item.worldcoords().T())))
        elif isinstance(item, tuple) and len(item) == 2 \
                and isinstance(item[1], trimesh.Trimesh):
            parts.append((str(item[0]), item[1]))
        else:
            raise TypeError(
                'environment items are RobotModel, Link or (name, Trimesh); '
                'got {!r}'.format(type(item)))
    return parts


class _LinkHull(object):
    """A link's convex hull in its own frame, as half-spaces."""

    def __init__(self, mesh):
        hull = mesh.convex_hull
        self.normals = np.asarray(hull.face_normals, dtype=np.float64)
        self.offsets = np.einsum('fi,fi->f', self.normals,
                                 hull.triangles[:, 0])
        self.lower = hull.bounds[0]
        self.upper = hull.bounds[1]
        self.center = (self.lower + self.upper) / 2
        self.radius = float(np.max(np.linalg.norm(
            hull.vertices - self.center, axis=1)))

    def signed_distance(self, points):
        """Largest plane distance: exact inside, a lower bound outside."""
        return np.max(points @ self.normals.T - self.offsets, axis=1)


def link_transforms(robot, angle_vectors=None, base_pose=None, links=None):
    """World transforms of links for a batch of angle vectors.

    The robot is posed once per row and put back as it was.

    Parameters
    ----------
    robot : RobotModel
        Robot to pose.
    angle_vectors : array_like (B, n), optional
        Rows of ``robot.angle_vector()``; None takes the current posture.
    base_pose : Coordinates, optional
        Pose of the root link; None keeps the current one.
    links : list of Link, optional
        Links to report; defaults to ``robot.link_list``.

    Returns
    -------
    numpy.ndarray (B, len(links), 4, 4)
    """
    links = robot.link_list if links is None else links
    saved_av = robot.angle_vector().copy()
    saved_base = robot.copy_worldcoords()
    try:
        if base_pose is not None:
            robot.newcoords(base_pose.copy_worldcoords())
        if angle_vectors is None:
            return np.stack([lk.worldcoords().T() for lk in links])[None]
        out = []
        for av in np.atleast_2d(np.asarray(angle_vectors, dtype=np.float64)):
            robot.angle_vector(av)
            out.append(np.stack([lk.worldcoords().T() for lk in links]))
        return np.stack(out)
    finally:
        robot.newcoords(saved_base)
        robot.angle_vector(saved_av)


class EnvironmentCollision(object):
    """Robot links against a static, possibly concave, environment.

    Built once per environment; :meth:`check` then takes any number of
    postures at once and returns a :class:`CollisionReport` naming, per
    frame, the robot link, the environment part, the depth and the point.

    Parameters
    ----------
    robot : RobotModel
        Robot whose links are checked.
    environment : list
        :class:`~skrobot.model.RobotModel` (each link with a collision
        mesh is a part, e.g. a fixture loaded from URDF), any
        :class:`~skrobot.model.Link` with a collision mesh (the
        :mod:`skrobot.model.primitives`), or ``(name, trimesh.Trimesh)`` in
        the world frame. Taken where they are now.
    links : list of Link, optional
        Robot links to check; defaults to every link with a collision
        mesh.
    spacing : float
        Largest gap between environment samples, m. Overlaps shallower than
        about this may be missed; the cost grows with its inverse square.
    ignore : iterable of (str, str), optional
        ``(robot link, environment part)`` name pairs to skip, e.g. the
        robot's base on the plinth it stands on.

    Examples
    --------
    >>> from skrobot.collision import EnvironmentCollision
    >>> from skrobot.model.primitives import Box
    >>> from skrobot.models import Panda
    >>> robot = Panda()
    >>> table = Box(extents=[0.2, 0.2, 0.02], pos=[0.25, 0.0, 0.64],
    ...             name='table')
    >>> checker = EnvironmentCollision(robot, [table])
    >>> report = checker.check([robot.reset_pose()])
    >>> print(report.summary())  # doctest: +SKIP
    1 of 1 frames in collision, deepest 15.9 mm
      panda_hand x table: 15.9 mm at frame 0 (0), 1 frames 0
      panda_leftfinger x table: 9.8 mm at frame 0 (0), 1 frames 0
      panda_rightfinger x table: 9.8 mm at frame 0 (0), 1 frames 0
    """

    def __init__(self, robot, environment, links=None, spacing=0.002,
                 ignore=None):
        from scipy.spatial import cKDTree

        self.robot = robot
        if links is None:
            links = [lk for lk in robot.link_list
                     if lk.collision_mesh is not None
                     and len(lk.collision_mesh.faces) > 0]
        self.links = list(links)
        self.robot_links = [lk.name for lk in self.links]
        self._hulls = [_LinkHull(lk.collision_mesh) for lk in self.links]
        self.spacing = float(spacing)

        parts = _environment_parts(environment)
        if not parts:
            raise ValueError('the environment has no meshes')
        self.environment_links = [name for name, _ in parts]
        samples = [sample_surface(mesh, self.spacing) for _, mesh in parts]
        self.points = np.concatenate(samples)
        self.owner = np.repeat(np.arange(len(parts)),
                               [len(s) for s in samples])
        # One tree per part, so that a part out of a link's reach, or
        # ignored for it, costs nothing.
        self._trees = [cKDTree(s) for s in samples]
        self._offsets = np.cumsum([0] + [len(s) for s in samples])[:-1]
        self._bounds = np.stack([mesh.bounds for _, mesh in parts])

        ignore = set() if ignore is None else set(map(tuple, ignore))
        unknown = {r for r, _ in ignore} - set(self.robot_links)
        unknown |= {e for _, e in ignore} - set(self.environment_links)
        if unknown:
            raise ValueError('unknown link names in ignore: {}'.format(
                sorted(unknown)))
        env_index = {n: i for i, n in enumerate(self.environment_links)}
        # Per robot link, a mask over the parts it is checked against.
        self._checked = np.ones((len(self.links), len(parts)), dtype=bool)
        for r, e in ignore:
            self._checked[self.robot_links.index(r), env_index[e]] = False

    def check(self, angle_vectors=None, link_transforms=None, base_pose=None,
              margin=0.0, labels=None, chunk=1 << 20):
        """Contacts for a batch of postures.

        Parameters
        ----------
        angle_vectors : array_like (B, n), optional
            Rows of ``robot.angle_vector()``.
        link_transforms : array_like (B, L, 4, 4) or dict, optional
            World transforms instead of angle vectors, e.g. a recorded
            trajectory: an array over ``robot.link_list`` or a dict from
            link name to ``(B, 4, 4)``. Wins over ``angle_vectors``.
        base_pose : Coordinates, optional
            Root-link pose to use with ``angle_vectors``.
        margin : float
            Report the triples closer than this; 0 reports overlaps only.
        labels : list, optional
            One label per frame (a waypoint name, a step number) carried
            into the report; defaults to the frame index.
        chunk : int
            Candidate points handled per vectorized step, bounding memory.

        Returns
        -------
        CollisionReport
        """
        transforms = self._transforms(angle_vectors, link_transforms,
                                      base_pose)
        n_frames = transforms.shape[0]
        if labels is None:
            labels = list(range(n_frames))
        if len(labels) != n_frames:
            raise ValueError('{} labels for {} frames'.format(
                len(labels), n_frames))
        n_env = len(self.environment_links)
        found = []
        for li, hull in enumerate(self._hulls):
            # Each distinct pose once: a link near the root, or the whole
            # arm during a pause, stays put over many frames.
            poses, pose_of_frame = np.unique(
                transforms[:, li].reshape(n_frames, 16), axis=0,
                return_inverse=True)
            pose_of_frame = pose_of_frame.ravel()
            poses = poses.reshape(-1, 4, 4)
            rot = poses[:, :3, :3]
            pos = poses[:, :3, 3]
            centers = np.einsum('bij,j->bi', rot, hull.center) + pos
            reach = hull.radius + max(margin, 0.0)
            frames, points = [], []
            for ei in np.flatnonzero(self._checked[li]):
                lower, upper = self._bounds[ei]
                gap = np.maximum(lower - centers, 0) \
                    + np.maximum(centers - upper, 0)
                sel = np.flatnonzero(np.einsum('bi,bi->b', gap, gap)
                                     <= reach ** 2)
                if len(sel) == 0:
                    continue
                near = self._trees[ei].query_ball_point(centers[sel], reach)
                counts = np.fromiter((len(n) for n in near), dtype=np.int64,
                                     count=len(sel))
                if counts.sum() == 0:
                    continue
                frames.append(np.repeat(sel, counts))
                points.append(self._offsets[ei] + np.concatenate(
                    [np.asarray(n, dtype=np.int64) for n in near if len(n)]))
            if not frames:
                continue
            frames = np.concatenate(frames)
            points = np.concatenate(points)
            for s in range(0, len(points), chunk):
                f = frames[s:s + chunk]
                p = points[s:s + chunk]
                local = np.einsum('bji,bj->bi', rot[f],
                                  self.points[p] - pos[f])
                inside = np.all((local > hull.lower - margin)
                                & (local < hull.upper + margin), axis=1)
                f, p, local = f[inside], p[inside], local[inside]
                dist = hull.signed_distance(local)
                close = dist < margin
                found.append(_frames_of_poses(
                    f[close], p[close], dist[close], li, pose_of_frame))
        if found:
            frame, robot_index, point, dist = (
                np.concatenate(x) for x in zip(*found))
        else:
            frame = robot_index = point = np.zeros(0, dtype=np.int64)
            dist = np.zeros(0)
        env_index = self.owner[point]
        # Deepest point per (frame, robot link, environment part).
        key = (frame * len(self.links) + robot_index) * n_env + env_index
        order = np.lexsort((dist, key))
        first = np.ones(len(order), dtype=bool)
        first[1:] = key[order][1:] != key[order][:-1]
        rows = order[first]
        return CollisionReport(
            self.robot_links, self.environment_links, labels, frame[rows],
            robot_index[rows], env_index[rows], dist[rows],
            self.points[point[rows]], margin)

    def check_placement(self, result, margin=0.0):
        """Contacts of a :class:`~skrobot.kinematics.BasePlacementResult`:
        the robot on its base, at each target's angle vector.

        Usable as a ``score`` for
        :func:`~skrobot.kinematics.find_base_placement`, e.g.
        ``score=lambda r: r.min_joint_margin
        - checker.check_placement(r).max_depth * 100``.
        """
        n = len(result.angle_vectors)
        return self.check(result.angle_vectors, base_pose=result.base_pose,
                          margin=margin,
                          labels=['target {}'.format(i) for i in range(n)])

    def _transforms(self, angle_vectors, transforms, base_pose):
        if transforms is None:
            return link_transforms(self.robot, angle_vectors, base_pose,
                                   self.links)
        if isinstance(transforms, dict):
            return np.stack([np.asarray(transforms[n], dtype=np.float64)
                             for n in self.robot_links], axis=1)
        transforms = np.asarray(transforms, dtype=np.float64)
        index = [self.robot.link_list.index(lk) for lk in self.links]
        return transforms[:, index]
