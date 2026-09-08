"""A robot's own collision model, built once and reused.

Everything that avoids collisions -- the collision checker, the trajectory
optimizer's self-collision cost, a collision-aware IK -- needs the same two
things from a robot: a cheap proxy geometry for every link, and the list of
link pairs that are worth checking against each other. Until now each caller
rebuilt both by hand (``RobotCollisionChecker.add_link`` per link, then
``setup_self_collision_pairs``), and none of them excluded the pairs that
touch by design, so a legitimate rest pose read as "in collision" on robots
like the PR2 whose links overlap at rest.

:class:`RobotCollisionModel` derives both from the robot itself, the way
MoveIt's setup assistant derives an SRDF: it drops parent/child pairs, the
pairs already in contact at the robot's default pose, and the pairs that
collide in nearly every random configuration (which means their proxies
overlap by construction). The result is keyed by the collision meshes and
joint limits and persisted, so a robot pays for the derivation once.
"""

import hashlib
import json
import os
import tempfile
import warnings

import numpy as np

from skrobot.collision.robot_collision import RobotCollisionChecker
from skrobot.collision.self_collision import is_fcl_available
from skrobot.collision.self_collision import REST_MARGIN
from skrobot.collision.self_collision import SelfCollision


_CACHE_VERSION = 1


def _default_cache_dir():
    """Return the directory holding cached collision models.

    Returns
    -------
    str
        ``<skrobot cache dir>/collision_model``. Resolved on every call
        because the skrobot cache dir honours ``SKROBOT_CACHE_DIR``.
    """
    from skrobot.data import get_cache_dir
    return os.path.join(get_cache_dir(), 'collision_model')


def _mesh_digest(mesh):
    """Return a content hash of a mesh's vertices and faces.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to hash.

    Returns
    -------
    str
        SHA-1 hex digest. A translated copy hashes differently, which is
        what a cache keyed on link-frame geometry needs.
    """
    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    h = hashlib.sha1()
    h.update(json.dumps([list(vertices.shape), list(faces.shape)]).encode())
    h.update(vertices.tobytes())
    h.update(faces.tobytes())
    return h.hexdigest()


def _ancestor_names(link):
    """Return the names of ``link`` and its ancestors, nearest first."""
    names = []
    current = link
    while current is not None:
        names.append(current.name)
        current = getattr(current, 'parent_link', None)
    return names


def kinematic_distance(link_a, link_b):
    """Return the number of links between two links along the tree.

    Parameters
    ----------
    link_a : skrobot.model.Link
        First link.
    link_b : skrobot.model.Link
        Second link.

    Returns
    -------
    int or float
        Steps from ``link_a`` up to the closest common ancestor plus steps
        from ``link_b`` up to it: 1 for parent and child, 2 for siblings or
        grandparent and grandchild. ``inf`` if the links share no ancestor.
    """
    ancestors_a = _ancestor_names(link_a)
    ancestors_b = _ancestor_names(link_b)
    index_a = {name: i for i, name in enumerate(ancestors_a)}
    for i, name in enumerate(ancestors_b):
        if name in index_a:
            return index_a[name] + i
    return float('inf')


def _has_collision_mesh(link):
    mesh = getattr(link, 'collision_mesh', None)
    if mesh is None:
        return False
    if getattr(mesh, 'is_empty', False):
        return False
    return len(getattr(mesh, 'faces', ())) > 0


def _atomic_write_json(path, payload):
    """Write ``payload`` as JSON to ``path`` without exposing a partial file."""
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, staging = tempfile.mkstemp(prefix='.collision_model.', suffix='.tmp',
                                   dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(staging, path)
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


class RobotCollisionModel(object):
    """Collision proxies and self-collision pairs derived from a robot.

    Built lazily by :attr:`skrobot.model.RobotModel.collision_model`; you
    rarely construct it yourself. Construct it directly (or call
    :meth:`RobotModel.build_collision_model`) to change how the pairs are
    derived.

    Self-collision pairs start from every unordered pair of links that have
    a collision mesh, then drop:

    - ``adjacent``: parent/child pairs (kinematic distance below 2), which
      touch at their joint by construction;
    - ``default_pose``: pairs already in contact at the robot's default pose
      (all joints at zero, clipped into their limits), i.e. parts that touch
      by design;
    - ``always``: pairs colliding in at least ``always_fraction`` of
      ``n_samples`` uniformly random configurations, which means their proxy
      geometry overlaps whatever the robot does.

    Contacts are tested with FCL on the exact collision meshes, using each
    mesh's convex hull only as the broadphase. Without the optional
    ``python-fcl`` package the default-pose test falls back to the
    sphere/capsule proxies and the random sampling is skipped;
    :attr:`method` records which was used. Proxies are inflated, so that
    path excludes more pairs than the mesh-based one and a warning says so.

    Two query surfaces come out of it. :attr:`checker` is a
    :class:`RobotCollisionChecker` over sphere/capsule proxies -- cheap,
    differentiable, but conservative (:attr:`proxy_overlap_pairs` says how
    much for this robot). :meth:`in_self_collision` answers on the meshes.

    The derivation is persisted under ``cache_dir``, keyed by the collision
    meshes, joint limits and these settings, so it runs once per robot.

    Parameters
    ----------
    robot_model : skrobot.model.RobotModel
        The robot. Its pose is restored after the derivation.
    n_samples : int
        Random configurations sampled for the ``always`` test.
    always_fraction : float
        Fraction of samples a pair must collide in to count as ``always``.
    margin : float
        Penetration depth (m) below which a contact is not a collision.
    seed : int
        Seed for the random configurations, so the result is reproducible.
    cache_dir : str or None
        Where derivations are persisted. ``None`` uses the skrobot cache.
    use_cache : bool
        If False, always derive and overwrite the cached entry.

    Attributes
    ----------
    link_list : list[skrobot.model.Link]
        The links that carry a collision mesh, in robot order.
    method : str
        ``'fcl'`` or ``'primitive'`` -- how contacts were tested.
    from_cache : bool
        Whether the pairs were read back from disk.
    excluded : dict
        The dropped pairs by category: ``'adjacent'``, ``'default_pose'``
        and ``'always'``, each a list of ``(name, name)`` tuples.
    proxy_overlap_pairs : list[tuple[str, str]]
        Checked pairs whose proxies overlap at the default pose although
        the meshes do not -- the pairs :attr:`checker` over-reports.
    """

    def __init__(self, robot_model, n_samples=200, always_fraction=0.95,
                 margin=REST_MARGIN, seed=0, cache_dir=None, use_cache=True):
        self.robot_model = robot_model
        self.link_list = [link for link in robot_model.link_list
                          if _has_collision_mesh(link)]
        self.n_samples = int(n_samples)
        self.always_fraction = float(always_fraction)
        self.margin = float(margin)
        self.seed = int(seed)
        self._cache_dir = cache_dir if cache_dir is not None \
            else _default_cache_dir()
        self._checker = None
        self._gridsdf_data = {}

        self.method = None
        self.from_cache = False
        self.excluded = {}
        self.proxy_overlap_pairs = []
        self._pair_names = []
        self._self_collision = None

        cached = self._read_cache() if use_cache else None
        if cached is not None:
            self.method = cached['method']
            self.excluded = {key: [tuple(p) for p in value]
                             for key, value in cached['excluded'].items()}
            self._pair_names = [tuple(p) for p in cached['pairs']]
            self.proxy_overlap_pairs = [
                tuple(p) for p in cached['proxy_overlap']]
            self.from_cache = True
        else:
            self._pair_names, self.excluded, self.method = self._derive()
            self.proxy_overlap_pairs = self._proxy_overlaps(self._pair_names)
            self._write_cache()

    # ------------------------------------------------------------------
    # derivation
    # ------------------------------------------------------------------
    def _derive(self):
        robot = self.robot_model
        links = self.link_list
        names = [link.name for link in links]

        candidates = []
        adjacent = []
        for i in range(len(links)):
            for j in range(i + 1, len(links)):
                pair = tuple(sorted((names[i], names[j])))
                if kinematic_distance(links[i], links[j]) < 2:
                    adjacent.append(pair)
                else:
                    candidates.append(pair)
        candidate_set = set(candidates)

        pose = robot.angle_vector()
        try:
            robot.init_pose()
            if is_fcl_available():
                default_pose, always = self._derive_with_fcl(candidate_set)
                method = 'fcl'
            else:
                warnings.warn(
                    "python-fcl is not installed, so self-collision pairs "
                    "for '{}' were derived from sphere/capsule proxies at "
                    "the default pose only. Proxies are inflated, so more "
                    "pairs are excluded than with the mesh-based test, and "
                    "pairs that always collide are not detected. Install "
                    "it with: pip install python-fcl".format(
                        getattr(robot, 'name', type(robot).__name__)),
                    RuntimeWarning, stacklevel=3)
                default_pose = self._default_pose_contacts_primitive(
                    candidate_set)
                always = set()
                method = 'primitive'
        finally:
            robot.angle_vector(pose)

        pairs = [pair for pair in candidates
                 if pair not in default_pose and pair not in always]
        excluded = {
            'adjacent': adjacent,
            'default_pose': sorted(default_pose),
            'always': sorted(always),
        }
        return pairs, excluded, method

    def _proxy_overlaps(self, pair_names):
        """Kept pairs whose sphere/capsule proxies overlap at the default pose.

        The proxies are fatter than the meshes, so some pairs that are clear
        on the mesh read as colliding through :attr:`checker`. Recording
        them tells a user how conservative the proxy checker is for this
        robot.
        """
        robot = self.robot_model
        checker = RobotCollisionChecker(robot)
        checker.add_links(self.link_list)
        indices = self._geometry_indices(checker)
        index_pairs = []
        owners = []
        for pair in pair_names:
            for i in indices[pair[0]]:
                for j in indices[pair[1]]:
                    index_pairs.append((i, j))
                    owners.append(pair)
        if not index_pairs:
            return []
        checker.set_self_collision_pairs(index_pairs)
        pose = robot.angle_vector()
        try:
            robot.init_pose()
            distances = np.asarray(checker.compute_self_collision_distances())
        finally:
            robot.angle_vector(pose)
        return sorted({pair for pair, distance in zip(owners, distances)
                       if distance <= 0.0})

    def _make_self_collision(self):
        """Build the exact-mesh self-collision query at the default pose."""
        robot = self.robot_model
        meshes = {link.name: link.collision_mesh for link in self.link_list}
        pose = robot.angle_vector()
        try:
            robot.init_pose()
            # Convex hulls for the broadphase, every candidate verified on
            # the exact mesh: hulls alone report contacts the real geometry
            # never makes (the tucked Fetch arm against its torso, say).
            return SelfCollision(robot, meshes, margin=self.margin,
                                 hull=True, confirm=True)
        finally:
            robot.angle_vector(pose)

    def _derive_with_fcl(self, candidate_set):
        """Default-pose contacts and always-colliding pairs via FCL hulls."""
        robot = self.robot_model
        # Built at the default pose, so its baseline is exactly the set of
        # pairs in contact there (plus adjacency, already removed above).
        model = self._make_self_collision()
        # The same object answers in_self_collision() later; its baseline is
        # exactly the default-pose contact set this derivation excludes.
        self._self_collision = model
        default_pose = {tuple(sorted(pair)) for pair in model.baseline}
        default_pose &= candidate_set

        always = set()
        if self.n_samples > 0:
            lower = np.array(robot.joint_min_angles, dtype=np.float64)
            upper = np.array(robot.joint_max_angles, dtype=np.float64)
            lower = np.where(np.isfinite(lower), lower, -np.pi)
            upper = np.where(np.isfinite(upper), upper, np.pi)
            rng = np.random.RandomState(self.seed)
            hits = {}
            for _ in range(self.n_samples):
                robot.angle_vector(rng.uniform(lower, upper))
                for pair in model.colliding_pairs():
                    key = tuple(sorted(pair))
                    hits[key] = hits.get(key, 0) + 1
            threshold = self.always_fraction * self.n_samples
            always = {pair for pair, count in hits.items()
                      if pair in candidate_set and count >= threshold}
            always -= default_pose
        return default_pose, always

    def _default_pose_contacts_primitive(self, candidate_set):
        """Default-pose contacts via the sphere/capsule proxies."""
        checker = RobotCollisionChecker(self.robot_model)
        checker.add_links(self.link_list)
        indices = self._geometry_indices(checker)
        index_pairs = []
        name_pairs = []
        for pair in sorted(candidate_set):
            for i in indices[pair[0]]:
                for j in indices[pair[1]]:
                    index_pairs.append((i, j))
                    name_pairs.append(pair)
        checker.set_self_collision_pairs(index_pairs)
        distances = np.asarray(checker.compute_self_collision_distances())
        return {pair for pair, distance in zip(name_pairs, distances)
                if distance <= 0.0}

    @staticmethod
    def _geometry_indices(checker):
        """Map link name to the indices of its geometries in ``checker``."""
        indices = {}
        for i, geometry in enumerate(checker.link_geometries):
            indices.setdefault(geometry.link.name, []).append(i)
        return indices

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def _cache_key(self):
        robot = self.robot_model
        payload = {
            'version': _CACHE_VERSION,
            'links': [
                (link.name,
                 link.parent_link.name if link.parent_link is not None
                 else None,
                 _mesh_digest(link.collision_mesh))
                for link in self.link_list],
            'joints': [(joint.name, float(joint.min_angle),
                        float(joint.max_angle))
                       for joint in robot.joint_list],
            'params': {
                'n_samples': self.n_samples,
                'always_fraction': self.always_fraction,
                'margin': self.margin,
                'seed': self.seed,
            },
            'fcl': is_fcl_available(),
        }
        h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode())
        return h.hexdigest()[:16]

    def _cache_path(self):
        return os.path.join(self._cache_dir, self._cache_key() + '.json')

    def _read_cache(self):
        try:
            with open(self._cache_path(), encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get('version') != _CACHE_VERSION:
            return None
        if not all(key in data for key in
                   ('method', 'pairs', 'excluded', 'proxy_overlap')):
            return None
        return data

    def _write_cache(self):
        payload = {
            'version': _CACHE_VERSION,
            'method': self.method,
            'pairs': [list(pair) for pair in self._pair_names],
            'excluded': {key: [list(pair) for pair in value]
                         for key, value in self.excluded.items()},
            'proxy_overlap': [list(pair) for pair in self.proxy_overlap_pairs],
        }
        try:
            _atomic_write_json(self._cache_path(), payload)
        except OSError as exc:
            warnings.warn(
                'could not persist the collision model to {}: {}'.format(
                    self._cache_path(), exc),
                RuntimeWarning, stacklevel=3)

    # ------------------------------------------------------------------
    # public surface
    # ------------------------------------------------------------------
    @property
    def self_collision_pair_names(self):
        """list[tuple[str, str]]: The pairs to check, as sorted link names."""
        return list(self._pair_names)

    @property
    def self_collision_pairs(self):
        """list[tuple[Link, Link]]: The pairs to check, as links."""
        by_name = {link.name: link for link in self.link_list}
        return [(by_name[a], by_name[b]) for a, b in self._pair_names]

    @property
    def self_collision_index_pairs(self):
        """list[tuple[int, int]]: The pairs as indices into ``link_list``."""
        index = {link.name: i for i, link in enumerate(self.link_list)}
        return [(index[a], index[b]) for a, b in self._pair_names]

    @property
    def self_collision(self):
        """SelfCollision: exact-mesh self-collision query over the pairs.

        Built on first access with the rest baseline taken at the default
        pose, so :meth:`SelfCollision.new_pairs` reports exactly the pairs
        this model checks. Needs the optional ``python-fcl`` package.
        """
        if self._self_collision is None:
            if not is_fcl_available():
                raise RuntimeError(
                    'the exact self-collision query needs the optional '
                    "'python-fcl' package -- install it with: pip install "
                    'python-fcl (the proxy-based `checker` works without it)')
            self._self_collision = self._make_self_collision()
        return self._self_collision

    def self_colliding_pairs(self):
        """Return the pairs in contact at the current pose, on the meshes.

        Returns
        -------
        list[tuple[str, str]]
            Sorted link-name pairs among :attr:`self_collision_pair_names`
            whose meshes penetrate by more than ``margin`` right now.
        """
        always = set(self.excluded.get('always', ()))
        return sorted(tuple(sorted(pair))
                      for pair in self.self_collision.new_pairs()
                      if tuple(sorted(pair)) not in always)

    def in_self_collision(self):
        """Return whether the robot self-collides at its current pose.

        Mesh-accurate (see :meth:`self_colliding_pairs`), unlike
        ``checker.is_collision_free()`` which uses the inflated proxies.

        Returns
        -------
        bool
            True if any checked pair is in contact.
        """
        return len(self.self_colliding_pairs()) > 0

    @property
    def checker(self):
        """RobotCollisionChecker: proxies for every link, pairs already set.

        Built on first access. Evaluates the robot's *current* pose; add
        world obstacles to it with ``add_world_obstacle``.

        The sphere/capsule proxies are conservative: a link that clears
        another on the mesh can still overlap it through its proxy.
        :attr:`proxy_overlap_pairs` lists the checked pairs for which that
        is already so at the default pose, and :meth:`in_self_collision`
        answers on the meshes instead.
        """
        if self._checker is None:
            checker = RobotCollisionChecker(self.robot_model)
            checker.add_links(self.link_list)
            indices = self._geometry_indices(checker)
            index_pairs = []
            for a, b in self._pair_names:
                for i in indices[a]:
                    for j in indices[b]:
                        index_pairs.append((i, j))
            checker.set_self_collision_pairs(index_pairs)
            self._checker = checker
        return self._checker

    def gridsdf_self_data(self, dim_grid=40, n_surface=48):
        """Per-link GridSDF data for the differentiable self-collision cost.

        The same arrays
        :func:`~skrobot.planner.trajectory_optimization.gridsdf_collision.build_gridsdf_self_data`
        returns, but over the derived pairs rather than list-adjacency.
        Cached per ``(dim_grid, n_surface)``.

        Parameters
        ----------
        dim_grid : int
            GridSDF resolution per axis.
        n_surface : int
            Surface sample points kept per link.

        Returns
        -------
        dict
            See ``build_gridsdf_self_data``.
        """
        key = (int(dim_grid), int(n_surface))
        if key not in self._gridsdf_data:
            from skrobot.planner.trajectory_optimization.gridsdf_collision import build_gridsdf_self_data
            self._gridsdf_data[key] = build_gridsdf_self_data(
                self.robot_model, self.link_list,
                dim_grid=dim_grid, n_surface=n_surface,
                link_pairs=self.self_collision_index_pairs)
        return self._gridsdf_data[key]

    def describe(self):
        """Return a one-paragraph summary of what the model holds."""
        return (
            '{} collision links, {} self-collision pairs '
            '(method={}, {}); excluded {} adjacent, {} touching at the '
            'default pose, {} always colliding; the sphere/capsule proxies '
            'already overlap for {} of the checked pairs at the default '
            'pose'.format(
                len(self.link_list), len(self._pair_names), self.method,
                'from cache' if self.from_cache else 'derived now',
                len(self.excluded.get('adjacent', ())),
                len(self.excluded.get('default_pose', ())),
                len(self.excluded.get('always', ())),
                len(self.proxy_overlap_pairs)))

    def __repr__(self):
        return '<RobotCollisionModel {}>'.format(self.describe())
