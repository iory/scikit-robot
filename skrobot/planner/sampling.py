"""Sampling-based planning in joint space.

An optimizer such as :func:`skrobot.planner.collision_aware.plan_trajectory`
improves the trajectory it is started from and stays in its basin: seeded
with a straight joint-space path that runs through an obstacle, it can end
up pressed against that obstacle rather than around it. A sampling planner
has the opposite profile -- it finds a way around, in whatever homotopy
class, but the way is jagged -- so the two are run in sequence: this module
finds a collision-free path, the optimizer smooths it.

:func:`rrt_connect` is the bidirectional RRT of Kuffner and LaValle [1]_,
which grows one tree from the start and one from the goal and tries to
join them after every extension; :func:`shortcut` removes the detours a
random tree leaves behind.

References
----------
.. [1] J. J. Kuffner and S. M. LaValle. "RRT-Connect: An Efficient
   Approach to Single-Query Path Planning." IEEE International Conference
   on Robotics and Automation (ICRA), 2000.
"""

import time

import numpy as np


_BATCH_CHUNK = 8


def _segment_is_valid(is_valid, a, b, resolution, is_valid_batch=None):
    """Whether the straight motion from ``a`` to ``b`` is valid throughout.

    Checked at points no more than ``resolution`` apart in max-norm, end
    excluded (the caller has checked the endpoints). With
    ``is_valid_batch`` every point is checked in one call.
    """
    n = int(np.ceil(np.max(np.abs(b - a)) / resolution))
    if n <= 1:
        return True
    if is_valid_batch is not None:
        # A few points per call: most candidate motions a shortcut tries
        # fail, and a long one checked all at once would pay for every
        # point where the first bad one settles it. The coarse points
        # (the midpoint, the quarters) go first, so a motion through an
        # obstacle usually fails on the first call.
        fractions = np.arange(1, n) / float(n)
        order = np.argsort(np.abs(fractions - 0.5) + 1e-3 * fractions,
                           kind='stable')
        fractions = fractions[order]
        for chunk in range(0, len(fractions), _BATCH_CHUNK):
            part = fractions[chunk:chunk + _BATCH_CHUNK]
            points = a[None, :] + (b - a)[None, :] * part[:, None]
            if not np.all(is_valid_batch(points)):
                return False
        return True
    for i in range(1, n):
        if not is_valid(a + (b - a) * (i / float(n))):
            return False
    return True


class _Tree(object):

    def __init__(self, root):
        self.nodes = [np.asarray(root, dtype=np.float64)]
        self.parents = [-1]

    def nearest(self, q):
        nodes = np.asarray(self.nodes)
        return int(np.argmin(np.max(np.abs(nodes - q), axis=1)))

    def add(self, q, parent):
        self.nodes.append(np.asarray(q, dtype=np.float64))
        self.parents.append(int(parent))
        return len(self.nodes) - 1

    def path_to_root(self, index):
        path = []
        while index >= 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        return path


def rrt_connect(start, goal, is_valid, lower, upper, step=0.2,
                resolution=0.05, max_iterations=5000, time_limit=5.0,
                goal_bias=0.1, line_bias=0.5, line_sigma=0.5, seed=0,
                is_valid_batch=None):
    """Find a collision-free joint-space path with bidirectional RRT.

    Parameters
    ----------
    start : array-like
        Joint configuration, valid.
    goal : array-like
        A valid joint configuration, or several as an ``(n, n_joints)``
        array -- the goal tree is then rooted at all of them and the
        path ends at whichever the start tree reaches, which on an arm
        with several ways to reach a pose finds a way far more often.
    is_valid : callable
        ``q -> bool``, whether a configuration is collision-free.
    lower, upper : array-like
        Joint limits the samples are drawn from.
    step : float
        How far (max-norm, rad) a tree grows toward a sample at a time.
    resolution : float
        Spacing (max-norm) of the validity checks along an extension.
        This is what the returned path is checked at and all it can be
        trusted to: a violation narrower than ``resolution`` can pass
        between two check points, so a caller that needs more than that
        must either check the path itself at a finer spacing or give
        ``is_valid`` a margin of its own.
    max_iterations : int
        Sample budget.
    time_limit : float
        Wall-clock budget in seconds; the search stops at whichever of
        the two budgets runs out first.
    goal_bias : float
        Fraction of samples that aim straight at the other tree's root.
    line_bias : float
        Fraction of the remaining samples drawn around the straight
        line from start to goal -- a point on it plus Gaussian noise of
        ``line_sigma`` per joint -- rather than uniformly over the joint
        limits. An arm's detour around an obstacle lies near that line;
        most of joint space does not, and uniform samples there are
        wasted.
    line_sigma : float
        Spread (rad, or m) of the samples around the line.
    seed : int
        Random seed, so a run is reproducible.

    Returns
    -------
    list[numpy.ndarray] or None
        Configurations from ``start`` to the goal reached whose
        consecutive straight motions are valid, or ``None`` when no
        path was found within the budgets.
    """
    start = np.asarray(start, dtype=np.float64)
    goals = np.atleast_2d(np.asarray(goal, dtype=np.float64))
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    rng = np.random.RandomState(seed)
    goal_tree = _Tree(goals[0])
    for extra in goals[1:]:
        goal_tree.add(extra, -1)
    trees = (_Tree(start), goal_tree)
    deadline = time.time() + float(time_limit)

    def extend(tree, q_target):
        """Grow ``tree`` toward ``q_target``; return (status, node index).

        ``status`` is 'reached' when the target itself was added,
        'advanced' when a step toward it was, 'trapped' otherwise.
        """
        near = tree.nearest(q_target)
        q_near = tree.nodes[near]
        delta = q_target - q_near
        span = float(np.max(np.abs(delta)))
        if span <= step:
            q_new, status = q_target, 'reached'
        else:
            q_new, status = q_near + delta * (step / span), 'advanced'
        if is_valid_batch is not None:
            n_sub = int(np.ceil(span / resolution))
            fractions = np.arange(1, max(n_sub, 1) + 1) / float(max(n_sub, 1))
            points = q_near[None, :] + (q_new - q_near)[None, :] \
                * fractions[:, None]
            if not np.all(is_valid_batch(points)):
                return 'trapped', None
        elif not is_valid(q_new) or not _segment_is_valid(
                is_valid, q_near, q_new, resolution):
            return 'trapped', None
        return status, tree.add(q_new, near)

    def connect(tree, q_target):
        status = 'advanced'
        index = None
        while status == 'advanced':
            status, index = extend(tree, q_target)
        return status, index

    for iteration in range(int(max_iterations)):
        if time.time() > deadline:
            return None
        a, b = trees[iteration % 2], trees[(iteration + 1) % 2]
        if rng.uniform() < goal_bias:
            roots = [i for i, p in enumerate(b.parents) if p < 0]
            sample = b.nodes[roots[rng.randint(len(roots))]]
        elif rng.uniform() < line_bias:
            end = goals[rng.randint(len(goals))]
            along = start + (end - start) * rng.uniform()
            sample = np.clip(along + rng.normal(0.0, line_sigma, start.shape),
                             lower, upper)
        else:
            sample = rng.uniform(lower, upper)
        status, index_a = extend(a, sample)
        if status == 'trapped':
            continue
        status, index_b = connect(b, a.nodes[index_a])
        if status != 'reached':
            continue
        path_a = a.path_to_root(index_a)[::-1]
        path_b = b.path_to_root(index_b)
        path = path_a + path_b[1:]
        if a is trees[1]:
            path = path[::-1]
        return path
    return None


def shortcut(path, is_valid, resolution=0.05, iterations=100, seed=0,
             is_valid_batch=None):
    """Remove detours from a path by straight-line shortcuts.

    Repeatedly picks two points on the path and replaces what lies
    between them by the straight motion when that motion is valid.

    Parameters
    ----------
    path : list[numpy.ndarray]
        Configurations whose consecutive straight motions are valid.
    is_valid : callable
        ``q -> bool``.
    resolution : float
        Spacing (max-norm) of the validity checks along a shortcut.
    iterations : int
        Shortcut attempts.
    seed : int
        Random seed.

    Returns
    -------
    list[numpy.ndarray]
        The shortened path, same endpoints.
    """
    path = [np.asarray(q, dtype=np.float64) for q in path]
    rng = np.random.RandomState(seed)
    for _ in range(int(iterations)):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), 2, replace=False))
        if j - i < 2:
            continue
        if _segment_is_valid(is_valid, path[i], path[j], resolution,
                             is_valid_batch=is_valid_batch):
            path = path[:i + 1] + path[j:]
    return path


def resample(path, n_points):
    """Spread ``n_points`` configurations evenly along a path.

    Even in joint-space arc length (max-norm), so the seed a planner
    hands to an optimizer has comparable steps everywhere.

    Parameters
    ----------
    path : list[numpy.ndarray]
        Configurations; consecutive ones are joined by straight motions.
    n_points : int
        Number of configurations wanted, endpoints included.

    Returns
    -------
    numpy.ndarray
        ``(n_points, n_joints)``, first and last rows the path's ends.
    """
    path = np.asarray(path, dtype=np.float64)
    if len(path) == 1:
        return np.tile(path, (int(n_points), 1))
    lengths = np.max(np.abs(np.diff(path, axis=0)), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    targets = np.linspace(0.0, total, int(n_points))
    out = np.empty((int(n_points), path.shape[1]))
    for k, s in enumerate(targets):
        i = int(min(np.searchsorted(cumulative, s, side='right') - 1,
                    len(lengths) - 1))
        seg = lengths[i]
        t = 0.0 if seg <= 0.0 else (s - cumulative[i]) / seg
        out[k] = path[i] + (path[i + 1] - path[i]) * min(max(t, 0.0), 1.0)
    out[0] = path[0]
    out[-1] = path[-1]
    return out


def smooth_path(path, is_valid, n_points, rounds=3, resolution=0.05,
                iterations=200, seed=0, is_valid_batch=None):
    """Shortcut a path, resample it and round its corners, all checked.

    The sampling planner's own finishing pass, with no optimizer: random
    shortcuts remove the tree's detours, the path is resampled to
    ``n_points`` evenly spaced waypoints, and each interior waypoint is
    then pulled toward the mean of its neighbours whenever the moved
    waypoint and both motions around it stay valid. Every step keeps
    the path valid under ``is_valid``; nothing is left to a solver.

    Parameters
    ----------
    path : numpy.ndarray
        ``(m, n_joints)`` waypoints from start to goal.
    is_valid : callable
        ``f(q) -> bool``.
    n_points : int
        Waypoints in the result.
    rounds : int
        Passes of the corner rounding.
    resolution : float
        Largest max-norm step between validity checks along a motion.
    iterations : int
        Shortcut attempts.
    seed : int
        Seed of the shortcut's random pairs.
    is_valid_batch : callable or None
        ``f(points) -> (m,) bools`` checking many configurations in one
        call; used for every motion when given.

    Returns
    -------
    numpy.ndarray
        ``(n_points, n_joints)`` waypoints, endpoints unchanged.
    """
    path = shortcut(np.asarray(path, dtype=np.float64), is_valid,
                    resolution=resolution, iterations=iterations, seed=seed,
                    is_valid_batch=is_valid_batch)
    path = resample(path, n_points)
    for _ in range(int(rounds)):
        rounded = path.copy()
        for i in range(1, len(path) - 1):
            candidate = 0.25 * path[i - 1] + 0.5 * path[i] + 0.25 * path[i + 1]
            if _segment_is_valid(is_valid, rounded[i - 1], candidate,
                                 resolution, is_valid_batch=is_valid_batch) \
                    and _segment_is_valid(is_valid, candidate, path[i + 1],
                                          resolution,
                                          is_valid_batch=is_valid_batch) \
                    and (is_valid_batch(candidate[None])[0]
                         if is_valid_batch is not None
                         else is_valid(candidate)):
                rounded[i] = candidate
        path = rounded
    return path
