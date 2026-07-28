"""Mesh-based self-collision queries and a joint-limit sweep.

Contacts present at the REST pose (parts touching by design) and parent/child
adjacency form the allowed baseline; :class:`SelfCollision` reports colliding
link pairs beyond it at the current pose, and :func:`sweep_limits` rotates each
joint until a NEW colliding pair appears -- that position (minus a small
margin) is the suggested limit.  A revolute joint that never collides within
``max_deg`` is suggested as ``continuous``.

For interactive use the collision geometry is each link's convex hull, so a
full query is ~1-2 ms (a raw CAD mesh is ~200 ms); the baseline is computed
with the same hulls, so hull-induced static overlaps cancel out.

Needs the optional ``python-fcl`` package (``pip install python-fcl``) on top
of trimesh.
"""

from __future__ import annotations

import numpy as np


REST_MARGIN = 0.002      # m of penetration before a pair counts as colliding


def is_fcl_available():
    """Return True if the optional ``python-fcl`` package is importable."""
    import importlib.util
    return importlib.util.find_spec('fcl') is not None


def _hull(mesh):
    """Convex-hull proxy of a link mesh (fast, watertight for fcl)."""
    try:
        h = mesh.convex_hull
        if h is not None and len(h.faces):
            return h
    except Exception:
        pass
    return mesh


def _rotational_joints(robot):
    from skrobot.model.joint import RotationalJoint
    return [j for j in robot.joint_list if isinstance(j, RotationalJoint)]


def link_visual_mesh(link):
    """A single local-frame trimesh for a skrobot link's visual geometry, or
    None.  ``visual_mesh`` is a single mesh, a list, or None depending on the
    URDF; empty meshes are dropped and a multi-mesh link is concatenated."""
    import trimesh
    vm = getattr(link, "visual_mesh", None)
    ms = (vm if isinstance(vm, (list, tuple)) else [vm]) \
        if vm is not None else []
    ms = [m for m in ms
          if m is not None and hasattr(m, "vertices") and len(m.vertices)]
    if not ms:
        return None
    return trimesh.util.concatenate(ms) if len(ms) > 1 else ms[0]


def link_meshes(robot):
    """``{link name -> local-frame trimesh}`` for every link that has visual
    geometry -- the per-link mesh map both :class:`SelfCollision` and
    :func:`sweep_limits` take.  One place so the webserver, the autolimits
    subprocess and the web editor build it identically."""
    out = {}
    for l in robot.link_list:
        m = link_visual_mesh(l)
        if m is not None:
            out[l.name] = m
    return out


class SelfCollision:
    """Self-collision model over a skrobot robot.

    ``meshes`` maps link name -> local-frame trimesh (the UI already has these).
    ``parts`` (optional) maps link name -> list of CONVEX part meshes (CoACD
    decomposition); when given, those drive the broadphase -- fast convex-convex
    queries that need no exact-mesh confirmation (a link without parts falls back
    to its convex hull).  Otherwise the old convex-hull model is used.
    """

    def __init__(self, robot, meshes, margin=REST_MARGIN, hull=True,
                 confirm=False, parts=None):
        if not is_fcl_available():
            raise RuntimeError(
                "self-collision queries need the optional 'python-fcl' "
                'package -- install it with: pip install python-fcl')
        from trimesh.collision import CollisionManager

        self.robot = robot
        self.margin = float(margin)
        self._link = {l.name: l for l in robot.link_list}
        self.names = [n for n in meshes if n in self._link]
        self.cm = CollisionManager()
        # object id -> link name; an object id is the link name (single-mesh /
        # hull) or "<link>\x00<i>" (the i-th CoACD convex part of that link)
        self._obj2link = {}
        self._raw = None

        if parts:
            # CoACD convex parts: accurate AND convex-fast, so no exact-mesh
            # confirm.  A link without parts falls back to its hull.
            self.confirm = False
            for n in self.names:
                ps = parts.get(n)
                tf = self._link[n].worldcoords().T()
                if ps:
                    for i, pm in enumerate(ps):
                        oid = f"{n}\x00{i}"
                        self.cm.add_object(oid, pm, transform=tf)
                        self._obj2link[oid] = n
                else:
                    self.cm.add_object(n, _hull(meshes[n]), transform=tf)
                    self._obj2link[n] = n
        else:
            # hull=True: convex-hull proxies (fast, ~2 ms/query) for the live
            # drag check.  hull=False: the ACTUAL meshes -- slower but exact, for
            # the limit sweep (hulls touch at rest and mask collisions that only
            # the real concave geometry develops, giving limits too wide).
            #
            # confirm=True: hulls for the cheap BROADPHASE, but every candidate
            # pair is verified on the EXACT mesh below -- so the live highlight
            # matches the exact-mesh limit sweep instead of the fatter hulls
            # lighting up red before the joint reaches its (mesh-derived) limit.
            self.confirm = bool(confirm)
            use_hull = hull or self.confirm
            self._hulls = {n: (_hull(meshes[n]) if use_hull else meshes[n])
                           for n in self.names}
            for n in self.names:
                self.cm.add_object(n, self._hulls[n],
                                   transform=self._link[n].worldcoords().T())
                self._obj2link[n] = n
            # second manager over the exact meshes, queried pairwise (not as a
            # broadphase) only on the candidates the hull broadphase flags.
            if self.confirm:
                self._raw = CollisionManager()
                for n in self.names:
                    self._raw.add_object(
                        n, meshes[n], transform=self._link[n].worldcoords().T())
        # trimesh 4.x's add_object(transform=) sets the fcl object's pose but
        # does NOT refresh the broadphase AABB-tree node, so a part that only
        # overlaps once moved to its world pose can be missed by
        # in_collision_internal.  At query time _sync() calls set_transform,
        # which DOES refresh the tree -- so without this, those rest-pose
        # contacts escape the baseline and then light up red as false "new"
        # collisions the instant the page loads (even on parts that never
        # move).  Re-sync now so the baseline is built over the same broadphase
        # state the queries use.
        self._sync()
        self.baseline = self._pairs(0.0)
        for j in robot.joint_list:
            if j.parent_link and j.child_link:
                self.baseline.add(
                    frozenset((j.parent_link.name, j.child_link.name)))

    def _link_pairs(self, name_pairs) -> set:
        """Map broadphase OBJECT-name pairs to LINK pairs, dropping intra-link
        pairs (two CoACD parts of the SAME link overlap at their shared seams --
        not a collision)."""
        out = set()
        for p in name_pairs:
            a, b = tuple(p)
            la, lb = self._obj2link.get(a, a), self._obj2link.get(b, b)
            if la != lb:
                out.add(frozenset((la, lb)))
        return out

    def _pairs(self, margin) -> set:
        _, names, data = self.cm.in_collision_internal(
            return_names=True, return_data=True)
        if self.confirm:
            # Candidates must be EVERY overlapping hull pair (boolean), not the
            # depth-thresholded set: the hull penetration depth is NOT a sound
            # upper bound on the mesh depth (deeply interlocking concave parts
            # can have only shallow-touching hulls), so filtering candidates by
            # hull depth would drop real collisions.  Boolean containment (mesh
            # overlap => hull overlap) is what makes the candidate set complete;
            # each is then verified on the exact mesh at the real margin.
            cand = self._link_pairs(names)
            # new_pairs() subtracts the rest baseline anyway, so don't spend an
            # exact-mesh query on the dozens of links that touch at rest -- drop
            # them from the candidates first.  (During __init__, before the
            # baseline exists, getattr returns None so every pair is verified,
            # which is exactly what building the baseline needs.)
            base = getattr(self, "baseline", None)
            if base:
                cand = cand - base
            return self._confirm(cand, margin) if cand else cand
        if margin <= 0:
            return self._link_pairs(names)
        depth = {}
        for d in data:
            a, b = tuple(d.names)
            la, lb = self._obj2link.get(a, a), self._obj2link.get(b, b)
            if la == lb:
                continue
            k = frozenset((la, lb))
            depth[k] = max(depth.get(k, 0.0), abs(d.depth))
        return {k for k, v in depth.items() if v > margin}

    def _confirm(self, cand, margin) -> set:
        """Keep only the candidate pairs whose EXACT meshes actually collide.  A
        convex hull is a superset of its mesh, so the hull broadphase never
        misses a real pair -- it only over-reports, and this verifies that away
        with one pairwise exact-mesh query per candidate (candidates are few:
        just the links whose hulls overlap right now)."""
        import fcl
        # enough contacts that the true max penetration depth isn't truncated
        req = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)
        objs = self._raw._objs
        for n in {n for p in cand for n in p}:        # sync only candidate links
            self._raw.set_transform(n, self._link[n].worldcoords().T())
        out = set()
        for pair in cand:
            a, b = tuple(pair)
            res = fcl.CollisionResult()
            fcl.collide(objs[a]["obj"], objs[b]["obj"], req, res)
            if margin > 0:
                hit = max((abs(c.penetration_depth) for c in res.contacts),
                          default=0.0) > margin
            else:
                hit = len(res.contacts) > 0
            if hit:
                out.add(pair)
        return out

    def _sync(self):
        # set each broadphase object to its link's world pose (several objects
        # may share a link when CoACD parts are used)
        cache = {}
        for oid, link in self._obj2link.items():
            tf = cache.get(link)
            if tf is None:
                tf = cache[link] = self._link[link].worldcoords().T()
            self.cm.set_transform(oid, tf)

    def new_pairs(self) -> set:
        """Colliding link pairs (beyond the rest baseline) at the current pose."""
        self._sync()
        return self._pairs(self.margin) - self.baseline

    def offenders(self):
        """``(new_pairs, offending_link_names)`` at the current pose -- the live
        drag highlight wants the flat set of links to tint as well as the
        pairs."""
        new = self.new_pairs()
        links = set()
        for p in new:
            links |= set(p)
        return new, links

    def min_distance(self):
        """``(distance_m, (link_a, link_b))`` of the closest non-adjacent pair,
        or ``(inf, None)`` if nothing is near.  Negative distance = penetration.

        fcl's ``min_distance_internal`` ignores objects already in contact, so a
        zero/negative reading is reported via ``new_pairs`` instead."""
        self._sync()
        try:
            d, names = self.cm.min_distance_internal(return_names=True)
        except Exception:
            return float("inf"), None
        if not names:
            return float(d), None
        a, b = tuple(names)
        la, lb = self._obj2link.get(a, a), self._obj2link.get(b, b)
        if la == lb:                  # closest pair is two parts of one link
            return float("inf"), None
        return float(d), tuple(sorted((la, lb)))


def _prismatic_joints(robot):
    from skrobot.model.joint import LinearJoint
    return [j for j in robot.joint_list if isinstance(j, LinearJoint)]


def sweep_limits(robot, meshes, step_deg=6.0, max_deg=180.0, margin_deg=2.0,
                 step_mm=5.0, max_mm=300.0, margin_mm=2.0,
                 margin_m=REST_MARGIN, only=None, progress=None, refine=True,
                 refine_tol_deg=0.4, refine_tol_mm=0.2, sc=None, hull=False,
                 on_start=None):
    """Per-joint self-collision limit sweep, from the HOME pose (all at 0).

    Sweeps REVOLUTE joints in radians (``step_deg`` / ``max_deg``) and PRISMATIC
    joints in metres (``step_mm`` / ``max_mm``).  Returns
    ``{joint_name: {lower, upper, continuous, hit_lower, hit_upper, child}}`` --
    revolute limits in radians, prismatic limits in metres.  ``hit_*`` is the
    colliding link pair that stopped that direction (or ``None`` if it reached
    the max freely).  A revolute joint free both ways is flagged ``continuous``.

    The sweep is a COARSE linear scan to bracket the first new self-collision,
    then -- when ``refine`` -- a BISECTION to pin the boundary.  The reported
    limit is the last clear position **backed off** by ``margin_deg`` (revolute)
    or ``margin_mm`` (prismatic) -- the user-facing safety margin from the
    colliding edge.

    ``only`` (joint names) restricts the sweep.  Contacts + adjacency form the
    HOME baseline; every other joint stays at 0 while one sweeps.  Pass a built
    ``sc`` to skip rebuilding it.  Pre-call joint angles are restored on return.
    """
    rjoints = _rotational_joints(robot)
    pjoints = _prismatic_joints(robot)
    if only is not None:
        only = set(only)
        rjoints = [j for j in rjoints if j.name in only]
        pjoints = [j for j in pjoints if j.name in only]
    joints = rjoints + pjoints

    # snapshot to restore at the end; widen limits so the sweep is not clamped
    # keyed by the joint object: names may be missing or duplicated
    snapshot = {j: float(j.joint_angle()) for j in robot.joint_list}
    saved_lims = {}
    for j in rjoints + pjoints:
        saved_lims[j] = (j.min_angle, j.max_angle)
        wide = 10.0 if j in pjoints else 4 * np.pi   # metres vs radians
        j.min_angle, j.max_angle = -wide, wide
    # baseline (rest contacts + adjacency) is defined at the HOME pose
    for j in joints:
        j.joint_angle(0.0)
    if sc is not None and any(oid != link for oid, link
                              in sc._obj2link.items()):
        raise ValueError(
            'sweep_limits does not support a parts-based SelfCollision '
            '(CoACD convex parts register several broadphase objects per '
            'link); pass a hull/mesh SelfCollision or sc=None')
    if sc is None:
        # hull=False by default here: the limit sweep needs the EXACT meshes
        # (convex hulls touch at rest and mask real collisions -> limits too
        # wide).  Callers that already built a hull `sc` (the live UI) pass it.
        sc = SelfCollision(robot, meshes, margin=margin_m, hull=hull)
    # Hull PRE-FILTER world.  A convex hull CONTAINS its mesh, so if two hulls
    # do not collide the meshes inside them cannot either: a hull-clear pose is
    # provably mesh-clear.  The coarse scan runs on these cheap hulls and only
    # escalates to the exact mesh where a hull flags a possible collision, so a
    # free joint (the common, expensive case -- it scans the whole range finding
    # nothing) costs only hull queries.  Result-identical to the pure-mesh scan;
    # see `_hull_clear`.  (When the caller already asked for a hull `sc`, reuse
    # it rather than build a redundant second hull world.)
    sc_hull = sc if hull else SelfCollision(robot, meshes, margin=margin_m,
                                            hull=True)

    import fcl

    _req = fcl.CollisionRequest(num_max_contacts=1000, enable_contact=True)

    # Reuse the fcl objects SelfCollision already built (their BVH is built
    # ONCE here).  Per joint we register the prebuilt objects into throwaway
    # broadphase managers -- registerObjects does NOT rebuild the BVH, so a
    # query is ~1 ms even on the real (non-hull) meshes; rebuilding a manager
    # via trimesh.add_object instead cost ~2 s/joint (the whole sweep was 55 s).
    def _world(world):
        objs = {n: world.cm._objs[n]["obj"] for n in world.names}
        geom2name = {id(world.cm._objs[n]["geom"]): n for n in world.names}

        def set_T(n):
            T = world._link[n].worldcoords().T()
            objs[n].setTranslation(np.ascontiguousarray(T[:3, 3]))
            objs[n].setRotation(np.ascontiguousarray(T[:3, :3]))
        return objs, geom2name, set_T

    raw_objs, raw_g2n, raw_set_T = _world(sc)
    hull_objs, hull_g2n, hull_set_T = _world(sc_hull)

    def _moving_set(J, probe):
        # Which links ACTUALLY move when only J moves?  Determined by probing
        # (the skrobot parent/child tree-walk under-reports the subtree for
        # this model -- it returned just the immediate child while 18 links
        # really move).  The subtree is rigid, so one probe reveals all of it;
        # the full transform comparison catches on-axis rotation too.  ``probe``
        # is in the joint's native unit (radians / metres).
        home = {n: sc._link[n].worldcoords().T() for n in sc.names}
        J.joint_angle(probe)
        moved = {n for n in sc.names
                 if not np.allclose(sc._link[n].worldcoords().T(),
                                    home[n], atol=1e-7)}
        J.joint_angle(0.0)
        return moved

    out = {}
    try:
        for _idx, J in enumerate(joints, 1):
            # fire BEFORE the (possibly multi-second) sweep of this joint so a UI
            # shows the joint currently being worked, not the last one finished
            if on_start:
                on_start(J.name, _idx, len(joints))
            is_prismatic = J in pjoints
            # per-joint sweep parameters in the joint's native unit
            if is_prismatic:
                u_step = step_mm / 1000.0
                u_max = max_mm / 1000.0
                u_margin = margin_mm / 1000.0
                u_tol = refine_tol_mm / 1000.0
                u_probe = max(u_step, 0.005)
            else:
                u_step = np.radians(step_deg)
                u_max = np.radians(max_deg)
                u_margin = np.radians(margin_deg)
                u_tol = np.radians(refine_tol_deg)
                u_probe = np.radians(7.0)
            nmax = max(1, int(u_max / u_step))
            moving = _moving_set(J, u_probe)
            static = [n for n in sc.names if n not in moving]
            if not moving or not static:
                # nothing can collide; treat as free
                out[J.name] = {
                    "lower": round(-u_max, 5),
                    "upper": round(u_max, 5),
                    "continuous": (not is_prismatic) and max_deg >= 180.0,
                    "hit_lower": None, "hit_upper": None,
                    "child": J.child_link.name if J.child_link else None}
                if progress:
                    progress(J.name, out[J.name])
                continue
            # static links don't move while J sweeps -> set their transforms
            # once; only the moving set updates per angle.  Build the same
            # throwaway broadphase managers for BOTH the exact-mesh world and
            # the hull pre-filter world, over the same moving/static split.

            def _make_pairs(objs, geom2name, set_T, margin):
                st_mgr = fcl.DynamicAABBTreeCollisionManager()
                st_mgr.registerObjects([objs[n] for n in static])
                for n in static:
                    set_T(n)
                st_mgr.setup()
                mv_mgr = fcl.DynamicAABBTreeCollisionManager()
                mv_mgr.registerObjects([objs[n] for n in moving])
                mv_mgr.setup()

                def _pairs(mag, direction):
                    J.joint_angle(direction * mag)
                    for n in moving:
                        set_T(n)
                    mv_mgr.update()          # refresh AABBs after the move
                    cdata = fcl.CollisionData(request=_req)
                    mv_mgr.collide(st_mgr, cdata, fcl.defaultCollisionCallback)
                    depth = {}
                    for c in cdata.result.contacts:
                        a = geom2name.get(id(c.o1))
                        b = geom2name.get(id(c.o2))
                        if a and b:
                            k = frozenset((a, b))
                            depth[k] = max(depth.get(k, 0.0),
                                           abs(c.penetration_depth))
                    return {k for k, v in depth.items() if v > margin}
                return _pairs

            raw_pairs = _make_pairs(raw_objs, raw_g2n, raw_set_T, sc.margin)
            hull_pairs = _make_pairs(hull_objs, hull_g2n, hull_set_T,
                                     sc_hull.margin)

            # baseline: moving-vs-static pairs already in contact at HOME (mesh)
            ignore = raw_pairs(0.0, 1) | sc.baseline

            def _collides(mag, direction):
                new = raw_pairs(mag, direction) - ignore
                return (tuple(sorted(next(iter(new)))) if new else None)

            def _hull_clear(mag, direction):
                # Hull CONTAINS the mesh, so no hull collision => no mesh
                # collision: this angle is skipped without an exact-mesh query.
                # Subtract the MESH `ignore` (not a hull baseline) so a pair that
                # only the hull touches at rest is never absorbed into the
                # baseline and so can never mask a real mesh collision -- this is
                # what keeps the pre-filter result-identical to a pure-mesh scan.
                return not (hull_pairs(mag, direction) - ignore)

            lim = {}
            hit = {}
            for direction in (+1, -1):
                clear_mag, hit_mag, hit_pair = 0.0, None, None
                # The hull pre-filter pays off only while the hulls actually
                # separate as J turns.  Some links have fat hulls that overlap a
                # static link at rest and never separate (a non-adjacent pair the
                # mesh never touches); for them every step would escalate, so the
                # hull query is pure overhead.  Treat the hull as "innocent until
                # useless": the first time it flags a step that the mesh finds
                # clear (a persistent false positive), stop consulting it for the
                # rest of this direction and scan the mesh directly.
                use_hull = True
                for k in range(1, nmax + 1):
                    mag = k * u_step
                    if use_hull and _hull_clear(mag, direction):
                        clear_mag = mag          # hull clear => mesh clear
                        continue
                    pair = _collides(mag, direction)  # hull flagged: check mesh
                    if pair:
                        hit_mag, hit_pair = mag, pair
                        break
                    clear_mag = mag              # hull false positive; mesh clear
                    use_hull = False             # hull is useless here; drop it
                # bisect [clear_mag, hit_mag] to the precise boundary
                if refine and hit_mag is not None:
                    a, b = clear_mag, hit_mag
                    while b - a > u_tol:
                        m = 0.5 * (a + b)
                        if _collides(m, direction):
                            b = m
                        else:
                            a = m
                    clear_mag = a
                J.joint_angle(0.0)   # back to home before the other direction
                if hit_mag is None:
                    lim[direction] = direction * u_max
                else:
                    lim[direction] = direction * max(0.0, clear_mag - u_margin)
                hit[direction] = hit_pair
            lo, up = sorted([lim[-1], lim[+1]])
            continuous = (not is_prismatic) and hit[+1] is None \
                and hit[-1] is None \
                and max_deg >= 180.0
            out[J.name] = {
                "lower": round(float(lo), 5),
                "upper": round(float(up), 5),
                "continuous": continuous,
                "hit_lower": hit[-1],
                "hit_upper": hit[+1],
                "child": J.child_link.name if J.child_link else None,
            }
            if progress:
                progress(J.name, out[J.name])
    finally:
        for j in rjoints + pjoints:
            j.min_angle, j.max_angle = saved_lims[j]
        for j in robot.joint_list:
            try:
                j.joint_angle(snapshot[j])
            except Exception:
                pass
    return out
