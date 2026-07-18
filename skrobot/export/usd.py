"""Convert a URDF into a USD physics articulation for Isaac Sim / PhysX.

Walks the raw parsed URDF and emits a USD stage with a UsdPhysics
articulation -- one rigid body per link (mass + regularized inertia), visual meshes embedded with
per-vertex ``displayColor`` (USD renders vertex colours natively, unlike STL/
MJCF), convex collision meshes, and revolute/prismatic/fixed joints (limits +
position drives). A fixed joint welds the base to the world.

USD authoring only needs ``usd-core`` (pip), so this runs anywhere -- author the
USD on a laptop, load it in Isaac Sim on a GPU box. Entry point: :func:`urdf_to_usd`.
"""

from __future__ import annotations

import warnings

import numpy as np

from skrobot.urdf.sanitize import sanitize_name
from skrobot.utils.convex_decomposition import convex_decomposition
from skrobot.utils.convex_decomposition import is_coacd_available


# Floors for massless URDF frame links (see urdf_to_usd). A rigid body with no
# collider and no authored mass makes PhysX assign a negative mass -> NaN
# articulation dynamics; these tiny positive values keep such frames valid.
_MASSLESS_FLOOR_KG = 1e-5
_MASSLESS_INERTIA_FLOOR = 1e-6

# decompose_links=True auto-decomposes a link only when its single convex hull
# is at least this many times the true mesh volume, i.e. the link is measurably
# concave. Near-convex links (ratio ~1) keep the cheap single hull.
_DECOMPOSE_RATIO = 2.0

_NO_MESH_INERTIA_REASON = 'no collision or visual mesh geometry was found'


def _mat_to_gf(pxr, matrix):
    """4x4 column-vector numpy matrix -> USD Gf.Matrix4d (row-major, row-vector)."""
    from pxr import Gf
    m = np.asarray(matrix, dtype=float).T  # USD uses row vectors: transpose
    return Gf.Matrix4d(*[float(v) for v in m.flatten()])


def _quat_wxyz(pxr, rot3x3):
    from pxr import Gf

    from skrobot.coordinates.math import matrix2quaternion
    w, x, y, z = matrix2quaternion(np.asarray(rot3x3, dtype=float))
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _align_z_to(axis):
    """3x3 rotation mapping +Z onto the unit ``axis`` (so a USD joint whose token
    axis is Z rotates about the URDF axis)."""
    from skrobot.coordinates.math import rotation_matrix_z_to_axis
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return np.eye(3)
    return rotation_matrix_z_to_axis(a)


def _world_transforms(urdf):
    """World transform of every link at the zero configuration (FK over joint
    origins from the base)."""
    children = {}
    for j in urdf.joints:
        children.setdefault(j.parent, []).append(j)
    world = {urdf.base_link.name: np.eye(4)}

    def recurse(link_name):
        for j in children.get(link_name, []):
            world[j.child] = (
                world[link_name] @ _joint_child_pose(j)
            )
            recurse(j.child)
    recurse(urdf.base_link.name)
    return world


def _joint_child_pose(joint, cfg=None):
    axis = np.asarray(getattr(joint, "axis", [0, 0, 1]), dtype=float)
    if joint.joint_type in ("revolute", "continuous") and np.linalg.norm(axis) < 1e-9:
        return np.asarray(joint.origin, dtype=float)
    return np.asarray(joint.get_child_pose(cfg), dtype=float)


def _fk_at(urdf, cfg):
    """World transform of every link with joint configuration ``cfg`` applied
    (name -> angle[rad]/displacement[m]). Like ``_world_transforms`` but at a
    non-zero pose, so gravity torques are evaluated at the resting (home) pose."""
    cfg = dict(cfg or {})
    children = {}
    for j in urdf.joints:
        children.setdefault(j.parent, []).append(j)
    world = {urdf.base_link.name: np.eye(4)}

    def recurse(link_name):
        for j in children.get(link_name, []):
            q = cfg.get(j.name, None)
            world[j.child] = (
                world[link_name] @ _joint_child_pose(j, q)
            )
            recurse(j.child)
    recurse(urdf.base_link.name)
    return world


def _worstcase_gravity_torque(urdf, cfg, gravity):
    """Per-joint worst-case gravity torque (name -> |tau|), evaluated with the
    distal subtree at pose ``cfg``.

    For a revolute joint the gravity moment arm of a distal mass is its
    *perpendicular distance to the joint axis* -- which the joint's own rotation
    cannot exceed -- so ``sum(m_i * r_perp_i) * |g|`` is the max torque the drive
    must hold at ANY angle of that joint (config-independent for its own DOF).
    For a prismatic joint it is the gravity force projected on the slide axis.
    This is what the per-joint drive stiffness is sized against."""
    world = _fk_at(urdf, cfg)
    g = np.asarray(gravity, dtype=float)
    gmag = float(np.linalg.norm(g))
    children = {}
    for j in urdf.joints:
        children.setdefault(j.parent, []).append(j.child)

    def subtree(link):
        out = [link]
        for c in children.get(link, []):
            out += subtree(c)
        return out

    mass_com = {}
    for link in urdf.links:
        inr = getattr(link, "inertial", None)
        if inr is not None and getattr(inr, "mass", 0.0) and link.name in world:
            com_local = np.asarray(inr.origin, dtype=float)[:3, 3]
            com_world = (world[link.name] @ np.append(com_local, 1.0))[:3]
            mass_com[link.name] = (float(inr.mass), com_world)

    tau = {}
    for j in urdf.joints:
        if j.joint_type not in ("revolute", "continuous", "prismatic"):
            continue
        if j.parent not in world:
            continue
        tj = world[j.parent] @ np.asarray(j.origin, dtype=float)
        p = tj[:3, 3]
        a = tj[:3, :3] @ (np.asarray(getattr(j, "axis", [0, 0, 1]), dtype=float))
        a = a / (np.linalg.norm(a) or 1.0)
        distal = subtree(j.child)
        if j.joint_type == "prismatic":
            m_tot = sum(mass_com.get(l, (0.0, None))[0] for l in distal)
            tau[j.name] = abs(m_tot * float(a @ g))
        else:
            moment = 0.0
            for link in distal:
                if link in mass_com:
                    m, c = mass_com[link]
                    d = c - p
                    r_perp = float(np.linalg.norm(d - (d @ a) * a))
                    moment += m * r_perp
            tau[j.name] = gmag * moment
    return tau


def _regularized_inertia(inertial):
    """Return (mass, com3, diag3, principal_axes_3x3) from a URDF inertial,
    forced positive-definite + triangle-inequality (PhysX rejects bad inertia)."""
    # Keep this regularization logic in sync with
    # skrobot.utils.mjcf_converter._inertial_attrib.
    origin = np.asarray(getattr(inertial, "origin", np.eye(4)), dtype=float)
    com = origin[:3, 3]
    rot = origin[:3, :3]
    inertia = rot @ np.asarray(inertial.inertia, dtype=float) @ rot.T
    inertia = 0.5 * (inertia + inertia.T)
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    moments = np.clip(moments, 1e-9, None)
    lo, mid, hi = moments
    if hi > lo + mid:
        hi = lo + mid
    return float(inertial.mass), com, np.array([lo, mid, hi]), axes


def _diag_inertia(inertia, mass, com):
    """Diagonalize a 3x3 inertia tensor into (mass, com, diag3, axes), forced
    positive-definite + triangle-inequality (PhysX rejects bad inertia)."""
    inertia = 0.5 * (np.asarray(inertia, dtype=float) + np.asarray(inertia, dtype=float).T)
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    moments = np.clip(moments, 1e-9, None)
    lo, mid, hi = moments
    if hi > lo + mid:
        hi = lo + mid
    return float(mass), np.asarray(com, dtype=float), np.array([lo, mid, hi]), axes


def _is_placeholder_inertia(inertial):
    """True if the URDF inertia is (near-)isotropic with zero products of
    inertia -- almost always a default placeholder (e.g. ixx=iyy=izz=1e-4 for
    every arm link) rather than a real, geometry-derived tensor."""
    try:
        mat = np.asarray(inertial.inertia, dtype=float)
    except Exception:  # noqa: BLE001
        return False
    diag = np.diag(mat)
    mag = max(float(np.max(np.abs(diag))), 1e-12)
    # products of inertia negligible relative to the diagonal, and the three
    # diagonal moments (near-)equal -> isotropic placeholder.
    off = float(np.max(np.abs(mat - np.diag(diag))))
    return off <= 1e-3 * mag and float(np.ptp(diag)) <= 1e-3 * mag


def _drop_outlier_parts(parts):
    """Drop mesh submeshes whose centroid is a far outlier from the cluster.

    Some URDF meshes ship stray/degenerate submeshes (inverted normals -> negative
    volume, or geometry placed metres away from the link frame).
    Concatenating those into a collision convex hull makes a huge
    degenerate collider, and computing an inertia tensor from them yields
    mass*distance^2 garbage. Either destabilises the PhysX articulation. Keep only
    submeshes within a sane link scale of the median centroid."""
    if len(parts) <= 1:
        return parts
    coms = np.array([np.asarray(p.center_mass, dtype=float) for p in parts])
    med = np.median(coms, axis=0)
    dist = np.linalg.norm(coms - med, axis=1)
    threshold = max(0.15, 3.0 * float(np.median(dist)))
    kept = [p for p, d in zip(parts, dist) if d <= threshold]
    return kept or parts


def _inertia_from_mesh(link, mass, return_reason=False):
    """Compute (mass, com, diag3, axes) from a link's collision (else visual)
    meshes at the given mass.

    Parameters
    ----------
    link : skrobot.utils.urdf.Link
        Link whose geometry is inspected.
    mass : float
        Link mass to assign as uniform density over the measured mesh volume.
    return_reason : bool, optional
        If True, return ``(result_or_None, reason_or_None)`` so callers can
        distinguish "no mesh to measure" from a measurement failure.

    A well-conditioned inertia tensor is essential: a reduced-coordinate PhysX
    articulation with placeholder inertias (all links isotropic ~1e-4) is
    ill-conditioned and the position drives cannot hold against gravity at any
    stiffness. Deriving the tensor from the actual mesh geometry (uniform
    density scaled to the URDF mass) makes it physical, like the shipped robots."""
    import trimesh
    saw_mesh_geometry = False
    saw_triangle_mesh = False
    last_error = None
    for source in ("collisions", "visuals"):
        parts = []
        for vc in (getattr(link, source, []) or []):
            geom = getattr(vc, "geometry", None)
            mesh_geom = getattr(geom, "mesh", None) if geom is not None else None
            if mesh_geom is None:
                continue
            saw_mesh_geometry = True
            origin = np.asarray(getattr(vc, "origin", np.eye(4)), dtype=float)
            scale = getattr(mesh_geom, "scale", None)
            for m in (getattr(mesh_geom, "meshes", None) or []):
                if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0:
                    continue
                saw_triangle_mesh = True
                mm = m.copy()
                if scale is not None:
                    mm.apply_scale(np.asarray(scale, dtype=float))
                if not mm.is_watertight:
                    try:
                        mm = mm.convex_hull
                    except Exception as e:  # noqa: BLE001
                        last_error = (
                            'building a convex hull raised {}: {}'
                            .format(type(e).__name__, e))
                        continue
                mm.apply_transform(origin)
                parts.append(mm)
        if not parts:
            continue
        parts = _drop_outlier_parts(parts)
        try:
            combined = trimesh.util.concatenate(parts)
            vol = float(combined.volume)
            if vol <= 1e-12:
                last_error = 'the measured mesh volume was zero'
                continue
            combined.density = float(mass) / vol
            inertia = np.asarray(combined.moment_inertia, dtype=float)
            com = np.asarray(combined.center_mass, dtype=float)
            if not np.all(np.isfinite(inertia)):
                last_error = 'the measured inertia tensor was not finite'
                continue
            result = _diag_inertia(inertia, mass, com)
            if return_reason:
                return result, None
            return result
        except Exception as e:  # noqa: BLE001
            last_error = 'mesh inertia computation raised {}: {}'.format(
                type(e).__name__, e)
            continue
    if not return_reason:
        return None
    if not saw_mesh_geometry:
        return None, _NO_MESH_INERTIA_REASON
    if not saw_triangle_mesh:
        return None, 'mesh geometry existed but had no usable triangle meshes'
    if last_error is None:
        last_error = (
            'mesh geometry existed but no valid watertight parts remained')
    return None, last_error


def _submesh_geometry(mesh_geometry):
    """Yield (points, face_counts, face_indices, per_vertex_rgb_or_None, scale)
    for each sub-mesh of a URDF Mesh geometry."""
    import trimesh
    subs = [m for m in (getattr(mesh_geometry, "meshes", None) or [])
            if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
    scale = getattr(mesh_geometry, "scale", None)
    for m in subs:
        pts = np.asarray(m.vertices, dtype=float)
        if scale is not None:
            pts = pts * np.asarray(scale, dtype=float)
        faces = np.asarray(m.faces, dtype=int)
        counts = [3] * len(faces)
        indices = faces.flatten().tolist()
        rgb = None
        try:
            vc = m.visual.vertex_colors  # (N,4) uint8
            if vc is not None and len(vc) == len(pts):
                rgb = (np.asarray(vc)[:, :3] / 255.0).tolist()
        except Exception:
            pass
        if rgb is None:
            try:
                mc = m.visual.main_color
                rgb = [[mc[0] / 255.0, mc[1] / 255.0, mc[2] / 255.0]]
            except Exception:
                rgb = None
        if rgb is None:
            # TextureVisuals has neither vertex_colors nor main_color (both
            # raise AttributeError), so read the colour off the material.
            mat = getattr(m.visual, "material", None)
            for attr in ("main_color", "baseColorFactor", "diffuse"):
                c = getattr(mat, attr, None) if mat is not None else None
                if c is None:
                    continue
                c = np.asarray(c, dtype=float).ravel()
                if len(c) < 3:
                    continue
                if c.max() > 1.0:          # uint8 0..255 vs float 0..1
                    c = c / 255.0
                rgb = [c[:3].tolist()]
                break
        yield pts.tolist(), counts, indices, rgb


def urdf_to_usd(urdf, out_path, floating_base=False, drive_stiffness=1000.0,
                drive_damping=200.0, gravity=(0.0, 0.0, -9.81),
                self_collision=False, home_positions=None, mobile_base=False,
                base_drive_stiffness=1e5, base_drive_damping=1e4,
                auto_gain=True, gain_max_scale=25.0, gain_ref_percentile=35.0,
                recompute_inertia=True, armature=0.0, inertia_floor=0.0,
                decompose_links=None, coacd_params=None):
    """Convert a URDF to a USD physics articulation written to ``out_path``
    (.usd/.usda). Returns the USD stage.

    Parameters
    ----------
    self_collision : bool
        If False (default), disable PhysX intra-articulation self-collision.
        Convex-hull link colliders overlap heavily at the zero configuration
        (a body's hull swallows the shoulder links mounted into it), so with
        self-collision on PhysX explodes the articulation to NaN on the first
        step. Ground/external collision is unaffected. This matches Isaac's
        URDF importer default; precise self-collision is handled one layer up
        (axon_safety), not by the physics engine.
    home_positions : dict[str, float] or None
        Per-joint target angle (radians for revolute, metres for prismatic) that
        each joint's position drive holds. Defaults to 0 for any joint not
        listed. Use a collision-free reference pose here: the zero configuration
        of many robots self-collides, and although self-collision is disabled in
        PhysX the links still visibly interpenetrate at zero. Driving to a
        reset pose gives a clean, collision-free resting posture.
    decompose_links : bool, list[str], or None
        Which links get their collision mesh convex-DECOMPOSED with CoACD
        instead of reduced to a single convex hull. ``None`` (default) decomposes
        nothing; ``True`` decomposes every link whose single hull is measurably
        concave (hull volume >= _DECOMPOSE_RATIO x the mesh); a list of names (or
        substrings) decomposes exactly those links. Requires the ``coacd``
        package; falls back to a single hull if it is missing.

        A single hull fills in a concave part, so a cavity becomes solid and
        nothing fits into it (this is what stops a gripper from closing on an
        object). Decomposition costs seconds per unique mesh (cached across links
        that share it) and adds many PhysX shapes, so ``None`` is the default and
        near-convex links are skipped even under ``True``.
    coacd_params : dict or None
        Parameters forwarded to
        :func:`skrobot.utils.convex_decomposition.convex_decomposition`
        (``threshold``, ``max_convex_hull``, ``quality``, ...) for the
        decomposed links. Defaults to a tight fit suitable for gripper cups;
        pass ``{'quality': 'balanced'}`` for a coarser, faster result.
    """
    from pxr import Gf
    from pxr import Sdf
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics
    from pxr import Vt

    from skrobot.utils.urdf import URDF

    home_positions = dict(home_positions or {})
    if coacd_params is None:
        # tight enough that a gripper's concave cup is not filled in
        coacd_params = {'threshold': 0.05, 'max_convex_hull': 32,
                                    'preprocess_resolution': 50,
                                    'mcts_iterations': 150}

    if isinstance(urdf, str):
        urdf = URDF.load(urdf)

    world = _world_transforms(urdf)

    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    g = np.asarray(gravity, dtype=float)
    n = np.linalg.norm(g)
    if n > 0:
        scene.CreateGravityDirectionAttr(Gf.Vec3f(*(g / n)))
        scene.CreateGravityMagnitudeAttr(float(n))

    root = UsdGeom.Xform.Define(stage, Sdf.Path("/robot"))
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    # PhysxArticulationAPI lives in the Omniverse physx schema, absent from pip
    # usd-core. AddAppliedSchema registers the API by name (no type needed) so
    # PhysX actually honors enabledSelfCollisions; authoring the raw attribute
    # alone is silently ignored.
    root.GetPrim().AddAppliedSchema("PhysxArticulationAPI")
    root.GetPrim().CreateAttribute(
        "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
    ).Set(bool(self_collision))

    link_path = {}
    for link in urdf.links:
        path = Sdf.Path("/robot/" + sanitize_name(link.name))
        link_path[link.name] = path
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTransformOp().Set(_mat_to_gf(None, world.get(link.name, np.eye(4))))
        UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())

        inertial = getattr(link, "inertial", None)
        mapi = UsdPhysics.MassAPI.Apply(xf.GetPrim())
        if inertial is not None and getattr(inertial, "mass", 0.0):
            mesh_inertia = None
            mesh_inertia_reason = None
            placeholder = recompute_inertia and _is_placeholder_inertia(inertial)
            if placeholder:
                mesh_inertia, mesh_inertia_reason = _inertia_from_mesh(
                    link, inertial.mass, return_reason=True)
            if mesh_inertia is not None:
                mass, com, diag, axes = mesh_inertia
            else:
                if placeholder and mesh_inertia_reason not in (
                        None, _NO_MESH_INERTIA_REASON):
                    warnings.warn(
                        'recompute_inertia=True was requested for link "{}", '
                        'but mesh-based inertia recomputation failed ({}); '
                        'the exporter used the URDF inertia instead. If that '
                        'URDF inertia is a placeholder, drive stability and '
                        'gravity response can be physically wrong.'.format(
                            link.name, mesh_inertia_reason),
                        RuntimeWarning, stacklevel=2)
                mass, com, diag, axes = _regularized_inertia(inertial)
            if inertia_floor > 0.0:
                diag = np.maximum(np.asarray(diag, dtype=float), inertia_floor)
            mapi.CreateMassAttr(mass)
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in diag]))
            mapi.CreatePrincipalAxesAttr(_quat_wxyz(None, axes))
        else:
            # Massless URDF frames (base_footprint, sensor/optical frames) still
            # get a RigidBodyAPI as articulation links. PhysX must NOT be left to
            # auto-compute their mass -- with no collider it yields a negative
            # mass and NaNs the whole reduced-coordinate articulation. Author a
            # tiny positive mass + isotropic inertia floor so they are inert but
            # valid dynamic bodies.
            _mif = max(_MASSLESS_INERTIA_FLOOR, inertia_floor)
            mapi.CreateMassAttr(_MASSLESS_FLOOR_KG)
            mapi.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
            mapi.CreateDiagonalInertiaAttr(Gf.Vec3f(_mif, _mif, _mif))
            mapi.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        if decompose_links is True:
            _decompose = "auto"      # decide per link by concavity (in _emit_one)
        elif decompose_links:
            _decompose = "force" if any(pat in link.name
                                        for pat in decompose_links) else False
        else:
            _decompose = False
        _emit_geoms(stage, UsdGeom, UsdPhysics, Gf, Vt, path, link,
                    decompose=_decompose, coacd_params=coacd_params)

    base = link_path[urdf.base_link.name]
    if mobile_base and not floating_base:
        # Planar mobile base: world -(X)-> px -(Y)-> py -(yaw)-> base, so the
        # robot can drive around the floor holonomically (matches the mecanum
        # base) while still being a proper articulation. Two massless dummy links
        # carry the X and Y prismatic DOFs; the yaw revolute closes onto base.
        # The three joints are named base_x_joint / base_y_joint / base_yaw_joint
        # and are position-driven, so a driver node can command them via
        # /joint_command and read them back for odom.
        # base_anchor is FIXED to the world so the articulation is fixed-base and
        # all three planar joints are INTERNAL (a prismatic/revolute joint whose
        # body0 is the world is the articulation's root joint, which PhysX does
        # not expose as a controllable/reported DOF -- so the anchor is required).
        for dummy in ("base_anchor", "base_planar_x", "base_planar_y"):
            dpath = Sdf.Path("/robot/" + dummy)
            dxf = UsdGeom.Xform.Define(stage, dpath)
            dxf.AddTransformOp().Set(_mat_to_gf(None, np.eye(4)))
            UsdPhysics.RigidBodyAPI.Apply(dxf.GetPrim())
            dm = UsdPhysics.MassAPI.Apply(dxf.GetPrim())
            dm.CreateMassAttr(_MASSLESS_FLOOR_KG)
            dm.CreateDiagonalInertiaAttr(Gf.Vec3f(_MASSLESS_INERTIA_FLOOR,
                                                  _MASSLESS_INERTIA_FLOOR,
                                                  _MASSLESS_INERTIA_FLOOR))
            link_path[dummy] = dpath
        anchor_fix = UsdPhysics.FixedJoint.Define(
            stage, Sdf.Path("/robot/joints/base_anchor_fixed"))
        anchor_fix.CreateBody1Rel().SetTargets([link_path["base_anchor"]])
        _emit_planar_joint(stage, UsdPhysics, Gf, Sdf, "base_x_joint", "prismatic",
                           "X", link_path["base_anchor"], link_path["base_planar_x"],
                           base_drive_stiffness, base_drive_damping, limit=6.0)
        _emit_planar_joint(stage, UsdPhysics, Gf, Sdf, "base_y_joint", "prismatic",
                           "Y", link_path["base_planar_x"], link_path["base_planar_y"],
                           base_drive_stiffness, base_drive_damping, limit=6.0)
        _emit_planar_joint(stage, UsdPhysics, Gf, Sdf, "base_yaw_joint", "revolute",
                           "Z", link_path["base_planar_y"], base,
                           base_drive_stiffness, base_drive_damping, limit=None)
    elif not floating_base:
        # fixed base: weld base_link to the world
        fj = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/robot/joints/base_fixed"))
        fj.CreateBody1Rel().SetTargets([base])

    # Per-joint drive gains sized to each joint's worst-case gravity torque, so a
    # joint carrying a whole arm gets proportionally more stiffness than a wrist
    # joint (a single global stiffness makes the loaded joints sag under gravity).
    # Stiffness scales with tau relative to a reference (a lightly-loaded joint
    # keeps the baseline), so every driven joint holds with about the same small
    # error; damping scales with it to keep the baseline damping ratio.
    joint_gain = {}
    if auto_gain:
        tau = _worstcase_gravity_torque(urdf, home_positions, gravity)
        loaded = sorted(v for v in tau.values() if v > 1e-9)
        tau_ref = float(np.percentile(loaded, gain_ref_percentile)) if loaded else 1.0
        tau_ref = max(tau_ref, 1e-9)
        for name, t in tau.items():
            scale = min(float(gain_max_scale), max(1.0, t / tau_ref))
            joint_gain[name] = (drive_stiffness * scale, drive_damping * scale)

    for joint in urdf.joints:
        stiff, damp = joint_gain.get(joint.name, (drive_stiffness, drive_damping))
        _emit_joint(stage, UsdPhysics, Gf, Sdf, joint, link_path,
                    stiff, damp,
                    float(home_positions.get(joint.name, 0.0)),
                    armature=armature)

    stage.GetRootLayer().Save()
    return stage


def _srgb_to_linear(c):
    """sRGB channel (0..1) -> linear. USD displayColor is a linear albedo."""
    c = float(c)
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


_COACD_CACHE = {}


def _hull_inflation(all_pts, all_idx):
    """Ratio of a mesh's single convex hull volume to its own volume.

    ~1 for a convex link; large for a concave one (a cup, a C-shape). Used by
    decompose="auto" to decompose only the links that are actually concave.
    Returns 1.0 if trimesh is unavailable or the mesh is degenerate.
    """
    try:
        import trimesh as _tm
        m = _tm.Trimesh(vertices=np.asarray(all_pts, dtype=float),
                        faces=np.asarray(all_idx, dtype=int).reshape(-1, 3),
                        process=False)
        if m.volume <= 1e-12:
            return 1.0
        return float(m.convex_hull.volume / m.volume)
    except Exception:  # noqa: BLE001
        return 1.0


def _warn_hull_fallback(reason):
    """Warn that a requested convex decomposition silently degraded to a hull.

    Parameters
    ----------
    reason : str
        Why the decomposition did not run, phrased for the caller.
    """
    warnings.warn(
        'convex decomposition was requested for this link but did not run '
        '({}); the collider falls back to a SINGLE CONVEX HULL. A hull fills '
        'every cavity, so a room, shell or gripper becomes a solid block that '
        'nothing can enter or grasp.'.format(reason),
        RuntimeWarning, stacklevel=3)


def _coacd_parts(all_pts, all_counts, all_idx, coacd_params):
    """Convex-decompose a collision mesh with CoACD via
    :func:`skrobot.utils.convex_decomposition.convex_decomposition`; return each
    part as (points, faces) ready to author as a convex collider.

    A single convex hull fills in a concave part, so any cavity becomes solid.

    Returns None (caller falls back to a single hull) if the mesh is not
    triangles or CoACD is unavailable. Every such path warns first: silently
    authoring a hull where a decomposition was asked for produces a collider
    that is wrong in a way nothing downstream can detect. Results are cached on
    the mesh contents and parameters, since one mesh file is often reused
    across links.
    """
    import hashlib

    if any(int(c) != 3 for c in all_counts):
        # convex_decomposition needs a triangle mesh
        _warn_hull_fallback('the collision mesh is not triangulated')
        return None
    if not is_coacd_available():
        _warn_hull_fallback(
            "the optional 'coacd' package is not installed -- "
            'install it with: pip install coacd')
        return None
    import trimesh as _tm

    verts = np.asarray(all_pts, dtype=float)
    faces = np.asarray(all_idx, dtype=int).reshape(-1, 3)
    key = (hashlib.md5(verts.tobytes()).hexdigest(),
           hashlib.md5(faces.tobytes()).hexdigest(),
           tuple(sorted((coacd_params or {}).items())))
    if key in _COACD_CACHE:
        cached = _COACD_CACHE[key]
        if cached is None:
            _warn_hull_fallback('CoACD already failed on this mesh')
        return cached

    try:
        parts = convex_decomposition(
            _tm.Trimesh(vertices=verts, faces=faces, process=False),
            **(coacd_params or {}))
    except Exception as e:  # noqa: BLE001
        _COACD_CACHE[key] = None
        _warn_hull_fallback('CoACD raised {}: {}'.format(
            type(e).__name__, e))
        return None

    # convex_decomposition already returns watertight convex hulls
    out = [(p.vertices.tolist(), np.asarray(p.faces, dtype=int))
           for p in parts if p.is_watertight and len(p.vertices) >= 4]
    if not out:
        _warn_hull_fallback(
            'CoACD returned no usable watertight convex parts')
        out = None
    _COACD_CACHE[key] = out
    return out


def _emit_geoms(stage, UsdGeom, UsdPhysics, Gf, Vt, link_path, link,
                decompose=False, coacd_params=None):
    for i, vis in enumerate(getattr(link, "visuals", []) or []):
        _emit_one(stage, UsdGeom, UsdPhysics, Gf, Vt, link_path, vis, i,
                  collision=False, link_name=link.name)
    for i, col in enumerate(getattr(link, "collisions", []) or []):
        _emit_one(stage, UsdGeom, UsdPhysics, Gf, Vt, link_path, col, i,
                  collision=True, decompose=decompose,
                  coacd_params=coacd_params, link_name=link.name)


def _visual_rgb(vc):
    """Return visual colour from URDF material as linear RGB, or None."""
    _mat = getattr(vc, "material", None)
    _col = getattr(_mat, "color", None) if _mat is not None else None
    if _col is None or len(_col) < 3:
        return None
    return [_srgb_to_linear(float(c)) for c in _col[:3]]


def _emit_one(stage, UsdGeom, UsdPhysics, Gf, Vt, link_path, vc, index, collision,
              decompose=False, coacd_params=None, link_name=None):
    geom = getattr(vc, "geometry", None)
    if geom is None:
        return
    origin = np.asarray(getattr(vc, "origin", np.eye(4)), dtype=float)

    if getattr(geom, "mesh", None) is None:
        prim_name = "{}_{}".format("collision" if collision else "visual", index)
        prim_path = link_path.AppendChild(prim_name)
        prim = None
        kind = "unknown"

        if getattr(geom, "box", None) is not None:
            kind = "box"
            # URDF box sizes are full XYZ extents; emit a unit USD cube and
            # bake non-uniform scaling into the local transform.
            box = geom.box
            prim = UsdGeom.Cube.Define(stage, prim_path)
            prim.CreateSizeAttr(1.0)
            xfm = origin.copy()
            xfm[:3, :3] = xfm[:3, :3] @ np.diag(np.asarray(box.size, dtype=float))
            prim.AddTransformOp().Set(_mat_to_gf(None, xfm))
        elif getattr(geom, "cylinder", None) is not None:
            kind = "cylinder"
            cyl = geom.cylinder
            prim = UsdGeom.Cylinder.Define(stage, prim_path)
            prim.CreateRadiusAttr(float(cyl.radius))
            prim.CreateHeightAttr(float(cyl.length))
            prim.CreateAxisAttr(UsdGeom.Tokens.z)
            prim.AddTransformOp().Set(_mat_to_gf(None, origin))
        elif getattr(geom, "sphere", None) is not None:
            kind = "sphere"
            sph = geom.sphere
            prim = UsdGeom.Sphere.Define(stage, prim_path)
            prim.CreateRadiusAttr(float(sph.radius))
            prim.AddTransformOp().Set(_mat_to_gf(None, origin))

        if prim is None:
            warnings.warn(
                'link "{}" has URDF {} primitive "{}"; this primitive type is '
                'not yet supported by the USD exporter and was omitted.'.format(
                    link_name or str(link_path).split("/")[-1],
                    "collision" if collision else "visual", kind),
                RuntimeWarning, stacklevel=2)
            return

        if collision:
            UsdGeom.Imageable(prim.GetPrim()).MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
        else:
            urdf_rgb = _visual_rgb(vc)
            if urdf_rgb is not None:
                dc = UsdGeom.Gprim(prim.GetPrim()).CreateDisplayColorPrimvar(
                    UsdGeom.Tokens.constant)
                dc.Set(Vt.Vec3fArray([Gf.Vec3f(*urdf_rgb)]))
        return

    submeshes = list(_submesh_geometry(geom.mesh))

    if collision:
        # ONE convex hull per collision element. PhysX's convex cooker only uses
        # the point cloud, so concatenate all submeshes into a single hull:
        # a hull per visual submesh produces degenerate slivers whose AABBs trip
        # the PhysX broadphase ("Illegal BroadPhaseUpdateData") and NaN the
        # articulation.
        # Drop stray/degenerate far-outlier submeshes before hulling: some URDFs
        # ship a submesh placed metres from the link frame, which
        # would make a huge degenerate convex hull that NaNs the articulation.
        if len(submeshes) > 1:
            cents = [np.mean(np.asarray(pts, dtype=float), axis=0)
                     if pts else np.zeros(3) for pts, _, _, _ in submeshes]
            med = np.median(np.array(cents), axis=0)
            dist = [float(np.linalg.norm(c - med)) for c in cents]
            threshold = max(0.15, 3.0 * float(np.median(dist)))
            submeshes = [sm for sm, d in zip(submeshes, dist) if d <= threshold] or submeshes
        all_pts = []
        all_counts = []
        all_idx = []
        offset = 0
        for pts, counts, idx, _rgb in submeshes:
            all_pts.extend(pts)
            all_counts.extend(counts)
            all_idx.extend(int(j) + offset for j in idx)
            offset += len(pts)
        if len(all_pts) < 4:
            return  # cannot form a 3D convex hull

        def _emit_convex(path_name, pts, counts, idx):
            p = link_path.AppendChild(path_name)
            mesh = UsdGeom.Mesh.Define(stage, p)
            mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in pts]))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
            mesh.AddTransformOp().Set(_mat_to_gf(None, origin))
            UsdGeom.Imageable(mesh).MakeInvisible()  # collider is not rendered
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            mc = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
            mc.CreateApproximationAttr(UsdPhysics.Tokens.convexHull)

        # Each part becomes its own convexHull collider; PhysX unions the
        # shapes on the link. Under decompose="auto", only decompose when the
        # single hull is measurably bigger than the true mesh (i.e. concave).
        _do_decompose = decompose == "force" or (
            decompose == "auto" and _hull_inflation(all_pts, all_idx)
            >= _DECOMPOSE_RATIO)
        if _do_decompose:
            parts = _coacd_parts(all_pts, all_counts, all_idx, coacd_params)
            if parts:
                for k, (pts_k, faces_k) in enumerate(parts):
                    _emit_convex("collision_{}_part_{}".format(index, k),
                                 pts_k, [3] * len(faces_k),
                                 faces_k.flatten().tolist())
                return

        # Reduce to an ACTUAL convex hull (a few clean vertices) instead of
        # handing PhysX the raw, dense, non-watertight point soup: dense or
        # degenerate input cooks into a bad hull that destabilises the
        # articulation.
        try:
            import trimesh as _tm
            _hull = _tm.Trimesh(vertices=np.asarray(all_pts, dtype=float)).convex_hull
            if _hull.is_watertight and len(_hull.vertices) >= 4:
                all_pts = _hull.vertices.tolist()
                _hf = np.asarray(_hull.faces, dtype=int)
                all_counts = [3] * len(_hf)
                all_idx = _hf.flatten().tolist()
        except Exception:  # noqa: BLE001
            pass
        _emit_convex("collision_{}".format(index), all_pts, all_counts, all_idx)
        return

    # The URDF's <material> on a visual OVERRIDES the colour baked into the mesh
    # file, which is how RViz and other URDF consumers read it. Links with no
    # <material> keep the mesh's own colours.
    # displayColor is consumed as a LINEAR albedo while URDF rgba is written as
    # an sRGB display colour, so it must be converted or the hue shifts: 0.5 sRGB
    # is only 0.214 linear.
    urdf_rgb = _visual_rgb(vc)

    # visual: URDF material colour if given, else keep per-submesh colors
    for sub_i, (pts, counts, idx, rgb) in enumerate(submeshes):
        p = link_path.AppendChild("visual_{}_{}".format(index, sub_i))
        mesh = UsdGeom.Mesh.Define(stage, p)
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*v) for v in pts]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
        mesh.AddTransformOp().Set(_mat_to_gf(None, origin))
        if urdf_rgb is not None:
            dc = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant)
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*urdf_rgb)]))
        elif rgb is not None:
            # Mesh colours are sRGB too (COLLADA diffuse, glTF baseColorFactor),
            # so they need the same linearisation as the URDF ones.
            lin = np.asarray(rgb, dtype=float)
            lin = np.where(lin <= 0.04045, lin / 12.92,
                           ((lin + 0.055) / 1.055) ** 2.4)
            dc = mesh.CreateDisplayColorPrimvar(
                UsdGeom.Tokens.vertex if len(rgb) == len(pts)
                else UsdGeom.Tokens.constant)
            dc.Set(Vt.Vec3fArray([Gf.Vec3f(*c) for c in lin.tolist()]))


def _emit_joint(stage, UsdPhysics, Gf, Sdf, joint, link_path, stiffness, damping,
                home=0.0, armature=0.0):
    if joint.parent not in link_path or joint.child not in link_path:
        return
    jtype = joint.joint_type
    jpath = Sdf.Path("/robot/joints/" + sanitize_name(joint.name))
    origin = np.asarray(joint.origin, dtype=float)
    pos0 = origin[:3, 3]
    align = _align_z_to(getattr(joint, "axis", [0, 0, 1]))
    rot0 = origin[:3, :3] @ align

    if jtype == "fixed":
        j = UsdPhysics.FixedJoint.Define(stage, jpath)
        j.CreateBody0Rel().SetTargets([link_path[joint.parent]])
        j.CreateBody1Rel().SetTargets([link_path[joint.child]])
        j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
        j.CreateLocalRot0Attr(_quat_wxyz(None, origin[:3, :3]))
        return
    if jtype in ("revolute", "continuous"):
        j = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        angular = True
    elif jtype == "prismatic":
        j = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        angular = False
    else:
        warnings.warn(
            'joint "{}" has unsupported URDF type "{}"; exported as a fixed '
            'weld. This removes that DOF, so robot kinematics and contact can '
            'be physically wrong.'.format(joint.name, jtype),
            RuntimeWarning, stacklevel=2)
        j = UsdPhysics.FixedJoint.Define(stage, jpath)
        j.CreateBody0Rel().SetTargets([link_path[joint.parent]])
        j.CreateBody1Rel().SetTargets([link_path[joint.child]])
        j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
        j.CreateLocalRot0Attr(_quat_wxyz(None, origin[:3, :3]))
        return

    j.CreateBody0Rel().SetTargets([link_path[joint.parent]])
    j.CreateBody1Rel().SetTargets([link_path[joint.child]])
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    j.CreateLocalRot0Attr(_quat_wxyz(None, rot0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(_quat_wxyz(None, align))
    j.CreateAxisAttr(UsdPhysics.Tokens.z)

    limit = getattr(joint, "limit", None)
    if jtype != "continuous" and limit is not None and \
            limit.lower is not None and limit.upper is not None:
        if angular:
            j.CreateLowerLimitAttr(float(np.degrees(limit.lower)))
            j.CreateUpperLimitAttr(float(np.degrees(limit.upper)))
        else:
            j.CreateLowerLimitAttr(float(limit.lower))
            j.CreateUpperLimitAttr(float(limit.upper))

    # Position drive holding the home configuration (0). Mimic joints get a
    # drive too: proper mimic->USD coupling is not implemented yet, so without a
    # drive they flop to their limits under gravity (grippers hanging open).
    # Holding them at 0 keeps the robot in a clean pose until mimic coupling
    # lands. (targetPosition/limits are in DEGREES for angular drives.)
    drive = UsdPhysics.DriveAPI.Apply(
        j.GetPrim(), "angular" if angular else "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    target = float(np.degrees(home)) if angular else float(home)
    drive.CreateTargetPositionAttr(target)

    # Joint armature (reflected motor/rotor inertia) is summed into the
    # joint-space inertia. It is essential for stable position drives on light
    # links: without it the articulation solver is ill-conditioned and drives
    # sag under gravity regardless of stiffness (this is what the shipped robots
    # -- e.g. Franka -- set and hand-authored URDFs omit).
    if armature and armature > 0.0:
        j.GetPrim().AddAppliedSchema("PhysxJointAPI")
        j.GetPrim().CreateAttribute(
            "physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(armature))


def _emit_planar_joint(stage, UsdPhysics, Gf, Sdf, name, kind, axis, parent, child,
                       stiffness, damping, limit):
    """Emit one world-axis-aligned prismatic/revolute joint for the planar mobile
    base, position-driven to 0. ``parent`` None means body0 = world."""
    jpath = Sdf.Path("/robot/joints/" + name)
    if kind == "prismatic":
        j = UsdPhysics.PrismaticJoint.Define(stage, jpath)
        angular = False
    else:
        j = UsdPhysics.RevoluteJoint.Define(stage, jpath)
        angular = True
    if parent is not None:
        j.CreateBody0Rel().SetTargets([parent])
    j.CreateBody1Rel().SetTargets([child])
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateAxisAttr(getattr(UsdPhysics.Tokens, axis.lower()))
    if limit is not None:
        j.CreateLowerLimitAttr(-float(limit))
        j.CreateUpperLimitAttr(float(limit))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if angular else "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(stiffness))
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetPositionAttr(0.0)
