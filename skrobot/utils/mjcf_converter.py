"""Convert a URDF into a MuJoCo MJCF model.

MuJoCo's own URDF importer is strict (it rejects empty ``<collision>`` geometry,
ignores ``mimic`` joints, and needs ``<mujoco>`` extension tags for actuators and
sensible defaults), so robots such as the JSK "jedy" fail to load directly. This
converter walks the *raw* parsed URDF (``skrobot.utils.urdf.URDF``, which retains
fixed joints, per-geometry origins, primitive shapes, mesh filenames+scale, and
mimic couplings) and emits a clean MJCF:

* URDF link tree -> MuJoCo ``<body>`` tree (a fixed joint simply produces a body
  with no ``<joint>``, i.e. a weld -- no collapsing needed).
* revolute/continuous -> ``hinge``, prismatic -> ``slide`` (limits -> range).
* URDF ``<inertial>`` -> MJCF ``<inertial ... fullinertia=...>``.
* box/cylinder/sphere primitives and meshes -> geoms; meshes exported to STL
  assets and referenced from ``<asset>``. Empty geometry is skipped.
* ``mimic`` -> ``<equality><joint polycoef=...>``.
* one ``<position>`` actuator per actuated (non-mimic) joint (optional).
* ``<compiler angle="radian">`` so radians match ROS.

Entry point: :func:`urdf_to_mjcf`.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates.math import matrix2translation_quaternion_wxyz


def _fmt(values):
    return " ".join("{:.9g}".format(float(v)) for v in np.ravel(values))


class _MeshAssets:
    """Collects unique meshes, exports them to STL, and hands out asset names."""

    # MuJoCo's STL decoder rejects meshes with more than this many faces.
    MAX_FACES = 200000

    def __init__(self, mesh_dir):
        self.mesh_dir = mesh_dir
        self._by_id = {}      # id(trimesh) -> asset name
        self.entries = []     # (name, filename, scale)
        os.makedirs(mesh_dir, exist_ok=True)

    def _decimate(self, mesh):
        if len(mesh.faces) <= self.MAX_FACES:
            return mesh
        target = int(self.MAX_FACES * 0.9)
        try:  # quadric decimation (needs fast-simplification); best visual result
            simplified = mesh.simplify_quadric_decimation(face_count=target)
            if simplified is not None and len(simplified.faces) > 0:
                return simplified
        except Exception:
            pass
        try:  # guaranteed-valid fallback: convex hull (few faces)
            return mesh.convex_hull
        except Exception:
            return mesh

    def add(self, mesh_geometry):
        """Return a list of ``(asset_name, scale, rgba)`` -- one entry PER
        sub-mesh. A single URDF visual mesh (e.g. a .glb) often bundles several
        differently-coloured parts; emitting one geom per sub-mesh keeps those
        colours, since MuJoCo has only a single colour per geom."""
        import trimesh
        from trimesh.exchange.stl import export_stl

        submeshes = [m for m in (getattr(mesh_geometry, "meshes", None) or [])
                     if isinstance(m, trimesh.Trimesh) and len(m.faces) > 0]
        scale = getattr(mesh_geometry, "scale", None)
        base = os.path.splitext(os.path.basename(
            getattr(mesh_geometry, "filename", "") or "mesh"))[0]

        out = []
        for sub in submeshes:
            # colour from the ORIGINAL sub-mesh (decimation's convex-hull
            # fallback drops vertex colours); STL cannot carry it either.
            color = None
            try:
                mc = sub.visual.main_color  # RGBA uint8
                color = [c / 255.0 for c in mc]
            except Exception:
                pass

            key = id(sub)
            if key in self._by_id:
                name = self._by_id[key]
            else:
                mesh = self._decimate(sub)
                if len(mesh.faces) == 0:
                    continue
                name = "{}_{}".format(base or "mesh", len(self.entries))
                fname = name + ".stl"
                with open(os.path.join(self.mesh_dir, fname), "wb") as f:
                    f.write(export_stl(mesh))  # MuJoCo needs BINARY STL
                self._by_id[key] = name
                self.entries.append((name, fname, scale))
            out.append((name, scale, color))
        return out


def _add_geom(parent_el, visual_or_collision, assets, *, collision, rgba=None):
    """Emit one <geom> for a Visual/Collision; return True if emitted."""
    geom = getattr(visual_or_collision, "geometry", None)
    if geom is None:
        return False
    origin = getattr(visual_or_collision, "origin", None)
    if origin is None:
        origin = np.eye(4)
    pos, quat = matrix2translation_quaternion_wxyz(origin)

    base_attrib = {"pos": _fmt(pos), "quat": _fmt(quat)}
    if collision:
        base_attrib["group"] = "3"
    else:
        # visual-only: no contact, distinct group so viewers can toggle it.
        base_attrib["group"] = "2"
        base_attrib["contype"] = "0"
        base_attrib["conaffinity"] = "0"

    if geom.box is not None:
        a = dict(base_attrib, type="box",
                 size=_fmt(np.asarray(geom.box.size, dtype=float) / 2.0))
        if rgba is not None:
            a["rgba"] = _fmt(rgba)
        ET.SubElement(parent_el, "geom", a)
        return True
    if geom.cylinder is not None:
        a = dict(base_attrib, type="cylinder",
                 size=_fmt([geom.cylinder.radius, geom.cylinder.length / 2.0]))
        if rgba is not None:
            a["rgba"] = _fmt(rgba)
        ET.SubElement(parent_el, "geom", a)
        return True
    if geom.sphere is not None:
        a = dict(base_attrib, type="sphere", size=_fmt([geom.sphere.radius]))
        if rgba is not None:
            a["rgba"] = _fmt(rgba)
        ET.SubElement(parent_el, "geom", a)
        return True
    if geom.mesh is not None:
        subs = assets.add(geom.mesh)
        if not subs:
            return False
        for name, scale, mesh_color in subs:
            a = dict(base_attrib, type="mesh", mesh=name)
            # URDF <material> colour wins; else the sub-mesh's own colour.
            eff = rgba if rgba is not None else mesh_color
            if eff is not None:
                a["rgba"] = _fmt(eff)
            ET.SubElement(parent_el, "geom", a)
        return True
    return False


def _visual_rgba(visual):
    material = getattr(visual, "material", None)
    if material is not None and getattr(material, "color", None) is not None:
        return np.asarray(material.color, dtype=float)
    return None


def _emit_body(link, joint_from_parent, urdf, assets, children_map,
               parent_el, actuated, qpos_layout):
    """Recursively emit <body> for `link` under `parent_el`."""
    body_attrib = {"name": link.name}
    if joint_from_parent is not None:
        pos, quat = matrix2translation_quaternion_wxyz(joint_from_parent.origin)
        body_attrib["pos"] = _fmt(pos)
        body_attrib["quat"] = _fmt(quat)
    body = ET.SubElement(parent_el, "body", body_attrib)

    # joint (movable only; fixed joint -> no <joint> element = weld)
    if joint_from_parent is not None and joint_from_parent.joint_type != "fixed":
        jt = joint_from_parent.joint_type
        j = {"name": joint_from_parent.name, "pos": "0 0 0",
             "axis": _fmt(joint_from_parent.axis)}
        if jt == "prismatic":
            j["type"] = "slide"
        else:
            j["type"] = "hinge"
        limit = getattr(joint_from_parent, "limit", None)
        if jt != "continuous" and limit is not None and \
                limit.lower is not None and limit.upper is not None:
            j["range"] = _fmt([limit.lower, limit.upper])
        ET.SubElement(body, "joint", j)
        # every movable joint (mimic included) contributes one qpos entry, in
        # this creation order -- used to lay out the optional home keyframe.
        qpos_layout.append(joint_from_parent.name)
        if joint_from_parent.mimic is None:
            actuated.append(joint_from_parent)

    _emit_link_content(link, body, assets)

    # recurse
    for child_joint in children_map.get(link.name, []):
        child_link = urdf.link_map[child_joint.child]
        _emit_body(child_link, child_joint, urdf, assets, children_map,
                   body, actuated, qpos_layout)


def urdf_to_mjcf(urdf, out_path, mesh_dir=None, floating_base=False,
                 add_position_actuators=True, actuator_kp=50.0,
                 actuator_type="position", actuator_kv=1.0,
                 joint_armature=0.0, joint_damping=0.0,
                 add_ground=True, gravity=(0, 0, -9.81),
                 self_collision=False, add_actuator_forcerange=True,
                 home=None, home_base_height=0.0):
    """Convert a URDF to an MJCF file.

    Parameters
    ----------
    urdf : str or skrobot.utils.urdf.URDF
        Path to a ``.urdf`` file, or an already-parsed URDF object.
    out_path : str
        Where to write the ``.xml`` MJCF. Mesh assets go in ``mesh_dir``
        (default ``<out_dir>/assets``), referenced relatively via ``meshdir``.
    floating_base : bool
        If True the base link gets a free joint; otherwise it is welded to world.
    add_position_actuators : bool
        Emit one ``<position>`` actuator per actuated joint.
    actuator_kp : float
        Proportional gain for the position actuators.
    add_ground : bool
        Add a ground plane + light for a usable scene.
    self_collision : bool
        If False (default) the robot's collision geoms are put on a
        ``contype=2 conaffinity=1`` mask while the ground stays ``1/1``, so the
        robot collides with the ground but *not with itself*. Legged-locomotion
        training almost always wants this off unless an explicit adjacency
        filter is provided -- self intersections of neighbouring links cause
        spawn-time blow-ups. Set True to keep full self-collision.
    add_actuator_forcerange : bool
        If True (default) position/velocity actuators get a ``forcerange`` taken
        from the URDF joint effort limit, so RL policies cannot command
        physically impossible torques (a classic "the humanoid flies" exploit).
    home : dict[str, float] or None
        Optional joint-name -> angle (rad) map. When given, a ``<keyframe>``
        named ``home`` is emitted so callers can reset to a known stable pose.
        Joints absent from the map default to 0.
    home_base_height : float
        Base-link height (m) used for the free-joint part of the home keyframe
        when ``floating_base`` is True.

    Returns
    -------
    str
        The MJCF XML written to ``out_path``.
    """
    from skrobot.utils.urdf import URDF

    if isinstance(urdf, str):
        urdf = URDF.load(urdf)

    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    if mesh_dir is None:
        mesh_dir = os.path.join(out_dir, "assets")
    assets = _MeshAssets(mesh_dir)

    # children map: parent link name -> [joints]
    children_map = {}
    for joint in urdf.joints:
        children_map.setdefault(joint.parent, []).append(joint)

    root = ET.Element("mujoco", {"model": urdf.name or "robot"})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "autolimits": "true",
        "meshdir": os.path.relpath(mesh_dir, out_dir)})
    ET.SubElement(root, "option", {"gravity": _fmt(gravity)})

    # A little rotor inertia (armature) and damping on every joint models real
    # geared servos and keeps velocity/position actuators numerically stable --
    # bare URDF inertias are often too small for a stiff servo at these timesteps.
    if joint_armature or joint_damping or not self_collision:
        default = ET.SubElement(root, "default")
        if joint_armature or joint_damping:
            jd = {}
            if joint_armature:
                jd["armature"] = "{:.9g}".format(joint_armature)
            if joint_damping:
                jd["damping"] = "{:.9g}".format(joint_damping)
            ET.SubElement(default, "joint", jd)
        if not self_collision:
            # Default mask for every geom: contype=2 conaffinity=1. Collision
            # geoms inherit it; the ground overrides to 1/1 below and visual
            # geoms override to 0/0. Two robot geoms (2/1 vs 2/1) then never
            # collide, while robot(2/1)-vs-ground(1/1) does.
            ET.SubElement(default, "geom",
                          {"contype": "2", "conaffinity": "1"})

    asset_el = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    if add_ground:
        ET.SubElement(worldbody, "light", {"pos": "0 0 3", "dir": "0 0 -1",
                                           "directional": "true"})
        ET.SubElement(worldbody, "geom", {"name": "ground", "type": "plane",
                                          "size": "5 5 0.1",
                                          "contype": "1", "conaffinity": "1",
                                          "rgba": "0.8 0.8 0.8 1"})

    base_link = urdf.base_link
    has_free_joint = False
    if base_link.name == "world":
        # URDF virtual root: MuJoCo already has a built-in "world" body, so don't
        # create another (would collide); attach the base link's children to it.
        base_body = worldbody
    else:
        base_body = ET.SubElement(worldbody, "body",
                                  {"name": base_link.name, "pos": "0 0 0"})
        if floating_base:
            ET.SubElement(base_body, "freejoint", {"name": "floating_base"})
            has_free_joint = True
        _emit_link_content(base_link, base_body, assets)
    actuated = []
    qpos_layout = []  # ordered movable-joint names, for the home keyframe
    for child_joint in children_map.get(base_link.name, []):
        child_link = urdf.link_map[child_joint.child]
        _emit_body(child_link, child_joint, urdf, assets, children_map,
                   base_body, actuated, qpos_layout)

    # mesh assets
    for name, fname, scale in assets.entries:
        attrs = {"name": name, "file": fname}
        if scale is not None:
            attrs["scale"] = _fmt(scale)
        ET.SubElement(asset_el, "mesh", attrs)

    # mimic joints -> equality constraints
    mimic_joints = [j for j in urdf.joints
                    if getattr(j, "mimic", None) is not None]
    if mimic_joints:
        eq = ET.SubElement(root, "equality")
        for j in mimic_joints:
            m = j.mimic
            ET.SubElement(eq, "joint", {
                "joint1": j.name, "joint2": m.joint,
                "polycoef": _fmt([m.offset, m.multiplier, 0, 0, 0])})

    # actuators
    # One actuator per actuated joint. actuator_type selects the operating mode:
    #   position -> <position kp>  (joint tracks a target angle)
    #   velocity -> <velocity kv>  (joint tracks a target rate)
    #   motor    -> <motor>        (direct torque/force command)
    if add_position_actuators and actuated:
        act = ET.SubElement(root, "actuator")
        for j in actuated:
            limit = getattr(j, "limit", None)
            has_pos_range = (j.joint_type != "continuous" and limit is not None
                             and limit.lower is not None and limit.upper is not None)
            a = {"name": j.name + "_act", "joint": j.name}
            effort = getattr(limit, "effort", None) if limit is not None else None
            if actuator_type == "velocity":
                a["kv"] = "{:.9g}".format(actuator_kv)
                if limit is not None and getattr(limit, "velocity", None):
                    a["ctrlrange"] = _fmt([-limit.velocity, limit.velocity])
                if add_actuator_forcerange and effort:
                    a["forcerange"] = _fmt([-effort, effort])
                ET.SubElement(act, "velocity", a)
            elif actuator_type == "motor":
                if effort:
                    a["ctrlrange"] = _fmt([-effort, effort])
                ET.SubElement(act, "motor", a)
            else:  # position
                a["kp"] = "{:.9g}".format(actuator_kp)
                if has_pos_range:
                    a["ctrlrange"] = _fmt([limit.lower, limit.upper])
                # Cap the torque a position servo can apply to the real motor's
                # effort limit, so an RL policy cannot exploit unbounded torque
                # (the classic "small humanoid learns to fly" failure).
                if add_actuator_forcerange and effort:
                    a["forcerange"] = _fmt([-effort, effort])
                ET.SubElement(act, "position", a)

    # home keyframe: qpos = [free-joint (7)] + one entry per movable joint,
    # in creation order (qpos_layout). Joints absent from `home` default to 0.
    if home is not None:
        qpos = []
        if has_free_joint:
            qpos += [0.0, 0.0, float(home_base_height), 1.0, 0.0, 0.0, 0.0]
        for jname in qpos_layout:
            qpos.append(float(home.get(jname, 0.0)))
        key_el = ET.SubElement(root, "keyframe")
        ET.SubElement(key_el, "key", {"name": "home", "qpos": _fmt(qpos)})

    _indent(root)
    tree = ET.ElementTree(root)
    tree.write(out_path, encoding="unicode", xml_declaration=False)
    return ET.tostring(root, encoding="unicode")


def _inertial_attrib(inertial):
    """Return MJCF <inertial> attributes as pos + mass + diaginertia + quat,
    regularizing the tensor so MuJoCo accepts it (positive eigenvalues and the
    triangle inequality, both of which raw URDF inertias frequently violate)."""
    origin = np.asarray(getattr(inertial, "origin", np.eye(4)), dtype=float)
    pos = origin[:3, 3]
    rot = origin[:3, :3]
    inertia = rot @ np.asarray(inertial.inertia, dtype=float) @ rot.T
    inertia = 0.5 * (inertia + inertia.T)               # symmetrize
    moments, axes = np.linalg.eigh(inertia)             # ascending, orthonormal cols
    if np.linalg.det(axes) < 0:                         # make it a proper rotation
        axes[:, 0] = -axes[:, 0]
    moments = np.clip(moments, 1e-9, None)              # positive-definite
    lo, mid, hi = moments                               # ascending
    if hi > lo + mid:                                   # triangle inequality
        hi = lo + mid
    moments = np.array([lo, mid, hi])
    from skrobot.coordinates.math import matrix2quaternion
    quat = matrix2quaternion(axes)                      # principal-axes frame [wxyz]
    return {
        "pos": _fmt(pos),
        "mass": "{:.9g}".format(float(inertial.mass)),
        "diaginertia": _fmt(moments),
        "quat": _fmt(quat),
    }


def _emit_link_content(link, body_el, assets):
    inertial = getattr(link, "inertial", None)
    if inertial is not None and getattr(inertial, "mass", 0.0):
        try:
            ET.SubElement(body_el, "inertial", _inertial_attrib(inertial))
        except Exception:
            pass  # degenerate inertial -> let MuJoCo infer from geoms
    for vis in getattr(link, "visuals", []) or []:
        _add_geom(body_el, vis, assets, collision=False, rgba=_visual_rgba(vis))
    for col in getattr(link, "collisions", []) or []:
        _add_geom(body_el, col, assets, collision=True)


def _indent(elem, level=0):
    pad = "\n" + "  " * level
    if len(elem):
        if not (elem.text or "").strip():
            elem.text = pad + "  "
        for child in elem:
            _indent(child, level + 1)
        if not (elem.tail or "").strip():
            elem.tail = pad
        if not (child.tail or "").strip():
            child.tail = pad
    elif level and not (elem.tail or "").strip():
        elem.tail = pad
