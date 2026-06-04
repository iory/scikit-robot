"""LLM-assisted limb grouping for robot class generation.

The geometry generator in :mod:`skrobot.urdf.robot_class_generator` classifies
limbs with a fixed taxonomy (arms / legs / head / torso / gripper) and naming
heuristics. That fails on morphologies outside the taxonomy - multi-finger
hands, quadrupeds, tails, many-armed robots - where it simply emits no limbs.

This module keeps the kinematics deterministic but delegates *only the semantic
grouping* to an LLM:

1. Deterministically extract the actuated skeleton (movable chains, the
   kinematic tree, joint axes) - never hallucinated.
2. Ask an LLM to map those chains onto named limbs (e.g. ``thumb``, ``index``,
   ``rarm``) - the one step the fixed taxonomy cannot generalize.
3. Validate the LLM's answer against the real link names, then compute
   ``end_coords`` deterministically and hand the result to the same code
   emitter the geometry path uses.

The LLM is injected as a plain callable ``llm_fn(prompt: str) -> str``, so the
library stays provider-neutral and dependency-free: plug in whatever model (or
agent) you like.

Examples
--------
>>> from skrobot.model import RobotModel
>>> from skrobot.urdf import generate_robot_class_from_geometry
>>> robot = RobotModel()
>>> robot.load_urdf_file('hand_robot.urdf')
>>> def llm_fn(prompt):  # call your model here; must return the JSON grouping
...     ...
>>> code = generate_robot_class_from_geometry(
...     robot, grouping='llm', llm_fn=llm_fn, output_path='hand_robot.py')
"""

import json
import keyword
import logging

import networkx as nx
import numpy as np

from skrobot.coordinates.math import normalize_vector
from skrobot.coordinates.math import rpy_angle
from skrobot.urdf.robot_class_generator import _build_link_graph
from skrobot.urdf.robot_class_generator import _find_base_link
from skrobot.urdf.robot_class_generator import get_default_config
from skrobot.urdf.structure import kinematic_tree


logger = logging.getLogger(__name__)


def _is_safe_limb_name(name):
    """Whether ``name`` is usable as a generated method name.

    It becomes a ``def <name>(self)`` in the generated class, so it must be a
    lowercase identifier (snake_case) that is not a Python keyword. ``str``'s
    ``isidentifier`` alone is not enough: keywords like ``class`` and uppercase
    names like ``RightArm`` pass it but produce broken or odd code.
    """
    return (isinstance(name, str) and name.isidentifier()
            and not keyword.iskeyword(name) and name == name.lower())


MOVABLE_JOINT_CLASSES = ('RotationalJoint', 'LinearJoint')


def _movable_chains(robot):
    """Extract every base->leaf chain of movable links (deterministic).

    Returns
    -------
    list of dict
        Each chain is ``{'leaf': str, 'links': [movable link names in order]}``.
    """
    graph, link_map = _build_link_graph(robot)
    base_link = _find_base_link(graph)
    leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

    chains = []
    for leaf in leaves:
        try:
            path = nx.shortest_path(graph, base_link, leaf)
        except nx.NetworkXNoPath:
            continue
        movable = []
        for name in path:
            link = link_map.get(name)
            if link is None or link.joint is None:
                continue
            if type(link.joint).__name__ in MOVABLE_JOINT_CLASSES:
                movable.append(name)
        if movable:
            chains.append({'leaf': leaf, 'links': movable})
    return chains


def build_grouping_prompt(robot, config=None, extra_instructions=None):
    """Build the LLM prompt for limb grouping.

    Parameters
    ----------
    robot : RobotModel
        Loaded robot model.
    config : PatternConfig, optional
        Unused for now; reserved for naming hints.
    extra_instructions : str, optional
        Appended verbatim, e.g. domain knowledge about the robot.

    Returns
    -------
    str
        The prompt to pass to ``llm_fn``.
    """
    chains = _movable_chains(robot)
    # World-frame axes/positions are far more informative to the LLM than the
    # URDF-local axes (which are often all [0, 0, 1]).
    tree = kinematic_tree(robot, collapse_fixed=True, annotate=True, world=True)

    chain_lines = []
    for i, chain in enumerate(chains):
        chain_lines.append(
            '  chain {}: tip leaf "{}" | movable links: {}'.format(
                i, chain['leaf'], chain['links']))
    chains_text = '\n'.join(chain_lines)

    instructions = '''You are grouping a robot's actuated joints into named \
kinematic limbs for a scikit-robot RobotModel class.

Below is the robot's kinematic tree (movable skeleton) and the list of every \
actuated chain from the base to a leaf. Group the movable links into limbs that \
a roboticist would use.

Rules:
- Use ONLY the movable link names listed in the chains. Do not invent names.
- One limb per natural appendage. For a multi-finger hand, make one limb per \
finger (e.g. "thumb", "index", "middle", "ring", "little"). For an arm robot use \
"rarm"/"larm" (right/left) or "arm" for a single arm; legs "rleg"/"lleg"; also \
"head", "torso" when present.
- limb "name" must be a valid lowercase python identifier (snake_case).
- "links" must be the movable links of that limb, ordered from base to tip.
- "tip_link" is the link to attach the end-effector frame to; prefer the leaf \
of that chain (the fingertip / tool frame).
- Omit purely structural chains that are not useful limbs only if clearly \
irrelevant; otherwise include them.
- For a two-jaw / parallel gripper arm, you may add an "end_coords" field to \
that limb: {"type": "jaw_gripper", "wrist_link": "<link the jaws mount on>", \
"jaw_links": ["<one jaw link>", "<the opposing jaw link>"]}. The end frame is \
then placed at the grasp point with +x = approach and +y = closing. Use the \
world positions to identify the wrist link and the two opposing jaws. Do NOT \
also put the jaw links in that limb's "links": the arm chain that IK controls \
should stop at the wrist, not the gripper joints.

Respond with ONLY a JSON object, no prose, of the form:
{"limbs": [{"name": "thumb", "links": ["...", "..."], "tip_link": "...",
            "end_coords": {"type": "jaw_gripper", "wrist_link": "...",
                           "jaw_links": ["...", "..."]}}, ...]}'''

    parts = [instructions,
             '',
             'Robot name: {}'.format(getattr(robot, 'name', None) or 'robot'),
             '',
             'Kinematic tree:',
             tree,
             '',
             'Actuated chains:',
             chains_text]
    if extra_instructions:
        parts.extend(['', 'Additional context:', extra_instructions])
    return '\n'.join(parts)


def _tip_attached_to_chain(tip, links, link_by_name):
    """Whether ``tip`` is on the limb chain ``links``.

    True when ``tip`` is one of the limb's links, or hangs below them (i.e. one
    of its ancestors, reached by walking parent links, is in the chain). This
    rejects an LLM answer that parents the end-effector frame to an unrelated
    link - e.g. an arm limb whose ``tip_link`` is a head or tool link - which
    would otherwise silently produce an inconsistent class.
    """
    chain = set(links)
    if tip in chain:
        return True
    node = link_by_name.get(tip)
    visited = set()
    while node is not None and id(node) not in visited:
        visited.add(id(node))
        if node.name in chain:
            return True
        node = getattr(node, 'parent_link', None)
    return False


def parse_grouping_response(text, robot):
    """Parse and validate an LLM grouping response against the real robot.

    Parameters
    ----------
    text : str
        Raw LLM response (may be wrapped in markdown fences).
    robot : RobotModel
        The robot, used to validate that every referenced link exists and is
        movable.

    Returns
    -------
    list of dict
        Validated limbs, each ``{'name', 'links', 'tip_link'}``.

    Raises
    ------
    ValueError
        If the response is not valid JSON, or references links that do not
        exist / are not movable. Errors are surfaced, never silently dropped.
    """
    cleaned = text.strip()
    if cleaned.startswith('```'):
        # Strip ```json ... ``` fences.
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith('```')]
        cleaned = '\n'.join(lines).strip()

    # Tolerate leading/trailing prose by extracting the outermost JSON object.
    start = cleaned.find('{')
    end = cleaned.rfind('}')
    if start == -1 or end == -1 or end < start:
        raise ValueError(
            'LLM grouping response is not JSON: {!r}'.format(text[:200]))
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(
            'Failed to parse LLM grouping JSON: {}'.format(exc))

    limbs = data.get('limbs')
    if not isinstance(limbs, list) or not limbs:
        raise ValueError(
            "LLM grouping JSON must contain a non-empty 'limbs' list")

    link_names = {link.name for link in robot.link_list}
    link_by_name = {link.name: link for link in robot.link_list}
    movable_names = {
        link.name for link in robot.link_list
        if link.joint is not None
        and type(link.joint).__name__ in MOVABLE_JOINT_CLASSES}

    validated = []
    seen_names = set()
    for limb in limbs:
        name = limb.get('name')
        links = limb.get('links')
        tip = limb.get('tip_link')
        if not _is_safe_limb_name(name):
            raise ValueError(
                'Invalid limb name from LLM: {!r} (expected a lowercase '
                'identifier that is not a Python keyword)'.format(name))
        if name in seen_names:
            raise ValueError('Duplicate limb name from LLM: {!r}'.format(name))
        if not isinstance(links, list) or not links:
            raise ValueError(
                "Limb {!r} has no 'links'".format(name))
        bad = [l for l in links if l not in movable_names]
        if bad:
            raise ValueError(
                'Limb {!r} references unknown/non-movable links: {}'.format(
                    name, bad))
        if tip is None:
            tip = links[-1]
        if tip not in link_names:
            raise ValueError(
                'Limb {!r} tip_link {!r} is not a real link'.format(name, tip))
        if not _tip_attached_to_chain(tip, links, link_by_name):
            raise ValueError(
                'Limb {!r} tip_link {!r} is not on the limb chain (it must be '
                "one of the limb's links or hang below them); got a "
                'disconnected link.'.format(name, tip))

        end_coords = _validate_end_coords(
            limb.get('end_coords'), name, link_names)

        seen_names.add(name)
        entry = {'name': name, 'links': list(links), 'tip_link': tip}
        if end_coords is not None:
            entry['end_coords'] = end_coords
        validated.append(entry)
    return validated


def _validate_end_coords(spec, name, link_names):
    """Validate an optional per-limb ``end_coords`` spec against real links.

    Currently supports ``{"type": "jaw_gripper", "wrist_link": ...,
    "jaw_links": [a, b]}``. Returns the normalized spec, or None when absent.
    """
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError(
            "Limb {!r} 'end_coords' must be an object".format(name))
    kind = spec.get('type')
    if kind != 'jaw_gripper':
        raise ValueError(
            "Limb {!r} end_coords type {!r} is not supported "
            "(use 'jaw_gripper')".format(name, kind))
    wrist = spec.get('wrist_link')
    jaws = spec.get('jaw_links')
    if wrist not in link_names:
        raise ValueError(
            'Limb {!r} jaw_gripper wrist_link {!r} is not a real link'.format(
                name, wrist))
    if not isinstance(jaws, list) or len(jaws) != 2:
        raise ValueError(
            "Limb {!r} jaw_gripper needs exactly two 'jaw_links'".format(name))
    bad = [j for j in jaws if j not in link_names]
    if bad:
        raise ValueError(
            'Limb {!r} jaw_gripper jaw_links not real links: {}'.format(
                name, bad))
    return {'kind': 'jaw_gripper', 'wrist_link': wrist,
            'jaw_links': list(jaws)}


def _link_vertices(link):
    """Local mesh vertices of ``link`` (collision preferred), or None."""
    mesh = getattr(link, 'collision_mesh', None)
    if mesh is not None and getattr(mesh, 'vertices', None) is not None \
            and len(mesh.vertices):
        return np.asarray(mesh.vertices)
    vmesh = getattr(link, 'visual_mesh', None)
    if vmesh:
        first = vmesh[0] if isinstance(vmesh, (list, tuple)) else vmesh
        if getattr(first, 'vertices', None) is not None \
                and len(first.vertices):
            return np.asarray(first.vertices)
    return None


def _fingertip(link, ref_pos):
    """Farthest mesh point of ``link`` from ``ref_pos`` (world).

    Falls back to the link origin when no mesh is available.
    """
    verts = _link_vertices(link)
    if verts is None:
        return np.asarray(link.worldpos(), dtype=np.float64)
    world = link.worldrot().dot(verts.T).T + link.worldpos()
    dist = np.linalg.norm(world - ref_pos, axis=1)
    return world[int(np.argmax(dist))]


def _fingertip_along(link, origin, direction):
    """Mesh point of ``link`` reaching farthest along ``direction`` from
    ``origin`` (world) - i.e. the actual contact pad / fingertip.

    Using "farthest forward (along approach)" rather than "farthest from the
    origin" is robust when the link origin sits at the distal end and the mesh
    trails back to the knuckle: the latter would wrongly pick the knuckle.
    Falls back to ``origin`` when the link has no mesh.
    """
    verts = _link_vertices(link)
    if verts is None:
        return np.asarray(origin, dtype=np.float64)
    world = link.worldrot().dot(verts.T).T + link.worldpos()
    proj = (world - np.asarray(origin)).dot(direction)
    return world[int(np.argmax(proj))]


def _kinematic_closing(jaw_a, jaw_b, masters, probe,
                       open_angle, closed_angle, eps=1e-3):
    """Closing direction from kinematics alone (no mesh).

    With the gripper at the closed pose, nudge the driving joints a small step
    toward open and measure how the two finger material points that coincide at
    ``probe`` (the grasp center) move. Their relative velocity is the direction
    the jaws separate - i.e. the closing axis, up to sign.

    Because it tracks material points through ``worldcoords`` it follows each
    finger's joint screw (its ``world_axis``), so it is correct for prismatic
    and revolute jaws alike, and stays well defined when a finger rotates about
    an axis through its own origin (the origin never translates, but a point at
    the grasp center does - the case the link-origin separation cannot see).
    Returns None when there is nothing to actuate. Restores the closed pose.
    """
    if not masters:
        return None

    def _to_local(link, point):
        rot = np.asarray(link.worldrot())
        return rot.T.dot(np.asarray(point) - np.asarray(link.worldpos()))

    def _to_world(link, local):
        return np.asarray(link.worldrot()).dot(local) \
            + np.asarray(link.worldpos())

    local_a = _to_local(jaw_a, probe)
    local_b = _to_local(jaw_b, probe)
    for joint in masters:
        lo, hi = closed_angle[id(joint)], open_angle[id(joint)]
        joint.joint_angle(lo + eps * (hi - lo))
    vel_a = _to_world(jaw_a, local_a) - probe
    vel_b = _to_world(jaw_b, local_b) - probe
    for joint in masters:
        joint.joint_angle(closed_angle[id(joint)])
    relative = vel_a - vel_b
    if np.linalg.norm(relative) < 1e-12:
        return None
    return relative


def compute_jaw_gripper_frame(robot, wrist_link, jaw_links):
    """Compute a jaw-gripper end-effector frame by actuating the gripper.

    Actuates the gripper joints and reads the resulting link ``worldcoords``;
    everything below the level of "which links are the wrist and the two jaws"
    is pure kinematics, so it generalizes across grippers without per-robot
    tuning. Relative to ``wrist_link`` it derives:

    - orientation: local +x = approach (out of the gripper), +y = closing.
      The closing axis comes from the kinematic relative velocity of the two
      fingers at the grasp center (see :func:`_kinematic_closing`) - it follows
      each finger's joint axis, so prismatic and revolute jaws, and fingers
      rotating about their own origin, are all handled. Its sign is snapped to
      the wrist link's frame (hand-written grippers align +y with a wrist
      axis), so mirrored left/right grippers come out consistently.
    - position: laterally the jaw-origin midpoint (centered between the jaws);
      its forward depth is taken from the fingertips (mesh vertex reaching
      farthest along the approach), or the jaw origin when there is no mesh.

    Returns a dict ``{'parent_link', 'pos', 'rot'}`` where ``rot`` is the rpy
    angle the generated ``CascadedCoords`` understands. The robot's pose is
    restored before returning.
    """
    wrist = getattr(robot, wrist_link)
    jaw_a = getattr(robot, jaw_links[0])
    jaw_b = getattr(robot, jaw_links[1])

    # Collect the gripper's actuated joints: every movable, non-mimic joint on
    # the path from each jaw up to the wrist. This covers single-joint jaws
    # (e.g. PR2) as well as multi-joint fingers (e.g. a thumb-vs-index hand)
    # and lets the jaw links be fixed fingertip frames.
    masters = []
    path_joints = []
    seen = set()
    for jaw in (jaw_a, jaw_b):
        link = jaw
        while link is not None and link is not wrist:
            joint = getattr(link, 'joint', None)
            if (joint is not None and id(joint) not in seen
                    and type(joint).__name__ in MOVABLE_JOINT_CLASSES):
                seen.add(id(joint))
                path_joints.append(joint)
                # ``masters`` are the joints we actually actuate to open/close
                # the gripper: movable, non-mimic, with a real (bounded) range.
                # A continuous joint (full +-pi / 2-pi span) is excluded - its
                # nominal limits do not encode a real open/closed range, so
                # driving it to a limit just yields a garbage pose. Mimic joints
                # are driven by their master, so they are not actuated directly.
                if (not getattr(joint, 'mimic', None)
                        and np.isfinite(joint.min_angle)
                        and np.isfinite(joint.max_angle)
                        and (joint.max_angle - joint.min_angle)
                        < 2.0 * np.pi - 1e-3):
                    masters.append(joint)
            link = link.parent_link

    # Save every joint on the jaw->wrist paths, not only the actuated masters:
    # actuating a master also moves its mimic jaw joint, and gripper joints are
    # often excluded from ``robot.angle_vector()`` (joint_list), so an
    # angle_vector round-trip would not restore them. Saving them all keeps the
    # robot pose untouched for callers regardless of mimic coupling.
    saved = [(joint, joint.joint_angle()) for joint in path_joints]

    def _sep():
        # Tip-to-tip distance, used to tell the open pose from the closed one.
        # Measured between the fingertips, not the link origins: some grippers
        # rotate their fingers about a joint whose axis passes through the
        # finger link origin, so the origins never move apart and the origin
        # separation is constant - useless for finding the closed pose. The
        # fingertips do move, so they detect the closing.
        return np.linalg.norm(
            _fingertip(jaw_a, np.asarray(jaw_a.worldpos()))
            - _fingertip(jaw_b, np.asarray(jaw_b.worldpos())))

    try:
        # Find each driving joint's open vs closed extreme: the limit that
        # increases the jaw separation is "open". Done per joint so grippers
        # whose two fingers are driven by independent joints (no mimic) still
        # close symmetrically - not just by moving both the same way. (Grippers
        # with a single master + mimic joints just have one master here, and
        # the mimics follow it.)
        open_angle = {}
        closed_angle = {}
        for joint in masters:
            base = joint.joint_angle()
            joint.joint_angle(joint.min_angle)
            sep_min = _sep()
            joint.joint_angle(joint.max_angle)
            sep_max = _sep()
            joint.joint_angle(base)
            lo, hi = joint.min_angle, joint.max_angle
            open_angle[id(joint)], closed_angle[id(joint)] = \
                (hi, lo) if sep_max >= sep_min else (lo, hi)

        # Move to the closed grasp pose; everything below is read there.
        for joint in masters:
            joint.joint_angle(closed_angle[id(joint)])
        origin_a = np.asarray(jaw_a.worldpos(), dtype=np.float64)
        origin_b = np.asarray(jaw_b.worldpos(), dtype=np.float64)
        origin_mid = 0.5 * (origin_a + origin_b)
        wpos = np.asarray(wrist.worldpos(), dtype=np.float64)
        wrot = np.asarray(wrist.worldrot(), dtype=np.float64)

        # Closing axis: the kinematic relative velocity of the two fingers at
        # the grasp center (sign fixed later against the wrist frame). Falls
        # back, for grippers that cannot be actuated (only continuous joints),
        # to the static origin separation, then a jaw joint axis, then wrist y.
        closing = _kinematic_closing(
            jaw_a, jaw_b, masters, origin_mid, open_angle, closed_angle)
        if closing is None or np.linalg.norm(closing) < 1e-9:
            if np.linalg.norm(origin_a - origin_b) > 1e-9:
                closing = origin_a - origin_b
        if closing is None or np.linalg.norm(closing) < 1e-9:
            for jaw in (jaw_a, jaw_b):
                joint = getattr(jaw, 'joint', None)
                axis = getattr(joint, 'world_axis', None) if joint else None
                if axis is not None and np.linalg.norm(axis) > 1e-9:
                    closing = np.asarray(axis, dtype=np.float64)
                    break
        if closing is None or np.linalg.norm(closing) < 1e-9:
            closing = wrot[:, 1]
        closing = normalize_vector(closing)

        def _perp(v):
            v = np.asarray(v, dtype=np.float64)
            return v - np.dot(v, closing) * closing

        # Approach axis: forward, perpendicular to closing. Use the closed
        # origin midpoint -> wrist direction; if the wrist sits at that midpoint
        # (e.g. Fetch), fall back to the arm's incoming direction
        # (parent -> wrist), then a wrist axis. Origins (not fingertips) keep
        # the sign reliable - a fingertip can fall behind the wrist origin.
        approach = _perp(origin_mid - wpos)
        if np.linalg.norm(approach) < 1e-4:
            parent = getattr(wrist, 'parent_link', None)
            if parent is not None:
                approach = _perp(wpos - np.asarray(parent.worldpos()))
        if np.linalg.norm(approach) < 1e-4:
            for col in (2, 0, 1):
                cand = _perp(wrot[:, col])
                if np.linalg.norm(cand) > 1e-4:
                    approach = cand
                    break
        x_axis = normalize_vector(approach)

        # Grasp point. The lateral position (between the jaws) comes from the
        # jaw-origin midpoint, which is symmetric and stays on the gripper
        # centerline; the forward depth comes from the fingertips (the contact
        # pad reaching farthest along the approach). Taking the lateral part
        # from the fingertips instead would drift sideways whenever the two
        # finger meshes are not mirror-symmetric (e.g. JAXON at the closed
        # pose). So project the fingertip midpoint onto the approach ray through
        # the origin midpoint.
        tip_mid = 0.5 * (_fingertip_along(jaw_a, origin_a, x_axis)
                         + _fingertip_along(jaw_b, origin_b, x_axis))
        depth = np.dot(tip_mid - origin_mid, x_axis)
        grasp = origin_mid + max(0.0, depth) * x_axis

        y_axis = closing
        z_axis = normalize_vector(np.cross(x_axis, y_axis))
        y_axis = np.cross(z_axis, x_axis)
        rot_world = np.vstack([x_axis, y_axis, z_axis]).T
        rel_rot = wrot.T.dot(rot_world)
        # The closing-axis sign (which jaw is +y) is arbitrary and would flip
        # mirrored left/right grippers by 180deg. Hand-written end_coords align
        # +y with the wrist link's own frame, so snap the sign there: make the
        # gripper y-axis' dominant wrist-frame component positive (flip z too,
        # to stay right-handed).
        y_local = rel_rot[:, 1]
        dom = int(np.argmax(np.abs(y_local)))
        if y_local[dom] < 0:
            rel_rot[:, 1] = -rel_rot[:, 1]
            rel_rot[:, 2] = -rel_rot[:, 2]
        rel_pos = wrot.T.dot(grasp - wpos)
        rpy = rpy_angle(rel_rot)[0]
        return {
            'parent_link': wrist_link,
            'pos': [round(float(v), 6) for v in rel_pos],
            'rot': [round(float(v), 6) for v in rpy],
        }
    finally:
        for joint, angle in saved:
            joint.joint_angle(angle)


def generate_groups_from_llm(robot, llm_fn, config=None,
                             extra_instructions=None):
    """Produce limb groups via an LLM, in the geometry generator's format.

    The return value is the same 4-tuple as
    :func:`skrobot.urdf.robot_class_generator.generate_groups_from_geometry`,
    so it feeds directly into the same code emitter.

    Parameters
    ----------
    robot : RobotModel
        Loaded robot model.
    llm_fn : callable
        ``llm_fn(prompt: str) -> str``. Receives the grouping prompt, returns
        the model's text response.
    config : PatternConfig, optional
        Pattern configuration (used for deterministic end_coords detection).
    extra_instructions : str, optional
        Extra domain context appended to the prompt.

    Returns
    -------
    (groups, end_effectors, end_coords_info, robot_name)
    """
    if not callable(llm_fn):
        raise TypeError(
            "grouping='llm' requires llm_fn to be callable "
            '(prompt:str -> response:str). See skrobot.urdf.llm_grouping.')
    if config is None:
        config = get_default_config()

    prompt = build_grouping_prompt(robot, config, extra_instructions)
    response = llm_fn(prompt)
    if not isinstance(response, str):
        raise TypeError(
            'llm_fn must return a string, got {}'.format(type(response)))
    limbs = parse_grouping_response(response, robot)

    groups = {}
    end_effectors = {}
    end_coords_info = {}
    for limb in limbs:
        name = limb['name']
        links = limb['links']
        tip = limb['tip_link']
        spec = limb.get('end_coords')
        if spec and spec.get('kind') == 'jaw_gripper':
            # The jaw links are the gripper, not part of the arm chain that IK
            # controls; keep them out of this limb's joint group.
            jaws = set(spec['jaw_links'])
            stripped = [l for l in links if l not in jaws]
            if stripped:
                links = stripped
        groups[name] = {
            'links': links,
            'root_link': links[0],
            'tip_link': tip,
        }
        end_effectors[name] = tip
        if spec and spec.get('kind') == 'jaw_gripper':
            # Derive the grasp frame now (x=approach, y=closing, at the closed
            # endpoint) by actuating the gripper; the values are baked into the
            # generated CascadedCoords.
            end_coords_info[name] = compute_jaw_gripper_frame(
                robot, spec['wrist_link'], spec['jaw_links'])
        else:
            # Honor the tip the LLM chose: end_coords there with no offset.
            # The offset/rotation is left for the user to refine.
            end_coords_info[name] = {
                'parent_link': tip, 'pos': [0.0, 0.0, 0.0], 'rot': None}

    robot_name = getattr(robot, 'name', None) or 'robot'
    return groups, end_effectors, end_coords_info, robot_name
