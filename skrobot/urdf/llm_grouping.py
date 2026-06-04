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

Respond with ONLY a JSON object, no prose, of the form:
{"limbs": [{"name": "thumb", "links": ["...", "..."], "tip_link": "..."}, ...]}'''

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
        seen_names.add(name)
        validated.append({'name': name, 'links': list(links),
                          'tip_link': tip})
    return validated


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
        groups[name] = {
            'links': links,
            'root_link': links[0],
            'tip_link': tip,
        }
        end_effectors[name] = tip
        # Honor the tip the LLM chose: attach end_coords there with no offset.
        # The geometry tool-frame search is intentionally not used here. The
        # offset/rotation is left for the user to refine.
        end_coords_info[name] = {
            'parent_link': tip, 'pos': [0.0, 0.0, 0.0], 'rot': None}

    robot_name = getattr(robot, 'name', None) or 'robot'
    return groups, end_effectors, end_coords_info, robot_name
