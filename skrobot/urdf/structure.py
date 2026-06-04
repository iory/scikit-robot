"""URDF structural analysis: link trees and connectivity validation.

This module inspects the *structure* of a robot - the link/joint graph - rather
than its geometry. It provides three things, all in pure Python:

- :func:`print_urdf_tree` renders the full link hierarchy as an ASCII tree.
- :func:`validate_urdf_structure` checks connectivity (base/end links,
  connected components, cycles), link references, duplicate names, naming
  conventions and zero-velocity joints, returning a :class:`ValidationResult`.
- :func:`kinematic_tree` renders an LLM-friendly view that collapses fixed
  frames, annotates each edge with its joint type/axis, marks branch points and
  classifies leaf frames (tool / sensor / wheel / gripper).

Every function accepts the same ``source``: a path to a URDF file, a raw URDF
XML string/bytes, or an already loaded :class:`skrobot.model.RobotModel`.

Examples
--------
>>> from skrobot.urdf import print_urdf_tree, validate_urdf_structure
>>> print(print_urdf_tree('robot.urdf'))
>>> result = validate_urdf_structure('robot.urdf')
>>> print(result)
>>> from skrobot.models import PR2
>>> from skrobot.urdf import kinematic_tree
>>> print(kinematic_tree(PR2(), collapse_fixed=True))
"""

import os
import xml.etree.ElementTree as ET

import networkx as nx


MOVABLE_JOINT_TYPES = ('revolute', 'continuous', 'prismatic', 'planar',
                       'floating')


# ============================================================================
# Internal URDF structure representation
# ============================================================================

class JointInfo(object):
    """Minimal joint description used for structural analysis.

    Parameters
    ----------
    name : str
        Joint name.
    joint_type : str
        URDF joint type (``revolute``, ``continuous``, ``prismatic``,
        ``fixed``, ...).
    parent : str
        Parent link name.
    child : str
        Child link name.
    axis : tuple of float or None
        Joint axis (3,), or None when not applicable.
    velocity : float or None
        Velocity limit, when present.
    lower : float or None
        Lower position limit, when present.
    upper : float or None
        Upper position limit, when present.
    """

    def __init__(self, name, joint_type, parent, child,
                 axis=None, velocity=None, lower=None, upper=None):
        self.name = name
        self.joint_type = joint_type
        self.parent = parent
        self.child = child
        self.axis = axis
        self.velocity = velocity
        self.lower = lower
        self.upper = upper

    @property
    def is_movable(self):
        return self.joint_type in MOVABLE_JOINT_TYPES


class _UrdfStructure(object):
    """Link/joint graph extracted from a URDF or a RobotModel.

    Attributes
    ----------
    name : str or None
        Robot name.
    links : list of str
        Link names, in source order.
    joints : list of JointInfo
        Joints, in source order.
    child_to_parent : dict
        Maps a child link name to its parent link name.
    parent_to_children : dict
        Maps a parent link name to an ordered list of child link names.
    joint_to_child : dict
        Maps a child link name to the JointInfo that produced it.
    """

    def __init__(self, name, links, joints, link_objects=None,
                 joint_objects=None):
        self.name = name
        self.links = list(links)
        self.joints = list(joints)
        # Live skrobot objects, present only when built from a RobotModel.
        # They enable world-frame queries (link positions, joint world axes).
        self.link_objects = link_objects or {}
        self.joint_objects = joint_objects or {}

        self.child_to_parent = {}
        self.parent_to_children = {}
        self.joint_by_child = {}
        for joint in self.joints:
            self.child_to_parent[joint.child] = joint.parent
            self.parent_to_children.setdefault(joint.parent, []).append(
                joint.child)
            self.joint_by_child[joint.child] = joint

    @property
    def has_world(self):
        """Whether world-frame queries are available (built from a model)."""
        return bool(self.link_objects)

    def base_links(self):
        """Links that are never a child (graph roots)."""
        return [link for link in self.links
                if link not in self.child_to_parent]

    def end_links(self):
        """Links that are never a parent (graph leaves)."""
        return [link for link in self.links
                if link not in self.parent_to_children]


# ============================================================================
# Source loading
# ============================================================================

def _axis_tuple(axis):
    if axis is None:
        return None
    values = []
    for value in axis:
        value = float(value)
        rounded = round(value)
        values.append(rounded if abs(value - rounded) < 1e-9 else round(value, 6))
    return tuple(values)


def _structure_from_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    if root.tag != 'robot':
        raise ValueError(
            "Expected 'robot' root element, got '{}'".format(root.tag))

    name = root.attrib.get('name')
    links = []
    for link in root.findall('link'):
        link_name = link.attrib.get('name')
        if link_name is None:
            raise ValueError("Found link without 'name' attribute")
        links.append(link_name)

    joints = []
    for joint in root.findall('joint'):
        joint_name = joint.attrib.get('name')
        if joint_name is None:
            raise ValueError("Found joint without 'name' attribute")
        joint_type = joint.attrib.get('type', 'unknown')
        parent_elem = joint.find('parent')
        child_elem = joint.find('child')
        parent = parent_elem.attrib.get('link') if parent_elem is not None \
            else None
        child = child_elem.attrib.get('link') if child_elem is not None \
            else None

        axis = None
        axis_elem = joint.find('axis')
        if axis_elem is not None and axis_elem.attrib.get('xyz'):
            axis = _axis_tuple(axis_elem.attrib['xyz'].split())

        velocity = lower = upper = None
        limit_elem = joint.find('limit')
        if limit_elem is not None:
            if limit_elem.attrib.get('velocity') is not None:
                velocity = float(limit_elem.attrib['velocity'])
            if limit_elem.attrib.get('lower') is not None:
                lower = float(limit_elem.attrib['lower'])
            if limit_elem.attrib.get('upper') is not None:
                upper = float(limit_elem.attrib['upper'])

        joints.append(JointInfo(
            joint_name, joint_type, parent, child,
            axis=axis, velocity=velocity, lower=lower, upper=upper))

    return _UrdfStructure(name, links, joints)


def _joint_type_from_object(joint):
    """Infer the URDF joint type string from a skrobot joint object."""
    class_name = type(joint).__name__
    if class_name == 'FixedJoint':
        return 'fixed'
    if class_name == 'LinearJoint':
        return 'prismatic'
    if class_name == 'RotationalJoint':
        min_angle = getattr(joint, 'min_angle', None)
        max_angle = getattr(joint, 'max_angle', None)
        try:
            import numpy as np
            if min_angle is not None and max_angle is not None \
                    and (np.isinf(min_angle) or np.isinf(max_angle)):
                return 'continuous'
        except (TypeError, ValueError):
            pass
        return 'revolute'
    return 'unknown'


def _structure_from_robot_model(robot):
    joints = []
    for link in robot.link_list:
        parent = link.parent_link
        joint = link.joint
        if parent is None or joint is None:
            continue
        axis = getattr(joint, 'axis', None)
        joints.append(JointInfo(
            joint.name,
            _joint_type_from_object(joint),
            parent.name,
            link.name,
            axis=_axis_tuple(axis) if axis is not None else None,
            velocity=getattr(joint, 'max_joint_velocity', None),
            lower=getattr(joint, 'min_angle', None),
            upper=getattr(joint, 'max_angle', None)))
    links = [link.name for link in robot.link_list]
    link_objects = {link.name: link for link in robot.link_list}
    joint_objects = {link.name: link.joint for link in robot.link_list
                     if link.joint is not None}
    return _UrdfStructure(getattr(robot, 'name', None), links, joints,
                          link_objects=link_objects,
                          joint_objects=joint_objects)


def _load_robot_model(source):
    """Load ``source`` into a RobotModel (meshes skipped for speed)."""
    from skrobot.model import RobotModel
    from skrobot.utils.urdf import no_mesh_load_mode
    model = RobotModel()
    with no_mesh_load_mode():
        if isinstance(source, bytes):
            model.load_urdf(source.decode('utf-8'))
        elif isinstance(source, str) and source.lstrip().startswith('<'):
            model.load_urdf(source)
        else:
            model.load_urdf_file(source)
    return model


def _load_structure(source, with_objects=False):
    """Build a :class:`_UrdfStructure` from a path, XML string/bytes or model.

    When ``with_objects`` is True the result carries live skrobot Link/Joint
    objects so world-frame queries (positions, world axes) are possible; a
    path / XML source is loaded into a RobotModel to provide them.
    """
    # RobotModel (duck-typed to avoid importing the heavy model module).
    if hasattr(source, 'link_list') and hasattr(source, 'joint_list'):
        return _structure_from_robot_model(source)
    if with_objects:
        return _structure_from_robot_model(_load_robot_model(source))
    if isinstance(source, bytes):
        return _structure_from_xml(source)
    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith('<'):
            return _structure_from_xml(source.encode('utf-8'))
        if not os.path.exists(source):
            raise IOError('URDF file not found: {}'.format(source))
        with open(source, 'rb') as f:
            return _structure_from_xml(f.read())
    raise TypeError(
        'source must be a URDF path, XML string/bytes or a RobotModel, '
        'got {}'.format(type(source)))


# ============================================================================
# Tree rendering
# ============================================================================

def _render_subtree(structure, link_name, prefix, is_last, lines, visited,
                    world=False):
    branch = '└── ' if is_last else '├── '
    pos = _pos_suffix(structure, link_name, world)
    lines.append('{}{}{}{}'.format(prefix, branch, link_name, pos))
    if link_name in visited:
        return
    visited.add(link_name)

    children = structure.parent_to_children.get(link_name, [])
    child_prefix = prefix + ('    ' if is_last else '│   ')
    for i, child in enumerate(children):
        _render_subtree(structure, child, child_prefix,
                        i == len(children) - 1, lines, visited, world)


def print_urdf_tree(source, world=False):
    """Render the full URDF link hierarchy as an ASCII tree.

    Parameters
    ----------
    source : str or bytes or skrobot.model.RobotModel
        A URDF file path, raw URDF XML, or a loaded robot model.
    world : bool, optional
        When True, suffix each link with its world position ``@[x, y, z]`` at
        the initial pose. A path/XML source is loaded into a RobotModel
        (meshes skipped) to compute it. Default False.

    Returns
    -------
    str
        The tree as a multi-line string, headed by
        ``URDF Link Tree Structure:``.
    """
    structure = _load_structure(source, with_objects=world)
    base_links = structure.base_links()
    if not base_links:
        return 'No base links found - cannot build tree structure'

    lines = ['URDF Link Tree Structure:']
    visited = set()
    for base_link in base_links:
        _render_subtree(structure, base_link, '', True, lines, visited, world)
    return '\n'.join(lines)


# ============================================================================
# Validation
# ============================================================================

class ValidationResult(object):
    """Result of :func:`validate_urdf_structure`.

    Attributes
    ----------
    is_valid : bool
        True when no errors were found.
    errors : list of str
        Structural problems (disconnected links, cycles, bad references, ...).
    warnings : list of str
        Non-fatal issues (zero-velocity joints, naming, ...).
    summary : dict
        ``links_count``, ``joints_count``, ``base_links``, ``end_links``,
        ``connected_components``.
    """

    def __init__(self, is_valid, errors, warnings, summary):
        self.is_valid = is_valid
        self.errors = errors
        self.warnings = warnings
        self.summary = summary

    def format_summary(self):
        """Return the ``URDF Validation Summary`` text."""
        base = self.summary['base_links']
        end = self.summary['end_links']
        lines = [
            'URDF Validation Summary:',
            '  Links: {}'.format(self.summary['links_count']),
            '  Joints: {}'.format(self.summary['joints_count']),
            '  Base links: {} ({})'.format(
                len(base), ', '.join(base) if base else 'None'),
            '  End links: {} ({})'.format(
                len(end), ', '.join(end) if end else 'None'),
            '  Connected components: {}'.format(
                self.summary['connected_components']),
        ]
        return '\n'.join(lines)

    def __str__(self):
        lines = [self.format_summary()]
        if self.warnings:
            lines.append('Warnings:')
            for warning in self.warnings:
                lines.append('  - {}'.format(warning))
        if self.errors:
            lines.append('Validation Errors:')
            for error in self.errors:
                lines.append('  - {}'.format(error))
        else:
            lines.append('All validation checks passed!')
        return '\n'.join(lines)


def _is_valid_ros_name(name):
    if not name:
        return False
    if not name[0].isascii() or not name[0].isalpha():
        return False
    return all(c.isascii() and (c.isalnum() or c == '_') for c in name)


def _connected_components(structure):
    graph = nx.Graph()
    graph.add_nodes_from(structure.links)
    for joint in structure.joints:
        if joint.parent in graph and joint.child in graph:
            graph.add_edge(joint.parent, joint.child)
    return [sorted(component) for component in nx.connected_components(graph)]


def _detect_cycles(structure):
    graph = nx.DiGraph()
    graph.add_nodes_from(structure.links)
    for joint in structure.joints:
        if joint.parent in graph and joint.child in graph:
            graph.add_edge(joint.parent, joint.child)
    try:
        return list(nx.simple_cycles(graph))
    except nx.NetworkXError:
        return []


def validate_urdf_structure(source, verbose=False):
    """Validate the structural integrity of a URDF.

    Checks link reference integrity, duplicate names, ROS naming conventions,
    zero-velocity joints, connectivity (single connected component, exactly
    one base link) and cycle detection.

    Parameters
    ----------
    source : str or bytes or skrobot.model.RobotModel
        A URDF file path, raw URDF XML, or a loaded robot model.
    verbose : bool, optional
        When True, print the summary, warnings and errors to stdout.

    Returns
    -------
    ValidationResult
        Structured result with ``errors``, ``warnings`` and ``summary``.
    """
    structure = _load_structure(source)
    errors = []
    warnings = []

    link_set = set(structure.links)

    # Check 1: link references.
    for joint in structure.joints:
        if joint.parent is None or joint.child is None:
            errors.append(
                "Joint '{}' is missing a parent or child link".format(
                    joint.name))
            continue
        if joint.parent not in link_set:
            errors.append(
                "Joint '{}' references non-existent parent link '{}'".format(
                    joint.name, joint.parent))
        if joint.child not in link_set:
            errors.append(
                "Joint '{}' references non-existent child link '{}'".format(
                    joint.name, joint.child))

    # Check 2: duplicate names.
    seen = set()
    duplicates = set()
    for link in structure.links:
        if link in seen:
            duplicates.add(link)
        seen.add(link)
    for link in sorted(duplicates):
        errors.append("Duplicate link name '{}'".format(link))

    seen_joint = set()
    duplicate_joints = set()
    for joint in structure.joints:
        if joint.name in seen_joint:
            duplicate_joints.add(joint.name)
        seen_joint.add(joint.name)
    for joint_name in sorted(duplicate_joints):
        errors.append("Duplicate joint name '{}'".format(joint_name))

    # Check 3: naming conventions (warnings).
    invalid_links = [link for link in structure.links
                     if not _is_valid_ros_name(link)]
    if invalid_links:
        warnings.append(
            'Links violate ROS naming conventions: {}'.format(
                ', '.join(invalid_links)))
    invalid_joints = [joint.name for joint in structure.joints
                      if not _is_valid_ros_name(joint.name)]
    if invalid_joints:
        warnings.append(
            'Joints violate ROS naming conventions: {}'.format(
                ', '.join(invalid_joints)))

    # Check 4: zero-velocity joints (warnings).
    zero_velocity = [joint.name for joint in structure.joints
                     if joint.is_movable and joint.velocity == 0.0]
    if zero_velocity:
        warnings.append(
            'Movable joints with zero velocity limit (set a positive '
            'velocity): {}'.format(', '.join(sorted(zero_velocity))))

    # Check 5: connected components.
    components = _connected_components(structure)
    if len(components) > 1:
        message = ('Links are not all connected. Found {} disconnected '
                   'components:'.format(len(components)))
        for i, component in enumerate(components):
            message += '\n  Component {}: {}'.format(
                i + 1, ', '.join(component))
        errors.append(message)

    # Check 6: cycles.
    cycles = _detect_cycles(structure)
    for cycle in cycles:
        errors.append('Detected cycle in link graph: {}'.format(
            ' -> '.join(cycle)))

    # Check 7: base links.
    base_links = structure.base_links()
    if not base_links:
        errors.append(
            'No base link found (all links have parents - this creates a '
            'cycle)')
    elif len(base_links) > 1:
        errors.append(
            'Multiple base links found: {}. A valid URDF should have exactly '
            'one base link.'.format(
                ', '.join("'{}'".format(link) for link in base_links)))

    # Check 8: end links (warning only).
    end_links = structure.end_links()
    if not end_links:
        warnings.append('No end links found (all links have children)')

    summary = {
        'links_count': len(structure.links),
        'joints_count': len(structure.joints),
        'base_links': base_links,
        'end_links': end_links,
        'connected_components': len(components),
    }

    result = ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        summary=summary)

    if verbose:
        print(result)

    return result


# ============================================================================
# Kinematic (LLM-friendly) tree
# ============================================================================

# Leaf classification patterns (substring match, lowercase).
_LEAF_PATTERNS = (
    ('tool', ('tool', 'tcp', 'grasp', 'eef', 'end_effector')),
    ('sensor', ('optical_frame', 'camera', 'stereo', 'laser', 'imu',
                'cam_', 'accelerometer', 'led', 'sensor', 'depth', 'rgb',
                'lidar', 'sonar', 'gps')),
    ('wheel', ('wheel', 'caster', 'castor')),
    ('gripper', ('gripper', 'finger', 'palm', 'thumb', 'hand', 'knuckle')),
)


def _classify_leaf(link_name):
    low = link_name.lower()
    for label, patterns in _LEAF_PATTERNS:
        if any(p in low for p in patterns):
            return label
    return None


def _subtree_has_movable(structure, link_name, cache):
    """Whether the subtree rooted at ``link_name`` contains any movable joint."""
    if link_name in cache:
        return cache[link_name]
    cache[link_name] = False  # guard against cycles
    result = False
    for child in structure.parent_to_children.get(link_name, []):
        joint = structure.joint_by_child.get(child)
        if joint is not None and joint.is_movable:
            result = True
        elif _subtree_has_movable(structure, child, cache):
            result = True
    cache[link_name] = result
    return result


def _all_descendants(structure, link_name):
    result = []
    stack = list(structure.parent_to_children.get(link_name, []))
    while stack:
        node = stack.pop()
        result.append(node)
        stack.extend(structure.parent_to_children.get(node, []))
    return result


def _fmt_vec(vec):
    return '[{}]'.format(
        ', '.join('{:g}'.format(round(float(v), 3)) for v in vec))


def _edge_annotation(structure, link_name, world=False):
    joint = structure.joint_by_child.get(link_name)
    if joint is None:
        return ''
    label = joint.joint_type
    if joint.is_movable:
        axis = None
        if world and structure.has_world:
            jobj = structure.joint_objects.get(link_name)
            if jobj is not None and hasattr(jobj, 'world_axis'):
                axis = jobj.world_axis
        if axis is None:
            axis = joint.axis
        if axis is not None:
            return '[{} {}] '.format(label, _fmt_vec(axis))
    return '[{}] '.format(label)


def _pos_suffix(structure, link_name, world):
    """World-position suffix ``@[x, y, z]`` for a link, when world is on."""
    if not (world and structure.has_world):
        return ''
    link = structure.link_objects.get(link_name)
    if link is None:
        return ''
    return ' @{}'.format(_fmt_vec(link.worldpos()))


def _movable_children(structure, link_name, movable_cache):
    """Children whose subtree (or own joint) is movement-relevant."""
    relevant = []
    fixed_only = []
    for child in structure.parent_to_children.get(link_name, []):
        joint = structure.joint_by_child.get(child)
        own_movable = joint is not None and joint.is_movable
        if own_movable or _subtree_has_movable(structure, child, movable_cache):
            relevant.append(child)
        else:
            fixed_only.append(child)
    return relevant, fixed_only


def _render_kinematic(structure, link_name, prefix, is_last, lines,
                      collapse_fixed, annotate, classify_leaves,
                      movable_cache, visited, world=False):
    branch = '└── ' if is_last else '├── '
    annotation = _edge_annotation(structure, link_name, world) if annotate \
        else ''

    movable_kids = structure.parent_to_children.get(link_name, [])
    n_movable = sum(
        1 for child in movable_kids
        if (structure.joint_by_child.get(child) is not None
            and structure.joint_by_child[child].is_movable))
    tags = []
    if n_movable >= 2:
        tags.append('◆BRANCH({})'.format(n_movable))
    if classify_leaves and link_name not in structure.parent_to_children:
        leaf_type = _classify_leaf(link_name)
        if leaf_type is not None:
            tags.append(leaf_type)
    tag_str = '  ' + ' '.join(tags) if tags else ''
    pos = _pos_suffix(structure, link_name, world)

    lines.append('{}{}{}{}{}{}'.format(prefix, branch, annotation,
                                       link_name, pos, tag_str))

    if link_name in visited:
        return
    visited.add(link_name)

    child_prefix = prefix + ('    ' if is_last else '│   ')
    if collapse_fixed:
        relevant, fixed_only = _movable_children(
            structure, link_name, movable_cache)
    else:
        relevant = structure.parent_to_children.get(link_name, [])
        fixed_only = []

    collapsed = []
    for child in fixed_only:
        collapsed.append(child)
        collapsed.extend(_all_descendants(structure, child))

    n_items = len(relevant) + (1 if collapsed else 0)
    for i, child in enumerate(relevant):
        last = (i == n_items - 1)
        _render_kinematic(structure, child, child_prefix, last, lines,
                          collapse_fixed, annotate, classify_leaves,
                          movable_cache, visited, world)
    if collapsed:
        names = ', '.join(collapsed[:6])
        if len(collapsed) > 6:
            names += ', ...'
        lines.append('{}└── (+{} fixed frame{}: {})'.format(
            child_prefix, len(collapsed),
            's' if len(collapsed) != 1 else '', names))


def kinematic_tree(source, collapse_fixed=True, annotate=True,
                   classify_leaves=True, world=False):
    """Render an LLM-friendly kinematic tree of a URDF.

    Compared with :func:`print_urdf_tree`, this view is tuned for reasoning
    about how to build a robot model:

    - fixed-only subtrees are collapsed into a single ``(+N fixed frames: ...)``
      line so the actuated skeleton stands out (``collapse_fixed``),
    - each link is annotated with the joint type and axis that connects it to
      its parent, e.g. ``[revolute [0, 0, 1]]`` (``annotate``),
    - links with two or more movable children are marked ``◆BRANCH(n)`` - these
      are the natural limb boundaries,
    - leaf frames are tagged ``tool`` / ``sensor`` / ``wheel`` / ``gripper``
      (``classify_leaves``).

    Parameters
    ----------
    source : str or bytes or skrobot.model.RobotModel
        A URDF file path, raw URDF XML, or a loaded robot model.
    collapse_fixed : bool, optional
        Collapse subtrees that contain no movable joints. Default True.
    annotate : bool, optional
        Annotate each edge with its joint type and axis. Default True.
    classify_leaves : bool, optional
        Tag leaf frames by inferred role. Default True.
    world : bool, optional
        When True, joint axes are shown in the *world* frame at the initial
        pose (via ``joint.world_axis``) instead of the URDF-local frame, and
        each link is suffixed with its world position ``@[x, y, z]``. This is
        usually more interpretable (which way a joint really turns, where a
        link sits). A path/XML source is loaded into a RobotModel (meshes
        skipped) to compute the forward kinematics. Default False.

    Returns
    -------
    str
        The kinematic tree as a multi-line string.
    """
    structure = _load_structure(source, with_objects=world)
    base_links = structure.base_links()
    if not base_links:
        return 'No base links found - cannot build tree structure'

    if world:
        header = ('Kinematic Tree (movable skeleton; world axes & '
                  'positions @ at init pose):')
    else:
        header = 'Kinematic Tree (movable skeleton):'
    lines = [header]
    movable_cache = {}
    visited = set()
    for base_link in base_links:
        _render_kinematic(structure, base_link, '', True, lines,
                          collapse_fixed, annotate, classify_leaves,
                          movable_cache, visited, world)
    return '\n'.join(lines)
