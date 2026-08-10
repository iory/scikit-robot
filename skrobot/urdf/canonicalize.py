"""Rewrite a URDF in a canonical, diff-friendly form.

Two URDFs that describe the same robot rarely diff cleanly: exporters order
links and joints differently, shuffle attributes, and format numbers with
different precision (including ``-0`` artifacts).  ``canonicalize_urdf``
rewrites a URDF so that semantically equal documents become textually equal:
links and joints are sorted by name, child elements and attributes follow a
fixed order, and every number is formatted with one rule (``%.10g``, values
below ``1e-12`` snapped to exact ``0``).  The output is still a valid URDF,
so ``diff -u`` / ``git diff --no-index`` between two canonicalized files
shows only meaningful differences.

XML comments and processing instructions are dropped from the canonical
form.
"""

import xml.etree.ElementTree as ET


__all__ = [
    'canonicalize_urdf',
    'canonicalize_urdf_file',
]

# Fixed child-element order inside <joint> and <link>; unknown tags follow,
# sorted by tag name, so nothing is silently dropped.
_JOINT_CHILD_ORDER = ['origin', 'parent', 'child', 'axis', 'limit', 'mimic',
                      'dynamics', 'calibration', 'safety_controller']
_LINK_CHILD_ORDER = ['inertial', 'visual', 'collision']

# Elements removed by ``kinematics_only`` (top level and inside <link>).
_NON_KINEMATIC_TAGS = frozenset(['visual', 'collision', 'inertial',
                                 'material', 'transmission', 'gazebo'])

# Attributes holding a number or a whitespace-separated number list.
_NUMERIC_ATTRS = frozenset([
    'xyz', 'rpy', 'lower', 'upper', 'effort', 'velocity', 'value',
    'ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz', 'multiplier', 'offset',
    'scale', 'radius', 'length', 'size', 'rgba',
    'soft_lower_limit', 'soft_upper_limit', 'k_position', 'k_velocity',
    'damping', 'friction', 'rising', 'falling',
])

_ZERO_SNAP = 1e-12


def _format_number_token(token):
    """Format one attribute token; non-numeric tokens pass through.

    Parameters
    ----------
    token : str
        One whitespace-separated token from an attribute value.

    Returns
    -------
    str
        ``%.10g`` rendering with ``|value| < 1e-12`` snapped to ``0`` (which
        also removes ``-0``), or the original token when it is not a number.
    """
    try:
        value = float(token)
    except ValueError:
        return token
    if abs(value) < _ZERO_SNAP:
        value = 0.0
    return '%.10g' % value


def _canonical_attribute(key, value):
    if key in _NUMERIC_ATTRS:
        return ' '.join(_format_number_token(t) for t in value.split())
    return value


def _child_sort_key(order):
    def key(elem):
        if elem.tag in order:
            return (0, order.index(elem.tag), elem.get('name', ''))
        return (1, 0, elem.tag + '\x00' + elem.get('name', ''))
    return key


def _rebuild(elem, child_order):
    """Copy ``elem`` with canonical attributes and ordered children."""
    out = ET.Element(elem.tag)
    for key in sorted(elem.keys(), key=lambda a: (a != 'name', a)):
        out.set(key, _canonical_attribute(key, elem.get(key)))
    for child in sorted(list(elem), key=_child_sort_key(child_order)):
        out.append(_rebuild(child, []))
    if elem.text is not None and elem.text.strip():
        out.text = elem.text.strip()
    return out


def _indent(elem, level=0):
    pad = '\n' + '  ' * (level + 1)
    if len(elem):
        elem.text = pad
        for child in elem:
            _indent(child, level + 1)
            child.tail = pad
        elem[-1].tail = '\n' + '  ' * level
    return elem


def canonicalize_urdf(urdf_xml, kinematics_only=False):
    """Return ``urdf_xml`` rewritten in canonical, diff-friendly form.

    Links and joints are sorted by ``name``, child elements and attributes
    follow a fixed order, and all numbers share one format (``%.10g`` with
    ``|value| < 1e-12`` snapped to exact ``0``).  The result is a valid URDF
    and the transform is idempotent.

    Parameters
    ----------
    urdf_xml : str
        URDF document text.
    kinematics_only : bool, optional
        When True, drop ``<visual>``, ``<collision>``, ``<inertial>``,
        ``<material>``, ``<transmission>`` and ``<gazebo>`` elements so only
        the kinematic skeleton (links, joints, origins, axes, limits, mimic)
        remains.  Default is False.

    Returns
    -------
    str
        The canonicalized URDF text.

    Raises
    ------
    ValueError
        If ``urdf_xml`` is not parseable XML or its root element is not
        ``<robot>``.

    Examples
    --------
    >>> from skrobot.urdf import canonicalize_urdf
    >>> a = canonicalize_urdf(open('a.urdf').read())   # doctest: +SKIP
    >>> b = canonicalize_urdf(open('b.urdf').read())   # doctest: +SKIP
    >>> a == b                                         # doctest: +SKIP
    True
    """
    try:
        root = ET.fromstring(urdf_xml)
    except ET.ParseError as e:
        raise ValueError('invalid URDF XML: {}'.format(e))
    if root.tag != 'robot':
        raise ValueError(
            "root element is <{}>, expected <robot>".format(root.tag))

    out = ET.Element('robot')
    if root.get('name') is not None:
        out.set('name', root.get('name'))

    links = []
    joints = []
    others = []
    for elem in root:
        if elem.tag == 'link':
            links.append(elem)
        elif elem.tag == 'joint':
            joints.append(elem)
        else:
            others.append(elem)

    for elem in sorted(others, key=lambda e: (e.tag, e.get('name', ''))):
        if kinematics_only and elem.tag in _NON_KINEMATIC_TAGS:
            continue
        out.append(_rebuild(elem, []))
    for elem in sorted(links, key=lambda e: e.get('name', '')):
        rebuilt = _rebuild(elem, _LINK_CHILD_ORDER)
        if kinematics_only:
            for tag in _NON_KINEMATIC_TAGS:
                for victim in rebuilt.findall(tag):
                    rebuilt.remove(victim)
        out.append(rebuilt)
    for elem in sorted(joints, key=lambda e: e.get('name', '')):
        out.append(_rebuild(elem, _JOINT_CHILD_ORDER))

    return ET.tostring(_indent(out), encoding='unicode') + '\n'


def canonicalize_urdf_file(urdf_path, output_path=None,
                           kinematics_only=False):
    """Canonicalize the URDF at ``urdf_path``.

    Parameters
    ----------
    urdf_path : str
        Path of the URDF file to read.
    output_path : str or None, optional
        When given, the canonical text is also written there (UTF-8, LF
        newlines).  Writing to ``urdf_path`` itself is allowed.
    kinematics_only : bool, optional
        Forwarded to :func:`canonicalize_urdf`.  Default is False.

    Returns
    -------
    str
        The canonicalized URDF text.
    """
    with open(urdf_path, encoding='utf-8') as f:
        text = canonicalize_urdf(f.read(), kinematics_only=kinematics_only)
    if output_path is not None:
        with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    return text
