"""Sanitize URDF link / joint names.

Downstream tools (xacro, ROS, MoveIt) choke on names carrying hyphens,
spaces, dots or other punctuation; these helpers rewrite identifiers to the
safe ``[A-Za-z0-9_]`` alphabet while keeping every cross reference in the
document consistent.
"""

import re


__all__ = [
    'sanitize_name',
    'sanitize_urdf_names',
]


def sanitize_name(raw):
    """Sanitize one URDF identifier (link / joint name).

    Keeps ``[A-Za-z0-9_]``, collapses every other run of characters to a
    single ``_``, strips leading/trailing underscores and never lets the
    result start with a digit (or be empty) by prefixing ``c_``.  Idempotent
    on names that are already valid.

    Parameters
    ----------
    raw : str
        The identifier to sanitize.

    Returns
    -------
    name : str
        A valid URDF identifier.

    Examples
    --------
    >>> from skrobot.urdf.sanitize import sanitize_name
    >>> sanitize_name('base link (rev.2)')
    'base_link_rev_2'
    >>> sanitize_name('42_arm')
    'c_42_arm'
    """
    name = re.sub(r'[^0-9A-Za-z_]+', '_', raw)
    name = name.strip('_')
    if not name or name[0].isdigit():
        name = 'c_' + name
    return name


def sanitize_urdf_names(root):
    """Sanitize every link / joint name in a parsed URDF, in place.

    Rewrites all ``<link>`` and ``<joint>`` names -- and their cross
    references (joint ``<parent>``/``<child>`` links, ``<mimic>`` joints and
    ``<transmission>`` joint entries) -- through :func:`sanitize_name`.  When
    two distinct dirty names collapse to the same clean one, a numeric suffix
    keeps them unique.

    A no-op when the names are already clean, so it can be used as a safety
    net over hand-edited or externally imported URDFs.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element or lxml.etree._Element
        Root ``<robot>`` element; modified in place.
    """
    def _remap(elems, attr):
        mapping = {}
        used = set()
        for el in elems:
            orig = el.get(attr)
            if orig is None or orig in mapping:
                continue
            cand = sanitize_name(orig)
            i = 1
            while cand in used:      # two dirty names -> the same clean one
                i += 1
                cand = '{}_{}'.format(sanitize_name(orig), i)
            mapping[orig] = cand
            used.add(cand)
        return mapping

    links = root.findall('link')
    joints = root.findall('joint')
    link_map = _remap(links, 'name')
    joint_map = _remap(joints, 'name')
    for link in links:
        if link.get('name') in link_map:
            link.set('name', link_map[link.get('name')])
    for joint in joints:
        if joint.get('name') in joint_map:
            joint.set('name', joint_map[joint.get('name')])
        for tag in ('parent', 'child'):
            el = joint.find(tag)
            if el is not None and el.get('link') in link_map:
                el.set('link', link_map[el.get('link')])
        mimic = joint.find('mimic')
        if mimic is not None and mimic.get('joint') in joint_map:
            mimic.set('joint', joint_map[mimic.get('joint')])
    for transmission in root.findall('transmission'):
        for el in transmission.findall('joint'):
            if el.get('name') in joint_map:
                el.set('name', joint_map[el.get('name')])
