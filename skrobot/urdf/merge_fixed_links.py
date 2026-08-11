"""Collapse a URDF's fixed joints so each rigid body is one link.

CAD exporters emit one link per part, joined by fixed joints, so a robot that
moves in six degrees of freedom can arrive as a hundred links.  Simulators pay
for every one of them and the kinematic tree is unreadable.

:func:`merge_fixed_links` folds each fixed-joint child into its parent: the
child's visual and collision geometry moves over with its origin pre-composed
by the joint transform, the two inertials are combined about the merged centre
of mass, joints hanging off the child are re-parented, and the joint and link
are removed.  It iterates to a fixpoint, so a chain of fixed joints collapses
onto the nearest movable (or root) link.

Links carrying no geometry are treated as coordinate frames and kept by
default: they are usually TF frames -- tool centre points, sensor mounts,
connector markers -- that something downstream looks up by name.
"""

from logging import getLogger
import xml.etree.ElementTree as ET

from skrobot.coordinates.math import matrix2xyzrpy
from skrobot.utils.inertia import combine_inertials
from skrobot.utils.inertia import transform_inertial
from skrobot.utils.urdf import parse_origin


logger = getLogger(__name__)


__all__ = [
    'merge_fixed_links',
    'merge_fixed_links_file',
]

_INERTIA_KEYS = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')


def _fmt(value):
    """Format a number for a URDF attribute, folding ``-0.0`` to ``0``."""
    value = float(value)
    return '0' if value == 0.0 else '{:.10g}'.format(value)


def _set_origin(element, matrix):
    """Write ``element``'s ``<origin>`` child from a 4x4 matrix."""
    xyz, rpy = matrix2xyzrpy(matrix)
    origin = element.find('origin')
    if origin is None:
        origin = ET.SubElement(element, 'origin')
    origin.set('xyz', ' '.join(_fmt(v) for v in xyz))
    origin.set('rpy', ' '.join(_fmt(v) for v in rpy))


def _has_geometry(link):
    """Whether a link carries geometry, as opposed to being a bare frame."""
    return link.find('visual') is not None or link.find('collision') is not None


def _vector(element, attribute):
    """A 3-vector attribute, defaulting to zeros when absent or empty."""
    raw = (element.get(attribute) if element is not None else None) or '0 0 0'
    return [float(v) for v in raw.split()]


def _read_inertial(link):
    """A link's ``<inertial>`` as an inertial dict in the LINK frame.

    None when the block is missing or its mass is not positive.  The inertial
    ``<origin>`` is applied here, so the tensor comes back in link axes about
    the centre of mass -- what :func:`~skrobot.utils.inertia.combine_inertials`
    expects.
    """
    inertial = link.find('inertial')
    if inertial is None:
        return None
    tensor = inertial.find('inertia')
    if tensor is None:
        return None
    mass_element = inertial.find('mass')
    mass = float(mass_element.get('value', '0')) \
        if mass_element is not None else 0.0
    components = tuple(float(tensor.get(k, '0')) for k in _INERTIA_KEYS)
    origin = inertial.find('origin')
    return transform_inertial(mass, [0.0, 0.0, 0.0], components,
                              _vector(origin, 'xyz'), _vector(origin, 'rpy'))


def _write_inertial(link, info):
    """Replace ``link``'s ``<inertial>`` with ``info``.

    Written with ``rpy="0 0 0"``: the tensor is already in link axes about
    ``com``, so the origin only has to carry the centre of mass.
    """
    old = link.find('inertial')
    if old is not None:
        link.remove(old)
    inertial = ET.SubElement(link, 'inertial')
    origin = ET.SubElement(inertial, 'origin')
    origin.set('xyz', ' '.join(_fmt(v) for v in info['com']))
    origin.set('rpy', '0 0 0')
    ET.SubElement(inertial, 'mass').set('value', _fmt(info['mass']))
    tensor = ET.SubElement(inertial, 'inertia')
    for key, value in zip(_INERTIA_KEYS, info['inertia']):
        tensor.set(key, _fmt(value))


def _merge_inertials(parent, child, transform):
    """Lump the child's inertial into the parent's.

    ``transform`` is the 4x4 parent-from-child transform.  No-op when the child
    has no usable inertial; seeds the parent when the parent had none.
    """
    info = _read_inertial(child)
    if info is None:
        return
    xyz, rpy = matrix2xyzrpy(transform)
    info = transform_inertial(info['mass'], info['com'], info['inertia'],
                              xyz, rpy)
    combined = combine_inertials(_read_inertial(parent), info)
    if combined is not None:
        _write_inertial(parent, combined)


def merge_fixed_links(root, keep_frames=True, force_merge=None, only=None):
    """Collapse fixed joints so each rigid body is a single link, in place.

    Iterates to a fixpoint, so a chain of fixed joints collapses onto the
    nearest movable or root link.  Movable joints, and the frames they are
    expressed in, are preserved exactly: every merge pre-composes the fixed
    transform into the origins it moves.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        The ``<robot>`` element, modified in place.
    keep_frames : bool
        Keep links that carry no geometry of their own, which are usually TF
        frames something downstream looks up by name.  They are then neither
        merged away nor used as a merge target -- a frame that absorbed
        geometry could be lumped onward itself, silently dropping the frame.
        Set False to collapse them too.
    force_merge : iterable of str or None
        Child links to merge even though they carry no geometry, so that a
        mass-only link's ``<inertial>`` still reaches its parent.  Such a child
        may fold into a frame parent even under ``keep_frames``, since it adds
        only mass, never geometry, and the parent stays a frame.
    only : iterable of str or None
        Restrict merging to these child links, leaving every other fixed child
        in place.  Implies ``force_merge`` for the same names when
        ``force_merge`` is not given, so a mass-only link folds without also
        collapsing the rest of the fixed structure.

    Returns
    -------
    merged : int
        How many links were folded away.

    Examples
    --------
    >>> import xml.etree.ElementTree as ET
    >>> from skrobot.urdf import merge_fixed_links
    >>> root = ET.fromstring('''<robot name="r">
    ...   <link name="base">
    ...     <visual><geometry><box size="1 1 1"/></geometry></visual>
    ...   </link>
    ...   <link name="bolt">
    ...     <visual><geometry><box size="0.1 0.1 0.1"/></geometry></visual>
    ...   </link>
    ...   <joint name="j" type="fixed">
    ...     <origin xyz="1 0 0"/>
    ...     <parent link="base"/><child link="bolt"/>
    ...   </joint>
    ... </robot>''')
    >>> merge_fixed_links(root)
    1
    >>> [link.get('name') for link in root.findall('link')]
    ['base']
    >>> root.find('link').findall('visual')[1].find('origin').get('xyz')
    '1 0 0'
    """
    if only is not None and not force_merge:
        # naming a child in `only` is the request to fold it; an explicitly
        # empty force_merge should not quietly cancel that
        force_merge = only
    force_merge = set(force_merge or ())
    only = None if only is None else set(only)

    # a link reached by more than one joint cannot be folded: merging it into
    # one parent would leave the other joint pointing at a link that is gone
    incoming = {}
    for joint in root.findall('joint'):
        child_ref = joint.find('child')
        if child_ref is not None:
            name = child_ref.get('link')
            incoming[name] = incoming.get(name, 0) + 1
    multi_parent = {name for name, count in incoming.items() if count > 1}
    # snapshot before merging: a link that receives geometry mid-run must not
    # then count as a frame, nor stop being one
    frames = set()
    if keep_frames:
        frames = {link.get('name') for link in root.findall('link')
                  if not _has_geometry(link)} - force_merge

    merged = 0
    while True:
        links = {link.get('name'): link for link in root.findall('link')}
        joints = root.findall('joint')
        target = None
        for joint in joints:
            if joint.get('type') != 'fixed':
                continue
            parent_ref, child_ref = joint.find('parent'), joint.find('child')
            if parent_ref is None or child_ref is None:
                continue
            parent_name = parent_ref.get('link')
            child_name = child_ref.get('link')
            if parent_name == child_name:
                # a self-loop is not a joint between two bodies; folding it
                # would delete the one link it names
                logger.warning('joint %s has %s as both parent and child; '
                               'skipping it', joint.get('name'), child_name)
                continue
            if child_name in multi_parent:
                logger.warning('link %s is the child of more than one joint; '
                               'skipping it', child_name)
                continue
            if only is not None and child_name not in only:
                continue
            parent = links.get(parent_name)
            child = links.get(child_name)
            if parent is None or child is None:
                continue
            # geometry is what makes a child mergeable while frames are being
            # kept, unless it was named in force_merge to carry its mass over
            if keep_frames and not (_has_geometry(child)
                                    or child_name in force_merge):
                continue
            if child_name in frames:
                continue
            if parent_name in frames and not (child_name in force_merge
                                              and not _has_geometry(child)):
                continue
            target = (joint, parent, child, parent_name, child_name)
            break
        if target is None:
            return merged
        joint, parent, child, parent_name, child_name = target
        transform = parse_origin(joint)

        for geometry in list(child):
            if geometry.tag not in ('visual', 'collision'):
                continue
            _set_origin(geometry, transform.dot(parse_origin(geometry)))
            child.remove(geometry)
            parent.append(geometry)

        _merge_inertials(parent, child, transform)

        for other in joints:
            other_parent = other.find('parent')
            if other_parent is not None \
                    and other_parent.get('link') == child_name:
                other_parent.set('link', parent_name)
                _set_origin(other, transform.dot(parse_origin(other)))

        root.remove(joint)
        root.remove(child)
        merged += 1


def merge_fixed_links_file(input_path, output_path=None, keep_frames=True,
                           force_merge=None, only=None):
    """Apply :func:`merge_fixed_links` to a URDF file.

    Comments survive the round trip, so per-link provenance written by an
    exporter is not lost.

    Parameters
    ----------
    input_path : str
        URDF to read.
    output_path : str or None
        Where to write.  Defaults to ``input_path``, editing it in place.
    keep_frames, force_merge, only
        Forwarded to :func:`merge_fixed_links`.

    Returns
    -------
    merged : int
        How many links were folded away.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    with open(input_path, encoding='utf-8') as f:
        root = ET.fromstring(f.read(), parser=parser)
    merged = merge_fixed_links(root, keep_frames=keep_frames,
                               force_merge=force_merge, only=only)
    text = ET.tostring(root, encoding='unicode')
    if not text.startswith('<?xml'):
        text = '<?xml version="1.0"?>\n' + text
    if not text.endswith('\n'):
        text += '\n'
    with open(output_path or input_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return merged
