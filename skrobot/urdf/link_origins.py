"""Re-centre link frames onto the geometry they carry.

CAD exporters routinely place a link's frame far away from its parts: a
sub-assembly inherits the CAD origin of the whole machine, so its frame can
float a metre from anything visible.  The model still renders correctly --
every ``<visual>`` carries the compensating ``<origin>`` -- but the TF frame
is useless to look at in RViz and the numbers in the file are hard to read.

:func:`normalize_link_origins` moves such a frame onto its geometry.  For a
shift ``d`` (the geometry centroid expressed in the link frame) the parent
joint's origin becomes ``F @ T(+d)``, which slides the link frame onto the
geometry, and everything expressed in that frame -- child joints,
``<inertial>``, ``<visual>``, ``<collision>`` -- has its origin turned into
``T(-d) @ F``.  The two cancel, so the link frame is the only thing that
moves: every piece of geometry, every mass and every other link stays
exactly where it was, in every joint configuration.

Only links whose parent joint is ``fixed`` are re-centred, and the root link
is never touched.  See :func:`normalize_link_origins` for why.
"""

from logging import getLogger
import os
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates.math import rpy2matrix


logger = getLogger(__name__)


__all__ = [
    'geometry_centroid',
    'normalize_link_origins',
    'normalize_link_origins_file',
]


def _origin_translation(elem):
    """Translation of ``elem``'s ``<origin>`` child; zeros when absent."""
    origin = elem.find('origin')
    if origin is None or 'xyz' not in origin.attrib:
        return np.zeros(3)
    return np.array([float(v) for v in origin.get('xyz').split()],
                    dtype=np.float64)


def _origin_rotation(elem):
    """Rotation of ``elem``'s ``<origin>`` child; identity when absent."""
    origin = elem.find('origin')
    if origin is None or 'rpy' not in origin.attrib:
        return np.eye(3)
    roll, pitch, yaw = [float(v) for v in origin.get('rpy').split()]
    return rpy2matrix(roll, pitch, yaw)


def _format_xyz(xyz):
    """Format a translation for an ``xyz`` attribute, round-trip exact."""
    tokens = []
    for value in xyz:
        value = float(value)
        if value == 0.0:      # also turns -0.0 into 0
            value = 0.0
        tokens.append(repr(value))
    return ' '.join(tokens)


def _set_origin_translation(elem, xyz):
    """Write ``elem``'s ``<origin>`` translation, in place.

    Only the ``xyz`` attribute is rewritten: every transform applied here is
    a pure translation, so the ``rpy`` attribute keeps its exact original
    text and contributes no round-trip error.
    """
    origin = elem.find('origin')
    if origin is None:
        origin = ET.SubElement(elem, 'origin')
        origin.set('rpy', '0 0 0')
    origin.set('xyz', _format_xyz(xyz))


def geometry_centroid(geometry, base_path):
    """Centroid of one URDF ``<geometry>`` element, in the geometry's frame.

    This is the default measurement used by :func:`normalize_link_origins`.
    A ``<mesh>`` is loaded through :func:`skrobot.utils.urdf.load_meshes`
    (``package://`` URIs are resolved relative to ``base_path``) and its
    area-weighted centroid is returned, scaled by the mesh ``scale``
    attribute.  ``<box>``, ``<sphere>`` and ``<cylinder>`` are centred on
    their own frame by definition, so they measure as zero.

    Parameters
    ----------
    geometry : xml.etree.ElementTree.Element
        A ``<geometry>`` element.
    base_path : str
        Directory the URDF lives in, used to resolve relative and
        ``package://`` mesh paths.

    Returns
    -------
    centroid : numpy.ndarray or None
        ``(3,)`` centroid in the geometry's own frame, or None when the
        geometry holds no shape this function understands or its mesh
        cannot be loaded.
    """
    # Imported here so callers that supply their own ``centroid_fn`` -- and
    # the rest of this module, which is pure XML -- never pull in the mesh
    # loading stack.
    from skrobot.utils.urdf import get_filename
    from skrobot.utils.urdf import load_meshes

    for tag in ('box', 'sphere', 'cylinder'):
        if geometry.find(tag) is not None:
            return np.zeros(3)

    mesh = geometry.find('mesh')
    if mesh is None or mesh.get('filename') is None:
        return None
    filename = get_filename(base_path, mesh.get('filename'))
    if filename is None:
        return None
    try:
        meshes = load_meshes(filename)
    except Exception as e:
        logger.warning('could not load %s: %s', filename, e)
        return None
    if not meshes:
        return None

    # Area-weighted mean of the sub-mesh centroids, so a multi-body mesh is
    # measured as the single body it visually is.
    weights = np.array([float(m.area) for m in meshes], dtype=np.float64)
    centroids = np.array([np.asarray(m.centroid, dtype=np.float64)
                          for m in meshes])
    if weights.sum() > 0.0:
        centroid = (centroids * weights[:, None]).sum(axis=0) / weights.sum()
    else:
        centroid = centroids.mean(axis=0)

    scale = mesh.get('scale')
    if scale is not None:
        # the spec asks for three values, but a single uniform one is common
        values = np.array([float(v) for v in scale.split()],
                          dtype=np.float64)
        centroid = centroid * (values if values.size == 3 else values[0])
    return centroid


def _external_frames(root, link_name):
    """Elements outside the link that place something in ITS frame.

    A ``<gazebo reference="L">`` sensor pose and a top-level ``<sensor>``
    parented to ``L`` are expressed in ``L``'s frame, so moving that frame
    moves them -- and nothing here rewrites them.  ``<transmission>`` is not
    listed: it references joints, which carry no frame of their own.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        The ``<robot>`` element.
    link_name : str
        Link whose frame is about to move.

    Returns
    -------
    tags : list of str
        Tag names found, empty when the frame is safe to move.
    """
    found = []
    for gazebo in root.findall('gazebo'):
        if gazebo.get('reference') == link_name and len(gazebo):
            found.append('<gazebo>')
            break
    for sensor in root.findall('sensor'):
        parent = sensor.find('parent')
        if parent is not None and parent.get('link') == link_name:
            found.append('<sensor>')
            break
    return found


def _translated_origins(elements, offsets):
    """Pre-compute every ``(element, new xyz)`` pair for one link's move.

    Parsing all of them before writing any keeps a malformed origin -- a
    xacro ``${...}`` substitution, say -- from leaving the tree half
    corrected, with the geometry moved and the joint not.

    Parameters
    ----------
    elements : list
        Elements whose ``<origin>`` moves.
    offsets : list
        Translation to add to each, in the same order.

    Returns
    -------
    updates : list of tuple
        ``(element, (3,) array)`` pairs to apply.

    Raises
    ------
    ValueError
        If any origin holds a value that is not a number.
    """
    updates = []
    for element, offset in zip(elements, offsets):
        updates.append((element, _origin_translation(element) + offset))
    return updates


def normalize_link_origins(root, base_path=None, centroid_fn=None,
                           link_names=None, tolerance=1e-6):
    """Re-centre fixed links' frames onto their geometry, in place.

    Each eligible link's frame is moved by ``d``, the centroid of its
    ``<visual>`` geometry expressed in that frame.  The shift is absorbed
    exactly: the parent joint's origin becomes ``F @ T(+d)`` and the
    origins of the child joints, the ``<inertial>``, the ``<visual>`` and
    the ``<collision>`` elements become ``T(-d) @ F``.  The re-centred
    frame is therefore the only thing that moves -- every other frame in
    the model keeps its world pose, in every joint configuration.

    A link is eligible only when its parent joint is of type ``fixed``.  A
    movable joint's ``<origin>`` places its axis of motion, so moving the
    child frame would move that axis and the model would no longer agree
    with itself away from the zero configuration.  The root link is skipped
    for the same reason in reverse: it has no parent joint to absorb
    ``T(+d)``, so re-centring it would move the whole robot in the world.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        Root ``<robot>`` element; modified in place.
    base_path : str or None, optional
        Directory the URDF lives in, used by the default measurement to
        resolve relative and ``package://`` mesh paths.  Required unless
        ``centroid_fn`` is given.
    centroid_fn : callable or None, optional
        ``centroid_fn(geometry) -> (3,) array or None``, measuring the
        centroid of a ``<geometry>`` element **in the geometry's own
        frame** (the ``<visual>`` origin is applied by this function, not
        by the callable).  Returning None makes that geometry unmeasurable
        and it is left out of the average.  Defaults to
        :func:`geometry_centroid` bound to ``base_path``.
    link_names : iterable of str or None, optional
        Restrict the operation to these links.  Default is every link.
    tolerance : float, optional
        Links whose centroid is within this Euclidean distance of their
        frame are left untouched.  Default is ``1e-6`` metres: measuring a
        centroid off a binary STL carries several nanometres of float32
        error, so a tighter threshold makes re-centring depend on which
        export of the same CAD it was handed.

    Returns
    -------
    shifts : dict
        ``{link_name: (3,) numpy.ndarray}`` for the links that moved, each
        value the shift ``d`` in the link's original frame.  Links that
        were skipped or already centred do not appear.

    Raises
    ------
    ValueError
        If neither ``base_path`` nor ``centroid_fn`` is given.

    Notes
    -----
    Frames declared by extensions that reference a link -- ``<gazebo>``
    bodies, ``<sensor>`` elements, ``<transmission>`` blocks -- are not
    rewritten, so a URDF relying on those should be re-centred with care.

    Examples
    --------
    >>> import xml.etree.ElementTree as ET
    >>> from skrobot.urdf import normalize_link_origins
    >>> root = ET.parse('robot.urdf').getroot()      # doctest: +SKIP
    >>> normalize_link_origins(root, base_path='.')  # doctest: +SKIP
    {'gripper_base': array([0.  , 0.  , 0.42])}
    """
    measure = centroid_fn
    if measure is None:
        if base_path is None:
            raise ValueError(
                'either base_path or centroid_fn must be given')

        def measure(geometry):
            return geometry_centroid(geometry, base_path)

    parent_joint = {}
    child_joints = {}
    ambiguous = set()
    for joint in root.findall('joint'):
        child = joint.find('child')
        parent = joint.find('parent')
        if child is None or parent is None:
            continue
        child_name = child.get('link')
        if child_name in parent_joint:
            # a URDF is a tree; with two parents there is no single frame to
            # absorb the shift, and correcting one branch moves the other
            ambiguous.add(child_name)
        parent_joint[child_name] = joint
        child_joints.setdefault(parent.get('link'), []).append(joint)

    seen = set()
    for link in root.findall('link'):
        name = link.get('name')
        if name in seen:
            # both elements would be processed, translating the shared parent
            # joint twice
            ambiguous.add(name)
        seen.add(name)

    if link_names is not None:
        link_names = set(link_names)

    shifts = {}
    for link in root.findall('link'):
        name = link.get('name')
        if link_names is not None and name not in link_names:
            continue
        joint = parent_joint.get(name)
        if joint is None or joint.get('type') != 'fixed':
            continue
        if name in ambiguous:
            logger.warning(
                'link %s is declared or parented more than once; leaving its '
                'frame alone', name)
            continue
        attached = _external_frames(root, name)
        if attached:
            logger.warning(
                'link %s carries %s, which name a frame this function does '
                'not rewrite; leaving it alone', name, ', '.join(attached))
            continue

        centroids = []
        for visual in link.findall('visual'):
            geometry = visual.find('geometry')
            if geometry is None:
                continue
            centroid = measure(geometry)
            if centroid is None:
                continue
            centroid = np.asarray(centroid, dtype=np.float64)
            centroids.append(_origin_rotation(visual).dot(centroid)
                             + _origin_translation(visual))
        if not centroids:
            logger.debug('link %s has no measurable visual geometry', name)
            continue
        shift = np.mean(centroids, axis=0)
        if np.linalg.norm(shift) <= tolerance:
            continue

        # F' = F @ T(+d) on the parent side; T(-d) @ F on everything the
        # link frame carries.  Both are pure translations, so only the xyz
        # attributes move and the rpy attributes stay byte for byte.
        elements = [joint]
        offsets = [_origin_rotation(joint).dot(shift)]
        for child_joint in child_joints.get(name, []):
            elements.append(child_joint)
            offsets.append(-shift)
        for tag in ('inertial', 'visual', 'collision'):
            for elem in link.findall(tag):
                elements.append(elem)
                offsets.append(-shift)
        try:
            updates = _translated_origins(elements, offsets)
        except ValueError as e:
            # every origin is read before any is written, so a link with an
            # unparsable one is skipped whole rather than half moved
            logger.warning('link %s has an origin this cannot move (%s); '
                           'leaving its frame alone', name, e)
            continue
        for element, xyz in updates:
            _set_origin_translation(element, xyz)
        shifts[name] = shift
    return shifts


def normalize_link_origins_file(input_file, output_file=None, **kwargs):
    """Re-centre fixed links' frames in a URDF file.

    Parameters
    ----------
    input_file : str
        Path of the URDF file to read.  Its directory is used as
        ``base_path`` when resolving mesh paths.
    output_file : str or None, optional
        When given, the rewritten URDF is written there.  Writing back to
        ``input_file`` is allowed.
    **kwargs
        Forwarded to :func:`normalize_link_origins` (``centroid_fn``,
        ``link_names``, ``tolerance``).  ``base_path`` defaults to the
        directory of ``input_file``.

    Returns
    -------
    shifts : dict
        ``{link_name: (3,) numpy.ndarray}``, as returned by
        :func:`normalize_link_origins`.

    Raises
    ------
    FileNotFoundError
        If ``input_file`` does not exist.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(
            'URDF file not found: {}'.format(input_file))

    ET.register_namespace('xacro', 'http://ros.org/wiki/xacro')
    tree = ET.parse(input_file)
    kwargs.setdefault(
        'base_path', os.path.dirname(os.path.abspath(input_file)))
    shifts = normalize_link_origins(tree.getroot(), **kwargs)
    if output_file is not None and shifts:
        # only rewrite when something moved: ElementTree drops comments and
        # reflows the markup, so writing a no-op still costs the file its
        # licence header and any per-link provenance
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
    return shifts
