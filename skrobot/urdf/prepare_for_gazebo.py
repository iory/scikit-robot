"""Make a description-only URDF loadable by Gazebo (gz-sim / sdformat).

A URDF written for ``robot_state_publisher`` often has no ``<inertial>``
blocks: TF and RViz do not need mass.  sdformat's URDF parser, which is what
``gz sim`` and ``gz sdf -p`` use, drops every link that has no mass, together
with the joint leading to it and everything below.  When the ROOT link is
massless the whole model is gone ("A model must have at least one link").

:func:`prepare_urdf_for_gazebo` fills that gap without touching the kinematics:

* every link with no ``<inertial>`` -- or with one whose mass is missing or
  not positive, which sdformat treats the same way -- gets a small token
  inertial (mass and diagonal inertia, both configurable).  Links that
  already carry a positive mass are left exactly as they are.
* optionally (``mass_from_geometry=True``) the inertial is estimated from the
  link's collision geometry, or its visual geometry when it has no collision,
  as a uniform-density solid.  Primitives use the closed-form formulas;
  meshes go through :func:`skrobot.utils.inertia.link_inertial_from_mesh`
  (falling back to the convex hull and then the bounding box for meshes that
  are not watertight).  A link whose geometry yields nothing usable still
  gets the token inertial, and the per-link ``method`` in the report says
  which one was used.
* optionally the URDF's ``<collision>`` elements are removed.  A token mass
  is enough to keep a link, not to simulate it: if the model is meant to
  collide, prefer ``mass_from_geometry`` or real inertials.
* ``<mimic>`` is kept by default: sdformat 14 (gz-sim 8, Harmonic) converts it
  into an SDF ``<joint><axis><mimic>``.  Whether it is enforced depends on the
  physics engine: in gz-sim 8 the bullet-featherstone engine
  (``gz sim --physics-engine gz-physics-bullet-featherstone-plugin``) builds
  the constraint, while the default DART engine logs "the chosen physics
  engine does not support mimic constraints" and leaves the joint free.
  ``strip_mimic=True`` removes the tags, for setups that drive every joint
  themselves (for example from ``/joint_states``) and want no constraint.

The edit is made on the raw XML, so ``<gazebo>``, ``<ros2_control>``,
``<transmission>`` and comments survive unchanged.
"""

from logging import getLogger
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.utils.inertia import combine_inertials
from skrobot.utils.inertia import DEFAULT_DENSITY
from skrobot.utils.inertia import link_inertial_from_mesh
from skrobot.utils.inertia import transform_inertial
from skrobot.utils.inertia import validate_inertia
from skrobot.utils.urdf_mesh import resolve_filepath


logger = getLogger(__name__)


__all__ = [
    'DEFAULT_TOKEN_MASS',
    'DEFAULT_TOKEN_INERTIA',
    'prepare_urdf_for_gazebo',
    'prepare_urdf_for_gazebo_file',
]


DEFAULT_TOKEN_MASS = 0.01  # kg
DEFAULT_TOKEN_INERTIA = 1e-6  # kg m^2, on each diagonal entry

_INERTIA_KEYS = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')


def _fmt(value):
    """Format a number for a URDF attribute, folding ``-0.0`` to ``0``."""
    value = float(value)
    return '0' if value == 0.0 else '{:.10g}'.format(value)


def _vector(element, attribute, default='0 0 0'):
    """A whitespace-separated float attribute, ``default`` when absent."""
    raw = (element.get(attribute) if element is not None else None) or default
    return [float(v) for v in raw.split()]


def _positive_mass(inertial):
    """Whether an ``<inertial>`` carries a finite positive mass."""
    mass = inertial.find('mass')
    if mass is None:
        return False
    try:
        value = float(mass.get('value'))
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _write_inertial(link, info):
    """Insert an ``<inertial>`` built from ``info`` as the link's first child."""
    inertial = ET.Element('inertial')
    origin = ET.SubElement(inertial, 'origin')
    origin.set('xyz', ' '.join(_fmt(v) for v in info['com']))
    origin.set('rpy', '0 0 0')
    ET.SubElement(inertial, 'mass').set('value', _fmt(info['mass']))
    tensor = ET.SubElement(inertial, 'inertia')
    for key, value in zip(_INERTIA_KEYS, info['inertia']):
        tensor.set(key, _fmt(value))
    # Inserted before the first child, so the whitespace that indented that
    # child now indents it -- cosmetic, but it keeps the output diffable.
    inertial.tail = link.text
    link.insert(0, inertial)


def _token_inertial(mass, inertia):
    return {'mass': float(mass), 'com': [0.0, 0.0, 0.0],
            'inertia': (float(inertia), 0.0, 0.0, float(inertia), 0.0,
                        float(inertia)),
            'method': 'token'}


def _primitive_properties(shape, density):
    """``(mass, inertia 3x3 about the centre, kind)`` of a URDF primitive.

    Returns None for a mesh or an unusable primitive.
    """
    if shape.tag == 'box':
        size = _vector(shape, 'size', default='0 0 0')
        if len(size) != 3:
            return None
        x, y, z = size
        mass = density * x * y * z
        diagonal = (mass / 12.0 * (y * y + z * z),
                    mass / 12.0 * (x * x + z * z),
                    mass / 12.0 * (x * x + y * y))
    elif shape.tag == 'cylinder':
        radius = float(shape.get('radius', 0.0))
        length = float(shape.get('length', 0.0))
        mass = density * math.pi * radius * radius * length
        side = mass * (3.0 * radius * radius + length * length) / 12.0
        diagonal = (side, side, 0.5 * mass * radius * radius)
    elif shape.tag == 'sphere':
        radius = float(shape.get('radius', 0.0))
        mass = density * 4.0 / 3.0 * math.pi * radius ** 3
        moment = 0.4 * mass * radius * radius
        diagonal = (moment, moment, moment)
    else:
        return None
    if not (math.isfinite(mass) and mass > 0.0):
        return None
    return mass, np.diag(diagonal), shape.tag


def _element_inertial(element, base_dir, density):
    """Link-frame inertial of one ``<collision>`` / ``<visual>``, or None."""
    geometry = element.find('geometry')
    if geometry is None or len(geometry) == 0:
        return None
    shape = geometry[0]
    origin = element.find('origin')
    xyz = _vector(origin, 'xyz')
    rpy = _vector(origin, 'rpy')
    if shape.tag == 'mesh':
        filename = shape.get('filename')
        if not filename:
            return None
        path = resolve_filepath(base_dir, filename)
        if path is None:
            logger.warning('mesh not found, cannot estimate its mass: %s',
                           filename)
            return None
        scale = _vector(shape, 'scale', default='1 1 1')
        return link_inertial_from_mesh(path, xyz, rpy, density=density,
                                       scale=scale)
    props = _primitive_properties(shape, density)
    if props is None:
        return None
    mass, inertia, kind = props
    return transform_inertial(mass, [0.0, 0.0, 0.0], inertia, xyz, rpy,
                              method=kind)


def _geometry_inertial(link, base_dir, density):
    """Inertial estimated from the link's collisions, else its visuals."""
    for tag in ('collision', 'visual'):
        infos = [_element_inertial(element, base_dir, density)
                 for element in link.findall(tag)]
        combined = combine_inertials(*infos)
        if combined is None:
            continue
        if validate_inertia(combined['mass'], combined['inertia']):
            logger.warning('link %s: %s geometry gives an invalid inertia, '
                           'using the token inertial', link.get('name'), tag)
            return None
        combined['method'] = '{}:{}'.format(tag, combined['method'])
        return combined
    return None


def prepare_urdf_for_gazebo(root, token_mass=DEFAULT_TOKEN_MASS,
                            token_inertia=DEFAULT_TOKEN_INERTIA,
                            mass_from_geometry=False,
                            density=DEFAULT_DENSITY,
                            strip_collision=False, strip_mimic=False,
                            base_dir=None):
    """Edit a parsed URDF in place so sdformat keeps every link.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        The ``<robot>`` element.  It is modified in place.
    token_mass : float
        Mass in kg given to a link that has no usable inertial.
    token_inertia : float
        Value of each diagonal entry of the token inertia tensor, in kg m^2.
    mass_from_geometry : bool
        When True, estimate a missing inertial from the link's collision (or,
        failing that, visual) geometry as a solid of ``density``; links whose
        geometry gives nothing usable fall back to the token inertial.
    density : float
        Density in kg/m^3 used by ``mass_from_geometry``.
    strip_collision : bool
        Remove every ``<collision>`` element.  Collisions are read for
        ``mass_from_geometry`` before they are removed.
    strip_mimic : bool
        Remove every ``<mimic>`` element, leaving those joints independent.
    base_dir : str or None
        Directory that relative mesh paths resolve against, normally the
        URDF's own directory.  Defaults to the current directory.

    Returns
    -------
    report : dict
        ``'added'``: ``{link: method}`` for links that got a new inertial,
        where ``method`` is ``'token'`` or the geometry it came from (for
        example ``'collision:mesh+box'``); ``'replaced'``: names of links
        whose existing inertial had no positive mass and was replaced;
        ``'kept'``: names of links whose inertial was left untouched;
        ``'collisions_removed'``: number of ``<collision>`` elements removed;
        ``'mimic_joints'``: ``{joint: mimicked joint}`` for every joint that
        had a ``<mimic>``; ``'mimic_stripped'``: whether those were removed.

    Raises
    ------
    ValueError
        If ``token_mass`` or ``token_inertia`` is not positive, or ``density``
        is not positive while ``mass_from_geometry`` is set.
    """
    if not (token_mass > 0.0 and token_inertia > 0.0):
        raise ValueError('token mass and inertia must be positive, got '
                         '{} and {}'.format(token_mass, token_inertia))
    if mass_from_geometry and not density > 0.0:
        raise ValueError('density must be positive, got {}'.format(density))
    if base_dir is None:
        base_dir = os.getcwd()

    report = {'added': {}, 'replaced': [], 'kept': [],
              'collisions_removed': 0, 'mimic_joints': {},
              'mimic_stripped': bool(strip_mimic)}
    for link in root.findall('link'):
        name = link.get('name')
        inertials = link.findall('inertial')
        if inertials and _positive_mass(inertials[0]):
            report['kept'].append(name)
            continue
        if inertials:
            for inertial in inertials:
                link.remove(inertial)
            report['replaced'].append(name)
        info = None
        if mass_from_geometry:
            info = _geometry_inertial(link, base_dir, density)
        if info is None:
            info = _token_inertial(token_mass, token_inertia)
        _write_inertial(link, info)
        report['added'][name] = info['method']

    if strip_collision:
        for link in root.findall('link'):
            for collision in link.findall('collision'):
                link.remove(collision)
                report['collisions_removed'] += 1

    for joint in root.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is None:
            continue
        report['mimic_joints'][joint.get('name')] = mimic.get('joint')
        if strip_mimic:
            joint.remove(mimic)
    return report


def prepare_urdf_for_gazebo_file(input_path, output_path=None, **kwargs):
    """Apply :func:`prepare_urdf_for_gazebo` to a URDF file.

    Relative mesh paths resolve against the input file's directory, and
    comments survive the round trip.

    Parameters
    ----------
    input_path : str
        URDF to read.
    output_path : str or None
        Where to write.  Defaults to ``input_path``, editing it in place.
    **kwargs
        Forwarded to :func:`prepare_urdf_for_gazebo` (``base_dir`` defaults
        to the input file's directory).

    Returns
    -------
    report : dict
        The report of :func:`prepare_urdf_for_gazebo`.
    """
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    with open(input_path, encoding='utf-8') as f:
        root = ET.fromstring(f.read(), parser=parser)
    kwargs.setdefault('base_dir',
                      os.path.dirname(os.path.abspath(input_path)))
    report = prepare_urdf_for_gazebo(root, **kwargs)
    text = ET.tostring(root, encoding='unicode')
    if not text.startswith('<?xml'):
        text = '<?xml version="1.0"?>\n' + text
    if not text.endswith('\n'):
        text += '\n'
    with open(output_path or input_path, 'w', encoding='utf-8') as f:
        f.write(text)
    return report
