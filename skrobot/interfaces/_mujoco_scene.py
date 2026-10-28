"""Edits that turn a converted robot MJCF into a scene to work in.

:func:`skrobot.urdf.urdf_to_mjcf` emits the robot alone. A
:class:`~skrobot.interfaces._mujoco.MujocoRobotInterface` session also needs
a base that can drive around, things on a table to pick up, and cameras on
the robot. MuJoCo cannot add bodies to a compiled model, so those go into the
XML before it is loaded; they are plain ``xml.etree`` edits so they run under
Pyodide as well.
"""

import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates.math import matrix2quaternion
from skrobot.model.primitives import Box
from skrobot.model.primitives import Cylinder
from skrobot.model.primitives import Sphere


PLANAR_BASE_JOINTS = ('base_x', 'base_y', 'base_yaw')

# Collision masks. The converter puts robot geoms on contype=2 conaffinity=1
# (unless self_collision is asked for); objects added here use 1/1. A ground
# on 4/5 then touches objects (1 & 5) but not the robot (2 & 5 and 4 & 1 are
# both 0), so a base driven by actuators is not dragged along the floor by
# its wheels and casters.
_OBJECT_MASK = {'contype': '1', 'conaffinity': '1'}
_GROUND_MASK_WITHOUT_ROBOT = {'contype': '4', 'conaffinity': '5'}


def _fmt(values):
    return ' '.join('{:.9g}'.format(float(v)) for v in np.ravel(values))


def _body(root, name):
    for body in root.iter('body'):
        if body.get('name') == name:
            return body
    return None


def add_planar_base(root, base_link_name):
    """Let the base body slide in x and y and turn about z.

    Three joints on the base body, each with a position actuator named
    ``<joint>_act``. The gains are left at placeholders: they depend on the
    robot's mass, which only the compiled model knows, so the interface sets
    them after loading.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        ``<mujoco>`` element.
    base_link_name : str
        Name of the body the converter made for the URDF base link.
    """
    base = _body(root, base_link_name)
    if base is None:
        raise ValueError(
            'planar_base needs a base body; the URDF root {} has none (a '
            'URDF rooted at "world" is welded to it).'.format(base_link_name))
    axes = ('1 0 0', '0 1 0', '0 0 1')
    types = ('slide', 'slide', 'hinge')
    # Joints go first in the body, ahead of its geoms and inertial.
    for i, (name, axis, jtype) in enumerate(
            zip(PLANAR_BASE_JOINTS, axes, types)):
        base.insert(i, ET.Element('joint', {
            'name': name, 'type': jtype, 'axis': axis,
            'limited': 'false'}))
    actuator = root.find('actuator')
    if actuator is None:
        actuator = ET.SubElement(root, 'actuator')
    for name in PLANAR_BASE_JOINTS:
        ET.SubElement(actuator, 'position', {
            'name': name + '_act', 'joint': name, 'kp': '1', 'kv': '0'})
    ground = _ground(root)
    if ground is not None:
        ground.attrib.update(_GROUND_MASK_WITHOUT_ROBOT)


def compensate_gravity(root):
    """Cancel gravity on every body now in the model.

    This has to be in the XML: MuJoCo decides at compile time whether any
    body uses gravity compensation, and setting ``MjModel.body_gravcomp``
    after loading a model that had none does nothing.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        ``<mujoco>`` element.
    """
    for body in root.find('worldbody').iter('body'):
        body.set('gravcomp', '1')


def _ground(root):
    worldbody = root.find('worldbody')
    for geom in worldbody.findall('geom'):
        if geom.get('name') == 'ground':
            return geom
    return None


def _primitive_geom(obj):
    """MJCF geom attributes for a skrobot primitive."""
    if isinstance(obj, Box):
        return {'type': 'box', 'size': _fmt(np.asarray(obj._extents) / 2.0)}
    if isinstance(obj, Cylinder):
        return {'type': 'cylinder',
                'size': _fmt([obj.radius, obj.height / 2.0])}
    if isinstance(obj, Sphere):
        return {'type': 'sphere', 'size': _fmt([obj._radius])}
    raise TypeError(
        'objects must be skrobot Box, Cylinder or Sphere, not {}'.format(
            type(obj).__name__))


def _primitive_rgba(obj):
    visual = getattr(obj.visual_mesh, 'visual', None)
    colors = getattr(visual, 'face_colors', None)
    if colors is None or len(colors) == 0:
        return None
    return _fmt(np.asarray(colors[0], dtype=np.float64) / 255.0)


def add_objects(root, objects, movable):
    """Put skrobot primitives into the world at their current coordinates.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        ``<mujoco>`` element.
    objects : list[skrobot.model.Link]
        Box, Cylinder or Sphere, each with a unique ``name``.
    movable : bool
        Give each a free joint (mass from MuJoCo's default density, that of
        water) instead of fixing it in place.
    """
    worldbody = root.find('worldbody')
    for obj in objects:
        if not obj.name:
            raise ValueError(
                'every object needs a name, to read its pose back by: '
                '{!r}'.format(obj))
        if _body(root, obj.name) is not None:
            raise ValueError('object name {} is already taken'.format(
                obj.name))
        coords = obj.worldcoords()
        body = ET.SubElement(worldbody, 'body', {
            'name': obj.name, 'pos': _fmt(coords.translation),
            'quat': _fmt(matrix2quaternion(coords.rotation))})
        if movable:
            ET.SubElement(body, 'freejoint', {'name': obj.name})
        geom = _primitive_geom(obj)
        geom.update(_OBJECT_MASK)
        geom['name'] = obj.name
        rgba = _primitive_rgba(obj)
        if rgba is not None:
            geom['rgba'] = rgba
        ET.SubElement(body, 'geom', geom)


# Orientation of a MuJoCo camera (looks along its -z, +y up) in the link it
# is mounted on, per link convention:
#   optical    ROS optical frame: +z forward, +x right, +y down
#   x_forward  ROS body frame:    +x forward, +y left,  +z up
_CAMERA_IN_LINK = {
    'optical': np.array([[1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0]]),
    'x_forward': np.array([[0.0, 0.0, -1.0],
                           [-1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]]),
}


def add_cameras(root, cameras):
    """Mount a camera on each named link, looking the way the link does.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        ``<mujoco>`` element.
    cameras : dict[str, dict]
        Camera name -> ``{'link': str, 'fovy': float [deg],
        'frame': 'optical' or 'x_forward'}``.
    """
    for name, spec in cameras.items():
        body = _body(root, spec['link'])
        if body is None:
            raise ValueError('camera {}: link {} is not in the model'.format(
                name, spec['link']))
        frame = spec['frame']
        if frame not in _CAMERA_IN_LINK:
            raise ValueError('camera {}: frame must be one of {}, not '
                             '{!r}'.format(name, sorted(_CAMERA_IN_LINK),
                                           frame))
        ET.SubElement(body, 'camera', {
            'name': name, 'pos': '0 0 0',
            'quat': _fmt(matrix2quaternion(_CAMERA_IN_LINK[frame])),
            'fovy': '{:.9g}'.format(spec['fovy'])})
