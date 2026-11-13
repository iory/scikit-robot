"""Edits that turn a converted robot MJCF into a scene to work in.

:func:`skrobot.urdf.urdf_to_mjcf` emits the robot alone. A
:class:`~skrobot.interfaces._mujoco.MujocoRobotInterface` session also needs
a base that can drive around, things on a table to pick up, and cameras on
the robot. MuJoCo cannot add bodies to a compiled model, so those go into the
XML before it is loaded; they are plain ``xml.etree`` edits so they run under
Pyodide as well.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates.math import matrix2quaternion
from skrobot.coordinates.math import quaternion2matrix
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
# A static model's collision geom whose top is no higher than this [m] is
# part of the floor (a room's floor slab), and collides as the ground does.
_FLOOR_HEIGHT = 1e-3

# (MJCF path, URDF path, URDF mtime) -> converted MJCF text, so a program
# run again does not convert the same room again.
_converted_models = {}


def _fmt(values):
    return ' '.join('{:.9g}'.format(float(v)) for v in np.ravel(values))


def _body(root, name):
    for body in root.iter('body'):
        if body.get('name') == name:
            return body
    return None


def _link_frame(root, name):
    """``(body, pos, rot)``: the body a link's frame rides on, and the link's
    position and orientation in that body; ``None`` if there is no such link.

    A link is a body, or -- for a frame link such as a camera's optical frame,
    which the converter writes as a site -- a site on its nearest real body.
    """
    body = _body(root, name)
    if body is not None:
        return body, np.zeros(3), np.eye(3)
    # a frame hung straight off a URDF 'world' root is a site on <worldbody>
    for body in [root.find('worldbody'), *root.iter('body')]:
        for site in body.findall('site'):
            if site.get('name') == name:
                pos = [float(v) for v in site.get('pos', '0 0 0').split()]
                quat = [float(v) for v in site.get('quat', '1 0 0 0').split()]
                return body, np.array(pos), quaternion2matrix(quat)
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


def _geom_top(geom, rot, pos):
    """Highest world z of a box, cylinder or sphere geom on a body at
    ``(rot, pos)``; None for other types."""
    size = [float(v) for v in geom.get('size', '0').split()]
    gtype = geom.get('type', 'sphere')
    if gtype == 'box':
        half = np.array(size[:3])
    elif gtype == 'cylinder':
        half = np.array([size[0], size[0], size[1]])
    elif gtype == 'sphere':
        half = np.full(3, size[0])
    else:
        return None
    geom_rot = rot @ quaternion2matrix(
        [float(v) for v in geom.get('quat', '1 0 0 0').split()])
    centre = rot @ np.array(
        [float(v) for v in geom.get('pos', '0 0 0').split()]) + pos
    # The box's corners reach np.abs(R) @ half above its centre.
    return float(centre[2] + np.abs(geom_rot[2]) @ half)


def _convert_model(model, out_dir):
    """Convert a RobotModel's URDF into ``out_dir``.

    Returns
    -------
    xml.etree.ElementTree.Element
        The converted ``<mujoco>`` element; its meshes and textures are in
        ``out_dir``.
    """
    from skrobot.urdf import urdf_to_mjcf

    urdf_path = getattr(model, 'urdf_path', None)
    if urdf_path is None:
        raise ValueError(
            'a RobotModel object must be loaded from a URDF file '
            '(urdf_path is None): {!r}'.format(model.name))
    out_path = os.path.join(out_dir, 'model.xml')
    key = (out_path, os.path.abspath(urdf_path), os.path.getmtime(urdf_path))
    xml = _converted_models.get(key)
    if xml is None or not os.path.exists(out_path):
        xml = urdf_to_mjcf(urdf_path, out_path, mesh_dir=out_dir,
                           add_ground=False, add_position_actuators=False,
                           self_collision=True)
        _converted_models[key] = xml
    return ET.fromstring(xml)


def add_static_model(root, model, mjcf_dir):
    """Put a model loaded from a URDF (a room, furniture) into the world,
    fixed as it is posed now.

    Each link becomes a body named after it, welded where the link is now:
    the model's joints (a desk top's hinge) are not simulated. Visual
    meshes keep their textures. Collision geoms collide with the robot and
    objects; those whose top is at the floor (z of at most 1 mm), such as
    a room's floor slab, collide as the ground does, so a planar base that
    does not touch the ground does not touch them either.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        ``<mujoco>`` element.
    model : skrobot.model.RobotModel
        Model with ``urdf_path`` set and a ``name``.
    mjcf_dir : str
        Directory of the MJCF ``root`` will be written to. The model's
        converted meshes go in a subdirectory of its ``meshdir`` named
        after the model, so the scene's files stay in one tree.
    """
    if not model.name:
        raise ValueError('every object needs a name: {!r}'.format(model))
    if _body(root, model.name) is not None:
        raise ValueError('object name {} is already taken'.format(
            model.name))
    compiler = root.find('compiler')
    if compiler is None:
        compiler = ET.Element('compiler')
        root.insert(0, compiler)
    asset_dir = os.path.join(mjcf_dir, compiler.get('meshdir', '.'),
                             model.name)
    converted = _convert_model(model, asset_dir)

    def relative(kind):
        # MuJoCo finds a mesh under meshdir and a texture under texturedir,
        # both relative to the MJCF's directory.
        return os.path.relpath(
            asset_dir, os.path.join(mjcf_dir, compiler.get(kind, '.')))

    prefix = model.name + '/'
    asset = root.find('asset')
    if asset is None:
        asset = ET.Element('asset')
        root.insert(list(root).index(root.find('worldbody')), asset)
    for element in converted.find('asset'):
        element = ET.SubElement(asset, element.tag, dict(element.attrib))
        element.set('name', prefix + element.get('name'))
        if element.tag == 'mesh':
            element.set('file', os.path.join(relative('meshdir'),
                                              element.get('file')))
        elif element.tag == 'texture':
            element.set('file', os.path.join(relative('texturedir'),
                                             element.get('file')))
        elif element.tag == 'material' and element.get('texture'):
            element.set('texture', prefix + element.get('texture'))

    ground = _ground(root)
    floor_mask = ({'contype': ground.get('contype', '1'),
                   'conaffinity': ground.get('conaffinity', '1')}
                  if ground is not None else _OBJECT_MASK)
    links = {link.name: link for link in model.link_list}
    worldbody = root.find('worldbody')
    for body in converted.find('worldbody').iter('body'):
        name = body.get('name')
        if name not in links:
            raise ValueError('body {} of {} is not a link of the model'.format(
                name, model.name))
        if _link_frame(root, name) is not None:
            raise ValueError('link {} of {} is already a name in the '
                             'scene'.format(name, model.name))
        coords = links[name].worldcoords()
        rot, pos = coords.rotation, coords.translation
        static = ET.SubElement(worldbody, 'body', {
            'name': name, 'pos': _fmt(pos),
            'quat': _fmt(matrix2quaternion(rot))})
        for child in body:
            if child.tag not in ('geom', 'site'):
                continue  # joints, inertia and the bodies below it
            child = ET.SubElement(static, child.tag, dict(child.attrib))
            for ref in ('mesh', 'material'):
                if child.get(ref):
                    child.set(ref, prefix + child.get(ref))
            if child.tag == 'geom' and child.get('group') == '3':
                top = _geom_top(child, rot, pos)
                child.attrib.update(
                    floor_mask if top is not None and top <= _FLOOR_HEIGHT
                    else _OBJECT_MASK)
    for child in converted.find('worldbody'):
        if child.tag in ('geom', 'site'):
            raise ValueError(
                '{}: geometry on the URDF root link "world" is not '
                'supported; attach it to a link'.format(model.name))


def add_objects(root, objects, movable, mjcf_dir=None):
    """Put skrobot primitives into the world at their current coordinates.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        ``<mujoco>`` element.
    objects : list[skrobot.model.Link or skrobot.model.RobotModel]
        Box, Cylinder or Sphere, each with a unique ``name``; or, when not
        ``movable``, a model loaded from a URDF (see
        :func:`add_static_model`).
    movable : bool
        Give each a free joint (mass from MuJoCo's default density, that of
        water) instead of fixing it in place.
    mjcf_dir : str or None
        Directory of the MJCF ``root`` will be written to; needed for a
        model loaded from a URDF.
    """
    from skrobot.model.robot_model import RobotModel

    worldbody = root.find('worldbody')
    for obj in objects:
        if isinstance(obj, RobotModel):
            if movable:
                raise TypeError(
                    'a RobotModel cannot be a movable object, only a fixed '
                    'one: {}'.format(obj.name))
            if mjcf_dir is None:
                raise ValueError('add_objects needs mjcf_dir for a RobotModel')
            add_static_model(root, obj, mjcf_dir)
            continue
        if not obj.name:
            raise ValueError(
                'every object needs a name, to read its pose back by: '
                '{!r}'.format(obj))
        if _link_frame(root, obj.name) is not None:
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
        mount = _link_frame(root, spec['link'])
        if mount is None:
            raise ValueError('camera {}: link {} is not in the model'.format(
                name, spec['link']))
        body, link_pos, link_rot = mount
        frame = spec['frame']
        if frame not in _CAMERA_IN_LINK:
            raise ValueError('camera {}: frame must be one of {}, not '
                             '{!r}'.format(name, sorted(_CAMERA_IN_LINK),
                                           frame))
        ET.SubElement(body, 'camera', {
            'name': name, 'pos': _fmt(link_pos),
            'quat': _fmt(matrix2quaternion(link_rot @ _CAMERA_IN_LINK[frame])),
            'fovy': '{:.9g}'.format(spec['fovy'])})
