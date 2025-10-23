import os
import xml.etree.ElementTree as ET


def scale_urdf(input_file, output_file, scale=1.0):
    """Scale a URDF file by a given factor.

    This function scales all geometric and physical properties of a URDF model:
    - Joint origins (xyz positions)
    - Link inertial origins (xyz positions)
    - Mesh geometries (via scale attribute)
    - Primitive geometries (box, cylinder, sphere dimensions)
    - Mass (scaled by scale^3, assuming constant density)
    - Inertia tensors (scaled by scale^5 = mass * length^2)

    Parameters
    ----------
    input_file : str
        Path to the input URDF file
    output_file : str
        Path for the output URDF file
    scale : float, optional
        Scale factor to apply. Values < 1.0 make the model smaller,
        values > 1.0 make it larger. Default: 1.0

    Raises
    ------
    FileNotFoundError
        If the input file does not exist
    ValueError
        If scale is not positive
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"URDF file not found: {input_file}")

    if scale <= 0:
        raise ValueError(f"Scale must be positive, got: {scale}")

    # Register xacro namespace if present
    ET.register_namespace('xacro', "http://ros.org/wiki/xacro")

    tree = ET.parse(input_file)
    root = tree.getroot()

    # Scale all joint origins
    for joint in root.findall('.//joint/origin[@xyz]'):
        xyz = joint.get('xyz')
        scaled_xyz = _scale_xyz(xyz, scale)
        joint.set('xyz', scaled_xyz)

    # Scale all link inertial origins
    for inertial_origin in root.findall('.//link/inertial/origin[@xyz]'):
        xyz = inertial_origin.get('xyz')
        scaled_xyz = _scale_xyz(xyz, scale)
        inertial_origin.set('xyz', scaled_xyz)

    # Scale all collision geometry origins
    for collision_origin in root.findall('.//link/collision/origin[@xyz]'):
        xyz = collision_origin.get('xyz')
        scaled_xyz = _scale_xyz(xyz, scale)
        collision_origin.set('xyz', scaled_xyz)

    # Scale all visual geometry origins
    for visual_origin in root.findall('.//link/visual/origin[@xyz]'):
        xyz = visual_origin.get('xyz')
        scaled_xyz = _scale_xyz(xyz, scale)
        visual_origin.set('xyz', scaled_xyz)

    # Scale mesh geometries
    for mesh in root.findall('.//geometry/mesh'):
        existing_scale = mesh.get('scale')
        if existing_scale:
            # Multiply existing scale by the new scale factor
            new_scale = _scale_xyz(existing_scale, scale)
        else:
            # Set uniform scale
            new_scale = f"{scale} {scale} {scale}"
        mesh.set('scale', new_scale)

    # Scale box geometries
    for box in root.findall('.//geometry/box'):
        size = box.get('size')
        if size:
            scaled_size = _scale_xyz(size, scale)
            box.set('size', scaled_size)

    # Scale cylinder geometries
    for cylinder in root.findall('.//geometry/cylinder'):
        radius = cylinder.get('radius')
        length = cylinder.get('length')
        if radius:
            scaled_radius = float(radius) * scale
            cylinder.set('radius', str(scaled_radius))
        if length:
            scaled_length = float(length) * scale
            cylinder.set('length', str(scaled_length))

    # Scale sphere geometries
    for sphere in root.findall('.//geometry/sphere'):
        radius = sphere.get('radius')
        if radius:
            scaled_radius = float(radius) * scale
            sphere.set('radius', str(scaled_radius))

    # Scale mass (mass scales with volume: scale^3)
    mass_scale = scale ** 3
    for mass in root.findall('.//inertial/mass'):
        value = mass.get('value')
        if value:
            scaled_mass = float(value) * mass_scale
            mass.set('value', str(scaled_mass))

    # Scale inertia (inertia scales with mass * length^2: scale^5)
    inertia_scale = scale ** 5
    for inertia in root.findall('.//inertial/inertia'):
        for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
            value = inertia.get(attr)
            if value:
                scaled_value = float(value) * inertia_scale
                inertia.set(attr, str(scaled_value))

    # Try to use pretty printing if available (Python 3.9+)
    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass

    tree.write(output_file, encoding='utf-8', xml_declaration=True)


def _scale_xyz(xyz_str, scale):
    """Scale an xyz string by a given factor.

    Parameters
    ----------
    xyz_str : str
        Space-separated xyz values (e.g., "1.0 2.0 3.0")
    scale : float
        Scale factor to apply

    Returns
    -------
    str
        Scaled xyz string
    """
    values = [float(v) * scale for v in xyz_str.split()]
    return ' '.join(str(v) for v in values)
