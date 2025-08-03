import hashlib
import logging
import os

from lxml import etree

from skrobot.utils.urdf import get_filename


def get_file_hash(filepath):
    """Calculate SHA-256 hash from file content.

    Parameters
    ----------
    filepath : str
        Path to the file to hash.

    Returns
    -------
    str
        SHA-256 hash of the file content, or 'file_not_found' if file doesn't exist.
    """
    if not os.path.exists(filepath):
        logging.warning("File not found at %s", filepath)
        return 'file_not_found'

    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        # Read in chunks to handle large files efficiently
        chunk = f.read(8192)
        while chunk:
            sha256.update(chunk)
            chunk = f.read(8192)
    return sha256.hexdigest()


def get_texture_hashes_from_dae(dae_path):
    """Extract texture file hashes from a DAE file.

    Parameters
    ----------
    dae_path : str
        Path to the DAE file.

    Returns
    -------
    list of str
        Sorted list of texture file hashes referenced by the DAE file.
    """
    hashes = []
    base_path = os.path.dirname(dae_path)
    try:
        # Parse with XML namespaces
        tree = etree.parse(dae_path)
        ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
        for init_from in tree.findall('.//c:init_from', namespaces=ns):
            if init_from.text:
                texture_path = os.path.join(base_path, init_from.text)
                hashes.append(get_file_hash(texture_path))
    except (etree.XMLSyntaxError, OSError) as e:
        logging.error("Error parsing DAE file %s: %s", dae_path, e)
    return sorted(hashes)  # Sort to ensure consistent order


def get_texture_hashes_from_mtl(mtl_path):
    """Extract texture file hashes from an MTL file.

    Parameters
    ----------
    mtl_path : str
        Path to the MTL file.

    Returns
    -------
    list of str
        Sorted list of texture file hashes referenced by the MTL file.
    """
    hashes = []
    base_path = os.path.dirname(mtl_path)
    try:
        with open(mtl_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Look for texture map specification lines
                parts = line.strip().split()
                if parts and parts[0] in ('map_Kd', 'map_Ks', 'map_Ka',
                                          'map_d', 'bump', 'norm'):
                    if len(parts) > 1:
                        texture_path = os.path.join(base_path, parts[-1])
                        hashes.append(get_file_hash(texture_path))
    except (OSError, UnicodeDecodeError) as e:
        logging.error("Error parsing MTL file %s: %s", mtl_path, e)
    return sorted(hashes)


def get_combined_mesh_hash(mesh_path):
    """Calculate combined hash of mesh file and all referenced textures.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh file (DAE or OBJ).

    Returns
    -------
    str
        Combined hash of mesh content and all referenced textures.
    """
    mesh_content_hash = get_file_hash(mesh_path)
    if mesh_content_hash == 'file_not_found':
        return 'mesh_file_not_found'

    texture_hashes = []
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext == '.dae':
        texture_hashes = get_texture_hashes_from_dae(mesh_path)
    elif ext == '.obj':
        # Find related MTL file references from OBJ file
        try:
            with open(mesh_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith('mtllib'):
                        mtl_filename = line.strip().split(maxsplit=1)[1]
                        mtl_path = os.path.join(os.path.dirname(mesh_path),
                                                mtl_filename)
                        texture_hashes.extend(
                            get_texture_hashes_from_mtl(mtl_path))
                        # Don't break to support multiple mtllib references
        except (OSError, UnicodeDecodeError) as e:
            logging.warning("Error parsing OBJ file %s: %s", mesh_path, e)

    # Combine mesh hash with sorted texture hashes for final hash
    combined_string = mesh_content_hash + ''.join(sorted(texture_hashes))
    return hashlib.sha256(combined_string.encode()).hexdigest()


def get_urdf_hash(urdf_path):
    """Calculate hash of URDF including all referenced meshes and textures.

    Parameters
    ----------
    urdf_path : str
        Path to the URDF file.

    Returns
    -------
    str
        Hash of the URDF with all asset references replaced by their hashes.
    """
    base_path = os.path.dirname(urdf_path)

    # Disable external entity resolution for security
    parser = etree.XMLParser(remove_blank_text=True, resolve_entities=False)
    tree = etree.parse(urdf_path, parser)
    root = tree.getroot()

    # 1. Process <mesh> tags
    for mesh_tag in root.findall('.//mesh'):
        filename_attr = mesh_tag.get('filename')
        if filename_attr:
            mesh_path = get_filename(base_path, filename_attr)
            combined_hash = get_combined_mesh_hash(mesh_path)
            mesh_tag.set('filename', f'hash://{combined_hash}')

    # 2. Process <texture> tags directly specified in URDF
    for texture_tag in root.findall('.//texture'):
        filename_attr = texture_tag.get('filename')
        if filename_attr:
            texture_path = get_filename(base_path, filename_attr)
            texture_hash = get_file_hash(texture_path)
            texture_tag.set('filename', f'hash://{texture_hash}')

    # 3. Canonicalize XML and calculate final hash
    canonical_xml = etree.tostring(root, method='c14n')
    urdf_hash = hashlib.sha256(canonical_xml).hexdigest()

    return urdf_hash
