import os
from pathlib import Path
import shutil
from urllib.parse import urlparse

from lxml import etree as ET

from skrobot.urdf.hash import get_urdf_hash
from skrobot.utils.archive import make_tarfile
from skrobot.utils.urdf import resolve_filepath


def _resolve_directory_structure(original_path, abs_path):
    """Generate directory name from package paths or relative paths.

    Preserves original structure as much as possible.

    Parameters
    ----------
    original_path : str
        Original path from URDF file
    abs_path : str
        Resolved absolute path

    Returns
    -------
    str
        directory path
    """
    # Handle package:// URLs
    parsed_url = urlparse(original_path)
    if parsed_url.scheme == 'package':
        package_name = parsed_url.netloc
        package_relative_path = parsed_url.path.lstrip('/')
        # Preserve package_name/relative_path structure (excluding filename)
        dir_path = Path(package_relative_path).parent
        if str(dir_path) == '.':
            return package_name
        else:
            return f"{package_name}/{dir_path}".replace('\\', '/')

    # Handle relative paths, preserve original directory structure
    if not os.path.isabs(original_path):
        dir_path = Path(original_path).parent
        if str(dir_path) == '.':
            return "meshes"  # Default directory name
        else:
            return str(dir_path).replace('\\', '/')

    # Handle absolute paths, infer structure from parent directories
    abs_path_obj = Path(abs_path)
    parent_dir = abs_path_obj.parent
    grandparent_dir = parent_dir.parent

    # Try to use parent and grandparent directory names
    if parent_dir.name and grandparent_dir.name:
        return f"{grandparent_dir.name}/{parent_dir.name}"
    elif parent_dir.name:
        return parent_dir.name
    else:
        return "meshes"


def aggregate_urdf_mesh_files(
    input_urdf_path, output_directory, compress=False, use_absolute_paths=True
):
    """Collect URDF and related files and rewrite mesh paths.

    Preserves original filenames and directory structure.

    Parameters
    ----------
    input_urdf_path : str or pathlib.Path
        Input URDF file path
    output_directory : str or pathlib.Path
        Output directory where aggregated files will be stored
    compress : bool, optional
        Whether to compress the output as tar.gz (default: False)
    use_absolute_paths : bool, optional
        Whether to use file:// absolute URLs (True, default) or relative paths (False)

    Returns
    -------
    str or pathlib.Path
        Path to output URDF file, or tar.gz file if compress=True

    Raises
    ------
    OSError
        If input URDF file doesn't exist
    """
    urdf_path = Path(input_urdf_path)
    if not urdf_path.exists():
        raise OSError(f"No such urdf {urdf_path}")

    # Parse URDF to get robot name
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    robot_name = root.get("name", "robot")

    # Calculate comprehensive URDF hash including all meshes and textures
    urdf_hash = get_urdf_hash(str(urdf_path))
    # Sanitize robot name (keep only filesystem-safe characters)
    safe_robot_name = "".join(
        c for c in robot_name if c.isalnum() or c in "._-"
    ).rstrip()
    # Sanitize robot name (keep only filesystem-safe characters, and avoid leading/trailing dots/spaces)
    safe_robot_name = "".join(
        c for c in robot_name if c.isalnum() or c in "._-"
    )
    # Strip leading/trailing dots and spaces
    safe_robot_name = safe_robot_name.strip(" .")
    # Avoid reserved names and empty string
    if not safe_robot_name or safe_robot_name in {".", ".."}:
        safe_robot_name = "robot"

    dir_name = f"{safe_robot_name}_{urdf_hash}"
    output_dir = Path(output_directory) / dir_name
    os.makedirs(output_dir, mode=0o755, exist_ok=True)

    # Keep file path mapping to avoid duplicates
    file_mapping = {}

    # Process <mesh> tags
    for mesh in root.findall(".//mesh"):
        filename = mesh.get("filename")
        if filename is None:
            continue

        abs_path = resolve_filepath(str(urdf_path), filename)
        if abs_path and os.path.exists(abs_path):
            # Check if file already processed
            if abs_path in file_mapping:
                mesh.set("filename", file_mapping[abs_path])
                continue

            original_filename = Path(abs_path).name

            # Generate directory name preserving original structure
            friendly_dir_name = _resolve_directory_structure(filename, abs_path)

            # Create directory and preserve original filename
            target_dir = output_dir / friendly_dir_name
            os.makedirs(target_dir, mode=0o755, exist_ok=True)
            target_path = target_dir / original_filename

            # Copy file
            shutil.copy(abs_path, target_path)

            # Copy MTL file if it exists (for OBJ files)
            if original_filename.lower().endswith(".obj"):
                mtl_path = Path(abs_path).with_suffix(".mtl")
                if mtl_path.exists():
                    mtl_target = target_dir / mtl_path.name
                    shutil.copy(mtl_path, mtl_target)

            # Set path based on use_absolute_paths setting
            if use_absolute_paths:
                absolute_target_path = os.path.abspath(target_path)
                file_url = f"file://{absolute_target_path}"
                mesh.set("filename", file_url)
                file_mapping[abs_path] = file_url
            else:
                relative_path = f"{friendly_dir_name}/{original_filename}"
                mesh.set("filename", relative_path)
                file_mapping[abs_path] = relative_path

    # Process <texture> tags similarly
    for texture in root.findall(".//texture"):
        filename = texture.get("filename")
        if filename is None:
            continue

        abs_path = resolve_filepath(str(urdf_path), filename)
        if abs_path and os.path.exists(abs_path):
            # Check if file already processed
            if abs_path in file_mapping:
                texture.set("filename", file_mapping[abs_path])
                continue

            original_filename = Path(abs_path).name

            # Generate directory name preserving original structure
            friendly_dir_name = _resolve_directory_structure(filename, abs_path)

            # Create directory and preserve original filename
            target_dir = output_dir / friendly_dir_name
            os.makedirs(target_dir, mode=0o755, exist_ok=True)
            target_path = target_dir / original_filename

            # Copy file
            shutil.copy(abs_path, target_path)

            # Set path based on use_absolute_paths setting
            if use_absolute_paths:
                absolute_target_path = os.path.abspath(target_path)
                file_url = f"file://{absolute_target_path}"
                texture.set("filename", file_url)
                file_mapping[abs_path] = file_url
            else:
                relative_path = f"{friendly_dir_name}/{original_filename}"
                texture.set("filename", relative_path)
                file_mapping[abs_path] = relative_path

    # Save modified URDF
    output_urdf_path = output_dir / f"{robot_name}.urdf"
    tree.write(str(output_urdf_path), encoding='utf-8', xml_declaration=True)

    if compress:
        return make_tarfile(output_dir, arcname=dir_name)

    return output_urdf_path
