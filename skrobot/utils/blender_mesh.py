from skrobot._lazy_imports import _lazy_trimesh


def _find_blender_executable():
    """Find Blender executable automatically.

    Returns
    -------
    str or None
        Path to Blender executable, or None if not found.
    """
    from pathlib import Path
    import platform
    import shutil

    # First try to find 'blender' in PATH
    blender_path = shutil.which('blender')
    if blender_path:
        return blender_path

    # Platform-specific search
    system = platform.system()

    if system == 'Darwin':  # macOS
        # Common macOS installation paths
        macos_paths = [
            '/Applications/Blender.app/Contents/MacOS/Blender',
            '~/Applications/Blender.app/Contents/MacOS/Blender',
        ]
        for path_str in macos_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                return str(path)
    elif system == 'Linux':
        # Common Linux installation paths
        linux_paths = [
            '/usr/bin/blender',
            '/usr/local/bin/blender',
            '~/blender/blender',
        ]
        for path_str in linux_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                return str(path)

    return None


def remesh_with_blender(
        mesh,
        voxel_size=0.002,
        output_format='dae',
        blender_executable=None,
        verbose=False,
        output_file=None):
    """Remesh a mesh using Blender's voxel remesher while preserving colors.

    This function uses Blender's voxel remeshing to create a cleaner mesh
    topology while preserving the original material colors by assigning them
    based on nearest face matching.

    Parameters
    ----------
    mesh : trimesh.Trimesh or str or pathlib.Path
        Mesh to be remeshed. Can be a trimesh object or path to a mesh file.
    voxel_size : float, optional
        Voxel size for remeshing. Smaller values create more detailed meshes.
        Default is 0.002.
    output_format : str, optional
        Output format for the remeshed mesh. Either 'dae' or 'stl'.
        Default is 'dae'.
    blender_executable : str, optional
        Path to Blender executable. If None, automatically searches for Blender
        in common installation locations. Default is None.
    verbose : bool, optional
        Whether to print progress information. Default is False.
    output_file : str or pathlib.Path, optional
        Path where the remeshed file should be saved. If None, a temporary file
        is used. Default is None.

    Returns
    -------
    trimesh.Trimesh or trimesh.Scene
        Remeshed mesh with preserved colors. Returns Scene for DAE files with
        multiple materials/colors.

    Raises
    ------
    RuntimeError
        If Blender remeshing fails or output file is not created.

    Notes
    -----
    This function requires Blender to be installed and accessible via
    the specified executable path.

    The function performs the following steps:
    1. Imports the mesh into Blender
    2. Collects original face colors from vertex colors
    3. Joins all mesh objects
    4. Applies voxel remeshing
    5. Assigns materials based on nearest original face colors
    6. Exports the remeshed mesh

    Examples
    --------
    >>> import trimesh
    >>> from skrobot.utils.blender_mesh import remesh_with_blender
    >>> mesh = trimesh.load('input.dae')
    >>> remeshed = remesh_with_blender(mesh, voxel_size=0.001)
    """
    from pathlib import Path
    import subprocess
    import tempfile

    trimesh = _lazy_trimesh()

    # Auto-detect Blender if not specified
    if blender_executable is None:
        blender_executable = _find_blender_executable()
        if blender_executable is None:
            raise RuntimeError(
                "Blender executable not found. Please install Blender or specify "
                "the path using the blender_executable parameter."
            )
        if verbose:
            print(f"Found Blender at: {blender_executable}")

    # Handle input mesh
    if isinstance(mesh, (str, Path)):
        input_path = Path(mesh)
        original_mesh = trimesh.load(input_path)
    else:
        # Save mesh to temporary file
        input_path = None
        original_mesh = mesh

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # If mesh object was provided, save it to temporary file
        if input_path is None:
            input_path = temp_path / 'input.dae'
            # Handle both Scene and Trimesh objects
            if isinstance(mesh, trimesh.Scene):
                # Scene.export() doesn't support DAE format well
                # Extract meshes and export using collada exporter
                geometries = list(mesh.dump())
                trimesh_objects = [g for g in geometries if isinstance(g, trimesh.Trimesh)]
                if trimesh_objects:
                    # Use trimesh's DAE exporter
                    dae_data = trimesh.exchange.dae.export_collada(trimesh_objects)
                    with open(str(input_path), 'wb') as f:
                        f.write(dae_data)
                else:
                    raise RuntimeError("Scene contains no Trimesh objects")
            else:
                mesh.export(str(input_path))

        # Create output path
        if output_format.lower() == 'stl':
            output_extension = '.stl'
        else:
            output_extension = '.dae'
        output_path = temp_path / f'output{output_extension}'

        # Create Blender script that imports and calls the core module
        script_path = temp_path / 'remesh_script.py'
        _create_blender_wrapper_script(
            script_path,
            input_path,
            output_path,
            voxel_size,
            output_format.upper()
        )

        # Run Blender
        cmd = [
            blender_executable,
            '--background',
            '--python', str(script_path)
        ]

        if verbose:
            print(f"Running Blender remesh with voxel_size={voxel_size}...")

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Blender remeshing failed with return code {result.returncode}"
            if not verbose and result.stderr:
                error_msg += f"\nError: {result.stderr}"
            raise RuntimeError(error_msg)

        # Check that Blender created the output file
        if not output_path.exists():
            raise RuntimeError(f"Remeshed output file not found: {output_path}")

        # If output_file is specified, copy Blender's output there
        if output_file is not None:
            import shutil
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(output_path), str(output_file))
            # Load from the final location
            remeshed_mesh = trimesh.load(str(output_file))
        else:
            # Load from temp location
            remeshed_mesh = trimesh.load(str(output_path))

        if verbose:
            # Count total faces from original mesh
            if isinstance(original_mesh, trimesh.Scene):
                original_geometries = list(original_mesh.dump())
                original_trimesh_objects = [g for g in original_geometries if isinstance(g, trimesh.Trimesh)]
                original_faces = sum(len(m.faces) for m in original_trimesh_objects)
            elif hasattr(original_mesh, 'faces'):
                original_faces = len(original_mesh.faces)
            else:
                original_faces = None

            # Count total faces from remeshed output
            if isinstance(remeshed_mesh, trimesh.Scene):
                geometries = list(remeshed_mesh.dump())
                trimesh_objects = [g for g in geometries if isinstance(g, trimesh.Trimesh)]
                total_faces = sum(len(m.faces) for m in trimesh_objects)
            elif hasattr(remeshed_mesh, 'faces'):
                total_faces = len(remeshed_mesh.faces)
            else:
                total_faces = None

            if original_faces is not None and total_faces is not None:
                print(f"Remeshing complete. Faces: {original_faces} -> {total_faces}")
            else:
                print("Remeshing complete.")

        return remeshed_mesh


def _create_blender_wrapper_script(script_path, input_path, output_path, voxel_size, export_format):
    """Create a wrapper script that imports the core Blender remeshing module.

    Parameters
    ----------
    script_path : pathlib.Path
        Path where the wrapper script will be written.
    input_path : pathlib.Path
        Path to input mesh file.
    output_path : pathlib.Path
        Path to output mesh file.
    voxel_size : float
        Voxel size for remeshing.
    export_format : str
        Export format ('DAE' or 'STL').
    """
    from pathlib import Path

    # Get the path to the core module
    core_module_path = Path(__file__).parent / '_blender_remesh_core.py'

    # Create a minimal wrapper script that imports and calls the core module
    script_content = f"""import sys
from pathlib import Path

# Add the skrobot utils directory to Python path
core_module_path = Path(r"{core_module_path}")
sys.path.insert(0, str(core_module_path.parent))

# Import and use the core remeshing function
from _blender_remesh_core import remesh_and_bake_file

input_path = Path(r"{input_path}")
output_path = Path(r"{output_path}")
voxel_size = {voxel_size}
export_format = "{export_format}"

remesh_and_bake_file(input_path, output_path, voxel_size, export_format)
"""

    script_path.write_text(script_content)
