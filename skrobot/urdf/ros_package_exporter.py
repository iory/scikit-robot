"""
ROS package exporter.

Exports a URDF plus its mesh assets as a ROS package that can be built
with catkin/colcon.

The exported package includes:
- URDF file (final combined robot description)
- Mesh files (3D model files)
- package.xml (ROS package manifest)
- CMakeLists.txt (build configuration)
"""

import logging
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Optional
from typing import Set
import zipfile

# The skeleton generators and resource-reference helpers live upstream in
# scikit-robot.
from skrobot.urdf.ros_package import extract_all_resource_references
from skrobot.urdf.ros_package import extract_mesh_references
from skrobot.urdf.ros_package import extract_registered_mesh_references
from skrobot.urdf.ros_package import generate_cmake_lists
from skrobot.urdf.ros_package import generate_package_xml
from skrobot.urdf.ros_package import generate_ros1_display_cmake_lists
from skrobot.urdf.ros_package import generate_ros1_display_launch
from skrobot.urdf.ros_package import generate_ros1_display_package_xml
from skrobot.urdf.ros_package import generate_ros1_rviz_config
from skrobot.urdf.ros_package import replace_package_references
from skrobot.urdf.ros_package import rewrite_mesh_package_references


logger = logging.getLogger(__name__)


class ROSPackageExporter:
    """
    Export robot assemblies as ROS packages.

    This class handles:
    - URDF/Xacro file generation
    - Mesh file collection and copying
    - Package manifest generation
    - ZIP archive creation

    Parameters
    ----------
    package_name : str
        Name of the output ROS package.
    source_assets_dir : Path, optional
        Directory containing source assets (mesh files).
        If None, meshes will not be copied.

    Examples
    --------
    >>> from skrobot.assembly.module_assembly import RobotAssembly
    >>> from skrobot.urdf.ros_package_exporter import ROSPackageExporter
    >>>
    >>> assembly = RobotAssembly("my_robot")
    >>> # ... add modules and connections ...
    >>> urdf_path = assembly.build()
    >>>
    >>> exporter = ROSPackageExporter("my_robot_pkg")
    >>> exporter.set_urdf(urdf_path)
    >>> zip_path = exporter.export_zip()
    """

    def __init__(self, package_name: str, source_assets_dir: Optional[Path] = None):
        self.package_name = package_name
        self.source_assets_dir = Path(source_assets_dir) if source_assets_dir else None
        self.urdf_content = None
        self.mesh_files = set()
        self.registered_mesh_files = set()  # Meshes from uploaded/registered modules
        self.resource_files = set()  # All file references (textures, etc.)

    def set_urdf(self, urdf_path: str, original_package_name: Optional[str] = None) -> None:
        """
        Set the URDF file to export.

        Parameters
        ----------
        urdf_path : str
            Path to the URDF file.
        original_package_name : str, optional
            Original package name to replace with the new package name.
        """
        with open(urdf_path, encoding="utf-8") as f:
            content = f.read()

        if original_package_name:
            content = replace_package_references(content, original_package_name, self.package_name)

        self.urdf_content = content
        self.mesh_files = extract_mesh_references(content)
        self.registered_mesh_files = extract_registered_mesh_references(content)
        self.resource_files = extract_all_resource_references(content)

    @staticmethod
    def _registered_mesh_replacer(match: re.Match) -> str:
        """Replace callback that includes a short hash to avoid collisions."""
        prefix = match.group(1)  # package://pkg/
        full_hash = match.group(2)  # full hash
        name = match.group(3)  # module name
        short_hash = full_hash[:8]
        return f"{prefix}meshes/{name}_{short_hash}/"

    def _rewrite_registered_mesh_paths(self, content: str) -> str:
        """Rewrite registered module mesh paths to namespaced meshes/ paths.

        Converts paths like
        ``package://pkg/registered/<hash>/<name>/meshes/<subpath>``
        to ``package://pkg/meshes/<name>_<hash8>/<subpath>`` so that
        modules with the same name but different content do not collide.
        """
        return re.sub(
            r"(package://[^/]+/)registered/([^/]+)/([^/]+)/meshes/",
            self._registered_mesh_replacer,
            content,
        )

    def _collect_mesh_directories(self) -> Set[str]:
        """
        Collect mesh directories from mesh file references.

        Returns
        -------
        Set[str]
            Set of mesh directory names.
        """
        directories = set()
        for mesh_path in self.mesh_files:
            if "/" in mesh_path:
                directory = mesh_path.split("/")[0]
                directories.add(directory)
        return directories

    def _copy_meshes(self, dest_meshes_dir: Path, progress_callback: Optional[callable] = None) -> int:
        """
        Copy required mesh files to destination directory.

        Handles both standard meshes (from meshes/ directory) and
        registered/uploaded module meshes (from registered/<hash>/<name>/meshes/).

        Parameters
        ----------
        dest_meshes_dir : Path
            Destination meshes directory.
        progress_callback : callable, optional
            Called with (message, current, total) for progress reporting.

        Returns
        -------
        int
            Number of mesh files copied.
        """
        if not self.source_assets_dir:
            return 0

        copied = 0

        # --- Copy standard meshes from meshes/ ---
        source_meshes_dir = self.source_assets_dir / "meshes"
        if source_meshes_dir.exists():
            total = len(self.mesh_files)

            mesh_dirs = self._collect_mesh_directories()
            individual_files = [m for m in self.mesh_files if "/" not in m]

            for mesh_dir in mesh_dirs:
                source_dir = source_meshes_dir / mesh_dir
                if source_dir.exists() and source_dir.is_dir():
                    dest_dir = dest_meshes_dir / mesh_dir
                    if not dest_dir.exists():
                        shutil.copytree(source_dir, dest_dir)
                    if progress_callback:
                        progress_callback(f"Copied mesh directory: {mesh_dir}", copied, total)

            for mesh_file in individual_files:
                source_file = source_meshes_dir / mesh_file
                if source_file.exists() and source_file.is_file():
                    dest_file = dest_meshes_dir / mesh_file
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_file, dest_file)
                    copied += 1
                    if progress_callback:
                        progress_callback(f"Copied mesh: {mesh_file}", copied, total)

        # --- Copy registered/uploaded module meshes ---
        # These have paths like: registered/<hash>/<name>/meshes/<subpath>
        # Copy into meshes/<name>_<hash8>/<subpath> to avoid collisions
        for reg_path in self.registered_mesh_files:
            source_file = self.source_assets_dir / reg_path
            if source_file.exists() and source_file.is_file():
                # reg_path = "registered/<hash>/<name>/meshes/<subpath>"
                match = re.match(r"registered/([^/]+)/([^/]+)/meshes/(.*)", reg_path)
                if match:
                    full_hash = match.group(1)
                    module_name = match.group(2)
                    mesh_subpath = match.group(3)
                    dir_name = f"{module_name}_{full_hash[:8]}"
                    dest_file = dest_meshes_dir / dir_name / mesh_subpath
                    dest_file.parent.mkdir(parents=True, exist_ok=True)
                    if not dest_file.exists():
                        shutil.copy2(source_file, dest_file)
                    copied += 1
                    if progress_callback:
                        progress_callback(f"Copied registered mesh: {dir_name}/{mesh_subpath}", copied, 0)

        return copied

    def _copy_resources(self, package_dir: Path, progress_callback: Optional[callable] = None) -> int:
        """Copy non-mesh resources (textures, materials, etc.) to the package.

        Copies files referenced via ``filename="package://..."`` that are
        outside the ``meshes/`` directory (e.g., ``materials/textures/``).

        Parameters
        ----------
        package_dir : Path
            Root of the exported package directory.
        progress_callback : callable, optional
            Progress callback function.

        Returns
        -------
        int
            Number of resource files copied.
        """
        if not self.source_assets_dir:
            return 0

        copied = 0
        for resource_path in self.resource_files:
            # Skip meshes/ — already handled by _copy_meshes
            if resource_path.startswith("meshes/"):
                continue

            source_file = self.source_assets_dir / resource_path
            if source_file.exists() and source_file.is_file():
                dest_file = package_dir / resource_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                if not dest_file.exists():
                    shutil.copy2(source_file, dest_file)
                copied += 1
                if progress_callback:
                    progress_callback(f"Copied resource: {resource_path}", copied, 0)

        return copied

    def build_package(
        self,
        output_dir: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        with_moveit_config: bool = False,
        ros1: bool = False,
    ) -> Path:
        """
        Build the ROS package directory structure.

        Parameters
        ----------
        output_dir : str, optional
            Output directory. If None, uses a temporary directory.
        progress_callback : callable, optional
            Progress callback function.
        with_moveit_config : bool
            If True, generate a MoveIt 2 package with launch files,
            SRDF, controllers, and ROS 2 package metadata.
        ros1 : bool
            If True, generate a ROS1 catkin package with a minimal
            ``display.launch`` (rviz + robot_state_publisher) and
            convert mesh files from GLB to DAE/STL so RViz can render
            them. Mutually exclusive with ``with_moveit_config``.

        Returns
        -------
        Path
            Path to the created package directory.
        """
        if ros1 and with_moveit_config:
            raise ValueError("ros1 and with_moveit_config cannot both be True")
        if output_dir is None:
            temp_dir = tempfile.mkdtemp(prefix=f"{self.package_name}_export_")
            output_dir = temp_dir

        package_dir = Path(output_dir) / self.package_name
        package_dir.mkdir(parents=True, exist_ok=True)

        # Create directory structure
        meshes_dir = package_dir / "meshes"
        meshes_dir.mkdir(exist_ok=True)

        if not with_moveit_config:
            # Standard package: URDF in urdf/ directory
            urdf_dir = package_dir / "urdf"
            urdf_dir.mkdir(exist_ok=True)

            if self.urdf_content:
                # Rewrite registered module mesh paths to standard meshes/ paths
                # e.g. package://pkg/registered/<hash>/<name>/meshes/foo.stl
                #   -> package://pkg/meshes/foo.stl
                export_content = self._rewrite_registered_mesh_paths(self.urdf_content)
                # The exported package is meant to be standalone (built and
                # roslaunched on the user's machine), so the ``package://``
                # mesh references must point at the *new* package name — not
                # at ``reconfigurable_assets`` or whatever the source package
                # was called.  This applies to both ROS 1 and ROS 2 standard
                # packages, since the mesh files are copied into
                # ``<package_name>/meshes/``.
                export_content = rewrite_mesh_package_references(export_content, self.package_name)
                urdf_path = urdf_dir / f"{self.package_name}.urdf"
                with open(urdf_path, "w", encoding="utf-8") as f:
                    f.write(export_content)
                if progress_callback:
                    progress_callback("Wrote URDF file", 1, 4)

        # Copy mesh files
        self._copy_meshes(meshes_dir, progress_callback)

        # Copy extra resources (textures, materials, etc.)
        self._copy_resources(package_dir, progress_callback)

        if with_moveit_config:
            # Generate MoveIt 2 package (launch files, configs, ROS 2 metadata)
            from skrobot.urdf.ros_config.moveit_package import _convert_urdf_meshes
            from skrobot.urdf.ros_config.moveit_package import build_moveit_package

            if self.urdf_content:
                moveit_urdf = self._rewrite_registered_mesh_paths(self.urdf_content)
                build_moveit_package(
                    package_dir=package_dir,
                    urdf_content=moveit_urdf,
                    robot_name=self.package_name,
                    package_name=self.package_name,
                )

                # Convert meshes: visual -> DAE, collision -> STL
                # This must run after build_moveit_package writes config/{name}.urdf
                # and after _copy_meshes copies the source meshes to meshes/
                urdf_path = package_dir / "config" / f"{self.package_name}.urdf"
                if urdf_path.exists():
                    try:
                        _convert_urdf_meshes(
                            urdf_path,
                            urdf_path,
                            visual_format="dae",
                            collision_format="stl",
                        )
                    except Exception as e:
                        logger.warning("Mesh conversion failed (meshes will remain in original format): %s", e)

            if progress_callback:
                progress_callback("Generated MoveIt package", 4, 4)
        else:
            # Standard package metadata (ROS 1 catkin)
            if ros1:
                package_xml = generate_ros1_display_package_xml(self.package_name)
                cmake_content = generate_ros1_display_cmake_lists(self.package_name)
            else:
                package_xml = generate_package_xml(self.package_name)
                cmake_content = generate_cmake_lists(self.package_name, include_xacro=False)

            with open(package_dir / "package.xml", "w", encoding="utf-8") as f:
                f.write(package_xml)
            if progress_callback:
                progress_callback("Generated package.xml", 3, 4)

            with open(package_dir / "CMakeLists.txt", "w", encoding="utf-8") as f:
                f.write(cmake_content)
            if progress_callback:
                progress_callback("Generated CMakeLists.txt", 4, 4)

            if ros1:
                self._add_ros1_display_assets(package_dir, progress_callback)

        return package_dir

    def _add_ros1_display_assets(
        self,
        package_dir: Path,
        progress_callback: Optional[callable] = None,
    ) -> None:
        """Add launch/rviz files and convert meshes for a ROS1 display build.

        Mesh conversion uses :func:`_convert_urdf_meshes` from
        :mod:`skrobot.urdf.ros_config.moveit_package` (visual → DAE,
        collision → STL) so that RViz on ROS Noetic / ROS-O can load
        the assembly.  GLB files do not load in stock ROS1 RViz.
        """
        from skrobot.urdf.ros_config.moveit_package import _convert_urdf_meshes

        launch_dir = package_dir / "launch"
        launch_dir.mkdir(parents=True, exist_ok=True)
        rviz_dir = package_dir / "rviz"
        rviz_dir.mkdir(parents=True, exist_ok=True)

        launch_content = generate_ros1_display_launch(self.package_name)
        with open(launch_dir / "display.launch", "w", encoding="utf-8") as f:
            f.write(launch_content)
        if progress_callback:
            progress_callback("Generated display.launch", 1, 2)

        rviz_content = generate_ros1_rviz_config()
        with open(rviz_dir / "urdf.rviz", "w", encoding="utf-8") as f:
            f.write(rviz_content)
        if progress_callback:
            progress_callback("Generated urdf.rviz", 2, 2)

        urdf_path = package_dir / "urdf" / f"{self.package_name}.urdf"
        if urdf_path.exists():
            try:
                _convert_urdf_meshes(
                    urdf_path,
                    urdf_path,
                    visual_format="dae",
                    collision_format="stl",
                )
            except Exception as e:
                logger.warning(
                    "Mesh conversion to DAE/STL failed; ROS1 RViz may not render meshes: %s",
                    e,
                )

    def export_zip(
        self,
        output_path: Optional[str] = None,
        progress_callback: Optional[callable] = None,
        with_moveit_config: bool = False,
        ros1: bool = False,
    ) -> str:
        """
        Export the package as a ZIP archive.

        Parameters
        ----------
        output_path : str, optional
            Path for the output ZIP file. If None, creates in temp directory.
        progress_callback : callable, optional
            Progress callback function.
        with_moveit_config : bool
            If True, generate a MoveIt 2 package.
        ros1 : bool
            If True, generate a ROS1 catkin package with display.launch
            and DAE/STL meshes (see :meth:`build_package`).

        Returns
        -------
        str
            Path to the created ZIP file.
        """
        # Build the package first
        temp_dir = tempfile.mkdtemp(prefix=f"{self.package_name}_zip_")
        package_dir = self.build_package(
            temp_dir,
            progress_callback,
            with_moveit_config=with_moveit_config,
            ros1=ros1,
        )

        # Create ZIP
        if output_path is None:
            output_path = os.path.join(temp_dir, f"{self.package_name}.zip")

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob("*"):
                if file_path.is_file():
                    arcname = str(file_path.relative_to(Path(temp_dir)))
                    zipf.write(file_path, arcname)

        return output_path
