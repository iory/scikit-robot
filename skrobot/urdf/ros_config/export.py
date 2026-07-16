"""
Configuration export utilities.

Creates ZIP archives containing all configuration files.
"""


from __future__ import annotations

import io
from typing import Any
import zipfile

from skrobot.urdf.ros_config.gazebo_generator import generate_gazebo_config
from skrobot.urdf.ros_config.gazebo_generator import generate_ros2_control_xacro
from skrobot.urdf.ros_config.moveit_generator import generate_controllers_yaml
from skrobot.urdf.ros_config.moveit_generator import generate_srdf


def export_all_configs(
    urdf_content: str,
    joints: list[dict[str, Any]],
    planning_groups: list[dict[str, Any]],
    controllers: list[dict[str, Any]],
    disabled_collision_pairs: list[tuple[str, str]],
    gazebo_physics: dict[str, Any],
    gazebo_plugins: list[dict[str, Any]],
    robot_name: str = "robot",
    export_options: dict[str, bool] | None = None,
    extra_files: dict[str, str] | None = None,
) -> bytes:
    """
    Export all configuration files as a ZIP archive.

    Parameters
    ----------
    urdf_content : str
        Original URDF content.
    joints : list
        Parsed joint information.
    planning_groups : list
        MoveIt planning group configurations.
    controllers : list
        Controller configurations.
    disabled_collision_pairs : list
        List of disabled collision link pairs.
    gazebo_physics : dict
        Gazebo physics settings.
    gazebo_plugins : list
        Gazebo plugin configurations.
    robot_name : str
        Name of the robot for file naming.
    extra_files : dict, optional
        Additional archive entries: relative path inside the ZIP mapped
        to file content.  Lets callers bundle their own configuration
        files alongside the generated ones.

    Returns
    -------
    bytes
        ZIP file contents as bytes.
    """
    # Default export options if not provided
    if export_options is None:
        export_options = {
            "includeUrdf": True,
            "includeMoveIt": True,
            "includeGazebo": True,
        }

    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # URDF (original)
        if export_options.get("includeUrdf", True):
            zf.writestr(f"{robot_name}/urdf/{robot_name}.urdf", urdf_content)

        # MoveIt SRDF and controllers
        if export_options.get("includeMoveIt", True):
            srdf_content = generate_srdf(
                robot_name=robot_name,
                planning_groups=planning_groups,
                disabled_collision_pairs=[tuple(p) for p in disabled_collision_pairs],
            )
            zf.writestr(f"{robot_name}/config/{robot_name}.srdf", srdf_content)

            controllers_yaml = generate_controllers_yaml(controllers)
            zf.writestr(f"{robot_name}/config/controllers.yaml", controllers_yaml)

        # Gazebo config
        if export_options.get("includeGazebo", True):
            gazebo_config = generate_gazebo_config(gazebo_physics, gazebo_plugins)
            zf.writestr(f"{robot_name}/config/gazebo.xml", gazebo_config)

            ros2_control_xacro = generate_ros2_control_xacro(
                joints, package_name=robot_name)
            zf.writestr(f"{robot_name}/urdf/ros2_control.xacro", ros2_control_xacro)

        # Caller-provided extra files
        if extra_files:
            for path, content in extra_files.items():
                zf.writestr(path, content)

        # README (always include)
        readme = _generate_readme(robot_name, export_options,
                                   extra_files)
        zf.writestr(f"{robot_name}/README.md", readme)

    return zip_buffer.getvalue()


def _generate_readme(
    robot_name: str,
    export_options: dict[str, bool],
    extra_files: dict[str, str] | None = None,
) -> str:
    """Generate a README file for the configuration package."""
    contents = []
    usage_sections = []

    if export_options.get("includeUrdf", True):
        contents.append(f"- `urdf/{robot_name}.urdf` - Robot URDF description")

    if export_options.get("includeGazebo", True):
        contents.append("- `urdf/ros2_control.xacro` - ros2_control hardware interface configuration")
        contents.append("- `config/gazebo.xml` - Gazebo physics and plugin configuration")
        usage_sections.append(f"""### Gazebo

Include the ros2_control xacro in your robot description:

```xml
<xacro:include filename="$(find {robot_name})/urdf/ros2_control.xacro" />
```""")

    if export_options.get("includeMoveIt", True):
        contents.append(f"- `config/{robot_name}.srdf` - MoveIt2 SRDF (semantic robot description)")
        contents.append("- `config/controllers.yaml` - ros2_control controller configuration")
        usage_sections.append(f"""### MoveIt2

Include the SRDF in your MoveIt2 launch files:

```python
srdf_file = os.path.join(pkg_share, 'config', '{robot_name}.srdf')
```

### Controllers

Load the controller configuration:

```yaml
ros2_control_node:
  ros__parameters:
    robot_description: $(command 'cat $(find {robot_name})/urdf/{robot_name}.urdf')
```""")

    for path in (extra_files or {}):
        contents.append(f"- `{path}`")

    contents_str = "\n".join(contents)
    usage_str = "\n\n".join(usage_sections)

    return f"""# {robot_name} Configuration Package

Generated by scikit-robot.

## Contents

{contents_str}

## Usage

{usage_str}
"""
