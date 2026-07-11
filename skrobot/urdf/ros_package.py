"""Generate the skeleton files of a ROS package around a URDF.

Generators for ``package.xml`` / ``CMakeLists.txt`` (plain and ROS 1
display variants), a minimal ``display.launch`` and its RViz
configuration, plus helpers that extract ``package://`` resource
references from URDF/xacro content.  Everything is plain string
templating on stdlib only.
"""

from __future__ import annotations

import re


__all__ = [
    'generate_package_xml',
    'generate_ros1_display_package_xml',
    'generate_cmake_lists',
    'generate_ros1_display_cmake_lists',
    'generate_ros1_display_launch',
    'generate_ros1_rviz_config',
    'extract_mesh_references',
    'extract_all_resource_references',
    'extract_registered_mesh_references',
    'extract_xacro_includes',
    'replace_package_references',
]


PACKAGE_XML_TEMPLATE = """\
<?xml version="1.0"?>
<package format="2">
  <name>{package_name}</name>
  <version>{version}</version>
  <description>{description}</description>

  <maintainer email="{maintainer_email}">{maintainer}</maintainer>
  <license>{license_name}</license>

  <buildtool_depend>catkin</buildtool_depend>

  <build_depend>rospy</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>urdf</build_depend>

  <run_depend>rospy</run_depend>
  <run_depend>std_msgs</run_depend>
  <run_depend>urdf</run_depend>
  <run_depend>robot_state_publisher</run_depend>
  <run_depend>joint_state_publisher</run_depend>

  <export>
    <architecture_independent/>
  </export>
</package>
"""


ROS1_DISPLAY_PACKAGE_XML_TEMPLATE = """\
<?xml version="1.0"?>
<package format="2">
  <name>{package_name}</name>
  <version>{version}</version>
  <description>{description}</description>

  <maintainer email="{maintainer_email}">{maintainer}</maintainer>
  <license>{license_name}</license>

  <buildtool_depend>catkin</buildtool_depend>

  <build_depend>rospy</build_depend>
  <build_depend>std_msgs</build_depend>
  <build_depend>urdf</build_depend>

  <exec_depend>rospy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>urdf</exec_depend>
  <exec_depend>robot_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher</exec_depend>
  <exec_depend>joint_state_publisher_gui</exec_depend>
  <exec_depend>rviz</exec_depend>
  <exec_depend>xacro</exec_depend>

  <export>
    <architecture_independent/>
  </export>
</package>
"""


CMAKE_LISTS_TEMPLATE = """\
cmake_minimum_required(VERSION 3.0.2)
project({package_name})

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  urdf
)

catkin_package()

install(DIRECTORY urdf/
  DESTINATION ${{CATKIN_PACKAGE_SHARE_DESTINATION}}/urdf
)
{xacro_install}
install(DIRECTORY meshes/
  DESTINATION ${{CATKIN_PACKAGE_SHARE_DESTINATION}}/meshes
)
"""

XACRO_INSTALL_BLOCK = """
install(DIRECTORY xacro/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/xacro
)
"""


ROS1_DISPLAY_CMAKE_LISTS_TEMPLATE = """\
cmake_minimum_required(VERSION 3.0.2)
project({package_name})

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  urdf
)

catkin_package()

install(DIRECTORY urdf/
  DESTINATION ${{CATKIN_PACKAGE_SHARE_DESTINATION}}/urdf
)

install(DIRECTORY meshes/
  DESTINATION ${{CATKIN_PACKAGE_SHARE_DESTINATION}}/meshes
)

install(DIRECTORY launch/
  DESTINATION ${{CATKIN_PACKAGE_SHARE_DESTINATION}}/launch
)

install(DIRECTORY rviz/
  DESTINATION ${{CATKIN_PACKAGE_SHARE_DESTINATION}}/rviz
)
"""


ROS1_DISPLAY_LAUNCH_TEMPLATE = """\
<launch>
  <arg name="model" default="$(find {package_name})/urdf/{package_name}.urdf"/>
  <arg name="rvizconfig" default="$(find {package_name})/rviz/urdf.rviz"/>
  <arg name="gui" default="true"/>

  <!-- The exporter inserts a <link name="world"/> as the URDF root,
       so robot_state_publisher already publishes the world TF tree. -->
  <param name="robot_description" textfile="$(arg model)"/>

  <node if="$(arg gui)" name="joint_state_publisher_gui"
        pkg="joint_state_publisher_gui" type="joint_state_publisher_gui"/>
  <node unless="$(arg gui)" name="joint_state_publisher"
        pkg="joint_state_publisher" type="joint_state_publisher"/>

  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>

  <node name="rviz" pkg="rviz" type="rviz" args="-d $(arg rvizconfig)" required="true"/>
</launch>
"""


ROS1_RVIZ_CONFIG = """\
Panels:
  - Class: rviz/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded: ~
      Splitter Ratio: 0.5
    Tree Height: 549
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.03
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz/RobotModel
      Collision Enabled: false
      Enabled: true
      Links:
        All Links Enabled: true
      Name: RobotModel
      Robot Description: robot_description
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz/TF
      Enabled: false
      Name: TF
      Value: false
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Default Light: true
    Fixed Frame: world
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz/Interact
      Hide Inactive Objects: true
    - Class: rviz/MoveCamera
    - Class: rviz/Select
    - Class: rviz/FocusCamera
    - Class: rviz/Measure
  Value: true
  Views:
    Current:
      Class: rviz/Orbit
      Distance: 1.5
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.06
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Field of View: 0.7853981852531433
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Name: Current View
      Near Clip Distance: 0.01
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Yaw: 0.7
  Window Geometry:
    Height: 800
    Width: 1200
"""


def generate_package_xml(
    package_name,
    version='1.0.0',
    description='Robot package generated by scikit-robot',
    maintainer='scikit-robot user',
    maintainer_email='user@example.com',
    license_name='MIT',
):
    """Generate ``package.xml`` content for a ROS package.

    Parameters
    ----------
    package_name : str
        Name of the ROS package.
    version : str
        Package version string.
    description : str
        Package description.
    maintainer : str
        Maintainer name.
    maintainer_email : str
        Maintainer email address.
    license_name : str
        License name (e.g., "MIT", "Apache-2.0", "BSD").

    Returns
    -------
    str
        XML content for ``package.xml``.
    """
    return PACKAGE_XML_TEMPLATE.format(
        package_name=package_name,
        version=version,
        description=description,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        license_name=license_name,
    )


def generate_ros1_display_package_xml(
    package_name,
    version='1.0.0',
    description='Robot package generated by scikit-robot',
    maintainer='scikit-robot user',
    maintainer_email='user@example.com',
    license_name='MIT',
):
    """Generate ``package.xml`` for a ROS 1 catkin package with display launch.

    Adds run dependencies on ``robot_state_publisher``,
    ``joint_state_publisher_gui`` and ``rviz`` so the generated
    ``display.launch`` can be executed via ``roslaunch``.

    Parameters
    ----------
    package_name : str
        Name of the ROS package.
    version : str
        Package version string.
    description : str
        Package description.
    maintainer : str
        Maintainer name.
    maintainer_email : str
        Maintainer email address.
    license_name : str
        License identifier.

    Returns
    -------
    str
        XML content for ``package.xml``.
    """
    return ROS1_DISPLAY_PACKAGE_XML_TEMPLATE.format(
        package_name=package_name,
        version=version,
        description=description,
        maintainer=maintainer,
        maintainer_email=maintainer_email,
        license_name=license_name,
    )


def generate_cmake_lists(package_name, include_xacro=True):
    """Generate ``CMakeLists.txt`` content for a ROS package.

    Parameters
    ----------
    package_name : str
        Name of the ROS package.
    include_xacro : bool
        Also install a ``xacro/`` directory.  Disable for packages
        without one -- ``catkin_make install`` fails on an install rule
        whose source directory does not exist.

    Returns
    -------
    str
        CMake content for ``CMakeLists.txt``.
    """
    xacro_install = XACRO_INSTALL_BLOCK if include_xacro else ''
    return CMAKE_LISTS_TEMPLATE.format(package_name=package_name,
                                       xacro_install=xacro_install)


def generate_ros1_display_cmake_lists(package_name):
    """Generate ``CMakeLists.txt`` for a ROS 1 display package.

    Adds ``install(DIRECTORY launch/ ...)`` and ``install(DIRECTORY
    rviz/ ...)`` rules in addition to the standard ``urdf/`` and
    ``meshes/`` rules so that ``catkin_make install`` ships the launch
    file and rviz config.

    Parameters
    ----------
    package_name : str
        Name of the ROS package.

    Returns
    -------
    str
        CMake content for ``CMakeLists.txt``.
    """
    return ROS1_DISPLAY_CMAKE_LISTS_TEMPLATE.format(package_name=package_name)


def generate_ros1_display_launch(package_name):
    """Generate a minimal ``display.launch`` for a ROS 1 package.

    The launch file starts ``robot_state_publisher``,
    ``joint_state_publisher_gui`` (or ``joint_state_publisher`` when
    ``gui:=false``) and ``rviz`` with a default configuration.  It
    expects the URDF at ``urdf/<package_name>.urdf`` and assumes the
    URDF's root frame is what rviz should use as its Fixed Frame.

    Parameters
    ----------
    package_name : str
        Name of the ROS package -- used in ``$(find ...)`` references.

    Returns
    -------
    str
        XML content for ``launch/display.launch``.
    """
    return ROS1_DISPLAY_LAUNCH_TEMPLATE.format(package_name=package_name)


def generate_ros1_rviz_config():
    """Return the default RViz configuration for the display launch."""
    return ROS1_RVIZ_CONFIG


def extract_mesh_references(urdf_content):
    """Extract mesh file references from URDF content.

    Parameters
    ----------
    urdf_content : str
        URDF XML content.

    Returns
    -------
    set of str
        Mesh file paths (relative paths from the ``meshes/`` directory).
    """
    mesh_refs = set()

    # Pattern for package:// URLs
    package_pattern = r'filename="package://[^/]+/meshes/([^"]+)"'
    mesh_refs.update(re.findall(package_pattern, urdf_content))

    # Pattern for file:// URLs
    file_pattern = r'filename="file://[^"]*meshes/([^"]+)"'
    mesh_refs.update(re.findall(file_pattern, urdf_content))

    # Pattern for absolute paths containing meshes/
    abs_pattern = r'filename="[^"]*meshes/([^"]+)"'
    mesh_refs.update(re.findall(abs_pattern, urdf_content))

    return mesh_refs


def extract_all_resource_references(urdf_content):
    """Extract all file references from URDF content (meshes, textures, ...).

    Extracts the path relative to the package root from any
    ``filename="package://<pkg>/<relative_path>"`` attribute.

    Parameters
    ----------
    urdf_content : str
        URDF XML content.

    Returns
    -------
    set of str
        Relative paths from the package root
        (e.g., ``materials/textures/foo.png``, ``meshes/visual/bar.stl``).
    """
    pattern = r'filename="package://[^/]+/([^"]+)"'
    refs = set(re.findall(pattern, urdf_content))
    # Exclude registered/ paths (handled separately)
    return {r for r in refs if not r.startswith('registered/')}


def extract_registered_mesh_references(urdf_content):
    """Extract mesh references for registered (uploaded) modules.

    These have paths like
    ``package://pkg/registered/<hash>/<name>/meshes/<subpath>``.

    Parameters
    ----------
    urdf_content : str
        URDF XML content.

    Returns
    -------
    set of str
        Relative paths from the assets root
        (e.g., ``registered/<hash>/<name>/meshes/visual/link0.dae``).
    """
    pattern = r'filename="package://[^/]+/(registered/[^"]*meshes/[^"]+)"'
    return set(re.findall(pattern, urdf_content))


def extract_xacro_includes(xacro_content):
    """Extract xacro include references from xacro content.

    Parameters
    ----------
    xacro_content : str
        Xacro XML content.

    Returns
    -------
    set of str
        Xacro file names referenced by includes.
    """
    includes = set()

    # Pattern for $(find package)/xacro/*.xacro
    find_pattern = r'<xacro:include filename="\$\(find [^)]+\)/xacro/([^"]+)"'
    includes.update(re.findall(find_pattern, xacro_content))

    # Pattern for absolute paths
    abs_pattern = r'<xacro:include filename="[^"]+/xacro/([^"]+)"'
    includes.update(re.findall(abs_pattern, xacro_content))

    return includes


def replace_package_references(content, old_package, new_package):
    """Replace ROS package references in URDF/Xacro content.

    Rewrites ``package://old_package/`` URIs and ``$(find old_package)``
    substitutions only, so other occurrences of the name -- including
    other package names that merely contain it as a substring -- are
    left untouched.

    Parameters
    ----------
    content : str
        Original content.
    old_package : str
        Original package name to replace.
    new_package : str
        New package name.

    Returns
    -------
    str
        Content with replaced package references.
    """
    escaped = re.escape(old_package)
    content = re.sub('package://{}/'.format(escaped),
                     'package://{}/'.format(new_package), content)
    content = re.sub(r'\$\(find {}\)'.format(escaped),
                     '$(find {})'.format(new_package), content)
    return content
