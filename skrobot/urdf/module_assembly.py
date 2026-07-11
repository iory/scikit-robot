"""Assemble multiple URDF modules into one robot by connecting named ports.

The architecture follows a 3-layer design:
1. RobotModule: catalog data for an individual URDF part
2. RobotAssembly: the undirected graph of module connections
3. skrobot.model.RobotModel: the final assembled robot for FK computation

``RobotAssembly.build()`` runs the ``zacro`` command-line tool (a pure-Python
xacro processor, ``pip install zacro``) to expand the generated master xacro
into the combined URDF.
"""

from collections import deque
from dataclasses import dataclass
from dataclasses import field
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import xml.etree.ElementTree as ET

from lxml import etree
import numpy as np

from skrobot.coordinates.math import matrix2xyzrpy
from skrobot.coordinates.math import matrix_relative
from skrobot.coordinates.math import rpy2matrix
from skrobot.coordinates.math import xyzrpy2matrix
from skrobot.urdf.modularize_urdf import transform_urdf_to_macro


@dataclass
class Port:
    """
    Represents a connection port on a robot module.

    A port is a link in the URDF that can be connected to another module's port.
    Typically these are "dummy_link" elements in the URDF files.

    Attributes
    ----------
    name : str
        The link name in the URDF that serves as a connection point.
    description : str, optional
        Human-readable description of this port's purpose.
    port_type : str, optional
        Type of port: "input", "output", or "bidirectional".
        - "input": Can only receive connections (e.g., base_link)
        - "output": Can only provide connections (e.g., end effector)
        - "bidirectional": Can be used as either input or output
    compatible_types : List[str], optional
        List of compatible port type names for connection validation.
    z_axis : Tuple[float, float, float], optional
        The Z-axis direction of this port in the module's root frame.
        Used for automatic port alignment when connecting modules.
        Default is (0, 0, 1) meaning Z points "up".
    """

    name: str
    description: str = ""
    port_type: str = "bidirectional"  # "input", "output", "bidirectional"
    compatible_types: List[str] = field(default_factory=list)
    z_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class RobotModule:
    """
    Represents an individual URDF part (module) in the robot catalog.

    This class does NOT hold parent-child relationships. It simply stores
    metadata about a single URDF file and its available connection ports.

    Attributes
    ----------
    module_id : str
        Unique identifier for this module type (e.g., "hinge_module", "branch_module").
    urdf_path : Path
        Path to the URDF file for this module.
    ports : List[Port]
        List of available connection ports (links that can connect to other modules).
    root_link : str
        The root link name in the original URDF (usually "base_link").
    xacro_path : Path, optional
        Path to the xacro macro file (if using existing xacro instead of URDF).
    macro_name : str, optional
        Name of the xacro macro to use.

    Examples
    --------
    >>> hinge = RobotModule(
    ...     module_id="hinge",
    ...     urdf_path=Path("/path/to/screw_hinge_module.urdf"),
    ...     ports=[Port("base_link"), Port("dummy_link1")],
    ...     root_link="base_link"
    ... )
    """

    module_id: str
    urdf_path: Path
    ports: List[Port] = field(default_factory=list)
    root_link: str = "base_link"
    xacro_path: Optional[Path] = None
    macro_name: Optional[str] = None

    @classmethod
    def from_urdf(cls, module_id: str, urdf_path: str) -> "RobotModule":
        """
        Create a RobotModule by parsing a URDF file.

        This factory method automatically extracts:
        - The root link name
        - Available ports (dummy_link* patterns)
        - Z-axis direction for each port (for auto-alignment)

        Parameters
        ----------
        module_id : str
            Unique identifier for this module.
        urdf_path : str
            Path to the URDF file.

        Returns
        -------
        RobotModule
            A new RobotModule instance with extracted port information.
        """
        path = Path(urdf_path)
        xml_root = ET.parse(path).getroot()
        return cls._build_from_xml(module_id, xml_root, path)

    @classmethod
    def from_urdf_string(
        cls,
        module_id: str,
        urdf_content: str,
        urdf_path: str = "",
    ) -> "RobotModule":
        """
        Create a RobotModule by parsing URDF content from a string.

        This method is useful when loading URDF from cloud storage
        where the content is already loaded into memory.

        Parameters
        ----------
        module_id : str
            Unique identifier for this module.
        urdf_content : str
            The URDF XML content as a string.
        urdf_path : str, optional
            Original path of the URDF file (for reference).

        Returns
        -------
        RobotModule
            A new RobotModule instance with extracted port information.
        """
        import io

        xml_root = ET.parse(io.StringIO(urdf_content)).getroot()
        path = Path(urdf_path) if urdf_path else Path(".")
        return cls._build_from_xml(module_id, xml_root, path)

    @classmethod
    def _build_from_xml(cls, module_id: str, xml_root, urdf_path: Path) -> "RobotModule":
        """
        Build a RobotModule from a parsed XML element tree root.

        Parameters
        ----------
        module_id : str
            Unique identifier for this module.
        xml_root : Element
            Root element of the parsed URDF XML.
        urdf_path : Path
            Path to the URDF file (or placeholder).

        Returns
        -------
        RobotModule
            A new RobotModule instance with extracted port information.
        """
        # Find all link names
        links = [link.get("name") for link in xml_root.findall(".//link")]

        # Find the root link (parent but never a child)
        parent_links = set()
        child_links = set()
        for joint in xml_root.findall(".//joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None:
                parent_links.add(parent.get("link"))
            if child is not None:
                child_links.add(child.get("link"))

        root_links = parent_links - child_links
        root_link = list(root_links)[0] if root_links else (links[0] if links else "base_link")

        # Build joint graph for Z-axis calculation
        joint_graph = {}  # child_link -> (parent_link, joint_element)
        for joint in xml_root.findall(".//joint"):
            parent = joint.find("parent")
            child = joint.find("child")
            if parent is not None and child is not None:
                joint_graph[child.get("link")] = (parent.get("link"), joint)

        # Calculate Z-axis for each link
        port_z_axes = cls._calculate_port_z_axes(root_link, links, joint_graph)

        # Identify ports (all links can potentially be ports)
        ports = [Port(name=link, z_axis=port_z_axes.get(link, (0.0, 0.0, 1.0))) for link in links]

        module = cls(
            module_id=module_id,
            urdf_path=urdf_path,
            ports=ports,
            root_link=root_link,
        )

        # Auto-detect port types based on naming conventions
        for port in module.ports:
            if port.name == root_link:
                port.port_type = "input"
                port.description = "Root link (default input)"
            elif port.name.startswith("dummy_link"):
                port.port_type = "bidirectional"
                port.description = "Connection port"

        return module

    @staticmethod
    def _calculate_port_z_axes(root_link, links, joint_graph):
        """
        Calculate the Z-axis direction for each link in the module's root frame.

        Parameters
        ----------
        root_link : str
            Name of the root link.
        links : List[str]
            List of all link names.
        joint_graph : Dict[str, Tuple[str, Element]]
            Mapping from child_link to (parent_link, joint_element).

        Returns
        -------
        Dict[str, Tuple[float, float, float]]
            Mapping from link name to Z-axis direction in root frame.
        """
        z_axes = {}

        for link in links:
            if link == root_link:
                z_axes[link] = (0.0, 0.0, 1.0)
                continue

            # Trace path from link to root
            path = []
            current = link
            while current in joint_graph:
                parent, joint = joint_graph[current]
                path.append(joint)
                current = parent
                if current == root_link:
                    break

            if current != root_link:
                # Link not connected to root
                z_axes[link] = (0.0, 0.0, 1.0)
                continue

            # Compute cumulative rotation from root to link
            # We need to compose rotations in reverse order (from root to link)
            rotation = np.eye(3)
            for joint in reversed(path):
                origin = joint.find("origin")
                if origin is not None:
                    rpy_str = origin.get("rpy", "0 0 0")
                    roll, pitch, yaw = [float(x) for x in rpy_str.split()]
                    joint_rot = rpy2matrix(roll, pitch, yaw)
                    rotation = rotation @ joint_rot

            # Transform Z-axis by cumulative rotation
            z_local = np.array([0.0, 0.0, 1.0])
            z_transformed = rotation @ z_local
            z_axes[link] = tuple(z_transformed.tolist())

        return z_axes

    def get_port_names(self) -> List[str]:
        """Return list of all available port names."""
        return [port.name for port in self.ports]


@dataclass
class Connection:
    """
    Represents a connection between two module ports.

    This is an edge in the undirected connection graph.

    Attributes
    ----------
    module_a : str
        Instance ID of the first module.
    port_a : str
        Port (link) name on module_a.
    module_b : str
        Instance ID of the second module.
    port_b : str
        Port (link) name on module_b.
    x : float
        X position offset from port_a to port_b (in port_a's frame).
    y : float
        Y position offset from port_a to port_b.
    z : float
        Z position offset from port_a to port_b.
    roll : float
        Roll (rotation around X axis) offset in radians.
    pitch : float
        Pitch (rotation around Y axis) offset in radians.
    yaw : float
        Yaw (rotation around Z axis) offset in radians.
    """

    module_a: str
    port_a: str
    module_b: str
    port_b: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def xyz(self) -> Tuple[float, float, float]:
        """Return position offset as tuple."""
        return (self.x, self.y, self.z)

    @property
    def rpy(self) -> Tuple[float, float, float]:
        """Return orientation offset as tuple."""
        return (self.roll, self.pitch, self.yaw)

    def involves(self, module_instance_id: str) -> bool:
        """Check if this connection involves the given module instance."""
        return self.module_a == module_instance_id or self.module_b == module_instance_id


@dataclass
class ModuleInstance:
    """
    Represents an instance of a RobotModule placed in the assembly.

    Multiple instances of the same module type can exist in an assembly.

    Attributes
    ----------
    instance_id : str
        Unique identifier for this instance (e.g., "arm_left", "arm_right").
    module : RobotModule
        Reference to the module definition.
    """

    instance_id: str
    module: RobotModule


@dataclass
class TreeNode:
    """
    Represents a node in the directed tree built from the assembly graph.

    Used internally by build() to track the tree structure.
    """

    instance_id: str
    parent_instance: Optional[str]
    parent_port: Optional[str]  # Port on parent that connects to this node
    child_port: Optional[str]  # Port on this module that connects to parent
    connection_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    connection_rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    children: List["TreeNode"] = field(default_factory=list)
    is_reversed: bool = False  # True if module's root was changed


class RobotAssembly:
    """
    Manages the assembly state of multiple robot modules.

    This class represents the undirected graph of module connections that
    the user has assembled through the UI. It does NOT represent the final
    kinematic tree - that is generated by the build() method.

    The assembly graph is stored as:
    - A dictionary of module instances (nodes)
    - A list of connections (edges)

    Attributes
    ----------
    name : str
        Name of the assembled robot.
    instances : Dict[str, ModuleInstance]
        Dictionary mapping instance_id to ModuleInstance.
    connections : List[Connection]
        List of all connections between module ports.
    root_instance : Optional[str]
        The instance ID that will become the root of the final URDF tree.
    root_port : Optional[str]
        The port on root_instance that will become the root link.

    Examples
    --------
    >>> # Create module definitions
    >>> hinge = RobotModule.from_urdf("hinge", "screw_hinge_module.urdf")
    >>> branch = RobotModule.from_urdf("branch", "screw_branch_module.urdf")
    >>>
    >>> # Create assembly
    >>> assembly = RobotAssembly("my_robot")
    >>> assembly.add_module_instance("base", branch)
    >>> assembly.add_module_instance("arm1", hinge)
    >>> assembly.add_module_instance("arm2", hinge)
    >>>
    >>> # Connect modules
    >>> assembly.connect("base", "dummy_link1", "arm1", "base_link")
    >>> assembly.connect("base", "dummy_link2", "arm2", "base_link")
    >>>
    >>> # Set root and build
    >>> assembly.set_root("base", "base_link")
    >>> combined_urdf = assembly.build()
    """

    def __init__(self, name: str):
        """
        Initialize a new RobotAssembly.

        Parameters
        ----------
        name : str
            Name for the assembled robot.
        """
        self.name: str = name
        self.instances: Dict[str, ModuleInstance] = {}
        self.connections: List[Connection] = []
        self.root_instance: Optional[str] = None
        self.root_port: Optional[str] = None

    def add_module_instance(self, instance_id: str, module: RobotModule) -> None:
        """
        Add a module instance to the assembly.

        Parameters
        ----------
        instance_id : str
            Unique identifier for this instance.
        module : RobotModule
            The module definition to instantiate.

        Raises
        ------
        ValueError
            If instance_id already exists.
        """
        if instance_id in self.instances:
            raise ValueError(f"Instance '{instance_id}' already exists in assembly")
        self.instances[instance_id] = ModuleInstance(instance_id=instance_id, module=module)

    def connect(
        self,
        module_a: str,
        port_a: str,
        module_b: str,
        port_b: str,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        """
        Create a connection between two module ports.

        Parameters
        ----------
        module_a : str
            Instance ID of the first module.
        port_a : str
            Port name on module_a.
        module_b : str
            Instance ID of the second module.
        port_b : str
            Port name on module_b.
        x : float, optional
            X position offset from port_a to port_b in meters (default: 0.0).
        y : float, optional
            Y position offset from port_a to port_b in meters (default: 0.0).
        z : float, optional
            Z position offset from port_a to port_b in meters (default: 0.0).
        roll : float, optional
            Roll (rotation around X axis) offset in radians (default: 0.0).
        pitch : float, optional
            Pitch (rotation around Y axis) offset in radians (default: 0.0).
        yaw : float, optional
            Yaw (rotation around Z axis) offset in radians (default: 0.0).

        Raises
        ------
        ValueError
            If either module instance doesn't exist, or if ports are invalid.

        Examples
        --------
        >>> assembly.connect('h1', 'dummy_link', 'p1', 'base_link')
        >>> assembly.connect('h1', 'dummy_link', 'p2', 'base_link', yaw=1.57)
        """
        # Validate module instances exist
        if module_a not in self.instances:
            raise ValueError(f"Module instance '{module_a}' not found")
        if module_b not in self.instances:
            raise ValueError(f"Module instance '{module_b}' not found")

        # Validate ports exist on modules
        mod_a = self.instances[module_a].module
        mod_b = self.instances[module_b].module
        if port_a not in mod_a.get_port_names():
            raise ValueError(f"Port '{port_a}' not found on module '{module_a}'")
        if port_b not in mod_b.get_port_names():
            raise ValueError(f"Port '{port_b}' not found on module '{module_b}'")

        # Check for duplicate connections
        for conn in self.connections:
            if (
                conn.module_a == module_a
                and conn.port_a == port_a
                and conn.module_b == module_b
                and conn.port_b == port_b
            ):
                raise ValueError("Connection already exists")
            if (
                conn.module_a == module_b
                and conn.port_a == port_b
                and conn.module_b == module_a
                and conn.port_b == port_a
            ):
                raise ValueError("Connection already exists (reverse)")

        self.connections.append(
            Connection(module_a, port_a, module_b, port_b, x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        )

    def set_root(self, instance_id: str, port_name: str) -> None:
        """
        Set the root link for the assembled robot.

        The specified port will become the root of the final kinematic tree.

        Parameters
        ----------
        instance_id : str
            Instance ID of the module containing the root.
        port_name : str
            Port (link) name to use as root.
        """
        if instance_id not in self.instances:
            raise ValueError(f"Module instance '{instance_id}' not found")
        module = self.instances[instance_id].module
        if port_name not in module.get_port_names():
            raise ValueError(f"Port '{port_name}' not found on module '{instance_id}'")

        self.root_instance = instance_id
        self.root_port = port_name

    def get_adjacency_list(self) -> Dict[str, List[Tuple[str, str, str, Tuple, Tuple]]]:
        """
        Get the assembly as an adjacency list.

        Returns
        -------
        Dict[str, List[Tuple[str, str, str, Tuple, Tuple]]]
            Maps instance_id to list of (connected_instance_id, this_port, other_port, xyz, rpy).
            When traversed from module_b to module_a, the connection
            transform is inverted (rigid inverse, not a component-wise
            negation -- the latter is only correct for single-axis
            rotations).
        """
        adj: Dict[str, List[Tuple[str, str, str, Tuple, Tuple]]] = {inst_id: [] for inst_id in self.instances}
        identity = np.eye(4)
        for conn in self.connections:
            # Forward direction: module_a -> module_b uses original xyz/rpy
            adj[conn.module_a].append((conn.module_b, conn.port_a, conn.port_b, conn.xyz, conn.rpy))
            # Reverse direction: module_b -> module_a uses the inverse transform
            transform = xyzrpy2matrix(conn.xyz, conn.rpy)
            inv_xyz, inv_rpy = matrix2xyzrpy(matrix_relative(transform, identity))
            adj[conn.module_b].append(
                (conn.module_a, conn.port_b, conn.port_a,
                 tuple(float(v) for v in inv_xyz),
                 tuple(float(v) for v in inv_rpy)))
        return adj

    def _build_forest_from_graph(self) -> Tuple[List[TreeNode], Dict[str, TreeNode]]:
        """
        Build a forest (multiple trees) from the graph, including disconnected components.

        This handles the case where there are multiple independent module trees
        that are not connected to each other.

        Returns
        -------
        Tuple[List[TreeNode], Dict[str, TreeNode]]
            (list of root nodes, dict mapping instance_id to TreeNode)
        """
        adj = self.get_adjacency_list()
        visited = set()
        node_map: Dict[str, TreeNode] = {}
        root_nodes: List[TreeNode] = []

        # Start with the designated root if set
        if self.root_instance is not None:
            root_node = TreeNode(
                instance_id=self.root_instance, parent_instance=None, parent_port=None, child_port=self.root_port
            )
            node_map[self.root_instance] = root_node
            visited.add(self.root_instance)
            root_nodes.append(root_node)

            # BFS to build tree from designated root
            queue = deque([root_node])
            while queue:
                current = queue.popleft()
                for neighbor_id, my_port, neighbor_port, xyz, rpy in adj[current.instance_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        child_node = TreeNode(
                            instance_id=neighbor_id,
                            parent_instance=current.instance_id,
                            parent_port=my_port,
                            child_port=neighbor_port,
                            connection_xyz=xyz,
                            connection_rpy=rpy,
                        )
                        current.children.append(child_node)
                        node_map[neighbor_id] = child_node
                        queue.append(child_node)

        # Find any remaining unvisited instances (disconnected components)
        for instance_id in self.instances:
            if instance_id not in visited:
                # This instance is part of a disconnected component
                # Find root_link of this module to use as default port
                instance = self.instances[instance_id]
                default_port = instance.module.root_link

                orphan_root = TreeNode(
                    instance_id=instance_id, parent_instance=None, parent_port=None, child_port=default_port
                )
                node_map[instance_id] = orphan_root
                visited.add(instance_id)
                root_nodes.append(orphan_root)

                # BFS to build tree from this orphan root
                queue = deque([orphan_root])
                while queue:
                    current = queue.popleft()
                    for neighbor_id, my_port, neighbor_port, xyz, rpy in adj[current.instance_id]:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            child_node = TreeNode(
                                instance_id=neighbor_id,
                                parent_instance=current.instance_id,
                                parent_port=my_port,
                                child_port=neighbor_port,
                                connection_xyz=xyz,
                                connection_rpy=rpy,
                            )
                            current.children.append(child_node)
                            node_map[neighbor_id] = child_node
                            queue.append(child_node)

        return root_nodes, node_map

    def build(self, output_path: Optional[str] = None) -> str:
        """
        Build a combined URDF from the assembly.

        This method:
        1. Traverses all connected components (forest) in the graph
        2. Uses modularize_urdf to add prefixes and create Xacro macros
        3. Combines all macros into a single URDF with proper connections

        Note: This method does NOT reverse kinematic chains. Each module
        maintains its original structure (base_link -> ... -> dummy_link),
        and modules are connected via fixed joints from parent's port to
        child's base_link.

        Parameters
        ----------
        output_path : str, optional
            Path to save the combined URDF. If None, uses a temporary file.

        Returns
        -------
        str
            Path to the generated combined URDF file.

        Raises
        ------
        ValueError
            If assembly is empty.
        """
        # Step 0: Validation
        if not self.instances:
            raise ValueError("Assembly is empty - add module instances first")

        # Auto-set root if not set (use first instance)
        if self.root_instance is None:
            first_instance_id = next(iter(self.instances))
            first_instance = self.instances[first_instance_id]
            self.root_instance = first_instance_id
            self.root_port = first_instance.module.root_link

        # Create temp directory for intermediate files
        temp_dir = tempfile.mkdtemp(prefix=f"{self.name}_build_")

        # Step 1: Build forest (multiple trees) from graph
        root_nodes, node_map = self._build_forest_from_graph()

        # Step 2: Process each module - create Xacro macro (no root link changes)
        xacro_files = {}  # instance_id -> (xacro_path, macro_name, connector_link)

        # BFS order to process all modules from all trees
        process_order = []
        for root_node in root_nodes:
            queue = deque([root_node])
            while queue:
                node = queue.popleft()
                process_order.append(node)
                for child in node.children:
                    queue.append(child)

        for node in process_order:
            instance = self.instances[node.instance_id]
            module = instance.module

            # Modules always attach through their root link (typically
            # base_link) so each keeps its original kinematic structure.
            # node.child_port records which child-side port the connection
            # was declared with (callers use it for placement math); the
            # attachment link is nevertheless the root, with the connection
            # transform expressed for the child root frame.
            connector_link = module.root_link

            # Use original URDF directly (no kinematic chain reversal)
            urdf_to_process = str(module.urdf_path)

            # No reversal needed
            node.is_reversed = False

            # Transform to Xacro macro with prefix
            # Use module_id for macro if instance_id is empty
            macro_base = node.instance_id if node.instance_id else module.module_id
            macro_name = f"{macro_base}_macro"
            xacro_element, final_macro_name = transform_urdf_to_macro(
                urdf_to_process, connector_link=connector_link, no_prefix=False, macro_name=macro_name
            )

            # Save Xacro file
            xacro_path = os.path.join(temp_dir, f"{node.instance_id}.xacro")
            with open(xacro_path, "wb") as f:
                f.write(etree.tostring(xacro_element, pretty_print=True, xml_declaration=True, encoding="UTF-8"))

            xacro_files[node.instance_id] = (xacro_path, final_macro_name, connector_link)

        # Step 3: Create master Xacro that combines all modules
        XACRO_NS = "http://ros.org/wiki/xacro"
        NSMAP = {"xacro": XACRO_NS}

        master_root = etree.Element("robot", nsmap=NSMAP)
        master_root.set("name", self.name)

        # Add world link as the absolute root
        world_link = etree.SubElement(master_root, "link")
        world_link.set("name", "world")

        # Include all xacro files
        for _, (xacro_path, _, _) in xacro_files.items():
            include = etree.SubElement(master_root, f"{{{XACRO_NS}}}include")
            include.set("filename", xacro_path)

        # Instantiate macros in tree order
        for node in process_order:
            instance_id = node.instance_id
            xacro_path, macro_name, connector_link = xacro_files[instance_id]

            # Determine parent link for this macro
            if node.parent_instance is None:
                parent_link = "world"
            else:
                parent_inst_id = node.parent_instance
                parent_port = node.parent_port
                # Parent link is prefixed with parent's instance_id (if non-empty)
                parent_link = f"{parent_inst_id}_{parent_port}" if parent_inst_id else parent_port

            # Connection rpy
            connection_rpy = list(node.connection_rpy)

            # Create macro instantiation with connection offset
            macro_call = etree.SubElement(master_root, f"{{{XACRO_NS}}}{macro_name}")
            # Only add prefix with underscore if instance_id is non-empty
            prefix_value = f"{instance_id}_" if instance_id else ""
            macro_call.set("prefix", prefix_value)
            macro_call.set("parent_link", parent_link)
            xyz_str = " ".join(str(x) for x in node.connection_xyz)
            rpy_str = " ".join(str(x) for x in connection_rpy)
            macro_call.set("xyz", xyz_str)
            macro_call.set("rpy", rpy_str)

        # Save master xacro
        master_xacro_path = os.path.join(temp_dir, f"{self.name}_master.xacro")
        with open(master_xacro_path, "wb") as f:
            f.write(etree.tostring(master_root, pretty_print=True, xml_declaration=True, encoding="UTF-8"))

        # Step 4: Process Xacro to final URDF
        if output_path is None:
            output_path = os.path.join(temp_dir, f"{self.name}.urdf")

        try:
            result = subprocess.run(
                ["zacro", master_xacro_path, "-o", output_path],
                capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "building an assembly needs the optional 'zacro' xacro "
                "processor -- install it with: pip install zacro")
        if result.returncode != 0:
            raise RuntimeError(f"Zacro processing failed: {result.stderr}")

        return output_path

    def to_dict(self) -> dict:
        """
        Serialize the assembly to a dictionary.

        Useful for saving/loading assembly state or sending to frontend.
        """
        return {
            "name": self.name,
            "instances": {
                inst_id: {"module_id": inst.module.module_id, "urdf_path": str(inst.module.urdf_path)}
                for inst_id, inst in self.instances.items()
            },
            "connections": [
                {"module_a": c.module_a, "port_a": c.port_a,
                 "module_b": c.module_b, "port_b": c.port_b,
                 "x": c.x, "y": c.y, "z": c.z,
                 "roll": c.roll, "pitch": c.pitch, "yaw": c.yaw}
                for c in self.connections
            ],
            "root": {"instance": self.root_instance, "port": self.root_port} if self.root_instance else None,
        }

    def __repr__(self) -> str:
        return (
            f"RobotAssembly(name={self.name!r}, "
            f"instances={len(self.instances)}, "
            f"connections={len(self.connections)}, "
            f"root={self.root_instance}:{self.root_port})"
        )
