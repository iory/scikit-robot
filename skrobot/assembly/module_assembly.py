"""Assemble multiple URDF modules into one robot by connecting named ports.

The architecture follows a 3-layer design:
1. RobotModule: catalog data for an individual URDF part
2. RobotAssembly: the undirected graph of module connections
3. skrobot.model.RobotModel: the final assembled robot for FK computation

``RobotAssembly.build()`` composes the combined URDF purely in memory --
no external tools are involved.

A closed linkage (four-bar, parallel mechanism) does not fit a URDF tree;
declare its cut edge with ``connect(..., loop=True)`` and ``build()`` will
export the closure as a ``loop_closures.yaml`` sidecar for runtime IK,
plus exact ``<mimic>`` tags when the loop is a parallelogram.
"""

from collections import deque
from dataclasses import dataclass
from dataclasses import field
import math
import os
from pathlib import Path
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
from skrobot.coordinates.math import rotation_matrix
from skrobot.coordinates.math import rpy2homogeneous
from skrobot.coordinates.math import xyzrpy2matrix
from skrobot.urdf.xml_root_link_changer import change_urdf_root_link


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
    xyz : Tuple[float, float, float], optional
        The port position in the module's root frame (composed from the
        joint origins along the root-to-port chain at the zero pose).
    rpy : Tuple[float, float, float], optional
        The port orientation (URDF roll-pitch-yaw) in the module's root
        frame, so a port is a full SE(3) frame -- like a keyed connector,
        mating is then fully determined.
    """

    name: str
    description: str = ""
    port_type: str = "bidirectional"  # "input", "output", "bidirectional"
    compatible_types: List[str] = field(default_factory=list)
    z_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)


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
        port_frames = cls._calculate_port_frames(root_link, links, joint_graph)

        # Identify ports (all links can potentially be ports)
        default_frame = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
        ports = [
            Port(name=link,
                 xyz=port_frames.get(link, default_frame)[0],
                 z_axis=port_frames.get(link, default_frame)[1],
                 rpy=port_frames.get(link, default_frame)[2])
            for link in links
        ]

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
    def _calculate_port_frames(root_link, links, joint_graph):
        """
        Calculate each link's frame (position + Z axis) in the root frame.

        Composes the full joint-origin transforms (rotation and translation,
        at the zero pose) along the root-to-link chain.

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
        Dict[str, Tuple]
            Mapping from link name to ``(xyz, z_axis, rpy)`` in the root
            frame.
        """
        frames = {}

        for link in links:
            if link == root_link:
                frames[link] = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                (0.0, 0.0, 0.0))
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
                frames[link] = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0),
                                (0.0, 0.0, 0.0))
                continue

            # Compose the full transforms from root to link
            transform = np.eye(4)
            for joint in reversed(path):
                origin = joint.find("origin")
                if origin is None:
                    continue
                xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
                transform = transform @ xyzrpy2matrix(xyz, rpy)

            xyz, rpy = matrix2xyzrpy(transform)
            frames[link] = (
                tuple(float(v) for v in xyz),
                tuple(float(v) for v in transform[:3, 2]),
                tuple(float(v) for v in rpy),
            )

        return frames

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
    loop : bool
        ``True`` marks this connection as a loop closure: the edge that a
        kinematic tree cannot carry and that the assembly graph cuts.  A
        loop connection never becomes a joint in the built URDF; instead
        it is exported as a runtime-IK closure constraint (the two port
        frames must stay coincident on their common hinge axis).
    dependent : Tuple[str, ...], optional
        For a loop closure: the final (prefixed) joint names this closure
        SOLVES.  The remaining movable joints on the loop ring stay
        independent (driven).  ``None`` (default) picks the split
        automatically: the movable joint nearest the root drives, the
        rest are solved.  A 1-DOF planar loop solves 2 joints; a 2-DOF
        loop (e.g. a five-bar) needs this made explicit.
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
    loop: bool = False
    dependent: Optional[Tuple[str, ...]] = None
    # How the CHILD-side node attaches when the tree is built.  The
    # connection is undirected: whichever endpoint ends up the child once
    # the tree is rooted attaches via ITS declared port.
    # "root" keeps that module's own root link as the attachment (the
    # connection transform is expressed for the child root frame);
    # "port" re-roots it so the declared port becomes the attachment link
    # (transform expressed for the port frame -- under which the rigid
    # inverse used for reverse traversal stays exact).
    attach: str = "root"

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
    attach: str = "root"
    children: List["TreeNode"] = field(default_factory=list)


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
        mate: bool = False,
        attach: str = "root",
        loop: bool = False,
        dependent: Optional[Tuple[str, ...]] = None,
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
        mate : bool
            Derive the transform so the child port seats onto the parent
            port (Z axes opposed); only ``yaw`` may be combined with it.
        attach : str
            ``'root'`` (default) attaches the child-side module through its
            own root link, treating its declared port as placement metadata.
            ``'port'`` re-roots the child-side module at its declared port
            during the build so that port becomes the real attachment link.
            Applies to whichever endpoint becomes the child once the tree
            is rooted.
        loop : bool
            ``True`` declares this connection as a loop closure -- the cut
            edge of a closed linkage (e.g. a four-bar).  It never becomes a
            joint in the built URDF; the two ports must already coincide at
            the zero pose of the rest of the assembly, and their common Z
            axis is the passive hinge axis.  ``build()`` exports the
            closure to a ``loop_closures.yaml`` sidecar for runtime IK and,
            for a parallelogram four-bar, writes exact ``<mimic>`` tags.
            A loop connection carries no transform, so the offset, ``mate``
            and ``attach`` arguments cannot be combined with it.
        dependent : Tuple[str, ...], optional
            Only with ``loop=True``: the final (prefixed, e.g.
            ``'arm1_elbow'``) joint names this closure solves; the other
            movable joints on the loop ring stay driven.  Validated at
            build time against the actual ring.  Default: the ring joint
            nearest the root drives, the rest are solved -- right for a
            1-DOF loop, but a multi-DOF loop (five-bar and up) needs the
            solved joints named explicitly.
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
        if attach not in ("root", "port"):
            raise ValueError(f"unknown attach mode: {attach!r} "
                             "(expected 'root' or 'port')")
        port_obj_a = next((p for p in self.instances[module_a].module.ports
                           if p.name == port_a), None)
        port_obj_b = next((p for p in self.instances[module_b].module.ports
                           if p.name == port_b), None)
        if port_obj_a is None:
            raise ValueError(
                f"Port '{port_a}' not found on module '{module_a}'")
        if port_obj_b is None:
            raise ValueError(
                f"Port '{port_b}' not found on module '{module_b}'")
        self._check_port_compatibility(module_a, port_obj_a,
                                       module_b, port_obj_b)
        if loop:
            if mate or attach != "root" or \
                    any(abs(v) > 1e-12 for v in (x, y, z, roll, pitch, yaw)):
                raise ValueError(
                    'a loop closure carries no transform: the cut hinge is '
                    'where the two ports coincide at the zero pose, so '
                    'xyz/rpy offsets, mate and attach cannot be combined '
                    'with loop=True')
            if dependent is not None:
                dependent = tuple(dependent)
                if not dependent:
                    raise ValueError(
                        'dependent must name at least one joint the '
                        'closure solves (or be None for the automatic '
                        'split)')
        elif dependent is not None:
            raise ValueError(
                'dependent only applies to a loop closure (loop=True): it '
                'names the joints solved to keep the loop closed')
        if mate:
            if any(abs(v) > 1e-12 for v in (x, y, z, roll, pitch)):
                raise ValueError(
                    'mate=True derives the connection transform from the '
                    'port frames; only yaw (spin about the parent port Z '
                    'axis) may be combined with it')
            x, y, z, roll, pitch, yaw = self._mate_transform(
                module_b, port_b, yaw, attach)

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
            Connection(module_a, port_a, module_b, port_b, x=x, y=y, z=z,
                       roll=roll, pitch=pitch, yaw=yaw, attach=attach,
                       loop=loop, dependent=dependent)
        )

    @staticmethod
    def _check_port_compatibility(module_a, port_a, module_b, port_b):
        """Reject connections the port metadata declares impossible.

        Two ``input`` ports (or two ``output`` ports) cannot mate;
        ``bidirectional`` mates with anything.  When a port lists
        ``compatible_types``, the other port's ``port_type`` or name must
        appear in it.  Ports created by :meth:`RobotModule.from_urdf` are
        unrestricted, so this only constrains explicitly curated catalogs.
        """
        type_a = port_a.port_type or "bidirectional"
        type_b = port_b.port_type or "bidirectional"
        if type_a == type_b and type_a in ("input", "output"):
            raise ValueError(
                f"cannot connect two '{type_a}' ports "
                f"({module_a}.{port_a.name} and {module_b}.{port_b.name})")
        for near, near_mod, far, far_mod in (
                (port_a, module_a, port_b, module_b),
                (port_b, module_b, port_a, module_a)):
            if near.compatible_types and \
                    far.port_type not in near.compatible_types and \
                    far.name not in near.compatible_types:
                raise ValueError(
                    f"port {near_mod}.{near.name} only accepts "
                    f"{sorted(near.compatible_types)}; got "
                    f"{far_mod}.{far.name} (type '{far.port_type}')")

    def _mate_transform(self, module_b: str, port_b: str, yaw: float,
                        attach: str = "root"):
        """Connection transform that mates the child port onto the parent port.

        Seats the child so its ``port_b`` frame coincides with the parent
        port frame in the keyed-connector convention: origins coincide, Z
        axes OPPOSED (plug into socket) and the child port X axis aligned
        with the parent port X axis, spun by ``yaw`` about the parent port
        Z.  Ports are full SE(3) frames, so the result is fully determined;
        ``yaw`` is an optional extra spin, not a free parameter left to
        guesswork.  The returned transform is expressed for the child ROOT
        frame, which is what the connection origin carries.
        """
        module = self.instances[module_b].module
        port = next((p for p in module.ports if p.name == port_b), None)
        if port is None:
            raise ValueError(
                f"Port '{port_b}' not found on module '{module_b}'")
        if attach == "port":
            # the child is re-rooted at the port, whose frame is then the
            # child root frame itself
            port_frame = np.eye(4)
        else:
            port_frame = xyzrpy2matrix(port.xyz, port.rpy)
        # desired child-port pose in parent-port coordinates:
        # 180-degree flip about X (Z opposed, X aligned), then the spin
        seat = rpy2homogeneous(math.pi, 0.0, float(yaw))
        transform = seat @ matrix_relative(port_frame, np.eye(4))
        out_xyz, out_rpy = matrix2xyzrpy(transform)
        return (float(out_xyz[0]), float(out_xyz[1]), float(out_xyz[2]),
                float(out_rpy[0]), float(out_rpy[1]), float(out_rpy[2]))

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
            rotations).  Loop closures (``loop=True``) are cut edges, not
            tree edges, so they do not appear here.
        """
        adj: Dict[str, List[Tuple[str, str, str, Tuple, Tuple, str]]] = {inst_id: [] for inst_id in self.instances}
        identity = np.eye(4)
        for conn in self.connections:
            if conn.loop:
                continue
            # Forward direction: module_a -> module_b uses original xyz/rpy
            adj[conn.module_a].append((conn.module_b, conn.port_a, conn.port_b, conn.xyz, conn.rpy, conn.attach))
            # Reverse direction: module_b -> module_a uses the inverse transform
            transform = xyzrpy2matrix(conn.xyz, conn.rpy)
            inv_xyz, inv_rpy = matrix2xyzrpy(matrix_relative(transform, identity))
            adj[conn.module_b].append(
                (conn.module_a, conn.port_b, conn.port_a,
                 tuple(float(v) for v in inv_xyz),
                 tuple(float(v) for v in inv_rpy),
                 conn.attach))
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
                for neighbor_id, my_port, neighbor_port, xyz, rpy, attach in adj[current.instance_id]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        child_node = TreeNode(
                            instance_id=neighbor_id,
                            parent_instance=current.instance_id,
                            parent_port=my_port,
                            child_port=neighbor_port,
                            connection_xyz=xyz,
                            connection_rpy=rpy,
                            attach=attach,
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
                    for neighbor_id, my_port, neighbor_port, xyz, rpy, attach in adj[current.instance_id]:
                        if neighbor_id not in visited:
                            visited.add(neighbor_id)
                            child_node = TreeNode(
                                instance_id=neighbor_id,
                                parent_instance=current.instance_id,
                                parent_port=my_port,
                                child_port=neighbor_port,
                                connection_xyz=xyz,
                                connection_rpy=rpy,
                                attach=attach,
                            )
                            current.children.append(child_node)
                            node_map[neighbor_id] = child_node
                            queue.append(child_node)

        # URDF is a tree: every non-loop connection must have been consumed
        # as a forest edge.  Anything left over closes an UNDECLARED cycle
        # and would be silently dropped by the traversal -- refuse instead.
        # Declared closures (loop=True) are cut edges by construction.
        n_tree_conns = sum(1 for c in self.connections if not c.loop)
        tree_edges = len(self.instances) - len(root_nodes)
        if n_tree_conns > tree_edges:
            raise ValueError(
                f'assembly graph contains an undeclared cycle: '
                f'{n_tree_conns} tree connections for '
                f'{len(self.instances)} instance(s) in '
                f'{len(root_nodes)} component(s); a URDF kinematic tree '
                'cannot represent closed loops -- declare the cut edge '
                'with connect(..., loop=True)')
        return root_nodes, node_map

    def build(self, output_path: Optional[str] = None) -> str:
        """
        Build a combined URDF from the assembly.

        Traverses the connection graph as a forest rooted at
        ``root_instance``, inserts a ``world`` link as the absolute root and
        attaches every module through a fixed joint from its parent's port
        to its own root link, with all element names prefixed by the
        instance id.  Modules keep their original kinematic structure (no
        chain reversal).

        Connections declared with ``loop=True`` are cut edges: they never
        become joints.  When any exist, a ``loop_closures.yaml`` sidecar
        (the runtime-IK relay config) is written next to the output URDF,
        and a parallelogram four-bar loop additionally gets exact
        ``<mimic>`` tags on its dependent joints.

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
            If the assembly is empty or contains a cycle.
        """
        self._validate_and_default_root()
        return self._build_inline(output_path)

    def _validate_and_default_root(self) -> None:
        if not self.instances:
            raise ValueError("Assembly is empty - add module instances first")
        # Auto-set root if not set (use first instance)
        if self.root_instance is None:
            first_instance_id = next(iter(self.instances))
            first_instance = self.instances[first_instance_id]
            self.root_instance = first_instance_id
            self.root_port = first_instance.module.root_link

    def _attachment_source(self, node, module, temp_dir):
        """``(urdf_path, connector_link)`` for one tree node, re-rooting the
        module URDF at the declared port when the connection was made with
        ``attach='port'``."""
        if node.attach == "port" and node.child_port \
                and node.child_port != module.root_link:
            rerooted = os.path.join(
                temp_dir, f"{node.instance_id}_rerooted.urdf")
            change_urdf_root_link(str(module.urdf_path), node.child_port,
                                  rerooted, robot_name=module.module_id)
            return rerooted, node.child_port
        return str(module.urdf_path), module.root_link

    def build_robot_model(self):
        """Assemble directly into a :class:`skrobot.model.RobotModel`.

        Composes the combined URDF in memory and loads it without writing
        the result to disk (re-rooted modules still go through temporary
        files).  Mesh resolution follows the same rules as loading the
        built URDF file would.

        Returns
        -------
        skrobot.model.RobotModel
            The assembled robot.
        """
        from skrobot.models.urdf import RobotModelFromURDF
        self._validate_and_default_root()
        temp_dir = tempfile.mkdtemp(prefix=f"{self.name}_build_")
        robot = self._compose_inline(temp_dir)
        # validates declared closures and injects parallelogram mimics;
        # the relay yaml sidecar only exists when building to a file
        self._apply_loop_closures(robot)
        return RobotModelFromURDF(
            urdf=etree.tostring(robot, encoding="unicode"))

    @staticmethod
    def _prefixed_module_elements(urdf_path: str, prefix: str):
        """Parse a module URDF and yield its top-level elements with every
        name -- and the joint parent/child/mimic and material references --
        prefixed."""
        def _prefixed(name):
            return prefix + name

        root = etree.parse(str(urdf_path)).getroot()
        for elem in root:
            if not isinstance(elem.tag, str):
                continue  # skip comments / processing instructions
            if "name" in elem.attrib:
                elem.attrib["name"] = _prefixed(elem.attrib["name"])
            if elem.tag == "joint":
                for sub in elem.findall("parent"):
                    sub.attrib["link"] = _prefixed(sub.attrib["link"])
                for sub in elem.findall("child"):
                    sub.attrib["link"] = _prefixed(sub.attrib["link"])
                for sub in elem.findall("mimic"):
                    if "joint" in sub.attrib:
                        sub.attrib["joint"] = _prefixed(sub.attrib["joint"])
            elif elem.tag == "link":
                for part in ("visual", "collision"):
                    for sub in elem.findall(part):
                        for material in sub.findall("material"):
                            if "name" in material.attrib:
                                material.attrib["name"] = \
                                    _prefixed(material.attrib["name"])
            elif elem.tag == "transmission":
                for part in ("joint", "actuator"):
                    for sub in elem.findall(part):
                        if "name" in sub.attrib:
                            sub.attrib["name"] = _prefixed(sub.attrib["name"])
            elif elem.tag == "gazebo":
                # <gazebo reference="..."> names a link or joint
                if "reference" in elem.attrib:
                    elem.attrib["reference"] = \
                        _prefixed(elem.attrib["reference"])
            yield elem

    def _compose_inline(self, temp_dir: str):
        """Compose the combined URDF in memory; returns the robot element."""
        root_nodes, _node_map = self._build_forest_from_graph()

        process_order = []
        for root_node in root_nodes:
            queue = deque([root_node])
            while queue:
                node = queue.popleft()
                process_order.append(node)
                for child in node.children:
                    queue.append(child)

        robot = etree.Element("robot")
        robot.set("name", self.name)
        etree.SubElement(robot, "link").set("name", "world")

        for node in process_order:
            instance = self.instances[node.instance_id]
            module = instance.module
            prefix = f"{node.instance_id}_" if node.instance_id else ""
            urdf_source, connector = self._attachment_source(
                node, module, temp_dir)

            if node.parent_instance is None:
                parent_link = "world"
            elif node.parent_instance:
                parent_link = f"{node.parent_instance}_{node.parent_port}"
            else:
                parent_link = node.parent_port

            child_link = prefix + connector
            joint = etree.SubElement(robot, "joint")
            joint.set("name", f"{parent_link}_to_{child_link}_joint")
            joint.set("type", "fixed")
            etree.SubElement(joint, "parent").set("link", parent_link)
            etree.SubElement(joint, "child").set("link", child_link)
            origin = etree.SubElement(joint, "origin")
            origin.set("xyz", " ".join(str(v) for v in node.connection_xyz))
            origin.set("rpy", " ".join(str(v) for v in node.connection_rpy))

            for elem in self._prefixed_module_elements(urdf_source, prefix):
                robot.append(elem)

        return robot

    _MOVABLE_JOINT_TYPES = ("revolute", "continuous", "prismatic")

    @staticmethod
    def _link_transforms(robot, joint_values=None):
        """World transform of every link in a composed robot element.

        Walks the joint tree from the root link(s).  ``joint_values`` maps
        joint names to positions; unlisted joints sit at zero, so with no
        argument this is the zero-pose forward kinematics.

        Parameters
        ----------
        robot : lxml.etree._Element
            The ``<robot>`` element with top-level ``<link>``/``<joint>``
            children.
        joint_values : Dict[str, float], optional
            Joint positions (radians / metres) for movable joints.

        Returns
        -------
        Dict[str, numpy.ndarray]
            Mapping from link name to its 4x4 world transform.
        """
        joint_values = joint_values or {}
        children: Dict[str, list] = {}
        child_links = set()
        for j in robot.findall("joint"):
            parent = j.find("parent").get("link")
            child = j.find("child").get("link")
            origin = j.find("origin")
            if origin is not None:
                xyz = [float(v) for v in
                       origin.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in
                       origin.get("rpy", "0 0 0").split()]
            else:
                xyz = [0.0, 0.0, 0.0]
                rpy = [0.0, 0.0, 0.0]
            transform = xyzrpy2matrix(xyz, rpy)
            q = float(joint_values.get(j.get("name"), 0.0))
            if q != 0.0:
                axis_elem = j.find("axis")
                axis = np.array(
                    [float(v) for v in axis_elem.get("xyz").split()]
                ) if axis_elem is not None else np.array([0.0, 0.0, 1.0])
                joint_type = j.get("type")
                motion = np.eye(4)
                if joint_type in ("revolute", "continuous"):
                    motion[:3, :3] = rotation_matrix(q, axis)
                elif joint_type == "prismatic":
                    motion[:3, 3] = axis / np.linalg.norm(axis) * q
                transform = transform @ motion
            children.setdefault(parent, []).append((child, transform))
            child_links.add(child)

        transforms = {}
        queue = deque()
        for link in robot.findall("link"):
            name = link.get("name")
            if name not in child_links:
                transforms[name] = np.eye(4)
                queue.append(name)
        while queue:
            link_name = queue.popleft()
            for child, transform in children.get(link_name, ()):
                transforms[child] = transforms[link_name] @ transform
                queue.append(child)
        return transforms

    def _apply_loop_closures(self, robot):
        """Export the declared loop closures against the composed robot.

        For every ``loop=True`` connection: validate that the two port
        links coincide at the zero pose and share a hinge axis (their port
        Z), split the movable joints on the loop ring into one driver
        (independent) and solved (dependent) joints, and -- when the ring
        is a parallelogram four-bar, verified numerically -- write exact
        linear ``<mimic>`` tags onto the dependent joints so even
        mimic-only consumers (robot_state_publisher) stay on the loop.

        Returns
        -------
        dict or None
            The runtime-IK relay config ``{closures: [{link_a, link_b,
            point, axis}], dependent: [...], independent: [...]}`` with
            ``point``/``axis`` in the URDF root frame at the zero pose,
            or None when no closure is declared.
        """
        loops = [c for c in self.connections if c.loop]
        if not loops:
            return None
        fk = self._link_transforms(robot)
        parent_joint = {j.find("child").get("link"): j
                        for j in robot.findall("joint")}
        all_movable = [j.get("name") for j in robot.findall("joint")
                       if j.get("type") in self._MOVABLE_JOINT_TYPES]

        def links_up(link):
            out = [link]
            while link in parent_joint:
                link = parent_joint[link].find("parent").get("link")
                out.append(link)
            return out

        def ring_side(link, stop):
            """Joint elements from ``link`` up to (excluding) ``stop``."""
            out = []
            while link != stop:
                joint = parent_joint[link]
                out.append(joint)
                link = joint.find("parent").get("link")
            return out

        closures = []
        dependent = set()
        drivers = set()
        for conn in loops:
            label = (f"{conn.module_a}.{conn.port_a} <-> "
                     f"{conn.module_b}.{conn.port_b}")
            link_a = (f"{conn.module_a}_{conn.port_a}"
                      if conn.module_a else conn.port_a)
            link_b = (f"{conn.module_b}_{conn.port_b}"
                      if conn.module_b else conn.port_b)
            t_a, t_b = fk[link_a], fk[link_b]
            gap = float(np.linalg.norm(t_a[:3, 3] - t_b[:3, 3]))
            if gap > 1e-6:
                raise ValueError(
                    f"loop closure {label}: the zero pose leaves the two "
                    f"ports {gap:.3g} m apart; a closure is the cut hinge "
                    "of an assembled loop, so the tree connections must "
                    "already place the ports coincident (e.g. via "
                    "mate=True) before the loop edge is declared")
            # the hinge is the LINE through the port origin along port Z;
            # the axis sign is irrelevant (the closure constrains witness
            # POINTS on that line, not a direction), so mated Z-opposed
            # ports are as valid as aligned ones
            axis_a, axis_b = t_a[:3, 2], t_b[:3, 2]
            if float(np.linalg.norm(np.cross(axis_a, axis_b))) > 1e-6:
                raise ValueError(
                    f"loop closure {label}: the two port Z axes are not "
                    "collinear at the zero pose, so they do not define a "
                    "common hinge axis")
            ancestors_b = set(links_up(link_b))
            lca = next(link for link in links_up(link_a)
                       if link in ancestors_b)
            side_a = ring_side(link_a, lca)
            side_b = ring_side(link_b, lca)
            ring = ([j for j in side_a
                     if j.get("type") in self._MOVABLE_JOINT_TYPES]
                    + [j for j in reversed(side_b)
                       if j.get("type") in self._MOVABLE_JOINT_TYPES])
            if not ring:
                raise ValueError(
                    f"loop closure {label}: no movable joint on the loop "
                    "ring -- the closure would weld a rigid ring, which a "
                    "kinematic tree already represents without it")
            driver_name, fixed, deps = self._split_loop_ring(
                robot, conn, label, ring, links_up, dependent, drivers)
            dependent.update(j.get("name") for j in deps)
            point = t_a[:3, 3]
            closures.append({
                "link_a": link_a, "link_b": link_b,
                "point": [round(float(v), 8) for v in point],
                "axis": [round(float(v), 8) for v in axis_a],
            })
            self._try_parallelogram_mimic(
                robot, fk, ring, driver_name, fixed, deps, point, axis_a,
                link_a, link_b)
        return {
            "closures": closures,
            "dependent": sorted(dependent),
            "independent": sorted(set(all_movable) - dependent),
        }

    @staticmethod
    def _resolve_mimic_chain(robot):
        """Resolve every ``<mimic>`` tag to its transitive root driver.

        Returns
        -------
        Dict[str, Tuple[str, float, float]]
            Mapping from a mimicking joint name to ``(root_driver,
            multiplier, offset)`` such that ``q_joint = multiplier *
            q_root + offset``.  Joints on a mimic cycle are omitted.
        """
        specs = {}
        for j in robot.findall("joint"):
            mimic = j.find("mimic")
            if mimic is not None and mimic.get("joint"):
                specs[j.get("name")] = (
                    mimic.get("joint"),
                    float(mimic.get("multiplier", 1.0)),
                    float(mimic.get("offset", 0.0)))
        resolved = {}
        for name in specs:
            multiplier, offset = 1.0, 0.0
            current = name
            seen = set()
            while current in specs:
                if current in seen:
                    break  # mimic cycle: unresolvable
                seen.add(current)
                target, m, o = specs[current]
                offset += multiplier * o
                multiplier *= m
                current = target
            else:
                resolved[name] = (current, multiplier, offset)
        return resolved

    def _split_loop_ring(self, robot, conn, label, ring, links_up,
                         dependent, drivers):
        """Split one closure's ring joints into driven vs solved.

        Three strategies, in priority order: the connection's explicit
        ``dependent`` list; following an existing ``<mimic>`` on the ring
        to its transitive root (so a chained loop never leaks a
        kinematically coupled joint into ``independent``); else the
        nearest-root heuristic (right for a 1-DOF loop).

        Returns
        -------
        Tuple[Optional[str], Dict[str, float], List]
            ``(driver_name, fixed, deps)``: the single driving joint the
            parallelogram certification runs against (None when there is
            no unique drivable one -- possibly a joint OUTSIDE the ring
            for a chained loop), ``fixed`` mapping ring joints that
            already mimic the driver to their known multiplier, and the
            ring joint elements this closure marks dependent.
        """
        ring_names = [j.get("name") for j in ring]
        if conn.dependent is not None:
            missing = [n for n in conn.dependent if n not in ring_names]
            if missing:
                raise ValueError(
                    f"loop closure {label}: dependent joint(s) {missing} "
                    f"are not movable joints on the loop ring "
                    f"{ring_names}")
            conflict = sorted(set(conn.dependent) & drivers)
            if conflict:
                raise ValueError(
                    f"loop closure {label}: {conflict} already drive "
                    "another closure and cannot be re-marked dependent")
            deps = [j for j in ring if j.get("name") in conn.dependent]
            rest = [j for j in ring if j.get("name") not in conn.dependent]
            drivers.update(j.get("name") for j in rest
                           if j.get("name") not in dependent)
            # certify only against a joint that really is independent
            driver_name = rest[0].get("name") \
                if len(rest) == 1 and rest[0].find("mimic") is None \
                and rest[0].get("name") not in dependent \
                else None
            return driver_name, {}, deps

        chain = self._resolve_mimic_chain(robot)
        mimicking = [j for j in ring if j.get("name") in chain]
        if mimicking:
            roots = {chain[j.get("name")][0] for j in mimicking}
            deps = [j for j in ring if j.get("name") not in roots
                    and j.get("name") not in drivers]
            # a root an earlier closure already solves is no driver: leave
            # it dependent and let the relay solve this ring numerically
            drivers.update(roots - dependent)
            exact = len(roots) == 1 and not (roots & dependent) and all(
                abs(chain[j.get("name")][2]) < 1e-12 for j in mimicking)
            if not exact:
                return None, {}, deps
            fixed = {j.get("name"): chain[j.get("name")][1]
                     for j in mimicking}
            return next(iter(roots)), fixed, deps

        # a joint an earlier closure marked dependent is solved --
        # it cannot drive
        drivable = [j for j in ring if j.get("name") not in dependent]
        if not drivable:
            raise ValueError(
                f"loop closure {label}: every movable joint on the loop "
                "ring is already solved by another closure, so nothing "
                "can drive the loop")
        driver = min(
            drivable,
            key=lambda j: (len(links_up(j.find("child").get("link"))),
                           j.get("name")))
        drivers.add(driver.get("name"))
        # never re-mark another closure's driver as dependent: a joint
        # must end up on exactly one side of the split
        deps = [j for j in ring if j is not driver
                and j.get("name") not in drivers]
        return driver.get("name"), {}, deps

    def _try_parallelogram_mimic(self, robot, fk, ring, driver_name,
                                 fixed, deps, cut_point, cut_axis,
                                 link_a, link_b):
        """Write exact ``<mimic>`` tags when the loop is a parallelogram.

        A four-bar whose hinge quadrilateral is a parallelogram (all four
        axes parallel) couples its joints LINEARLY (multipliers of exactly
        +/-1), which the URDF ``<mimic>`` tag can represent -- unlike a
        general four-bar, whose coupling is nonlinear and is left to the
        runtime-IK relay.  Candidate multipliers are certified by forward
        kinematics before anything is written: the cut-hinge witness
        points must still coincide at sampled nonzero driver angles.
        Silently does nothing when the ring is not such a loop.

        For a chained loop, ``driver_name`` may be a joint OUTSIDE the
        ring (the transitive root of an existing mimic) and ``fixed``
        carries the known multipliers of the ring joints that already
        follow it; only the remaining free dependents get new tags.
        """
        if driver_name is None or len(ring) != 3:
            return
        free = [j for j in deps if j.get("name") not in fixed]
        if not free or len(free) > 2:
            return
        # every ring joint must have a determined value during the FK
        # certification: the driver itself, a known follower, or a free
        # candidate
        covered = set(fixed) | {j.get("name") for j in free} | {driver_name}
        if any(j.get("name") not in covered for j in ring):
            return
        if any(j.get("type") not in ("revolute", "continuous")
               for j in ring):
            return
        if any(j.find("mimic") is not None and j.get("name") not in fixed
               for j in ring):
            return
        normal = cut_axis / np.linalg.norm(cut_axis)
        world_axes = []
        hinge_points = [np.asarray(cut_point, dtype=float)]
        for j in ring:
            child = j.find("child").get("link")
            axis_elem = j.find("axis")
            axis = np.array(
                [float(v) for v in axis_elem.get("xyz").split()]
            ) if axis_elem is not None else np.array([0.0, 0.0, 1.0])
            world_axes.append(fk[child][:3, :3] @ axis)
            hinge_points.append(fk[child][:3, 3])
        if any(float(np.linalg.norm(np.cross(normal, a)))
               > 1e-8 * np.linalg.norm(a) for a in world_axes):
            return
        # ring order is cut -> side_a joints -> side_b joints, a cyclic
        # quadrilateral; parallelogram iff one side pair matches (the
        # other follows), checked in the plane normal to the hinge axis
        planar = [p - np.dot(p, normal) * normal for p in hinge_points]
        if float(np.linalg.norm((planar[1] - planar[0])
                                - (planar[2] - planar[3]))) > 1e-6:
            return

        witness_local = []
        for w in (cut_point, cut_point + 0.03 * normal):
            w_h = np.append(w, 1.0)
            witness_local.append((np.linalg.solve(fk[link_a], w_h),
                                  np.linalg.solve(fk[link_b], w_h)))
        combos = [()]
        for _ in free:
            combos = [c + (m,) for c in combos for m in (1.0, -1.0)]
        for multipliers in combos:
            closed = True
            for q in (0.3, -0.45):
                joint_values = {driver_name: q}
                joint_values.update(
                    (name, m * q) for name, m in fixed.items())
                for dep, m in zip(free, multipliers):
                    joint_values[dep.get("name")] = m * q
                fk_q = self._link_transforms(robot, joint_values)
                for local_a, local_b in witness_local:
                    gap = np.linalg.norm(fk_q[link_a] @ local_a
                                         - fk_q[link_b] @ local_b)
                    if gap > 1e-9:
                        closed = False
                        break
                if not closed:
                    break
            if closed:
                for dep, m in zip(free, multipliers):
                    mimic = etree.SubElement(dep, "mimic")
                    mimic.set("joint", driver_name)
                    mimic.set("multiplier", str(m))
                    mimic.set("offset", "0")
                return

    @staticmethod
    def _write_loop_closures_yaml(config, directory):
        """Write the runtime-IK relay config next to the built URDF.

        ``loop_closures.yaml`` is the contract consumed by downstream
        loop-closure relay nodes: at every joint state they solve the
        dependent joints so each closure's two link points coincide.
        """
        import yaml
        head = ("# Closed-loop closures for a loop-closure relay "
                "(runtime IK).\n"
                "# independent = driven joints; dependent = solved so each "
                "closure's\n"
                "# two link points (the cut hinge) coincide.\n")
        path = os.path.join(directory, "loop_closures.yaml")
        with open(path, "w") as f:
            f.write(head + yaml.safe_dump(config, sort_keys=False,
                                          default_flow_style=None))
        return path

    def _build_inline(self, output_path: Optional[str] = None) -> str:
        """Compose the combined URDF and write it to disk."""
        temp_dir = tempfile.mkdtemp(prefix=f"{self.name}_build_")
        robot = self._compose_inline(temp_dir)
        closures = self._apply_loop_closures(robot)
        if output_path is None:
            output_path = os.path.join(temp_dir, f"{self.name}.urdf")
        with open(output_path, "wb") as f:
            f.write(etree.tostring(robot, pretty_print=True,
                                   xml_declaration=True, encoding="UTF-8"))
        if closures is not None:
            self._write_loop_closures_yaml(
                closures, os.path.dirname(os.path.abspath(output_path)))
        return output_path

    def to_dict(self) -> dict:
        """
        Serialize the assembly to a dictionary.

        Useful for saving/loading assembly state or sending to frontend.
        """
        return {
            "name": self.name,
            "root_instance": self.root_instance,
            "root_port": self.root_port,
            "instances": {
                inst_id: {"module_id": inst.module.module_id, "urdf_path": str(inst.module.urdf_path)}
                for inst_id, inst in self.instances.items()
            },
            "connections": [
                {"module_a": c.module_a, "port_a": c.port_a,
                 "module_b": c.module_b, "port_b": c.port_b,
                 "x": c.x, "y": c.y, "z": c.z,
                 "roll": c.roll, "pitch": c.pitch, "yaw": c.yaw,
                 "attach": c.attach, "loop": c.loop,
                 "dependent": list(c.dependent) if c.dependent else None}
                for c in self.connections
            ],
            "root": {"instance": self.root_instance, "port": self.root_port} if self.root_instance else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RobotAssembly":
        """Rebuild an assembly from :meth:`to_dict` output.

        Modules are re-parsed from the recorded ``urdf_path``s, so those
        files must still exist.
        """
        assembly = cls(data["name"])
        for inst_id, inst in data.get("instances", {}).items():
            module = RobotModule.from_urdf(inst["module_id"],
                                           inst["urdf_path"])
            assembly.add_module_instance(inst_id, module)
        for conn in data.get("connections", []):
            assembly.connect(
                conn["module_a"], conn["port_a"],
                conn["module_b"], conn["port_b"],
                x=conn.get("x", 0.0), y=conn.get("y", 0.0),
                z=conn.get("z", 0.0),
                roll=conn.get("roll", 0.0), pitch=conn.get("pitch", 0.0),
                yaw=conn.get("yaw", 0.0),
                attach=conn.get("attach", "root"),
                loop=conn.get("loop", False),
                dependent=(tuple(conn["dependent"])
                           if conn.get("dependent") else None))
        if data.get("root_instance"):
            assembly.set_root(data["root_instance"], data["root_port"])
        return assembly

    def __repr__(self) -> str:
        return (
            f"RobotAssembly(name={self.name!r}, "
            f"instances={len(self.instances)}, "
            f"connections={len(self.connections)}, "
            f"root={self.root_instance}:{self.root_port})"
        )
