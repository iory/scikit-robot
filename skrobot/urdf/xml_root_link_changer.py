import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

from skrobot.coordinates.math import invert_yaw_pitch_roll


class URDFXMLRootLinkChanger:
    """A class to change the root link of a URDF by directly manipulating XML.

    This class reads URDF XML files and modifies the kinematic tree structure
    to change the root link to any specified link, then writes the modified
    URDF back to a file.
    """

    def __init__(self, urdf_path):
        """Initialize the URDFXMLRootLinkChanger.

        Parameters
        ----------
        urdf_path : str
            Path to the input URDF file

        Raises
        ------
        FileNotFoundError
            If the URDF file does not exist
        """
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(
                "URDF file not found: {}".format(urdf_path))

        self.urdf_path = urdf_path
        self.tree = ET.parse(urdf_path)
        self.root = self.tree.getroot()

        ET.register_namespace('xacro', "http://ros.org/wiki/xacro")

        # Parse the URDF structure
        self.links = self._parse_links()
        self.joints = self._parse_joints()
        self.joint_tree = self._build_joint_tree()

    def _parse_links(self):
        """Parse all links from the URDF.

        Returns
        -------
        dict
            Dictionary mapping link names to their XML elements
        """
        links = {}
        for link in self.root.findall('link'):
            name = link.get('name')
            if name:
                links[name] = link
        return links

    def _parse_joints(self):
        """Parse all joints from the URDF.

        Returns
        -------
        dict
            Dictionary mapping joint names to their XML elements
        """
        joints = {}
        for joint in self.root.findall('joint'):
            name = joint.get('name')
            if name:
                joints[name] = joint
        return joints

    def _build_joint_tree(self):
        """Build a tree structure representing parent-child relationships.

        Returns
        -------
        dict
            Dictionary with link names as keys and dictionaries containing
            'parent', 'children', and 'joint' information
        """
        tree = {link_name: {'parent': None, 'children': [], 'joint': None}
                for link_name in self.links.keys()}

        for joint_name, joint in self.joints.items():
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')

            if parent_elem is not None and child_elem is not None:
                parent_link = parent_elem.get('link')
                child_link = child_elem.get('link')

                if parent_link and child_link:
                    if parent_link in tree and child_link in tree:
                        tree[child_link]['parent'] = parent_link
                        tree[child_link]['joint'] = joint_name
                        tree[parent_link]['children'].append(child_link)

        return tree

    def get_current_root_link(self):
        """Get the current root link name.

        Returns
        -------
        str or None
            Name of the current root link (link with no parent)
        """
        for link_name, info in self.joint_tree.items():
            if info['parent'] is None:
                return link_name
        return None

    def list_links(self):
        """List all links in the URDF.

        Returns
        -------
        list of str
            List of all link names
        """
        return list(self.links.keys())

    def change_root_link(self, new_root_link, output_path):
        """Change the root link and save the modified URDF.

        Parameters
        ----------
        new_root_link : str
            Name of the new root link
        output_path : str
            Path where the modified URDF will be saved

        Raises
        ------
        ValueError
            If the specified link name is not found in the URDF
        """
        if new_root_link not in self.links:
            raise ValueError(
                "Link '{}' not found in URDF".format(new_root_link))

        current_root = self.get_current_root_link()
        if new_root_link == current_root:
            # No change needed, just copy the file
            self._save_urdf(output_path)
            return

        # Build path from new root to current root
        path_to_new_root = self._find_path_to_link(new_root_link, current_root)
        if not path_to_new_root:
            raise ValueError(
                "No path found from {} to {}".format(
                    current_root, new_root_link))

        # Reverse the joints along the path
        self._reverse_joints_along_path(path_to_new_root)

        # Save the modified URDF
        self._save_urdf(output_path)

    def _find_path_to_link(self, start_link, target_link):
        """Find path from start_link to target_link.

        Parameters
        ----------
        start_link : str
            Starting link name
        target_link : str
            Target link name

        Returns
        -------
        list of tuple
            List of (parent_link, child_link, joint_name)
            tuples representing the path
        """
        def dfs(current, target, path, visited):
            if current == target:
                return True

            if current in visited:
                return False

            visited.add(current)

            # Check children
            for child in self.joint_tree[current]['children']:
                joint_name = self.joint_tree[child]['joint']
                path.append((current, child, joint_name))
                if dfs(child, target, path, visited):
                    return True
                path.pop()

            # Check parent
            parent = self.joint_tree[current]['parent']
            if parent and parent not in visited:
                joint_name = self.joint_tree[current]['joint']
                path.append((parent, current, joint_name))
                if dfs(parent, target, path, visited):
                    return True
                path.pop()

            visited.remove(current)
            return False

        path = []
        visited = set()
        if dfs(start_link, target_link, path, visited):
            return path
        return []

    def _reverse_joints_along_path(self, path):
        """Reverse joints along the given path.

        Parameters
        ----------
        path : list of tuple
            List of (parent_link, child_link, joint_name) tuples
        """

        # Cache origin/xyz,rpy values of each joint before modifying origin of joints
        joint_xyz_rpy_cache = {}
        for parent_link, child_link, joint_name in path:
            if joint_name in self.joints:
                joint = self.joints[joint_name]
                origin = joint.find('origin')

                # Get current xyz and rpy
                xyz_str = origin.get('xyz', '0 0 0')
                rpy_str = origin.get('rpy', '0 0 0')

                # Parse the values
                xyz = [float(x) for x in xyz_str.split()]
                rpy = [float(x) for x in rpy_str.split()]

                joint_xyz_rpy_cache[joint] = [xyz, rpy]

        # Modify joint origins before swapping parent and child
        prev_joint = None
        for parent_link, child_link, joint_name in path:
            if joint_name in self.joints:
                joint = self.joints[joint_name]
                parent_elem = joint.find('parent')
                child_elem = joint.find('child')
                if parent_elem is not None and child_elem is not None:
                    prev_joint_xyz = None
                    prev_joint_rpy = None
                    if prev_joint is not None:
                        prev_joint_xyz = joint_xyz_rpy_cache[prev_joint][0]
                        prev_joint_rpy = joint_xyz_rpy_cache[prev_joint][1]
                    self._reverse_joint_transform(joint, prev_joint_xyz, prev_joint_rpy, path)
                    prev_joint = joint

        for parent_link, child_link, joint_name in path:
            if joint_name in self.joints:
                joint = self.joints[joint_name]

                # Swap parent and child
                parent_elem = joint.find('parent')
                child_elem = joint.find('child')

                if parent_elem is not None and child_elem is not None:
                    # Swap the link attributes
                    parent_elem.set('link', child_link)
                    child_elem.set('link', parent_link)

                    # Update our internal tree structure
                    self.joint_tree[parent_link]['parent'] = child_link
                    self.joint_tree[child_link]['parent'] = None

                    if parent_link in self.joint_tree[child_link]['children']:
                        self.joint_tree[child_link]['children'].remove(
                            parent_link)
                    if child_link in self.joint_tree[parent_link]['children']:
                        self.joint_tree[parent_link]['children'].remove(child_link)
                    self.joint_tree[parent_link]['children'].append(child_link)
                    self.joint_tree[child_link]['children'].append(parent_link)

                    self.joint_tree[parent_link]['joint'] = joint_name
                    self.joint_tree[child_link]['joint'] = None

    def _get_inversed_joint_origin(self, xyz, rpy):
        if xyz is not None and rpy is not None:
            # Calculate inversed transform of origin
            inv_y, inv_p, inv_r = invert_yaw_pitch_roll(rpy[2], rpy[1], rpy[0])
            rpy_reversed = [inv_r, inv_p, inv_y]
            rot_inv = R.from_euler('xyz', rpy_reversed)
            xyz_reversed = -rot_inv.apply(xyz)
            return xyz_reversed, rpy_reversed
        return None, None

    # When the prev_joint_xyz and rpy is None, the child of this joint is the new root link
    def _reverse_joint_transform(self, joint, prev_joint_xyz, prev_joint_rpy, path_to_current_root):
        origin = joint.find('origin')

        # Cache current xyz and rpy
        current_xyz_str = origin.get('xyz', '0 0 0')
        current_rpy_str = origin.get('rpy', '0 0 0')
        current_xyz = [float(x) for x in current_xyz_str.split()]
        current_rpy = [float(x) for x in current_rpy_str.split()]
        inv_current_xyz, inv_current_rpy = self._get_inversed_joint_origin(current_xyz, current_rpy)
        inv_current_pose = self._pose_to_matrix(inv_current_xyz, inv_current_rpy)

        # Calculate new xyz and rpy from previous joint's origin
        new_xyz = [0, 0, 0]
        new_rpy = [0, 0, 0]
        if (prev_joint_xyz is not None) and (prev_joint_rpy is not None):
            # Set the reversed values of previous joint
            new_xyz, new_rpy = self._get_inversed_joint_origin(prev_joint_xyz, prev_joint_rpy)
        new_pose = self._pose_to_matrix(new_xyz, new_rpy)

        origin.set('xyz', ' '.join(map(str, new_xyz)))
        origin.set('rpy', ' '.join(map(str, new_rpy)))

        # Invert axis direction
        axis = joint.find('axis')
        axis_xyz_str = axis.get('xyz', '0 0 0')
        inv_axis_xyz = [-float(x) for x in axis_xyz_str.split()]
        axis.set('xyz', ' '.join(map(str, inv_axis_xyz)))

        # Adjust visual and collision origin
        parent_name = joint.find('parent').get('link')
        parent = self.links[parent_name]
        parent_visual = parent.find('visual')
        if parent_visual is not None:
            parent_visual_origin = parent_visual.find('origin')
            if parent_visual_origin is not None:
                parent_visual_origin.set('xyz', ' '.join(map(str, inv_current_xyz)))
                parent_visual_origin.set('rpy', ' '.join(map(str, inv_current_rpy)))
        parent_collision = parent.find('collision')
        if parent_collision is not None:
            parent_collision_origin = parent_collision.find('origin')
            if parent_collision_origin is not None:
                parent_collision_origin.set('xyz', ' '.join(map(str, inv_current_xyz)))
                parent_collision_origin.set('rpy', ' '.join(map(str, inv_current_rpy)))

        # When the parent link of this joint is the current base_link
        base_link_name = path_to_current_root[-1][0]
        if parent_name == base_link_name:
            base_link_child_joints = self._get_all_children_joints_of_link(base_link_name)
            for base_link_child_joint in base_link_child_joints:
                if self._joint_is_included_in_path(base_link_child_joint, path_to_current_root):
                    continue
                base_link_child_origin = base_link_child_joint.find('origin')
                if base_link_child_origin is not None:
                    # Get pose of child joint
                    base_link_child_xyz_str = base_link_child_origin.get('xyz', '0 0 0')
                    base_link_child_rpy_str = base_link_child_origin.get('rpy', '0 0 0')
                    base_link_child_xyz = [float(x) for x in base_link_child_xyz_str.split()]
                    base_link_child_rpy = [float(x) for x in base_link_child_rpy_str.split()]
                    base_link_child_pose = self._pose_to_matrix(base_link_child_xyz, base_link_child_rpy)
                    # Calculate and set new child pose
                    new_base_link_child_pose = np.dot(inv_current_pose, base_link_child_pose)
                    new_base_link_child_xyz, new_base_link_child_rpy = self._matrix_to_pose(new_base_link_child_pose)
                    base_link_child_origin.set('xyz', ' '.join(map(str, new_base_link_child_xyz)))
                    base_link_child_origin.set('rpy', ' '.join(map(str, new_base_link_child_rpy)))

        # Correct relative positions of child joints which are not on the path
        children_joints = self._get_all_children_joints(joint)
        for child_joint in children_joints:
            if self._joint_is_included_in_path(child_joint, path_to_current_root):
                continue
            child_origin = child_joint.find('origin')
            if child_origin is not None:
                # Get pose of child joint
                child_xyz_str = child_origin.get('xyz', '0 0 0')
                child_rpy_str = child_origin.get('rpy', '0 0 0')
                child_xyz = [float(x) for x in child_xyz_str.split()]
                child_rpy = [float(x) for x in child_rpy_str.split()]
                child_pose = self._pose_to_matrix(child_xyz, child_rpy)

                # Calculate new pose of child joint
                new_child_pose = np.dot(new_pose, child_pose)
                new_child_xyz, new_child_rpy = self._matrix_to_pose(new_child_pose)

                # Set new pose to child joint
                child_origin.set('xyz', ' '.join(map(str, new_child_xyz)))
                child_origin.set('rpy', ' '.join(map(str, new_child_rpy)))

    def _pose_to_matrix(self, xyz, rpy):
        rot = R.from_euler('xyz', rpy)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()
        T[:3, 3] = xyz
        return T

    def _matrix_to_pose(self, T):
        xyz = T[:3, 3]
        rot = R.from_matrix(T[:3, :3])
        rpy = rot.as_euler('xyz')
        return xyz.tolist(), rpy.tolist()

    def _get_all_children_links(self, parent_link):
        parent_link_name = parent_link.get('link')
        for link_name, info in self.joint_tree.items():
            if link_name == parent_link_name:
                children_names = info['children']
                children_links = [self.links[name] for name in children_names if name in self.links]
                return children_links

    def _get_all_children_joints(self, parent_joint):
        children_joints = []
        link_name = parent_joint.find('child').get('link')
        for joint_name, joint in self.joints.items():
            parent_link_name = joint.find('parent').get('link')
            if parent_link_name == link_name:
                children_joints.append(joint)
        return children_joints

    def _get_all_children_joints_of_link(self, link_name):
        children_joints = []
        for joint_name, joint in self.joints.items():
            parent_link_name = joint.find('parent').get('link')
            if parent_link_name == link_name:
                children_joints.append(joint)
        return children_joints

    def _link_is_included_in_path(self, target_link, path):
        target_link_name = target_link.get('name')
        for elem in path:
            parent_link_name = elem[0]
            child_link_name = elem[1]
            if (target_link_name == parent_link_name) or (target_link_name == child_link_name):
                return True
        return False

    def _joint_is_included_in_path(self, target_joint, path):
        target_joint_name = target_joint.get('name')
        for elem in path:
            joint_name = elem[2]
            if joint_name == target_joint_name:
                return True
        return False

    def _save_urdf(self, output_path):
        """Save the modified URDF to a file.

        Parameters
        ----------
        output_path : str
            Path where the URDF will be saved
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write the XML tree to file
        self.tree.write(output_path, encoding='utf-8', xml_declaration=True)

        # Format the output for better readability
        self._format_xml_file(output_path)

    def _format_xml_file(self, file_path):
        """Format XML file for better readability.

        Parameters
        ----------
        file_path : str
            Path to the XML file to format
        """
        try:
            import xml.dom.minidom

            # Parse and format
            dom = xml.dom.minidom.parse(file_path)
            formatted_xml = dom.toprettyxml(indent="  ")

            # Remove empty lines
            lines = [line for line in formatted_xml.split('\n')
                     if line.strip()]
            formatted_xml = '\n'.join(lines)

            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_xml)
        except ImportError:
            # If minidom is not available, keep the original formatting
            pass


def change_urdf_root_link(urdf_path, new_root_link, output_path):
    """Change the root link of a URDF file.

    Parameters
    ----------
    urdf_path : str
        Path to the input URDF file
    new_root_link : str
        Name of the new root link
    output_path : str
        Path where the modified URDF will be saved
    """
    changer = URDFXMLRootLinkChanger(urdf_path)
    changer.change_root_link(new_root_link, output_path)
