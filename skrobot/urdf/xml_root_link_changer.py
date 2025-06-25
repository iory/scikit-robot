import os
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation as R


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

        # Build path from current root to new root
        # path_to_new_root = self._find_path_to_link(current_root, new_root_link)
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

        # Cache origin/xyz,rpy values of each joint
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

        prev_joint = None
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

                    # Reverse the joint transformation if needed
                    prev_joint_xyz = None
                    prev_joint_rpy = None
                    if prev_joint is not None:
                        prev_joint_xyz = joint_xyz_rpy_cache[prev_joint][0]
                        prev_joint_rpy = joint_xyz_rpy_cache[prev_joint][1]
                    self._reverse_joint_transform_2(joint, prev_joint_xyz, prev_joint_rpy)
                    prev_joint = joint

    def _get_inversed_joint_origin(self, xyz, rpy):
        if xyz is not None and rpy is not None:
            # Calculate inversed transform of origin
            rot = R.from_euler('xyz', rpy)
            rot_matrix = rot.as_matrix()
            rot_matrix_inv = rot_matrix.T
            xyz_reversed = -np.dot(rot_matrix_inv, xyz)
            rpy_reversed = R.from_matrix(rot_matrix_inv).as_euler('xyz')
            xyz_reversed.tolist()
            rpy_reversed.tolist()

            return xyz_reversed, rpy_reversed

    # prev_joint is None: child of this joint is new root link
    def _reverse_joint_transform_2(self, joint, prev_joint_xyz, prev_joint_rpy):
        origin = joint.find('origin')
        if prev_joint_xyz is None and prev_joint_rpy is None:
            # Set origin of this joint to Zero
            origin.set('xyz', ' '.join(map(str, [0, 0, 0])))
            origin.set('rpy', ' '.join(map(str, [0, 0, 0])))
        else:
            # Set the reversed values of previous joint
            xyz_reversed, rpy_reversed = self._get_inversed_joint_origin(prev_joint_xyz, prev_joint_rpy)
            origin.set('xyz', ' '.join(map(str, xyz_reversed)))
            origin.set('rpy', ' '.join(map(str, rpy_reversed)))

    def _reverse_joint_transform(self, joint):
        """Reverse the transformation of a joint.

        Parameters
        ----------
        joint : ET.Element
            Joint XML element to reverse
        """
        # Find the origin element
        origin = joint.find('origin')
        if origin is not None:
            # Get current xyz and rpy
            xyz_str = origin.get('xyz', '0 0 0')
            rpy_str = origin.get('rpy', '0 0 0')

            # Parse the values
            xyz = [float(x) for x in xyz_str.split()]
            rpy = [float(x) for x in rpy_str.split()]

            # For simplicity, we negate the translation and rotation
            # In a full implementation, you would need proper matrix inversion
            # xyz_reversed = [-x for x in xyz]
            # rpy_reversed = [-r for r in rpy]

            # ============
            rot = R.from_euler('xyz', rpy)
            rot_matrix = rot.as_matrix()
            rot_matrix_inv = rot_matrix.T
            xyz_reversed = -np.dot(rot_matrix_inv, xyz)
            rpy_reversed = R.from_matrix(rot_matrix_inv).as_euler('xyz')
            xyz_reversed.tolist()
            rpy_reversed.tolist()
            # ============

            # Set the reversed values
            origin.set('xyz', ' '.join(map(str, xyz_reversed)))
            origin.set('rpy', ' '.join(map(str, rpy_reversed)))

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
