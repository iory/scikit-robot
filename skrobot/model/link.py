try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

import numpy as np

import skrobot
from skrobot._lazy_imports import _lazy_trimesh
from skrobot.coordinates import CascadedCoords


def _color_concatenate(meshes):
    """Concatenate meshes into one, each textured mesh baked to colours.

    ``trimesh.util.concatenate`` packs textured meshes' images into one
    atlas: slow, memory hungry for large images, and wrong for a tiled
    texture (UVs outside 0 .. 1 are squeezed into the atlas cell). The
    concatenated mesh is what the viewers draw as one piece and what
    ``set_color`` paints, so a colour per vertex is what it needs.

    Parameters
    ----------
    meshes : trimesh.Trimesh or sequence of trimesh.Trimesh
        A link's visual meshes.

    Returns
    -------
    trimesh.Trimesh
        The meshes as one, textured meshes carrying ``ColorVisuals``.
    """
    trimesh = _lazy_trimesh()
    if isinstance(meshes, trimesh.Trimesh):
        meshes = [meshes]
    elif not isinstance(meshes, Sequence):
        return trimesh.util.concatenate(meshes)
    colored = []
    for mesh in meshes:
        if not isinstance(mesh, trimesh.Trimesh) \
                or isinstance(mesh.visual, trimesh.visual.ColorVisuals):
            colored.append(mesh)
            continue
        # copy() drops the cache, where normals loaded from the file live
        normals = mesh._cache.cache.get('vertex_normals')
        baked = mesh.copy()
        baked.visual = mesh.visual.to_color()
        if baked.visual.vertex_colors.ndim == 1:
            baked.visual.vertex_colors = \
                baked.visual.vertex_colors[None].repeat(
                    len(baked.vertices), axis=0)
        if normals is not None and len(normals) == len(baked.vertices):
            baked.vertex_normals = normals
        colored.append(baked)
    return trimesh.util.concatenate(colored)


class Link(CascadedCoords):

    def __init__(self, centroid=None,
                 inertia_tensor=None,
                 collision_mesh=None,
                 visual_mesh=None,
                 mass=None,
                 *args, **kwargs):
        super(Link, self).__init__(*args, **kwargs)
        self.centroid = centroid
        self.joint = None
        self._child_links = []
        self._parent_link = None
        if inertia_tensor is None:
            inertia_tensor = np.eye(3)
        self.inertia_tensor = inertia_tensor
        self._collision_mesh = collision_mesh
        self.visual_mesh = visual_mesh
        self._visual_mesh_changed = False
        self._sdf = None

        # Dynamics properties
        self.mass = mass if mass is not None else 1.0  # kg

    @property
    def parent_link(self):
        return self._parent_link

    @property
    def child_links(self):
        return self._child_links

    def add_joint(self, j):
        self.joint = j

    def delete_joint(self):
        self.joint = None

    def add_child_link(self, child_link):
        """Add child link."""
        if child_link is not None and child_link not in self._child_links:
            self._child_links.append(child_link)

    def del_child_link(self, link):
        self._child_links.remove(link)

    def add_parent_link(self, parent_link):
        self._parent_link = parent_link

    def del_parent_link(self):
        self._parent_link = None

    @property
    def collision_mesh(self):
        """Return collision mesh

        Returns
        -------
        self._collision_mesh : trimesh.base.Trimesh
            A single collision mesh for the link.
            specified in the link frame,
            or None if there is not one.
        """
        return self._collision_mesh

    @collision_mesh.setter
    def collision_mesh(self, mesh):
        """Setter of collision mesh

        Parameters
        ----------
        mesh : trimesh.base.Trimesh
            A single collision mesh for the link.
            specified in the link frame,
            or None if there is not one.
        """
        if mesh is None or (isinstance(mesh, Sequence) and len(mesh) == 0):
            self._collision_mesh = None
            return
        trimesh = _lazy_trimesh()
        if mesh is not None and \
           not isinstance(mesh, trimesh.base.Trimesh):
            raise TypeError('input mesh is should be trimesh.base.Trimesh, '
                            'get type {}'.format(type(mesh)))
        self._collision_mesh = mesh

    @property
    def visual_mesh(self):
        """Return visual mesh

        Returns
        -------
        self._visual_mesh : None, trimesh.base.Trimesh, or
                            sequence of trimesh.Trimesh
            A set of visual meshes for the link in the link frame.
        """
        return self._visual_mesh

    @visual_mesh.setter
    def visual_mesh(self, mesh):
        """Setter of visual mesh

        Parameters
        ----------
        mesh : None, trimesh.Trimesh, sequence of trimesh.Trimesh,
               trimesh.points.PointCloud or str
            A set of visual meshes for the link in the link frame.
        """
        if mesh is None or (isinstance(mesh, Sequence) and len(mesh) == 0):
            self._visual_mesh = mesh
            self._concatenated_visual_mesh = None
            self._concatenated_stale = False
            self._color_overridden = False
            self._visual_mesh_changed = True
            return
        trimesh = _lazy_trimesh()
        if not (isinstance(mesh, trimesh.Trimesh)
                or (isinstance(mesh, Sequence)
                    and all(isinstance(m, trimesh.Trimesh) for m in mesh))
                or isinstance(mesh, trimesh.points.PointCloud)
                or isinstance(mesh, trimesh.path.path.Path3D)
                or isinstance(mesh, str)):
            raise TypeError(
                'mesh must be None, trimesh.Trimesh, sequence of '
                'trimesh.Trimesh, trimesh.points.PointCloud '
                'or path of mesh file, but got: {}'.format(type(mesh)))
        if isinstance(mesh, str):
            mesh = trimesh.load(mesh)
        self._visual_mesh = mesh
        # made when first asked for (concatenated_visual_mesh): a textured
        # link that a viewer draws mesh by mesh never needs it
        self._concatenated_visual_mesh = None
        self._concatenated_stale = True
        self._color_overridden = False
        self._visual_mesh_changed = True

    @property
    def visual_mesh_file_paths(self):
        """Return file paths of visual meshes.

        Returns a list of file paths extracted from trimesh metadata.
        Only available when meshes were loaded from URDF files.

        Returns
        -------
        file_paths : list of str
            File paths of the visual meshes.
        """
        if self._visual_mesh is None:
            return []
        meshes = self._visual_mesh
        if not isinstance(meshes, Sequence):
            meshes = [meshes]
        paths = []
        for m in meshes:
            path = getattr(m, 'metadata', {}).get('file_path')
            if path is not None:
                paths.append(path)
        return paths

    @property
    def concatenated_visual_mesh(self):
        """Concatenated visual mesh for visualization.

        Made when first asked for, with textured meshes baked to vertex
        colours (the meshes in ``visual_mesh`` keep their textures).

        Returns
        -------
        self._concatenated_visual_mesh : None, trimesh.base.Trimesh
            A concatenated visual meshes for the link in the link frame.
        """
        if getattr(self, '_concatenated_stale', False):
            self._concatenated_visual_mesh = _color_concatenate(
                self._visual_mesh)
            self._concatenated_stale = False
        return getattr(self, '_concatenated_visual_mesh', None)

    @property
    def color_overridden(self):
        """Whether ``set_color`` or ``set_alpha`` repainted the link.

        Returns
        -------
        bool
            True until ``reset_color`` or a new ``visual_mesh``. A viewer
            that draws the textured ``visual_mesh`` draws
            ``concatenated_visual_mesh`` instead while this is True.
        """
        return getattr(self, '_color_overridden', False)

    @property
    def visual_mesh_changed(self):
        return self._visual_mesh_changed

    @property
    def colors(self):
        mesh = self.concatenated_visual_mesh
        if mesh is not None:
            return mesh.visual.face_colors
        else:
            return None

    def set_color(self, color):
        mesh = self.concatenated_visual_mesh
        if mesh is None:
            return
        color = np.array(color)
        if color.ndim == 2:
            mesh.visual.face_colors = color
        else:
            n_facet = len(mesh.visual.face_colors)
            mesh.visual.face_colors = np.array([color] * n_facet)
        self._color_overridden = True
        self._visual_mesh_changed = True

    def reset_color(self):
        mesh = self.concatenated_visual_mesh
        if mesh is None:
            return
        mesh.visual.face_colors = \
            _color_concatenate(self._visual_mesh).visual.face_colors
        self._color_overridden = False
        self._visual_mesh_changed = True

    def set_alpha(self, alpha):
        """Set alpha (transparency) value for visual mesh.

        Parameters
        ----------
        alpha : float
            Alpha value between 0.0 (transparent) and 1.0 (opaque)
        """
        mesh = self.concatenated_visual_mesh
        if mesh is None:
            return
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure alpha is in valid range
        # Update alpha channel (4th component) of all face colors
        mesh.visual.face_colors[:, 3] = np.round(alpha * 255).astype(np.uint8)
        self._color_overridden = True
        self._visual_mesh_changed = True

    @property
    def sdf(self):
        """Return signed distance function.

        Returns
        -------
        self._sdf : None or skrobot.sdf.SignedDistanceFunction
            signed distance function.
        """
        return self._sdf

    @sdf.setter
    def sdf(self, sdf):
        """Setter of sdf.

        Parameters
        ----------
        sdf : skrobot.sdf.SignedDistanceFunction
            signed distance function.
        """
        if not isinstance(sdf, skrobot.sdf.SignedDistanceFunction):
            raise TypeError('sdf must be skrobot.sdf.SignedDistanceFunction.'
                            ' but is {}'.format(type(sdf)))
        self._sdf = sdf


def _find_link_path(src_link, target_link, previous_link=None,
                    include_target=False):
    if src_link == target_link:
        if include_target:
            return [target_link], True
        else:
            return [], True
    paths = []
    links = []
    if hasattr(src_link, 'parent'):
        links = [src_link.parent]
    if hasattr(src_link, '_descendants'):
        links.extend(src_link._descendants)
    for next_link in links:
        if next_link is None or next_link == previous_link:
            continue
        path, succ = _find_link_path(next_link, target_link, src_link)
        if succ is True:
            paths.append(next_link)
            paths.extend(path)
            return paths, True
    return [], False


def find_link_path(src_link, target_link, include_source=True,
                   include_target=True):
    """Find paths of src_link to target_link

    Parameters
    ----------
    src_link : skrobot.model.link.Link
        source link.
    target_link : skrobot.model.link.Link
        target link.
    include_source : bool
        If `True`, return link list includes `src_link`.
    include_target : bool
        If `True`, return link list includes `target_link`.

    Returns
    -------
    ret : tuple(List[skrobot.model.link.Link], bool)
        If the links are connected, return Link list and `True`.
        Otherwise, return an empty list and `False`.
    """
    paths, succ = _find_link_path(src_link, target_link,
                                  include_target=include_target)
    if succ:
        if include_source:
            return [src_link] + paths, succ
        else:
            return paths, succ
    else:
        return [], False
