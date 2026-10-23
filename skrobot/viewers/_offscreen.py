"""Render robots and scenes to images without a window, in one or several processes.

The interactive viewers redraw on a display; this draws frames on demand (EGL or
OSMesa through pyrender, so ``PYOPENGL_PLATFORM=egl`` on a headless machine) for
videos of a trajectory:

>>> from skrobot.viewers import OffscreenRenderer          # doctest: +SKIP
>>> r = OffscreenRenderer((640, 480))                       # doctest: +SKIP
>>> r.add(robot)                                            # doctest: +SKIP
>>> for av in trajectory:                                   # doctest: +SKIP
...     robot.angle_vector(av)
...     rgb = r.render(eye=[2, -2, 1.5], target=[0, 0, 0.8])

A robot is drawn one mesh per link (``link.concatenated_visual_mesh``, as
PyrenderViewer does) and posed from its links at every ``render``: pyrender's
Python-side cost is per mesh and per frame, so a robot of hundreds of visual
pieces drawn one by one costs more than the GPU does.

:func:`render_parallel` splits frames over processes, each with its own scene
and GL context.

This module does not import pyglet, so it loads where no display exists.
"""

import multiprocessing

import numpy as np
from PIL import Image
import pyrender
import trimesh
import trimesh.scene.cameras

from skrobot import model as model_module
from skrobot.coordinates import Coordinates


def look_at_pose(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera pose (4, 4) at eye looking at target, in pyrender's (OpenGL) convention.

    The camera looks along its -z, with +y up. (skrobot.coordinates.math's
    look_at_rotation uses the OpenCV convention: +z forward, +y down.)

    Parameters
    ----------
    eye, target : array-like (3,)
    up : array-like (3,)
        World up. Must not be parallel to target - eye.

    Returns
    -------
    pose : numpy.ndarray (4, 4)
    """
    eye = np.asarray(eye, dtype=np.float64)
    z = eye - np.asarray(target, dtype=np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(np.asarray(up, dtype=np.float64), z)
    n = np.linalg.norm(x)
    if n < 1e-9:
        raise ValueError('up {} is parallel to the view direction'.format(up))
    x /= n
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, np.cross(z, x), z, eye
    return pose


def merge_by_material(meshes):
    """Join meshes that draw alike into fewer meshes.

    Meshes join when they share a texture image, or a plain material (colour,
    metallic, roughness), or both carry vertex or face colours. A scanned room of
    ~430 pieces becomes a few dozen, which is most of what a frame costs.

    Parameters
    ----------
    meshes : list of trimesh.Trimesh
        In world coordinates (``trimesh.Scene.dump()``).

    Returns
    -------
    merged : list of trimesh.Trimesh
    """
    groups = {}
    for m in meshes:
        v = m.visual
        mat = getattr(v, 'material', None)
        img = None
        if mat is not None:
            img = getattr(mat, 'baseColorTexture', None)
            if img is None:
                img = getattr(mat, 'image', None)
        if v.kind == 'texture' and img is not None:
            key = ('tex', hash(np.asarray(img).tobytes()))
        elif v.kind == 'texture' and mat is not None:
            c = getattr(mat, 'baseColorFactor', None)
            if c is None:
                c = getattr(mat, 'main_color', None)
            key = ('col', None if c is None else tuple(np.asarray(c).ravel().tolist()),
                   getattr(mat, 'metallicFactor', None),
                   getattr(mat, 'roughnessFactor', None))
        elif v.kind in ('vertex', 'face'):
            key = ('vertex',)
        else:
            key = ('other', id(m))
        groups.setdefault(key, []).append(m)
    merged = []
    for key, ms in groups.items():
        if len(ms) == 1 or key[0] == 'other':
            merged.extend(ms)
            continue
        if key[0] == 'vertex':
            merged.append(trimesh.util.concatenate(
                [trimesh.Trimesh(m.vertices, m.faces,
                                 vertex_colors=m.visual.vertex_colors,
                                 process=False)
                 for m in ms]))
            continue
        verts = np.concatenate([m.vertices for m in ms])
        offsets = np.cumsum([0] + [len(m.vertices) for m in ms[:-1]])
        faces = np.concatenate([m.faces + o for m, o in zip(ms, offsets)])
        material = ms[0].visual.material
        if key[0] == 'tex':
            uv = np.concatenate([
                m.visual.uv if m.visual.uv is not None
                else np.zeros((len(m.vertices), 2)) for m in ms])
            visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        else:
            visual = trimesh.visual.TextureVisuals(material=material)
        merged.append(trimesh.Trimesh(verts, faces, visual=visual,
                                      process=False))
    return merged


def _camera_lights(intensity):
    """pyrender.Viewer's 'raymond' lights: three directional lights fixed to the camera,
    which PyrenderViewer turns on (use_raymond_lighting, lighting_intensity 3 / 3 each)."""
    nodes = []
    for phi in np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0]):
        theta = np.pi / 6.0
        z = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
        x = np.array([-z[1], z[0], 0.0])
        x /= np.linalg.norm(x)
        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, np.cross(z, x), z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
            matrix=matrix))
    return nodes


class OffscreenRenderer(object):
    """A pyrender scene of robots and meshes, rendered to arrays on demand.

    It takes the viewers' calls too (``add``, ``delete``, ``set_camera``,
    ``redraw``, ``show``, ``save_image``), so a script written for a viewer runs
    headless with ``create_viewer('offscreen')``; and :meth:`from_viewer` copies
    what an open PyrenderViewer or TrimeshSceneViewer shows (its links, camera
    and size) to render it without a window.

    By default it lights the scene as PyrenderViewer does (no ambient light,
    three directional lights fixed to the camera), so a frame looks like the
    viewer's.

    Parameters
    ----------
    resolution : tuple of int
        (width, height) in pixels.
    yfov : float
        Vertical field of view [rad]. PyrenderViewer's camera has pi / 3.
    ambient_light : array-like (3,)
    bg_color : array-like (4,)
        RGBA in [0, 1].
    camera_lights : float or None
        Intensity of each of the three lights fixed to the camera; None for
        none (light the scene yourself with :meth:`add_light`).
    """

    def __init__(self, resolution=(640, 480), yfov=np.pi / 3.0,
                 ambient_light=(0.0, 0.0, 0.0), bg_color=(1, 1, 1, 1),
                 camera_lights=1.0):
        if resolution is None:
            resolution = (640, 480)
        self.resolution = tuple(int(v) for v in resolution)
        # floats: pyrender reads integer colours as 0-255, so (1, 1, 1, 1) would be black
        self.scene = pyrender.Scene(ambient_light=np.asarray(ambient_light, dtype=np.float64),
                                    bg_color=np.asarray(bg_color, dtype=np.float64))
        self._camera = self.scene.add(pyrender.PerspectiveCamera(
            yfov=yfov, aspectRatio=self.resolution[0] / float(self.resolution[1])),
            pose=np.eye(4))
        if camera_lights is not None:
            for node in _camera_lights(camera_lights):
                self.scene.add_node(node, parent_node=self._camera)
        self._renderer = pyrender.OffscreenRenderer(*self.resolution)
        self._links = []   # (node, link): posed from the link at each render
        self._camera_set = False
        self.has_exit = False

    @classmethod
    def from_viewer(cls, viewer, resolution=None, **kwargs):
        """An OffscreenRenderer showing what ``viewer`` shows now.

        Copies the links added to a PyrenderViewer or TrimeshSceneViewer (they
        keep following their robots), the camera pose and field of view, and the
        window size unless ``resolution`` is given.

        Parameters
        ----------
        viewer : PyrenderViewer or TrimeshSceneViewer
        resolution : tuple of int, optional
        **kwargs
            Passed to the constructor (``bg_color``, ``camera_lights``, ...).
        """
        if hasattr(viewer, '_visual_mesh_map'):          # PyrenderViewer
            links = [link for _, link in viewer._visual_mesh_map.values()]
            pose = np.array(viewer._camera_node.matrix)
            yfov = viewer._camera_node.camera.yfov
            size = viewer.viewport_size
        elif hasattr(viewer, '_links') and hasattr(viewer, 'scene'):   # TrimeshSceneViewer
            links = list(viewer._links.values())
            pose = np.array(viewer.scene.camera_transform)
            yfov = np.deg2rad(viewer.scene.camera.fov[1])
            size = viewer.scene.camera.resolution
        else:
            raise TypeError('cannot copy a {}'.format(type(viewer).__name__))
        kwargs.setdefault('yfov', yfov)
        r = cls(resolution=size if resolution is None else resolution, **kwargs)
        seen = set()
        for link in links:
            if id(link) not in seen:
                seen.add(id(link))
                r._add_link(link, merge=True, smooth=False)
        r.set_camera(coords_or_transform=pose)
        return r

    def add(self, obj, pose=None, merge=True, smooth=False):
        """Add a robot, a link, a mesh or a scene.

        Parameters
        ----------
        obj : skrobot.model.CascadedLink, skrobot.model.Link, trimesh.Trimesh or trimesh.Scene
            A robot or a link follows its links' world poses at every
            :meth:`render`. A mesh is placed at ``pose`` and can be moved with
            :meth:`set_pose`. A scene is static, its meshes joined by
            :func:`merge_by_material` unless ``merge`` is False.
        pose : numpy.ndarray (4, 4), optional
            For a mesh. Default identity.
        merge : bool
            For a robot: one mesh per link (default) or one per visual piece.
            For a scene: join look-alike pieces (default) or keep them apart.
        smooth : bool
            Smooth shading (pyrender's ``from_trimesh``). Default False, as in
            PyrenderViewer.

        Returns
        -------
        nodes : pyrender.Node or list of pyrender.Node
            One node for a mesh (pass it to :meth:`set_pose`), else a list.
        """
        if isinstance(obj, model_module.CascadedLink):
            nodes = []
            for link in obj.link_list:
                nodes.extend(self._add_link(link, merge, smooth))
            return nodes
        if isinstance(obj, model_module.Link):
            return self._add_link(obj, merge, smooth)
        if isinstance(obj, trimesh.Scene):
            meshes = [m for m in obj.dump() if len(m.faces)]
            if merge:
                meshes = merge_by_material(meshes)
            return [self.scene.add(pyrender.Mesh.from_trimesh(m, smooth=smooth))
                    for m in meshes]
        if isinstance(obj, trimesh.Trimesh):
            return self.scene.add(pyrender.Mesh.from_trimesh(obj, smooth=smooth),
                                  pose=np.eye(4) if pose is None else pose)
        raise TypeError('cannot add {}'.format(type(obj).__name__))

    def _add_link(self, link, merge, smooth):
        if any(lk is link for _, lk in self._links):
            return []   # as the viewers do: a link added twice is drawn once
        if merge:
            meshes = [link.concatenated_visual_mesh]
        else:
            meshes = link.visual_mesh
            if not isinstance(meshes, (list, tuple)):
                meshes = [meshes]
        nodes = []
        for m in meshes:
            if not isinstance(m, trimesh.Trimesh) or not len(m.faces):
                continue
            node = self.scene.add(pyrender.Mesh.from_trimesh(m, smooth=smooth))
            self._links.append((node, link))
            nodes.append(node)
        return nodes

    def add_light(self, light=None, pose=None, eye=None, target=None):
        """Add a light (default a directional one of intensity 2.5).

        Place it by ``pose``, or by ``eye`` and ``target`` (it shines from eye
        toward target).
        """
        if light is None:
            light = pyrender.DirectionalLight(intensity=2.5)
        if pose is None:
            pose = np.eye(4) if eye is None else look_at_pose(eye, target)
        return self.scene.add(light, pose=pose)

    def set_pose(self, node, pose):
        """Move a mesh added by :meth:`add`."""
        self.scene.set_pose(node, np.asarray(pose))

    def remove(self, nodes):
        """Remove what :meth:`add` added (a node or a list of nodes)."""
        if isinstance(nodes, pyrender.Node):
            nodes = [nodes]
        gone = set(nodes)
        self._links = [(n, lk) for n, lk in self._links if n not in gone]
        for n in nodes:
            self.scene.remove_node(n)

    def delete(self, geometry):
        """Remove a robot or a link added by :meth:`add` (the viewers' delete)."""
        if isinstance(geometry, model_module.CascadedLink):
            links = set(id(lk) for lk in geometry.link_list)
        elif isinstance(geometry, model_module.Link):
            links = {id(geometry)}
        else:
            raise TypeError('geometry must be Link or CascadedLink')
        self.remove([n for n, lk in self._links if id(lk) in links])

    def _pose_links(self):
        for node, link in self._links:
            self.scene.set_pose(node, link.worldcoords().T())

    def set_camera(self, angles=None, distance=None, center=None,
                   resolution=None, fov=None, coords_or_transform=None):
        """Place the camera as PyrenderViewer.set_camera does.

        Parameters
        ----------
        angles : array-like (3,), optional
            Euler angles [rad] of a view of the whole scene (trimesh's
            ``cameras.look_at`` on the scene's bounds).
        distance : float, optional
            From ``center``; default fits the scene in ``fov``.
        center : array-like (3,), optional
            Default the scene's centre.
        resolution : ignored
            Fixed by the constructor (the GL buffers are sized there).
        fov : array-like (2,), optional
            [deg], to fit the scene. Default [60, 45].
        coords_or_transform : skrobot.coordinates.Coordinates or (4, 4), optional
            The camera pose itself, instead of ``angles``.
        """
        if angles is None and coords_or_transform is None:
            return
        if angles is not None:
            self._pose_links()
            rotation = trimesh.transformations.euler_matrix(*angles)
            pose = trimesh.scene.cameras.look_at(
                self.scene.bounds, fov=np.array([60, 45]) if fov is None else fov,
                rotation=rotation, distance=distance, center=center)
        elif isinstance(coords_or_transform, Coordinates):
            pose = coords_or_transform.worldcoords().T()
        else:
            pose = np.asarray(coords_or_transform, dtype=np.float64)
            if pose.shape != (4, 4):
                raise ValueError('coords_or_transform must be Coordinates or (4, 4), '
                                 'got shape {}'.format(pose.shape))
        self.scene.set_pose(self._camera, pose)
        self._camera_set = True

    @property
    def camera_pose(self):
        """The camera's pose (4, 4), OpenGL convention."""
        return np.array(self.scene.get_pose(self._camera))

    def render(self, eye=None, target=None, camera_pose=None, up=(0, 0, 1),
               depth=False):
        """Pose the robots' links, then render one view.

        Parameters
        ----------
        eye, target : array-like (3,), optional
            Camera position and the point it looks at.
        camera_pose : numpy.ndarray (4, 4), optional
            Instead of eye and target (OpenGL convention, see
            :func:`look_at_pose`). If neither is given the camera stays where
            the last call put it; if no call ever placed it, it takes
            PyrenderViewer's first view of the whole scene.
        up : array-like (3,)
        depth : bool
            Also return the depth image.

        Returns
        -------
        color : numpy.ndarray (height, width, 3) uint8
        depth : numpy.ndarray (height, width) float32
            Only with ``depth=True``.
        """
        self._pose_links()
        if camera_pose is None and eye is not None:
            camera_pose = look_at_pose(eye, target, up)
        if camera_pose is not None:
            self.set_camera(coords_or_transform=camera_pose)
        elif not self._camera_set:
            # PyrenderViewer.show's first view: 1.2 x the scene's diagonal away
            bounds = self.scene.bounds
            self.set_camera([np.deg2rad(45), 0.0, np.deg2rad(135)],
                            distance=1.2 * np.linalg.norm(bounds[1] - bounds[0]))
        color, d = self._renderer.render(self.scene)
        return (color, d) if depth else color

    # The viewers' calls, for scripts that take any viewer (create_viewer('offscreen')).

    @property
    def is_active(self):
        return not self.has_exit

    def show(self):
        pass

    def redraw(self):
        pass

    def save_image(self, file_obj):
        """Render the current camera's view and write it (a path or a binary file)."""
        Image.fromarray(self.render()).save(file_obj, format=None if isinstance(file_obj, str) else 'PNG')

    def close(self):
        """Free the GL context."""
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
        self.has_exit = True

    def wait_until_close(self, *args, **kwargs):
        pass

    def pause(self, duration, fps=30.0):
        """Return at once: no window to keep responsive, and waiting only slows a
        headless render down. ``fps`` is checked as the viewers check it."""
        if not (np.isfinite(fps) and fps > 0):
            raise ValueError('fps must be a positive finite number, got {}.'.format(fps))


_WORKER = None


def _init_worker(setup, setup_args):
    global _WORKER
    _WORKER = setup(*setup_args)


def _run_job(job):
    return _WORKER(job)


def render_parallel(setup, jobs, workers=1, setup_args=(), chunksize=4):
    """Run ``setup(*setup_args)(job)`` for every job, split over processes.

    ``setup`` builds a scene (robots, meshes, an :class:`OffscreenRenderer`) and
    returns a function that renders one job, typically saving an image and
    returning its path. Each process calls ``setup`` once, so a scene that takes
    seconds to load is loaded once per process, not per frame.

    Processes are started with 'spawn': a GL context does not survive a fork.
    So ``setup`` must be importable by name (a module-level function, not a
    lambda or a closure), its arguments picklable, and the calling script
    guarded by ``if __name__ == '__main__':``.

    Parameters
    ----------
    setup : callable
        ``setup(*setup_args) -> render_one(job)``.
    jobs : iterable
        One item per frame, passed to ``render_one`` (a time, an index, or
        (time, path)).
    workers : int
        Processes. 1 renders in this process.
    setup_args : tuple
    chunksize : int
        Jobs handed to a process at a time.

    Returns
    -------
    results : list
        ``render_one(job)`` for each job, in the order of ``jobs``.
    """
    jobs = list(jobs)
    if workers <= 1:
        render_one = setup(*setup_args)
        return [render_one(job) for job in jobs]
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(setup, setup_args)) as pool:
        return pool.map(_run_job, jobs, chunksize=chunksize)
