from __future__ import division

import collections
import os
from pathlib import Path
import platform
import threading

import numpy as np
from PIL import Image
import pyglet
from pyglet import compat_platform


# WSL2 Ubuntu 20.04 specific fix for pyrender
# Check if running in WSL2 environment
if platform.system() == 'Linux':
    # Check for WSL-specific indicators
    if os.path.exists('/proc/version'):
        try:
            with open('/proc/version', 'r') as f:
                version_info = f.read().lower()
        except (OSError, IOError, PermissionError):
            version_info = ""
        if 'microsoft' in version_info or 'wsl' in version_info:
            # Check Ubuntu version
            if os.path.exists('/etc/os-release'):
                try:
                    with open('/etc/os-release', 'r') as f:
                        os_info = f.read()
                except (OSError, IOError, PermissionError):
                    os_info = ""
                if '20.04' in os_info:
                    # Set PYOPENGL_PLATFORM to GLX for WSL2 Ubuntu 20.04
                    # This ensures proper rendering with X11 forwarding
                    if 'PYOPENGL_PLATFORM' not in os.environ:
                        os.environ['PYOPENGL_PLATFORM'] = 'glx'

import pyrender
from pyrender.trackball import Trackball
import trimesh
from trimesh import transformations
from trimesh.scene import cameras

from skrobot.coordinates import Coordinates

from .. import model as model_module


def _redraw_all_windows():
    try:
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event('on_draw')
            window.flip()
            window._legacy_invalid = False
    except RuntimeError:
        pass


class PyrenderViewer(pyrender.Viewer):

    """PyrenderViewer class implemented as a Singleton.

    This ensures that only one instance of the viewer
    is created throughout the program. Any subsequent attempts to create a new
    instance will return the existing one.

    Parameters
    ----------
    resolution : tuple, optional
        The resolution of the viewer. Default is (640, 480).
    update_interval : float, optional
        The update interval (in seconds) for the viewer. Default is
        1.0 seconds.
    title : str, optional
        The title of the viewer window. Default is 'scikit-robot PyrenderViewer'.
    enable_collision_toggle : bool, optional
        Enable collision/visual mesh toggle functionality with 'v' key.
        Default is True.

    Notes
    -----
    Since this is a singleton, the __init__ method might be called
    multiple times, but only one instance is actually used.

    Keyboard Controls
    -----------------
    v : Toggle between visual and collision meshes (if enable_collision_toggle=True)
        Collision meshes are displayed in orange/transparent color
    """

    # Class variable to hold the single instance of the class.
    _instance = None

    def __init__(self, resolution=None, update_interval=1.0,
                 render_flags=None, title=None, enable_collision_toggle=True):
        if getattr(self, '_initialized', False):
            return
        if resolution is None:
            resolution = (640, 480)

        self.thread = None
        self._visual_mesh_map = collections.OrderedDict()

        # Collision toggle functionality
        self.enable_collision_toggle = enable_collision_toggle
        if self.enable_collision_toggle:
            self._stored_links = []
            self.show_collision = False

        self._redraw = True

        refresh_rate = 1.0 / update_interval
        self._kwargs = dict(
            scene=pyrender.Scene(),
            viewport_size=resolution,
            run_in_thread=False,
            use_raymond_lighting=True,
            auto_start=False,
            render_flags=render_flags,
            refresh_rate=refresh_rate,
        )
        super(PyrenderViewer, self).__init__(**self._kwargs)
        window_title = title if title is not None else 'scikit-robot PyrenderViewer'
        self.viewer_flags['window_title'] = window_title
        self._initialized = True

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PyrenderViewer, cls).__new__(cls)
        return cls._instance

    def show(self):
        if self.thread is not None and self.thread.is_alive():
            return
        distance = self._calculate_camera_distance()
        self.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)],
                        distance=distance)
        if compat_platform == 'darwin':
            self._init_and_start_app()
            init_loop = 30
            for _ in range(init_loop):
                _redraw_all_windows()
        else:
            self.thread = threading.Thread(target=self._init_and_start_app)
            self.thread.daemon = True  # terminate when main thread exit
            self.thread.start()

    def _init_and_start_app(self):
        # Try multiple configs starting with target OpenGL version
        # and multisampling and removing these options if exception
        # Note: multisampling not available on all hardware
        from pyglet import clock
        from pyglet.gl import Config
        from pyrender.constants import MIN_OPEN_GL_MAJOR
        from pyrender.constants import MIN_OPEN_GL_MINOR
        from pyrender.constants import TARGET_OPEN_GL_MAJOR
        from pyrender.constants import TARGET_OPEN_GL_MINOR
        from pyrender.viewer import Viewer
        confs = [Config(sample_buffers=1, samples=4,
                        depth_size=24,
                        double_buffer=True,
                        major_version=TARGET_OPEN_GL_MAJOR,
                        minor_version=TARGET_OPEN_GL_MINOR),
                 Config(depth_size=24,
                        double_buffer=True,
                        major_version=TARGET_OPEN_GL_MAJOR,
                        minor_version=TARGET_OPEN_GL_MINOR),
                 Config(sample_buffers=1, samples=4,
                        depth_size=24,
                        double_buffer=True,
                        major_version=MIN_OPEN_GL_MAJOR,
                        minor_version=MIN_OPEN_GL_MINOR),
                 Config(depth_size=24,
                        double_buffer=True,
                        major_version=MIN_OPEN_GL_MAJOR,
                        minor_version=MIN_OPEN_GL_MINOR)]
        for conf in confs:
            try:
                super(Viewer, self).__init__(config=conf, resizable=True,
                                             width=self._viewport_size[0],
                                             height=self._viewport_size[1])
                break
            except (pyglet.window.NoSuchConfigException, pyglet.gl.ContextException):
                pass
            except pyglet.canvas.xlib.NoSuchDisplayException:
                print('No display found. Viewer is disabled.')
                self.has_exit = True
                return

        if not self.context:
            raise ValueError('Unable to initialize an OpenGL 3+ context')
        clock.schedule_interval(
            Viewer._time_event, 1.0 / self.viewer_flags['refresh_rate'], self
        )
        self.switch_to()
        self.set_caption(self.viewer_flags['window_title'])
        if compat_platform != 'darwin':
            pyglet.app.run()

    def redraw(self):
        self._redraw = True
        if compat_platform == 'darwin':
            _redraw_all_windows()

    def on_draw(self):
        if not self._redraw:
            with self._render_lock:
                super(PyrenderViewer, self).on_draw()
            return

        with self._render_lock:
            # apply latest angle-vector
            for link_id, (node, link) in self._visual_mesh_map.items():
                link.update(force=True)
                transform = link.worldcoords().T()
                if link.visual_mesh_changed:
                    mesh = link.concatenated_visual_mesh
                    pyrender_mesh = pyrender.Mesh.from_trimesh(
                        mesh, smooth=False)
                    self.scene.remove_node(node)
                    node = self.scene.add(pyrender_mesh, pose=transform)
                    self._visual_mesh_map[link_id] = (node, link)
                    link._visual_mesh_changed = False
                else:
                    node.matrix = transform
            super(PyrenderViewer, self).on_draw()

        self._redraw = False

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, symbol, modifiers, *args, **kwargs):
        """Handle key press events with collision toggle support."""
        # Handle 'v' key for collision toggle if enabled
        if self.enable_collision_toggle:
            from pyglet.window import key
            if symbol == key.V:
                # Toggle display mode
                self.show_collision = not self.show_collision

                # Rebuild scene with current mesh type
                self._rebuild_scene_for_toggle()

                mode_text = "Collision" if self.show_collision else "Visual"
                print(f"Switched to {mode_text.lower()} mesh display")

                self._redraw = True
                return True

        self._redraw = True
        return super(PyrenderViewer, self).on_key_press(symbol, modifiers, *args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super(PyrenderViewer, self).on_resize(*args, **kwargs)

    def _add_link(self, link):
        assert isinstance(link, model_module.Link)

        with self._render_lock:
            transform = link.worldcoords().T()
            link_id = str(id(link))
            mesh = link.concatenated_visual_mesh

            if link_id not in self._visual_mesh_map and mesh:
                node = None
                if isinstance(mesh, trimesh.path.Path3D):
                    pyrender_mesh = pyrender.Mesh(
                        primitives=[pyrender.Primitive(
                            mesh.vertices[mesh.vertex_nodes].reshape(-1, 3),
                            mode=pyrender.constants.GLTF.LINE_STRIP,
                            color_0=mesh.colors)])
                    node = self.scene.add(pyrender_mesh)
                elif isinstance(mesh, trimesh.PointCloud):
                    pyrender_mesh = pyrender.Mesh(
                        primitives=[pyrender.Primitive(
                            mesh.vertices,
                            mode=pyrender.constants.GLTF.POINTS,
                            color_0=mesh.colors)])
                    node = self.scene.add(pyrender_mesh)
                else:
                    pyrender_mesh = pyrender.Mesh.from_trimesh(
                        mesh, smooth=False)
                    # Check if the mesh has vertices
                    # before adding it to the scene
                    if len(mesh.vertices) != 0:
                        node = self.scene.add(pyrender_mesh, pose=transform)
                # Add the node and link to the
                # visual mesh map only if the node is successfully created
                if node is not None:
                    self._visual_mesh_map[link_id] = (node, link)

        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        # Store links for collision toggle if enabled
        if self.enable_collision_toggle:
            for link in links:
                if link not in self._stored_links:
                    self._stored_links.append(link)

        for link in links:
            self._add_link(link)

        self._redraw = True

    def delete(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        with self._render_lock:
            all_links = links
            while all_links:
                link = all_links[0]
                link_id = str(id(link))
                if link_id in self._visual_mesh_map:
                    self.scene.remove_node(self._visual_mesh_map[link_id][0])
                    self._visual_mesh_map.pop(link_id)
                all_links = all_links[1:]
                all_links.extend(link.child_links)
        self._redraw = True

    def set_camera(self, angles=None, distance=None, center=None,
                   resolution=None, fov=None, coords_or_transform=None):
        if angles is None and coords_or_transform is None:
            return
        if angles is not None:
            if fov is None:
                fov = np.array([60, 45])
            rotation = transformations.euler_matrix(*angles)
            pose = cameras.look_at(
                self.scene.bounds, fov=fov, rotation=rotation,
                distance=distance, center=center)
        else:
            if isinstance(coords_or_transform, Coordinates):
                pose = coords_or_transform.worldcoords().T()
        self._camera_node.matrix = pose
        self._trackball = Trackball(
            pose=pose,
            size=self.viewport_size,
            scale=self.scene.scale,
            target=self.scene.centroid
        )

    def capture_360_images(self, output_dir, num_frames=36,
                           distance=None, center=None, fov=None,
                           lighting_config=None, camera_elevation=45,
                           distance_margin=1.2, create_gif=True,
                           gif_duration=100, gif_loop=0,
                           transparent_background=True):
        """Capture 360-degree rotation images around the scene.

        Parameters
        ----------
        output_dir : str
            Directory to save the images
        num_frames : int
            Number of images to capture (default: 36, every 10 degrees)
        distance : float, optional
            Camera distance from center
        center : array-like, optional
            Center point to rotate around
        fov : array-like, optional
            Field of view [horizontal, vertical] in degrees
        lighting_config : dict, optional
            Lighting configuration with keys: 'positions', 'colors', 'intensity'
        camera_elevation : float
            Camera elevation angle in degrees (default: 45)
        distance_margin : float
            Margin factor for automatic distance calculation (default: 1.2)
        create_gif : bool
            Whether to create a GIF animation from captured images (default: True)
        gif_duration : int
            Duration between frames in milliseconds for GIF (default: 100)
        gif_loop : int
            Number of loops for GIF (0 = infinite loop, default: 0)
        transparent_background : bool
            Whether to render with transparent background (default: True)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if fov is None:
            fov = np.array([60, 45])

        # Create offscreen renderer
        offscreen_renderer = pyrender.OffscreenRenderer(
            viewport_width=self._viewport_size[0],
            viewport_height=self._viewport_size[1]
        )

        # Setup lighting
        added_lights = self._setup_scene_lighting(lighting_config)

        # Store original camera pose and scene settings to restore later
        original_pose = self._camera_node.matrix.copy()
        original_bg_color = self.scene.bg_color

        # Set transparent background if requested
        if transparent_background:
            self.scene.bg_color = np.array([0.0, 0.0, 0.0, 0.0])

        # Determine render flags
        render_flags = pyrender.RenderFlags.RGBA if transparent_background else pyrender.RenderFlags.NONE

        # Calculate rotation angles
        angles = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)

        try:
            with self._render_lock:
                # Update scene meshes
                self._update_scene_meshes()

                # Calculate optimal camera distance
                camera_distance_calc = self._calculate_camera_distance(distance_margin)

                for i, z_angle in enumerate(angles):
                    # Set camera position with rotation around Z axis
                    camera_angles = [np.deg2rad(camera_elevation), np.deg2rad(0), z_angle]
                    rotation = transformations.euler_matrix(*camera_angles)

                    # Use calculated distance if not provided
                    actual_distance = distance if distance is not None else camera_distance_calc
                    pose = cameras.look_at(
                        self.scene.bounds, fov=fov, rotation=rotation,
                        distance=actual_distance, center=center)

                    # Update camera node matrix
                    self._camera_node.matrix = pose

                    # Render image
                    color, depth = offscreen_renderer.render(self.scene, flags=render_flags)

                    # Save image with appropriate mode
                    if transparent_background and color.shape[2] == 4:
                        image = Image.fromarray(color, mode='RGBA')
                    else:
                        image = Image.fromarray(color)
                    image_path = output_path / f"frame_{i:03d}.png"
                    image.save(image_path)
                    print(f"Saved: {image_path}")
        finally:
            # Clean up
            self._cleanup_scene_lighting(added_lights)
            self._camera_node.matrix = original_pose
            self.scene.bg_color = original_bg_color
            offscreen_renderer.delete()

        print(f"360-degree image capture complete. {num_frames} images saved to {output_dir}")

        # Create GIF animation if requested
        if create_gif:
            gif_path = output_path / "animation.gif"
            self._create_gif_from_images(output_path, gif_path, gif_duration, gif_loop)

    def _create_gif_from_images(self, image_dir, output_gif, duration=100, loop=0):
        """Create GIF animation from captured images.

        Parameters
        ----------
        image_dir : Path
            Directory containing the images
        output_gif : Path
            Output path for the GIF file
        duration : int
            Duration between frames in milliseconds
        loop : int
            Number of loops (0 = infinite loop)
        """
        # Get all PNG files and sort them
        image_files = sorted(image_dir.glob("frame_*.png"))

        if not image_files:
            print("No images found to create GIF")
            return

        # Load images
        images = []
        for img_path in image_files:
            img = Image.open(img_path)
            images.append(img)

        # Save as GIF with proper disposal for transparency
        if images:
            images[0].save(
                output_gif,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=loop,
                optimize=True,
                disposal=2  # Clear frame before rendering next frame
            )
            print(f"GIF animation saved: {output_gif}")

    def _get_default_lighting_config(self):
        """Get default lighting configuration for uniform illumination."""
        return {
            'use_ambient': True,
            'ambient_intensity': 0.2
        }

    def _setup_scene_lighting(self, lighting_config=None):
        """Setup scene lighting and return list of added light nodes."""
        if lighting_config is None:
            lighting_config = self._get_default_lighting_config()

        added_lights = []

        if lighting_config.get('use_ambient', True):
            # Use uniform ambient lighting for shadowless rendering
            ambient_intensity = lighting_config.get('ambient_intensity', 0.2)

            # Set ambient light on the scene
            self.scene.ambient_light = np.array([ambient_intensity,
                                                 ambient_intensity,
                                                 ambient_intensity])

            # Add a single directional light for some definition
            # Scale its intensity with ambient_intensity
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                              intensity=ambient_intensity * 2.0)
            pose = np.eye(4)
            # Point downward from above
            pose[:3, :3] = transformations.euler_matrix(np.pi / 4, 0, 0)[:3, :3]
            light_node = self.scene.add(light, pose=pose)
            added_lights.append(light_node)
        else:
            # Use traditional point lights if specified
            bounds = self.scene.bounds
            center_z = (bounds[0][2] + bounds[1][2]) / 2

            positions = lighting_config.get('positions', [])
            colors = lighting_config.get('colors', [[1.0, 1.0, 1.0]] * len(positions))
            intensity = lighting_config.get('intensity', 10.0)
            distance_factors = lighting_config.get('distance_factors', [3] * len(positions))

            for i, (pos_factor, color) in enumerate(zip(positions, colors)):
                if isinstance(pos_factor, tuple):
                    pos_factor, height_offset = pos_factor
                else:
                    height_offset = [0, 0, 1]

                distance_factor = distance_factors[i] if i < len(distance_factors) else 3
                pos = [pos_factor[0] * distance_factor, pos_factor[1] * distance_factor,
                       center_z + height_offset[2] * distance_factor]

                light = pyrender.PointLight(color=color, intensity=intensity)
                pose = np.eye(4)
                pose[:3, 3] = pos
                light_node = self.scene.add(light, pose=pose)
                added_lights.append(light_node)

        return added_lights

    def _cleanup_scene_lighting(self, light_nodes):
        """Remove lighting nodes from scene and reset ambient light."""
        for light_node in light_nodes:
            self.scene.remove_node(light_node)
        # Reset ambient light to default
        self.scene.ambient_light = np.array([0., 0., 0.])

    def _update_scene_meshes(self):
        """Update scene meshes with latest transforms."""
        for link_id, (node, link) in self._visual_mesh_map.items():
            link.update(force=True)
            transform = link.worldcoords().T()
            if link.visual_mesh_changed:
                mesh = link.concatenated_visual_mesh
                pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
                self.scene.remove_node(node)
                node = self.scene.add(pyrender_mesh, pose=transform)
                self._visual_mesh_map[link_id] = (node, link)
                link._visual_mesh_changed = False
            else:
                node.matrix = transform

    def _calculate_camera_distance(self, distance_margin=1.2):
        """Calculate optimal camera distance based on scene bounds."""
        bounds = self.scene.bounds
        bbox_diagonal = np.linalg.norm(bounds[1] - bounds[0])
        return bbox_diagonal * distance_margin

    def _rebuild_scene_for_toggle(self):
        """Completely rebuild the scene with current mesh type for toggle functionality."""
        if not self.enable_collision_toggle:
            return

        with self._render_lock:
            # Clear all mesh nodes but preserve camera and lights
            mesh_nodes_to_remove = []

            for node in list(self.scene.nodes):
                if node.camera is None and node.light is None and node.mesh is not None:
                    mesh_nodes_to_remove.append(node)

            # Remove mesh nodes
            for node in mesh_nodes_to_remove:
                self.scene.remove_node(node)

            # Clear visual mesh map
            self._visual_mesh_map.clear()

            # Add meshes for current mode
            for link in self._stored_links:
                self._add_single_link_mesh_for_toggle(link)

    def _add_single_link_mesh_for_toggle(self, link):
        """Add a single mesh (visual or collision) for a link during toggle."""
        if not isinstance(link, model_module.Link):
            return

        link_id = str(id(link))
        transform = link.worldcoords().T()

        # Choose mesh based on current mode
        if self.show_collision:
            mesh = link.collision_mesh
            # Process collision mesh with orange coloring
            if mesh is not None:
                if isinstance(mesh, list):
                    colored_meshes = []
                    for m in mesh:
                        colored_mesh = m.copy()
                        colored_mesh.visual.face_colors = [255, 150, 100, 200]
                        colored_meshes.append(colored_mesh)
                    if colored_meshes:
                        mesh = trimesh.util.concatenate(colored_meshes)
                    else:
                        mesh = None
                else:
                    mesh = mesh.copy()
                    mesh.visual.face_colors = [255, 150, 100, 200]
        else:
            mesh = link.concatenated_visual_mesh

        if mesh is not None and len(mesh.vertices) > 0:
            # Create pyrender mesh
            pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

            # Create node with transformation matrix
            node = pyrender.Node(
                name=f"{'collision' if self.show_collision else 'visual'}_{link.name}_{link_id}",
                mesh=pyrender_mesh,
                matrix=transform
            )

            # Add to scene
            self.scene.add_node(node)

            # Update visual mesh map for compatibility (only for visual meshes)
            if not self.show_collision:
                self._visual_mesh_map[link_id] = (node, link)

        # Process child links
        for child_link in link.child_links:
            self._add_single_link_mesh_for_toggle(child_link)
