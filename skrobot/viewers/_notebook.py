from __future__ import division

import collections

import numpy as np
import trimesh

from skrobot import model as model_module


class JupyterNotebookViewer(object):
    """Jupyter Notebook viewer for scikit-robot.

    This viewer allows you to display robot models interactively
    in Jupyter notebooks using three.js via trimesh's notebook viewer.

    Parameters
    ----------
    resolution : tuple, optional
        The resolution of the viewer (width, height).
        Default is (640, 480).
    height : int, optional
        The height of the iframe in pixels. Default is 500.

    Examples
    --------
    Basic usage in Jupyter notebook:

    >>> import skrobot
    >>> robot = skrobot.models.PR2()
    >>> viewer = skrobot.viewers.JupyterNotebookViewer()
    >>> viewer.add(robot)
    >>> viewer.show()  # Display in Jupyter notebook

    Show different poses (creates new displays, camera view preserved):

    >>> # In the next cell:
    >>> robot.reset_pose()
    >>> viewer.show()  # Creates new display below

    >>> # In the next cell:
    >>> robot.reset_manip_pose()
    >>> viewer.update()  # Same as show()

    Update in place (replaces previous display, but camera view resets):

    >>> robot.reset_pose()
    >>> viewer.show(update_in_place=True)  # Replaces previous display
    """

    def __init__(self, resolution=None, height=500):
        if resolution is None:
            resolution = (640, 480)

        self.resolution = resolution
        self.height = height
        self._links = collections.OrderedDict()
        self._robots = []  # Store robot objects for updating root coords
        self.scene = trimesh.Scene()
        self._display_id = None  # For updating the display in Jupyter
        self._camera_set_by_user = False  # Track if user explicitly set camera

    def _add_link(self, link):
        """Add a link and its children to the scene.

        Parameters
        ----------
        link : skrobot.model.Link
            Link to add to the scene.
        """
        assert isinstance(link, model_module.Link)

        link_id = str(id(link))
        if link_id in self._links:
            return

        transform = link.worldcoords().T()
        mesh = link.concatenated_visual_mesh

        # Handle both single meshes and lists of meshes
        if (isinstance(mesh, list) or isinstance(mesh, tuple)) \
           and len(mesh) > 0:
            for m in mesh:
                link_mesh_id = link_id + str(id(m))
                self.scene.add_geometry(
                    geometry=m,
                    node_name=link_mesh_id,
                    geom_name=link_mesh_id,
                    transform=transform,
                )
                self._links[link_mesh_id] = link
        elif mesh is not None:
            self.scene.add_geometry(
                geometry=mesh,
                node_name=link_id,
                geom_name=link_id,
                transform=transform,
            )
            self._links[link_id] = link

        # Recursively add child links
        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry):
        """Add geometry to the scene.

        Parameters
        ----------
        geometry : skrobot.model.Link or skrobot.model.CascadedLink
            Geometry to add to the scene.
        """
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
            # Store robot object for updating root coordinates
            self._robots.append(geometry)
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        for link in links:
            self._add_link(link)

    def delete(self, geometry):
        """Delete geometry from the scene.

        Parameters
        ----------
        geometry : skrobot.model.Link or skrobot.model.CascadedLink
            Geometry to delete from the scene.
        """
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
            # Remove robot object from tracking
            if geometry in self._robots:
                self._robots.remove(geometry)
        else:
            raise TypeError('geometry must be Link or CascadedLink')

        for link in links:
            link_id = str(id(link))
            if link_id not in self._links:
                continue
            self.scene.delete_geometry(link_id)
            self._links.pop(link_id)

    def set_camera(self, angles=None, distance=None, **kwargs):
        """Set camera position and orientation.

        Parameters
        ----------
        angles : list or None
            Camera angles in radians [azimuth, elevation, roll].
        distance : float or None
            Camera distance from target.
        **kwargs
            Additional arguments passed to trimesh.Scene.set_camera.
        """
        camera_kwargs = kwargs.copy()
        if angles is not None:
            camera_kwargs['angles'] = angles
            # Store angles for HTML export
            self._camera_angles = angles
        if distance is not None:
            camera_kwargs['distance'] = distance
        self.scene.set_camera(**camera_kwargs)
        self._camera_set_by_user = True

    def show(self, update_in_place=False):
        """Display the scene in a Jupyter notebook.

        Automatically updates link transforms before displaying.

        Parameters
        ----------
        update_in_place : bool, optional
            If True, updates the previous display in-place.
            Note: This will reset the camera view to the initial position.
            If False (default), creates a new display below, preserving
            the camera view from any previous displays.
            Default is False.

        Returns
        -------
        IPython.display.HTML or None
            HTML widget containing the 3D scene.

        Notes
        -----
        By default, each call to show() creates a new display output.
        This preserves the camera view that you've adjusted with your mouse.

        If you want to update the same output (to avoid cluttering the notebook),
        set update_in_place=True. However, this will reset the camera view
        each time, which may be disorienting.

        The camera is automatically positioned to provide a good view of the
        scene on first display. You can override this by calling set_camera()
        before show().

        Examples
        --------
        >>> # Create new displays (camera view preserved for each)
        >>> viewer.show()  # Display 1
        >>> robot.reset_pose()
        >>> viewer.show()  # Display 2 - with new pose

        >>> # Update in place using show (also works)
        >>> viewer.show(update_in_place=True)  # Display 1
        >>> robot.reset_pose()
        >>> viewer.show(update_in_place=True)  # Updates Display 1

        >>> # Recommended: use redraw() for updates
        >>> viewer.show()  # Initial display
        >>> robot.reset_pose()
        >>> viewer.redraw()  # Update display
        """
        # If update_in_place, just call redraw()
        if update_in_place and self._display_id is not None:
            self.redraw()
            return None

        # Update transforms before showing
        self.redraw()

        # Set default camera if user hasn't explicitly set it
        # Use angles that provide a good view of the robot standing upright
        if not self._camera_set_by_user and len(self.scene.geometry) > 0:
            # This provides a left-front elevated view with the robot upright
            # in the GLB/three.js coordinate system (Y-up)
            default_angles = [np.deg2rad(-45), np.deg2rad(45), np.deg2rad(135)]
            self.scene.set_camera(angles=default_angles)
            # Store angles for HTML export
            self._camera_angles = default_angles
            # Don't mark as user-set, so it can be recalculated if geometry changes significantly

        # Create new display
        try:
            import uuid

            from IPython.display import display
            from IPython.display import HTML

            self._display_id = str(uuid.uuid4())

            # Get HTML with postMessage listener
            html_str = self.to_html(escape_quotes=True, enable_updates=True)

            html_content = HTML(
                f'<iframe id="viewer-frame-{self._display_id}" srcdoc="{html_str}" '
                f'width="100%" height="{self.height}px" style="border:none;"></iframe>'
            )

            display(html_content, display_id=self._display_id)
            return None

        except ImportError:
            # Not in IPython environment, just return HTML
            return self.to_html(escape_quotes=True)

    def to_html(self, escape_quotes=False, enable_updates=False):
        """Convert the scene to HTML string.

        Parameters
        ----------
        escape_quotes : bool
            If True, escapes quotes for use in srcdoc attribute.
        enable_updates : bool
            If True, enables postMessage listener for transform updates.

        Returns
        -------
        str
            HTML string containing the 3D scene.
        """
        import base64

        # Export scene to GLB
        glb_data = self.scene.export(file_type='glb')
        glb_base64 = base64.b64encode(glb_data).decode('utf-8')

        # Get camera angles (either user-set or default)
        if hasattr(self, '_camera_angles'):
            angles_deg = [np.rad2deg(a) for a in self._camera_angles]
        else:
            # Default angles
            angles_deg = [-45, 45, 135]

        # Create custom HTML with correct camera setup
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{ margin: 0; overflow: hidden; }}
    </style>
</head>
<body>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>

    <script type="module">
        import * as THREE from 'three';
        import {{ TrackballControls }} from 'three/addons/controls/TrackballControls.js';
        import {{ GLTFLoader }} from 'three/addons/loaders/GLTFLoader.js';

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new TrackballControls(camera, renderer.domElement);
        controls.rotateSpeed = 1.0;
        controls.zoomSpeed = 1.2;
        controls.panSpeed = 0.8;
        controls.noZoom = false;
        controls.noPan = false;
        controls.staticMoving = true;
        controls.dynamicDampingFactor = 0.3;

        // Lights - extremely bright setup
        const ambientLight = new THREE.AmbientLight(0xffffff, 2.0);
        scene.add(ambientLight);
        const light1 = new THREE.DirectionalLight(0xffffff, 1.75);
        light1.position.set(5, 5, 5);
        scene.add(light1);
        const light2 = new THREE.DirectionalLight(0xffffff, 1.0);
        light2.position.set(-5, 3, -5);
        scene.add(light2);
        const light3 = new THREE.DirectionalLight(0xffffff, 0.8);
        light3.position.set(0, -5, 0);
        scene.add(light3);

        // Set camera using our angles
        const xRad = {angles_deg[0]} * Math.PI / 180;
        const yRad = {angles_deg[1]} * Math.PI / 180;
        const zRad = {angles_deg[2]} * Math.PI / 180;

        const euler = new THREE.Euler(xRad, yRad, zRad, 'XYZ');
        const rotMat = new THREE.Matrix4().makeRotationFromEuler(euler);

        const forward = new THREE.Vector3(0, 0, 1);
        forward.applyMatrix4(rotMat);

        const up = new THREE.Vector3(0, 1, 0);
        up.applyMatrix4(rotMat);

        const distance = 3.5;
        camera.position.copy(forward.multiplyScalar(distance));
        camera.up.copy(up);
        camera.lookAt(0, 0, 0);

        controls.target.set(0, 0, 0);
        controls.update();

        // Load GLB
        const loader = new GLTFLoader();
        const glbData = atob('{glb_base64}');
        const arrayBuffer = new Uint8Array(glbData.length);
        for (let i = 0; i < glbData.length; i++) {{
            arrayBuffer[i] = glbData.charCodeAt(i);
        }}
        const blob = new Blob([arrayBuffer], {{ type: 'model/gltf-binary' }});
        const url = URL.createObjectURL(blob);

        // Store loaded objects for updates
        const loadedObjects = {{}};

        loader.load(url, (gltf) => {{
            scene.add(gltf.scene);

            // Store references to all meshes by name
            gltf.scene.traverse((child) => {{
                if (child.isMesh && child.name) {{
                    loadedObjects[child.name] = child;
                }}
            }});
        }});

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
            controls.handleResize();
        }});
'''

        # Add postMessage listener if enable_updates is True
        if enable_updates:
            html += '''
        // Listen for transform updates via postMessage
        window.addEventListener("message", (event) => {
            if (event.data && event.data.type === "updateTransforms") {
                const transforms = event.data.transforms;
                for (const [name, matrix] of Object.entries(transforms)) {
                    const obj = loadedObjects[name];
                    if (obj) {
                        // matrix is flat array of 16 floats (row-major)
                        const mat = new THREE.Matrix4();
                        mat.fromArray(matrix);
                        mat.transpose(); // Convert row-major to column-major
                        obj.matrix.copy(mat);
                        obj.matrixAutoUpdate = false;
                    }
                }
            }
        });
'''

        html += '''
    </script>
</body>
</html>'''

        if escape_quotes:
            return html.replace('"', '&quot;')

        return html

    def redraw(self):
        """Redraw the scene by updating link transforms and display.

        This method:
        1. Updates robot root coordinates (for base movements)
        2. Updates all link transforms in the scene graph
        3. In Jupyter notebooks, automatically updates the display via postMessage

        Examples
        --------
        >>> viewer.show()  # Initial display
        >>> robot.rarm.angle_vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        >>> viewer.redraw()  # Update display
        """
        # First update all robot objects (for root coordinate changes)
        for robot in self._robots:
            robot.update(force=True)

        # Update transforms for all links
        for link_id, link in self._links.items():
            link.update(force=True)
            transform = link.worldcoords().T()
            self.scene.graph.update(link_id, matrix=transform)

        # If in Jupyter and display exists, update it via postMessage
        if self._display_id is not None:
            try:
                import json

                from IPython.display import display
                from IPython.display import Javascript

                # Collect transform data for all geometries
                transforms = {}
                for geom_name in self.scene.geometry.keys():
                    transform, _ = self.scene.graph.get(geom_name)
                    if transform is not None:
                        transforms[geom_name] = transform.flatten().tolist()

                # Send update message to iframe
                js_code = f'''
                    var iframe = document.getElementById('viewer-frame-{self._display_id}');
                    if (iframe && iframe.contentWindow) {{
                        iframe.contentWindow.postMessage({{
                            type: 'updateTransforms',
                            transforms: {json.dumps(transforms)}
                        }}, '*');
                    }}
                '''
                display(Javascript(js_code))
            except ImportError:
                # Not in IPython environment
                pass

    def update(self, update_in_place=False):
        """Update and display the scene.

        This is a convenience method that calls show().
        Use this after modifying robot poses to see the changes.

        Parameters
        ----------
        update_in_place : bool, optional
            If True, updates the previous display in-place (camera resets).
            If False (default), creates a new display (camera preserved).
            Default is False.

        Examples
        --------
        >>> robot.reset_pose()
        >>> viewer.update()  # Creates new display with new pose

        >>> # Or update in place (camera will reset)
        >>> robot.reset_pose()
        >>> viewer.update(update_in_place=True)
        """
        self.show(update_in_place=update_in_place)

    # Compatibility methods with TrimeshSceneViewer interface
    def save_image(self, file_obj):
        """Save the scene as an image.

        Note: This method is not supported in JupyterNotebookViewer.
        Use TrimeshSceneViewer or PyrenderViewer for image saving.
        """
        raise NotImplementedError(
            "Image saving is not supported in JupyterNotebookViewer. "
            "Use TrimeshSceneViewer or PyrenderViewer instead.")

    @property
    def is_active(self):
        """bool: Always True for JupyterNotebookViewer."""
        return True

    @property
    def has_exit(self):
        """bool: Always False for JupyterNotebookViewer."""
        return False
