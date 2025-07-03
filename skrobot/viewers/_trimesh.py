from __future__ import division

import collections
import logging
import os
import tempfile
import threading
import time
import warnings

import numpy as np
import PIL.Image
import pyglet
from pyglet import compat_platform
import trimesh
from trimesh.transformations import euler_from_matrix
import trimesh.viewer
from trimesh.viewer.trackball import Trackball

from .. import model as model_module


logger = logging.getLogger('trimesh')
logger.setLevel(logging.ERROR)


def _check_trimesh_version():
    trimesh_version = tuple(map(int, trimesh.__version__.split('.')))
    if (4, 6, 1) < trimesh_version < (4, 6, 6):
        warnings.warn(
            "\033[31m"
            + "Trimesh version {} detected. ".format(trimesh.__version__)
            + "Versions >= 4.6.1 and <= 4.6.5 may cause models to "
            + "appear completely black."
            + "\033[0m",
            category=RuntimeWarning
        )


def _redraw_all_windows():
    for window in pyglet.app.windows:
        window.switch_to()
        window.dispatch_events()
        window.dispatch_event('on_draw')
        window.flip()
        window._legacy_invalid = False


class TrimeshSceneViewer(trimesh.viewer.SceneViewer):
    """TrimeshSceneViewer class implemented as a Singleton.

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

    Notes
    -----
    Since this is a singleton, the __init__ method might be called
    multiple times, but only one instance is actually used.
    """

    # Class variable to hold the single instance of the class.
    _instance = None
    _version_warning_issued = False

    def __init__(self, resolution=None, update_interval=1.0):
        if getattr(self, '_initialized', False):
            return
        if resolution is None:
            resolution = (640, 480)

        self.thread = None

        self._links = collections.OrderedDict()

        self._redraw = True
        pyglet.clock.schedule_interval(self.on_update, update_interval)

        self.scene = trimesh.Scene()
        self._kwargs = dict(
            scene=self.scene,
            resolution=resolution,
            offset_lines=False,
            start_loop=False,
            caption='scikit-robot TrimeshSceneViewer',
        )

        self.lock = threading.Lock()
        self._initialized = True

        # Recording related attributes
        self._recording = False
        self._recording_thread = None
        self._recording_frames = []
        self._recording_stop_event = threading.Event()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TrimeshSceneViewer, cls).__new__(cls)

            if not cls._version_warning_issued:
                _check_trimesh_version()
                cls._version_warning_issued = True

        return cls._instance

    def show(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.set_camera([np.deg2rad(45), -np.deg2rad(0), np.deg2rad(135)])
        if compat_platform == 'darwin':
            super(TrimeshSceneViewer, self).__init__(**self._kwargs)
            init_loop = 30
            for _ in range(init_loop):
                _redraw_all_windows()
        else:
            self.thread = threading.Thread(target=self._init_and_start_app)
            self.thread.daemon = True  # terminate when main thread exit
            self.thread.start()

    def _init_and_start_app(self):
        with self.lock:
            try:
                super(TrimeshSceneViewer, self).__init__(**self._kwargs)
            except pyglet.canvas.xlib.NoSuchDisplayException:
                print('No display found. Viewer is disabled.')
                self.has_exit = True
                return
        pyglet.app.run()

    def redraw(self):
        self._redraw = True
        if compat_platform == 'darwin':
            # On macOS, try to redraw without blocking
            try:
                if hasattr(self, 'window') and self.window:
                    _redraw_all_windows()
            except Exception:
                # If redraw fails, just continue - recording will capture next frame
                pass
        
        # Capture frame during redraw when recording is active
        if hasattr(self, '_recording') and self._recording:
            try:
                # Try to capture frame during redraw
                self.capture_frame()
            except Exception as e:
                # Frame capture failed, background thread will handle it
                pass

    def on_update(self, dt):
        self.on_draw()

    def on_draw(self):
        if not self._redraw:
            with self.lock:
                self._update_vertex_list()
                super(TrimeshSceneViewer, self).on_draw()
            # Skip automatic frame capture during redraw when background recording is active
            # Background thread handles frame capture automatically
            pass
            return

        with self.lock:
            self._update_vertex_list()

            # apply latest angle-vector
            for link_id, link in self._links.items():
                link.update(force=True)
                transform = link.worldcoords().T()
                self.scene.graph.update(link_id, matrix=transform)
            super(TrimeshSceneViewer, self).on_draw()

        self._redraw = False
        
        # Capture frame during redraw when recording is active
        if hasattr(self, '_recording') and self._recording:
            try:
                # Try to capture frame during redraw
                self.capture_frame()
            except Exception as e:
                # Frame capture failed, background thread will handle it
                pass

    def on_mouse_press(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_press(*args, **kwargs)

    def on_mouse_drag(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_drag(*args, **kwargs)

    def on_mouse_scroll(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_mouse_scroll(*args, **kwargs)

    def on_key_press(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_key_press(*args, **kwargs)

    def on_resize(self, *args, **kwargs):
        self._redraw = True
        return super(TrimeshSceneViewer, self).on_resize(*args, **kwargs)

    def _add_link(self, link):
        assert isinstance(link, model_module.Link)

        with self.lock:
            link_id = str(id(link))
            if link_id in self._links:
                return
            transform = link.worldcoords().T()
            mesh = link.concatenated_visual_mesh
            # TODO(someone) fix this at trimesh's scene.
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

        for child_link in link._child_links:
            self._add_link(child_link)

    def add(self, geometry):
        if isinstance(geometry, model_module.Link):
            links = [geometry]
        elif isinstance(geometry, model_module.CascadedLink):
            links = geometry.link_list
        else:
            raise TypeError('geometry must be Link or CascadedLink')

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

        with self.lock:
            for link in links:
                link_id = str(id(link))
                if link_id not in self._links:
                    continue
                self.scene.delete_geometry(link_id)
                self._links.pop(link_id)
            self.cleanup_geometries()

        self._redraw = True

    def set_camera(self, *args, **kwargs):
        if len(args) < 1 and 'angles' not in kwargs:
            if hasattr(self, "view"):
                kwargs['angles'] = euler_from_matrix(
                    self.view["ball"].pose[:3, :3])
        with self.lock:
            self.scene.set_camera(*args, **kwargs)
            if hasattr(self, "view"):
                self.view["ball"] = Trackball(
                    pose=self.scene.camera_transform,
                    size=self.scene.camera.resolution,
                    scale=self.scene.scale,
                    target=self.scene.centroid
                )

    def save_image(self, file_obj):
        # Check if viewer is initialized
        if not hasattr(self, '_initialized') or not self._initialized:
            raise RuntimeError("Viewer not initialized for save_image")
        
        # Check if we have a valid window (may be inherited from parent)
        if not hasattr(self, 'window') or self.window is None:
            # Try to get window from parent class or pyglet
            try:
                import pyglet
                windows = list(pyglet.app.windows)
                if not windows:
                    raise RuntimeError("No pyglet windows available for save_image")
                # Use the first available window
                self.window = windows[0]
            except:
                raise RuntimeError("No window available for save_image")
            
        try:
            # Make sure we have a current OpenGL context
            if hasattr(self, 'switch_to'):
                self.switch_to()
            
            # Dispatch events and draw
            if hasattr(self, 'dispatch_events'):
                self.dispatch_events()
            if hasattr(self, 'dispatch_event'):
                self.dispatch_event('on_draw')
            if hasattr(self, 'flip'):
                self.flip()
                
            return super(TrimeshSceneViewer, self).save_image(file_obj)
        except Exception as e:
            raise RuntimeError("Failed to save image: {}".format(e))
    
    def _safe_save_image(self, file_obj):
        """Safer version of save_image optimized for GUI environments."""
        if not hasattr(self, '_save_attempt_counter'):
            self._save_attempt_counter = 0
        self._save_attempt_counter += 1
        
        try:
            # In GUI environments, we can use the standard save_image more reliably
            import os
            if 'DISPLAY' in os.environ:
                # GUI environment - use standard approach
                result = super(TrimeshSceneViewer, self).save_image(file_obj)
                logger.debug("Successfully saved real frame #{} to {}".format(self._save_attempt_counter, file_obj))
                return result
            else:
                # Fallback for headless environments
                import pyglet
                windows = list(pyglet.app.windows)
                
                if windows and len(windows) > 0:
                    window = windows[0]
                    if not window.has_exit:
                        window.switch_to()
                        result = super(TrimeshSceneViewer, self).save_image(file_obj)
                        logger.debug("Successfully saved real frame #{} to {}".format(self._save_attempt_counter, file_obj))
                        return result
                    else:
                        raise RuntimeError("Window has exited")
                else:
                    raise RuntimeError("No pyglet windows available")
                
        except Exception as e:
            logger.warning("save_image failed on frame #{}: {}".format(self._save_attempt_counter, e))
            # Re-raise the exception to indicate failure
            raise e

    def record(self, fps=30.0, output_path=None):
        """Start recording the viewer's content as a video.

        Parameters
        ----------
        fps : float, optional
            Frames per second for the recording. Default is 30.0.
        output_path : str, optional
            Path to save the video file. If None, a temporary file
            will be created with a timestamp.

        Returns
        -------
        str
            The path where the video will be saved.

        Notes
        -----
        - Recording saves frames in memory and creates video on stop
        - Call stop_record() to stop recording and save the video
        - Requires imageio-ffmpeg to be installed for video encoding
        """
        if self._recording:
            raise RuntimeError("Recording is already in progress. Call stop_record() first.")

        # Set default output path if not provided
        if output_path is None:
            import os
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(tempfile.gettempdir(),
                                       "scikit_robot_recording_{}.mp4".format(timestamp))

        self._recording = True
        self._recording_frames = []
        self._recording_frame_timestamps = []  # Store actual capture times
        self._recording_start_time = None
        self._recording_stop_event.clear()
        self._recording_output_path = output_path
        self._recording_fps = fps
        self._recording_frame_count = 0

        # Start background recording thread for automatic capture
        import os
        if 'DISPLAY' in os.environ:
            # GUI environment - start background recording thread
            self._recording_thread = threading.Thread(
                target=self._background_record_loop,
                daemon=True
            )
            self._recording_thread.start()
            print("Recording started in background mode.")
            print("Frames will be captured automatically.")
        else:
            # Headless environment - use manual capture
            print("Recording started in manual mode.")
            print("Frames will be captured during redraw operations.")
        
        print("Call viewer.stop_record() when done.")
        print("Recording started. Output will be saved to: {}".format(output_path))
        return output_path

    def capture_frame(self):
        """Manually capture a frame during recording for video.
        
        Returns
        -------
        bool
            True if frame was captured successfully, False otherwise.
            
        Notes
        -----
        This method captures frames in memory for video creation.
        """
        if not self._recording:
            return False
            
        try:
            # Increment frame counter
            if not hasattr(self, '_recording_frame_count'):
                self._recording_frame_count = 0
            self._recording_frame_count += 1
            
            # Create temporary file for frame capture
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                temp_path = tmp_file.name

            # Save current frame with error handling
            try:
                # Use safer save_image approach - only real frames
                self._safe_save_image(temp_path)
            except Exception as e:
                logger.warning("Failed to save image: {}. Skipping this frame.".format(e))
                # Clean up and skip this frame
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return False
            
            # Check if file was created and has content
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                # Check frame limit to prevent memory issues
                if len(self._recording_frames) >= 300:  # Limit to 300 frames (~30 seconds at 10fps)
                    print("Frame limit reached (300 frames). Skipping frame capture.")
                    os.unlink(temp_path)
                    return True
                
                # Record the actual capture time
                import time
                current_time = time.time()
                
                # Set start time for the first frame
                if self._recording_start_time is None:
                    self._recording_start_time = current_time
                
                # Store relative timestamp from recording start
                relative_time = current_time - self._recording_start_time
                self._recording_frame_timestamps.append(relative_time)
                
                # Load the frame into memory
                frame = PIL.Image.open(temp_path)
                # Ensure consistent format (RGB, 3 channels)
                if frame.mode != 'RGB':
                    frame = frame.convert('RGB')
                frame_array = np.array(frame)
                self._recording_frames.append(frame_array)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                print("Frame {} captured at {:.2f}s".format(len(self._recording_frames), relative_time))
                return True
            else:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return False
                
        except Exception as e:
            logger.warning("Frame capture failed: {}".format(e))
            return False

    def _background_record_loop(self):
        """Background thread function for automatic frame recording."""
        # Capture frames at regular intervals while recording is active
        capture_interval = 0.1  # Capture every 100ms (10 FPS)
        
        logger.debug("Background recording thread started")
        
        # Wait for viewer to initialize
        time.sleep(2.0)
        
        while self._recording and not self._recording_stop_event.is_set():
            try:
                # Check if viewer is properly initialized and has a window
                if not hasattr(self, '_initialized') or not self._initialized:
                    time.sleep(0.1)
                    continue
                
                # Check if we have a window available
                import pyglet
                windows = list(pyglet.app.windows)
                if not windows:
                    logger.debug("No windows available for frame capture")
                    time.sleep(0.1)
                    continue
                
                # Capture frame in background
                success = self.capture_frame()
                if not success:
                    logger.debug("Background frame capture failed, retrying...")
                else:
                    logger.debug("Background frame captured successfully")
                
                # Wait before next capture
                time.sleep(capture_interval)
                
            except Exception as e:
                logger.debug("Background recording error: {}".format(e))
                # Continue recording despite errors
                time.sleep(capture_interval)
        
        logger.debug("Background recording loop ended")

    def _record_loop(self, fps):
        """Legacy background thread function for recording frames."""
        # This method is kept for compatibility but not used in GUI mode
        frame_interval = 1.0 / fps

        while not self._recording_stop_event.is_set():
            start_time = time.time()

            # Capture frame
            frame = self._capture_frame()
            if frame is not None:
                self._recording_frames.append(frame)

            # Sleep to maintain fps
            elapsed = time.time() - start_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


    def _capture_frame(self):
        """Capture a single frame from the viewer."""
        # This method is no longer used since we switched to manual recording
        # Just call capture_frame directly
        return self.capture_frame()


    def stop_record(self):
        """Stop recording and save the video.

        Returns
        -------
        str or None
            The path to the saved video file, or None if recording failed.

        Notes
        -----
        This method will block until the video is saved.
        """
        if not self._recording:
            raise RuntimeError("No recording in progress.")

        # Stop recording
        self._recording = False
        self._recording_stop_event.set()

        # Wait for recording thread to finish (if any)
        if hasattr(self, '_recording_thread') and self._recording_thread is not None:
            self._recording_thread.join(timeout=5.0)

        # Save video
        try:
            if len(self._recording_frames) > 0:
                try:
                    import imageio

                    # Calculate actual duration and effective FPS
                    if len(self._recording_frame_timestamps) > 1:
                        actual_duration = self._recording_frame_timestamps[-1] - self._recording_frame_timestamps[0]
                        effective_fps = (len(self._recording_frames) - 1) / actual_duration if actual_duration > 0 else self._recording_fps
                    else:
                        actual_duration = 1.0 / self._recording_fps
                        effective_fps = self._recording_fps
                    
                    print("Actual recording duration: {:.2f} seconds".format(actual_duration))
                    print("Effective FPS based on actual timing: {:.2f}".format(effective_fps))

                    # Create video with actual timing preserved
                    if len(self._recording_frame_timestamps) > 1:
                        # Use the effective FPS based on actual capture timing
                        writer = imageio.get_writer(
                            self._recording_output_path,
                            fps=effective_fps,
                            codec='libx264',
                            pixelformat='yuv420p'
                        )

                        # Simply add frames in order - the FPS will handle the timing
                        for frame in self._recording_frames:
                            writer.append_data(frame)
                        
                        writer.close()
                        
                        print("Recording saved to: {}".format(self._recording_output_path))
                        print("Total frames: {}".format(len(self._recording_frames)))
                        print("Actual duration: {:.2f} seconds".format(actual_duration))
                        print("Effective FPS: {:.2f}".format(effective_fps))
                        print("Video will play at the same speed as the actual recording")
                    else:
                        # Fallback for single frame
                        writer.append_data(self._recording_frames[0])
                        writer.close()
                        
                        print("Recording saved to: {}".format(self._recording_output_path))
                        print("Single frame recorded")

                    return self._recording_output_path

                except ImportError:
                    logger.error("imageio is required for video recording. "
                                 "Install it with: pip install imageio imageio-ffmpeg")
                    return None
                except Exception as e:
                    logger.error("Failed to save video: {}".format(e))
                    return None
            else:
                print("No frames captured during recording.")
                return None

        finally:
            # Clean up
            self._recording_frames = []
            self._recording_frame_timestamps = []
            self._recording_start_time = None
            if hasattr(self, '_recording_thread'):
                self._recording_thread = None
            self._recording_frame_count = 0

        return None
