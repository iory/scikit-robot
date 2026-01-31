==============
Visualization
==============

For visualization and viewer documentation, see:

- :doc:`../reference/viewers` - Viewer API reference
- :doc:`../examples/index` - Basic visualization examples

Scikit-robot supports multiple visualization backends:

- **TrimeshSceneViewer**: Lightweight, fast rendering
- **PyrenderViewer**: OpenGL-based, smoother rendering
- **JupyterNotebookViewer**: Browser-based, works in Jupyter and Google Colab
- **ViserViewer**: Web-based viewer with interactive joint sliders

Basic usage:

.. code-block:: python

   from skrobot.models import PR2
   from skrobot.viewers import TrimeshSceneViewer

   robot = PR2()
   viewer = TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()

ViserViewer
-----------

ViserViewer provides a web-based 3D visualization that opens in your browser.
It automatically generates GUI sliders for each joint, allowing real-time manipulation of joint angles.

.. code-block:: python

   from skrobot.models import PR2
   from skrobot.viewers import ViserViewer

   robot = PR2()
   viewer = ViserViewer()
   viewer.add(robot)
   viewer.show()  # Opens browser automatically

   # Keep the server running
   import time
   while viewer.is_active:
       viewer.redraw()
       time.sleep(0.1)

.. image:: ../../image/viser-viewer.jpg
   :width: 600px
   :align: center
   :alt: ViserViewer with joint angle sliders

Features:

- Web-based visualization accessible from any browser
- Interactive joint angle sliders organized by link groups
- Real-time robot pose updates
- Works in headless environments (no display server required)
- Remote access capability over network
