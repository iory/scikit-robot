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

Basic usage:

.. code-block:: python

   from skrobot.models import PR2
   from skrobot.viewers import TrimeshSceneViewer

   robot = PR2()
   viewer = TrimeshSceneViewer()
   viewer.add(robot)
   viewer.show()
