Robot Model Tips
================

Loading the Robot Model Without Mesh Data
-----------------------------------------

Sometimes, loading a robot model with full mesh data can be slow and resource-intensive. If mesh data is not necessary for your use case, you can improve the loading speed by disabling mesh data loading as shown below:

.. code-block:: python

    from datetime import datetime

    from skrobot.models import PR2
    from skrobot.utils.urdf import no_mesh_load_mode

    start = datetime.now()
    robot_no_mesh = PR2()
    end = datetime.now()
    print(end - start)
    # 0:00:00.269310

    # Load the PR2 model without mesh data for faster initialization
    start = datetime.now()
    with no_mesh_load_mode():
        robot_no_mesh = PR2()
    end = datetime.now()
    print(end - start)
    # 0:00:00.083222

This approach is useful when you only need the basic structure of the robot without the visual details of the mesh, which can be beneficial in scenarios where performance is prioritized over graphical fidelity.
