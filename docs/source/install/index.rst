.. _install:

Installation and Setup Guide
============================

Quick Install with uv (Recommended)
-----------------------------------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager that makes
installation simple and fast.

If you don't have uv installed, you can install it with:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Create a virtual environment and install scikit-robot:

.. code-block:: bash

   uv venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   uv pip install scikit-robot

To install with all optional dependencies (Pybullet, open3d, JAX, etc.):

.. code-block:: bash

   uv pip install "scikit-robot[all]"

Python Installation with pip
----------------------------
This package is pip-installable for any Python version. Simply run the
following command:

.. code-block:: bash

   pip install scikit-robot

To install with all optional dependencies:

.. code-block:: bash

   pip install "scikit-robot[all]"

Installing in Development Mode
------------------------------
If you're planning on contributing to this repository,
please see the :ref:`development`.
