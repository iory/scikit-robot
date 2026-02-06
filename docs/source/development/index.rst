.. _development:

Development Guide
=================

Read this guide before doing development in ``skrobot``.

Setting Up with uv (Recommended)
--------------------------------

`uv <https://docs.astral.sh/uv/>`_ is a fast Python package manager that makes
development setup simple and fast. First, install uv if you haven't already:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Then clone the repository, create a virtual environment, and install in development mode:

.. code-block:: bash

   git clone https://github.com/iory/scikit-robot.git
   cd scikit-robot
   uv venv
   source .venv/bin/activate  # Linux/macOS
   # .venv\Scripts\activate   # Windows
   uv pip install -e .

To install development dependencies (ruff, pytest, etc.):

.. code-block:: bash

   uv pip install -e ".[all]" ruff pytest

Setting Up with pip
-------------------

Alternatively, you can use pip to install ``skrobot`` in development mode:

.. code-block:: bash

   git clone https://github.com/iory/scikit-robot.git
   cd scikit-robot
   pip install -e .

Running Code Style Checks
-------------------------

We follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and partially `OpenStack Style Guidelines <https://docs.openstack.org/developer/hacking/>`_ as basic style guidelines.
Any contributions in terms of code are expected to follow these guidelines.

You can use ``ruff`` to check and automatically fix code style issues, including import ordering.
``ruff`` is a fast Python linter and formatter that replaces ``flake8``, ``isort``, and ``autopep8``.
Install it with the following command::

  $ pip install ruff pytest

Check your code with::

  $ ruff check path/to/your/code.py

``ruff`` can automatically fix many style issues::

  $ ruff check --fix path/to/your/code.py

To check the entire project::

  $ ruff check .

For more information, please see `the ruff documentation`_.

.. _the ruff documentation: https://docs.astral.sh/ruff/

Running Tests
-------------

This project uses `pytest`_, the standard Python testing framework.
Their website has tons of useful details, but here are the basics.

.. _pytest: https://docs.pytest.org/en/latest/

To run the testing suite, simply navigate to the top-level folder
in ``scikit-robot`` and run the following command:

.. code-block:: bash

   pytest -v tests

You should see the testing suite run. There are a few useful command line
options that are good to know:

- ``-s`` - Shows the output of ``stdout``. By default, this output is masked.
- ``--pdb`` - Instead of crashing, opens a debugger at the first fault.
- ``--lf`` - Instead of running all tests, just run the ones that failed last.
- ``--trace`` - Open a debugger at the start of each test.

You can see all of the other command-line options `here`_.

.. _here: https://docs.pytest.org/en/latest/usage.html

By default, ``pytest`` will look in the ``tests`` folder recursively.
It will run any function that starts with ``test_`` in any file that starts
with ``test_``. You can run ``pytest`` on a directory or on a particular file
by specifying the file path:

.. code-block:: bash

   pytest -v tests/skrobot_tests/coordinates_tests/test_math.py


Building Documentation
----------------------

To build ``scikit-robot``'s documentation, go to the ``docs`` directory and run
``make`` with the appropriate target.
For example,

.. code-block:: bash

    cd docs/
    make html

will generate HTML-based docs, which are probably the easiest to read.
The resulting index page is at ``docs/build/html/index.html``.
If the docs get stale, just run ``make clean`` to remove all build files.
