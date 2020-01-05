.. _development:

Development Guide
=================

Read this guide before doing development in ``skrobot``.

Setting Up
----------

To set up the tools you'll need for developing, you'll need to install
``skrobot`` in development mode. Start by installing the development
dependencies:

.. code-block:: bash

   git clone https://github.com/iory/scikit-robot.git
   cd scikit-robot
   pip install -e .

Running Code Style Checks
-------------------------

We follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ and partially `OpenStack Style Guidelines <https://docs.openstack.org/developer/hacking/>`_ as basic style guidelines.
Any contributions in terms of code are expected to follow these guidelines.

You can use the ``autopep8`` and the ``flake8`` commands to check whether or not your code follows the guidelines.
In order to avoid confusion from using different tool versions, we pin the versions of those tools.
Install them with the following command (from within the top directory of the Chainer repository)::

  $ pip install hacking pytest autopep8

And check your code with::

  $ autopep8 path/to/your/code.py
  $ flake8 path/to/your/code.py

``autopep8`` can automatically correct Python code to conform to the PEP 8 style guide::

  $ autopep8 --in-place path/to/your/code.py


For more information, please see `the flake8 documentation`_.

.. _the flake8 documentation: https://flake8.pycqa.org/en/latest/user/options.html

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
