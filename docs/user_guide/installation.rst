Installation
============

Install cyecca from source using Poetry:

.. code-block:: bash

   git clone https://github.com/CogniPilot/cyecca.git
   cd cyecca
   poetry install

Optional: Run tests to verify installation:

.. code-block:: bash

   poetry run ./tools/test.sh

Dependencies
------------

Core dependencies (automatically installed):

* **casadi**: Symbolic framework and optimization
* **numpy**: Numerical computing  
* **scipy**: Scientific computing
* **sympy**: Symbolic mathematics
* **beartype**: Runtime type checking
* **pydot**: Graph visualization

Development dependencies:

* **pytest**: Testing framework
* **black**: Code formatting
* **sphinx**: Documentation generation
* **jupyter**: Interactive notebooks

ROS 2 Integration
-----------------

For ROS 2 simulation and visualization:

1. Install ROS 2 (Humble or later)
2. Build the workspace:

   .. code-block:: bash

      colcon build
      source install/setup.bash

3. Run simulations:

   .. code-block:: bash

      ros2 launch cyecca rdd2_sim.xml
      ros2 launch cyecca fixedwing_sim.xml

See :doc:`../examples/index` for usage tutorials.
