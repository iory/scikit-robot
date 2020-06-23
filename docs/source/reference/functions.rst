Functions
=========

.. module:: skrobot.coordinates.math

Utilities functions
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.coordinates.math._wrap_axis
   skrobot.coordinates.math._check_valid_rotation
   skrobot.coordinates.math._check_valid_translation
   skrobot.coordinates.math.triple_product
   skrobot.coordinates.math.inverse_rodrigues
   skrobot.coordinates.math.rotation_angle
   skrobot.coordinates.math.make_matrix
   skrobot.coordinates.math.random_rotation
   skrobot.coordinates.math.random_translation
   skrobot.coordinates.math.midpoint
   skrobot.coordinates.math.midrot
   skrobot.coordinates.math.transform
   skrobot.coordinates.math.rotation_matrix
   skrobot.coordinates.math.rotate_vector
   skrobot.coordinates.math.rotate_matrix
   skrobot.coordinates.math.rpy_matrix
   skrobot.coordinates.math.rpy_angle
   skrobot.coordinates.math.normalize_vector
   skrobot.coordinates.math.matrix_log
   skrobot.coordinates.math.matrix_exponent
   skrobot.coordinates.math.outer_product_matrix
   skrobot.coordinates.math.rotation_matrix_from_rpy
   skrobot.coordinates.math.rotation_matrix_from_axis
   skrobot.coordinates.math.rodrigues
   skrobot.coordinates.math.rotation_angle
   skrobot.coordinates.math.rotation_distance
   skrobot.coordinates.math.axis_angle_from_matrix
   skrobot.coordinates.math.angle_between_vectors


Jacobian Functions
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.coordinates.math.sr_inverse
   skrobot.coordinates.math.sr_inverse_org
   skrobot.coordinates.math.manipulability


Quaternion Functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.coordinates.math.xyzw2wxyz
   skrobot.coordinates.math.wxyz2xyzw
   skrobot.coordinates.math.random_quaternion
   skrobot.coordinates.math.quaternion_multiply
   skrobot.coordinates.math.quaternion_conjugate
   skrobot.coordinates.math.quaternion_inverse
   skrobot.coordinates.math.quaternion_slerp
   skrobot.coordinates.math.quaternion_distance
   skrobot.coordinates.math.quaternion_absolute_distance
   skrobot.coordinates.math.quaternion_norm
   skrobot.coordinates.math.quaternion_normalize
   skrobot.coordinates.math.matrix2quaternion
   skrobot.coordinates.math.quaternion2matrix
   skrobot.coordinates.math.quaternion2rpy
   skrobot.coordinates.math.rpy2quaternion
   skrobot.coordinates.math.rpy_from_quat
   skrobot.coordinates.math.quat_from_rotation_matrix
   skrobot.coordinates.math.quat_from_rpy
   skrobot.coordinates.math.rotation_matrix_from_quat
   skrobot.coordinates.math.quaternion_from_axis_angle
   skrobot.coordinates.math.axis_angle_from_quaternion


.. module:: skrobot.coordinates.geo

Geometry functions
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   skrobot.coordinates.geo.rotate_points
