Functions
=========

.. module:: skrobot.coordinates.math

Utilities functions
-------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   convert_to_axis_vector
   _check_valid_rotation
   _check_valid_translation
   triple_product
   inverse_rodrigues
   rotation_angle
   make_matrix
   random_rotation
   random_translation
   midpoint
   midrot
   transform
   rotation_matrix
   rotate_vector
   rotate_matrix
   rpy_matrix
   rpy_angle
   normalize_vector
   matrix_log
   matrix_exponent
   outer_product_matrix
   skew_symmetric_matrix
   rotation_matrix_from_rpy
   rotation_matrix_from_axis
   rodrigues
   rotation_angle
   rotation_distance
   axis_angle_from_matrix
   angle_between_vectors
   matrix2ypr
   matrix2rpy
   ypr2matrix
   rpy2matrix


Jacobian Functions
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   sr_inverse
   sr_inverse_org
   manipulability


Quaternion Functions
--------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   xyzw2wxyz
   wxyz2xyzw
   random_quaternion
   quaternion_multiply
   quaternion_conjugate
   quaternion_inverse
   quaternion_slerp
   quaternion_distance
   quaternion_absolute_distance
   quaternion_norm
   quaternion_normalize
   matrix2quaternion
   quaternion2matrix
   quaternion2rpy
   rpy2quaternion
   rpy_from_quat
   quat_from_rotation_matrix
   quat_from_rpy
   rotation_matrix_from_quat
   quaternion_from_axis_angle
   axis_angle_from_quaternion


.. module:: skrobot.coordinates.geo

Geometry functions
------------------

.. autosummary::
   :toctree: generated/
   :nosignatures:

   rotate_points
