"""Joint limit table utilities.

This module provides classes and functions to create and apply dynamic joint
limit tables where one joint's limits depend on another joint's angle.

Classes:
- JointLimitTable: Dynamic joint limit table based on another joint's angle

Functions:
- create_joint_limit_table: Create and apply a joint limit table from arrays
"""

import numpy as np


class JointLimitTable:
    """Dynamic joint limit table based on another joint's angle.

    This class represents a joint limit table where joint B's min/max angles
    depend on joint A's current angle. The table uses linear interpolation
    between the discrete angle samples.

    Parameters
    ----------
    target_joint : Joint
        The joint whose angle determines the limits (joint A).
    target_min_angle : float
        Minimum angle of the target joint (in radians).
    target_max_angle : float
        Maximum angle of the target joint (in radians).
    min_angles : numpy.ndarray
        Array of minimum angles for the dependent joint at each sample point.
    max_angles : numpy.ndarray
        Array of maximum angles for the dependent joint at each sample point.

    Examples
    --------
    >>> table = JointLimitTable(target_joint, -np.pi/2, np.pi/2, min_arr, max_arr)
    >>> current_min = table.min_angle_function(target_joint.joint_angle())
    >>> current_max = table.max_angle_function(target_joint.joint_angle())
    """

    def __init__(self, target_joint, target_min_angle, target_max_angle,
                 min_angles, max_angles):
        self.target_joint = target_joint
        self.target_min_angle = float(target_min_angle)
        self.target_max_angle = float(target_max_angle)
        self.min_angles = np.asarray(min_angles, dtype=np.float64)
        self.max_angles = np.asarray(max_angles, dtype=np.float64)

        # Precompute sample x-coordinates for interpolation
        n_samples = len(self.min_angles)
        self.sample_angles = np.linspace(
            self.target_min_angle,
            self.target_max_angle,
            n_samples
        )

    def min_angle_function(self, target_angle):
        """Get minimum angle of dependent joint for given target angle.

        Parameters
        ----------
        target_angle : float
            Current angle of the target joint (in radians).

        Returns
        -------
        float
            Minimum allowable angle for the dependent joint.
        """
        return float(np.interp(target_angle, self.sample_angles, self.min_angles))

    def max_angle_function(self, target_angle):
        """Get maximum angle of dependent joint for given target angle.

        Parameters
        ----------
        target_angle : float
            Current angle of the target joint (in radians).

        Returns
        -------
        float
            Maximum allowable angle for the dependent joint.
        """
        return float(np.interp(target_angle, self.sample_angles, self.max_angles))

    def get_data_for_differentiable(self):
        """Get data for differentiable IK solvers.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'sample_angles': Array of sample angles for interpolation
            - 'min_angles': Array of min limits at each sample
            - 'max_angles': Array of max limits at each sample
        """
        return {
            'sample_angles': self.sample_angles,
            'min_angles': self.min_angles,
            'max_angles': self.max_angles,
        }


def create_joint_limit_table(
        robot,
        target_joint,
        dependent_joint,
        target_min_angle,
        target_max_angle,
        min_angles,
        max_angles,
        use_degrees=True):
    """Create and apply a joint limit table from arrays.

    This is the recommended way to create joint limit tables. It accepts
    numpy arrays or lists directly.

    Parameters
    ----------
    robot : RobotModel
        The robot model to apply the joint limit table to.
    target_joint : Joint or str
        The joint whose angle determines the limits. Can be a Joint object
        or a joint name string.
    dependent_joint : Joint or str
        The joint whose limits depend on the target joint. Can be a Joint
        object or a joint name string.
    target_min_angle : float
        Minimum angle of the target joint.
    target_max_angle : float
        Maximum angle of the target joint.
    min_angles : array-like
        Array of minimum angles for the dependent joint at each sample point.
    max_angles : array-like
        Array of maximum angles for the dependent joint at each sample point.
    use_degrees : bool
        If True (default), angles are in degrees. If False, angles are in radians.

    Returns
    -------
    JointLimitTable
        The created joint limit table object.

    Raises
    ------
    ValueError
        If joints are not found or array lengths don't match.

    Examples
    --------
    >>> # Using degrees (default)
    >>> table = create_joint_limit_table(
    ...     robot,
    ...     target_joint='RARM_JOINT5',
    ...     dependent_joint='RARM_JOINT6',
    ...     target_min_angle=-90,
    ...     target_max_angle=90,
    ...     min_angles=[-10, -15, -20, -15, -10],
    ...     max_angles=[10, 15, 20, 15, 10]
    ... )

    >>> # Using radians
    >>> import numpy as np
    >>> table = create_joint_limit_table(
    ...     robot,
    ...     target_joint='JOINT5',
    ...     dependent_joint='JOINT6',
    ...     target_min_angle=-np.pi/2,
    ...     target_max_angle=np.pi/2,
    ...     min_angles=np.deg2rad([-10, -15, -20]),
    ...     max_angles=np.deg2rad([10, 15, 20]),
    ...     use_degrees=False
    ... )
    """
    # Resolve joint names to Joint objects
    if isinstance(target_joint, str):
        target_joint_obj = None
        for j in robot.joint_list:
            if j.name == target_joint:
                target_joint_obj = j
                break
        if target_joint_obj is None:
            raise ValueError(f"Target joint '{target_joint}' not found in robot.")
        target_joint = target_joint_obj

    if isinstance(dependent_joint, str):
        dependent_joint_obj = None
        for j in robot.joint_list:
            if j.name == dependent_joint:
                dependent_joint_obj = j
                break
        if dependent_joint_obj is None:
            raise ValueError(f"Dependent joint '{dependent_joint}' not found in robot.")
        dependent_joint = dependent_joint_obj

    # Convert to numpy arrays
    min_angles = np.asarray(min_angles, dtype=np.float64)
    max_angles = np.asarray(max_angles, dtype=np.float64)

    if len(min_angles) != len(max_angles):
        raise ValueError(
            f"Length mismatch: min_angles has {len(min_angles)} elements, "
            f"max_angles has {len(max_angles)} elements."
        )

    # Convert to radians if using degrees
    if use_degrees:
        target_min_rad = np.deg2rad(target_min_angle)
        target_max_rad = np.deg2rad(target_max_angle)
        min_angles_rad = np.deg2rad(min_angles)
        max_angles_rad = np.deg2rad(max_angles)
    else:
        target_min_rad = float(target_min_angle)
        target_max_rad = float(target_max_angle)
        min_angles_rad = min_angles
        max_angles_rad = max_angles

    # Create the joint limit table
    table = JointLimitTable(
        target_joint=target_joint,
        target_min_angle=target_min_rad,
        target_max_angle=target_max_rad,
        min_angles=min_angles_rad,
        max_angles=max_angles_rad,
    )

    # Set up the joint properties
    dependent_joint.joint_min_max_table = table
    dependent_joint.joint_min_max_target = target_joint

    return table
