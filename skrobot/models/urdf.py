import warnings

from ..model import RobotModel


class RobotModelFromURDF(RobotModel):
    """Deprecated: Use RobotModel with urdf parameter instead.

    This class is kept for backward compatibility.
    Use ``RobotModel(urdf=urdf_input)`` or ``RobotModel.from_urdf(urdf_input)``
    instead.
    """

    def __init__(self, urdf=None, urdf_file=None):
        warnings.warn(
            "RobotModelFromURDF is deprecated. "
            "Use RobotModel(urdf=...) or RobotModel.from_urdf(...) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if urdf is not None and urdf_file is not None:
            raise ValueError(
                "'urdf' and 'urdf_file' cannot be given at the same time"
            )
        if urdf is not None:
            urdf_input = urdf
        elif urdf_file is not None:
            urdf_input = urdf_file
        else:
            urdf_input = self.default_urdf_path
        super(RobotModelFromURDF, self).__init__(urdf=urdf_input)

    @property
    def default_urdf_path(self):
        raise NotImplementedError
