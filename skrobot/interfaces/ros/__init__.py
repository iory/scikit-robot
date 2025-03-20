from skrobot._lazy_imports import LazyImportClass


PandaROSRobotInterface = LazyImportClass(
    ".panda", "PandaROSRobotInterface", "skrobot.interfaces.ros")
PR2ROSRobotInterface = LazyImportClass(
    ".pr2", "PR2ROSRobotInterface", "skrobot.interfaces.ros")

__all__ = ["PandaROSRobotInterface", "PR2ROSRobotInterface"]
