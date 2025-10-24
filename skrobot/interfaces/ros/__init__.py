from skrobot._lazy_imports import LazyImportClass


NextageROSRobotInterface = LazyImportClass(
    ".nextage", "NextageROSRobotInterface", "skrobot.interfaces.ros")
PandaROSRobotInterface = LazyImportClass(
    ".panda", "PandaROSRobotInterface", "skrobot.interfaces.ros")
PR2ROSRobotInterface = LazyImportClass(
    ".pr2", "PR2ROSRobotInterface", "skrobot.interfaces.ros")

__all__ = ["NextageROSRobotInterface", "PandaROSRobotInterface", "PR2ROSRobotInterface"]
