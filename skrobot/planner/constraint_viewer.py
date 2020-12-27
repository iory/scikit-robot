import numpy as np
from skrobot.model import Axis
from skrobot.model import Sphere
from skrobot.coordinates.math import rotation_matrix_from_rpy
from skrobot.planner import ConstraintManager
from skrobot.planner.constraint_manager import PoseConstraint

def rpy2ypr(rpy):
    # note that in skrobot, everythin is ypr
    return np.array([rpy[2], rpy[1], rpy[0]])

class ConstraintViewer(object):
    def __init__(self, viewer, constraint_manager):
        self.viewer = viewer
        self.cm = constraint_manager
        self.visual_object_list = []

    def show(self):
        desired_pose_list = []
        for constraint in self.cm.constraint_table.values():
            if isinstance(constraint, PoseConstraint):
                desired_pose_list.append(constraint.pose_desired)

        for pose in desired_pose_list:
            hasRotation = (len(pose) == 6) # pose can be 3dim position
            if hasRotation:
                pos = pose[:3]
                ypr = rpy2ypr(pose[3:])
                rot = rotation_matrix_from_rpy(ypr) # actually from ypr
                vis_obj = Axis(pos=pos, rot=rot)
            else:
                pos = pose[:3]
                yellow = [250, 250, 10, 200]
                vis_obj = Sphere(radius=0.02, pos=pos, color=yellow)
            self.visual_object_list.append(vis_obj)

        for vis_obj in self.visual_object_list:
            self.viewer.add(vis_obj)

    def delete(self):
        for vis_obj in self.visual_object_list:
            self.viewer.delete(vis_obj)
        self.visual_object_list = []
