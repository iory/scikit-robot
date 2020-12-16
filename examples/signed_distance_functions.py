#!/usr/bin/env python

import numpy as np

import skrobot
from skrobot.coordinates.math import rotation_matrix_from_axis
from skrobot.model import Axis
from skrobot.model import Box
from skrobot.model import MeshLink
from skrobot.sdf import UnionSDF

b = Box(extents=[0.05, 0.1, 0.05], with_sdf=True)
m = MeshLink(visual_mesh=skrobot.data.bunny_objpath(), with_sdf=True)
b.translate([0, 0.1, 0])
u = UnionSDF([b.sdf, m.sdf])
axis = Axis(axis_radius=0.001, axis_length=0.3)
viewer = skrobot.viewers.TrimeshSceneViewer(resolution=(640, 480))
viewer.add(b)
viewer.add(m)
viewer.show()
pts, sd_vals = u.surface_points()

for _ in range(100):
    idx = np.random.randint(len(pts))
    rot = rotation_matrix_from_axis(np.random.random(3), np.random.random(3))
    ax = Axis(axis_radius=0.001, axis_length=0.01, pos=pts[idx], rot=rot)
    viewer.add(ax)
