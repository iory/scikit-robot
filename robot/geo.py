from robot.coordinates import make_coords
from robot.math import midpoint
from robot.math import midrot


def midcoords(p, c1, c2):
    """Returns mid (or p) coordinates of given two coordinates c1 and c2

    Args:
        TODO
    """
    return make_coords(pos=midpoint(p, c1.worldpos(), c2.worldpos()),
                       rot=midrot(p, c1.worldrot(), c2.worldrot()))
