from cached_property import cached_property
import numpy as np

from skrobot.coordinates import CascadedCoords
from skrobot.data import griphis_urdfpath
from skrobot.models.urdf import RobotModelFromURDF


class _NailCentroidEndCoords(CascadedCoords):
    """End-effector coords whose world pose is the live centroid of four
    nail-tip CascadedCoords, oriented with +X along the tip-forward
    direction of ``parent_link``.

    The standard CascadedCoords caches a local offset from its parent and
    composes with ``parent.worldcoords()`` on update.  For Griphis the
    nail tips move with the mimic-driven ``worm_rotate`` joint, which
    sits *below* the parent link; a cached local offset would go stale
    every time worm_rotate changes.  This subclass overrides ``update``
    to recompute the world pose directly from the (live) tip world
    positions, so callers never have to manually refresh the frame.
    """

    def __init__(self, name, parent_link, tips, flip):
        # Assign attrs before super().__init__ because the parent.assoc()
        # call inside it triggers a worldcoords() update, which lands in
        # our overridden update() and needs these attrs to be present.
        self._tips = tips
        self._flip = np.asarray(flip, dtype=np.float64)
        self._parent_link = parent_link
        super(_NailCentroidEndCoords, self).__init__(
            parent=parent_link, name=name)

    def update(self, force=False):
        # Always recompute from live tip positions.  We bypass the usual
        # parent-world × local-offset composition because the tips'
        # positions can change without flagging ``_changed`` on this
        # coord (worm_rotate moves the tips but not the parent link).
        hook_disabled, original_hook = self.disable_hook()
        try:
            centroid = np.mean(
                [t.worldpos() for t in self._tips], axis=0)
            rot = self._parent_link.worldrot().dot(self._flip)
            self._worldcoords._translation = np.asarray(
                centroid, dtype=np.float64)
            self._worldcoords._rotation = np.asarray(
                rot, dtype=np.float64)
        finally:
            if hook_disabled:
                self._hook = original_hook
        self._changed = False


class Griphis(RobotModelFromURDF):
    """Griphis two-gripper wall-climbing robot.

    Each gripper has four "nail" fingers whose tips form a plane; the
    per-gripper end-effector frame is placed at their centroid with its
    +X axis aligned with the tip-forward direction.  Gripper 2's nail
    meshes are mirrored relative to gripper 1's (tips live at local -X
    instead of +X), so the class normalizes the frame for both grippers
    and callers can treat them uniformly in IK targets.

    ``gripper_{1,2}_end_coords.worldcoords()`` is computed on every
    access from the live nail-tip positions, so changes to
    ``worm_rotate_{1,2}`` (which move the tips via URDF mimic bindings)
    are reflected automatically — no manual refresh is required.
    """

    def __init__(self, *args, **kwargs):
        super(Griphis, self).__init__(*args, **kwargs)
        e1, tips1 = self._attach_gripper_end_coords(1)
        e2, tips2 = self._attach_gripper_end_coords(2)
        self.gripper_1_end_coords = e1
        self.gripper_2_end_coords = e2
        self.gripper_1_nail_tips = tips1
        self.gripper_2_nail_tips = tips2

    @cached_property
    def default_urdf_path(self):
        return griphis_urdfpath()

    def _attach_gripper_end_coords(self, gripper_id):
        """Attach per-nail tip coords and a live-centroid end_coords."""
        tips = []
        tip_x_signs = []
        for i in range(1, 5):
            link = getattr(self, 'nail{}_{}_link'.format(i, gripper_id))
            tip_pos = self._nail_tip_local(link)
            tip_x_signs.append(np.sign(tip_pos[0]))
            tip = CascadedCoords(
                pos=tip_pos,
                name='nail{}_{}_tip'.format(i, gripper_id))
            link.assoc(tip, relative_coords='local')
            tips.append(tip)

        roll_link = getattr(self, 'roll_{}_link'.format(gripper_id))
        # Gripper 2's tips sit at -X of their nail links, so the gripper's
        # forward direction is -X of roll_link too.  Flipping the x/y
        # columns of the world rotation keeps +X pointing at the tips
        # for both grippers.
        flip = np.eye(3)
        if np.mean(tip_x_signs) < 0:
            flip = np.array([[-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, 1.0]])

        end_coords = _NailCentroidEndCoords(
            name='gripper_{}_end'.format(gripper_id),
            parent_link=roll_link,
            tips=tips,
            flip=flip,
        )
        return end_coords, tips

    @staticmethod
    def _nail_tip_local(link):
        """Vertex with the largest |x| on the link's visual mesh, in the
        link's local frame.  The URDF loader has already parsed and cached
        the mesh on each link, so no extra trimesh.load is needed."""
        meshes = link.visual_mesh
        if not isinstance(meshes, list):
            meshes = [meshes]
        verts = np.concatenate(
            [np.asarray(m.vertices) for m in meshes], axis=0)
        return verts[np.argmax(np.abs(verts[:, 0]))]
