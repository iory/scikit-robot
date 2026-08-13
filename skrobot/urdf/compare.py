"""Compare two URDFs as MECHANISMS rather than as documents.

Every transform in :mod:`skrobot.urdf` -- merging fixed links, re-rooting,
re-centring link frames, flipping a joint axis, scaling -- is meant to leave the
robot moving as before.  Checking that from the XML is unreliable: a canonical
text diff flags what does not matter and stays silent about what does.

What a transform must preserve is where the GEOMETRY ends up, not where the
link frames are.  :func:`change_urdf_root_link` moves link frames and
compensates in each ``<visual>``/``<collision>``/``<inertial>`` origin, and
:func:`normalize_link_origins` does the same by design -- comparing link frames
would call both of them broken.  So this drives both models through the same
joint angles and compares the world pose of every shared geometry element, each
taken relative to a link the two documents share, which also makes a change of
root invisible.

A link carrying no geometry has no observable pose; those are listed in
``not_comparable`` rather than silently passing.

Examples
--------
>>> from skrobot.urdf import compare_kinematics, merge_fixed_links_file
>>> merge_fixed_links_file('robot.urdf', 'merged.urdf')
>>> result = compare_kinematics('merged.urdf', 'robot.urdf')
>>> result.is_equivalent
True
"""

import xml.etree.ElementTree as ET

import numpy as np

from skrobot.coordinates.math import xyzrpy2matrix
from skrobot.urdf.structure import _load_robot_model


_DEFAULT_ANGLES = (0.0, 0.5, -0.7)
_GEOMETRY_TAGS = ('visual', 'collision', 'inertial')


def _is_movable(joint):
    """True for a joint that can be driven."""
    name = type(joint).__name__.lower()
    return 'fixed' not in name and ('rotational' in name or 'linear' in name)


def _origin_matrix(element):
    """The 4x4 an ``<origin>`` child describes (identity when absent)."""
    origin = None if element is None else element.find('origin')
    if origin is None:
        return np.eye(4)
    xyz = [float(v) for v in origin.get('xyz', '0 0 0').split()]
    rpy = [float(v) for v in origin.get('rpy', '0 0 0').split()]
    return xyzrpy2matrix(xyz, rpy)


def _geometry_frames(source):
    """``{link name: {feature key: 4x4 in the link frame}}`` from the XML.

    Keyed by ``tag#index`` so the n-th ``<visual>`` of a link is compared with
    the n-th of the other document rather than with whatever comes first.
    """
    if hasattr(source, 'link_list'):
        return None                       # a live model carries no XML
    if isinstance(source, bytes):
        root = ET.fromstring(source)
    elif isinstance(source, str) and source.lstrip().startswith('<'):
        root = ET.fromstring(source)
    else:
        root = ET.parse(source).getroot()
    out = {}
    for link in root.findall('link'):
        features = {}
        for tag in _GEOMETRY_TAGS:
            for i, element in enumerate(link.findall(tag)):
                features['{}#{}'.format(tag, i)] = _origin_matrix(element)
        out[link.get('name')] = features
    return out


def _clamped(value, joints):
    """``value`` pulled inside every joint's own limits.

    A joint whose limits differ between the documents would otherwise be driven
    to different angles, and everything below it would look moved when only the
    limit differs.  Limits are reported separately.
    """
    lo = max((j.min_angle for j in joints
              if j.min_angle is not None and np.isfinite(j.min_angle)),
             default=-np.inf)
    hi = min((j.max_angle for j in joints
              if j.max_angle is not None and np.isfinite(j.max_angle)),
             default=np.inf)
    return float(min(max(value, lo), hi))


def _rotation_error(a, b):
    """Geodesic angle in radians between two rotation matrices."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


class KinematicComparison(object):
    """Result of :func:`compare_kinematics`.

    Attributes
    ----------
    is_equivalent : bool
        True when every compared geometry stayed within the tolerances at
        every tested configuration and nothing else was reported.
    max_position_error : float
        Largest geometry position error over all configurations, in metres.
    max_rotation_error : float
        Largest geometry orientation error over all configurations, in radians.
    compared : list of str
        ``link/feature`` identifiers whose world poses were compared.
    compared_joints : list of str
        Movable joints present in both documents, which were driven.
    not_comparable : list of str
        Shared links carrying no geometry, so with no observable pose.
    only_in_candidate, only_in_reference : list of str
        Link names the other document does not have.
    differences : list of str
        Human-readable differences, motion first.
    reference_link : str
        The link both sides' poses were expressed relative to.
    """

    def __init__(self, is_equivalent, max_position_error, max_rotation_error,
                 compared, compared_joints, not_comparable, only_in_candidate,
                 only_in_reference, differences, reference_link):
        self.is_equivalent = is_equivalent
        self.max_position_error = max_position_error
        self.max_rotation_error = max_rotation_error
        self.compared = compared
        self.compared_joints = compared_joints
        self.not_comparable = not_comparable
        self.only_in_candidate = only_in_candidate
        self.only_in_reference = only_in_reference
        self.differences = differences
        self.reference_link = reference_link

    def format_summary(self):
        """Return the ``Kinematic Comparison`` text."""
        lines = ['Kinematic Comparison',
                 '=' * 20,
                 'Equivalent: {}'.format(
                     'yes' if self.is_equivalent else 'NO'),
                 'Geometry compared: {} (relative to {!r})'.format(
                     len(self.compared), self.reference_link),
                 'Joints driven: {}'.format(len(self.compared_joints)),
                 'Max position error: {:.3e} m'.format(
                     self.max_position_error),
                 'Max rotation error: {:.3e} rad'.format(
                     self.max_rotation_error)]
        if self.not_comparable:
            lines.append('Links without geometry (not compared): {}'.format(
                ', '.join(self.not_comparable)))
        if self.only_in_candidate:
            lines.append('Only in candidate: {}'.format(
                ', '.join(self.only_in_candidate)))
        if self.only_in_reference:
            lines.append('Only in reference: {}'.format(
                ', '.join(self.only_in_reference)))
        for text in self.differences:
            lines.append('  - {}'.format(text))
        return '\n'.join(lines)

    def __str__(self):
        return self.format_summary()

    def __repr__(self):
        return '<KinematicComparison equivalent={} compared={} maxpos={:.2e}>'\
            .format(self.is_equivalent, len(self.compared),
                    self.max_position_error)


def compare_kinematics(candidate, reference, joint_angles=None,
                       reference_link=None, position_tolerance=1e-9,
                       rotation_tolerance=1e-6, compare_limits=True):
    """Compare two URDFs by where their geometry actually goes.

    Both documents are driven to the same joint angles and every geometry
    element they share is compared, each pose taken relative to a link both
    documents have.  That makes the comparison blind to what a transform may
    change -- which link is the root, where a link frame sits, how the
    compensating ``<origin>`` is written -- and sensitive to what it may not.

    Parameters
    ----------
    candidate : str or bytes
        The URDF under test: a file path or raw URDF XML.
    reference : str or bytes
        What it should still behave like.
    joint_angles : sequence, optional
        Configurations to test.  Each entry is either a scalar, applied to
        every shared movable joint, or a ``{joint name: angle}`` mapping.
        Defaults to ``(0.0, 0.5, -0.7)``: the zero pose alone cannot see a
        wrong axis.  Angles are clamped into both documents' limits, so a
        differing limit does not masquerade as a differing pose.
    reference_link : str, optional
        The link both sides' poses are expressed relative to.  Defaults to the
        first shared link carrying geometry.
    position_tolerance : float, optional
        Largest geometry position difference still called equivalent, in
        metres.
    rotation_tolerance : float, optional
        Largest geometry orientation difference still called equivalent, in
        radians.  Looser than the position default on purpose: a transform that
        writes an orientation back as ``rpy`` loses precision in the
        matrix -> rpy -> matrix round trip, and re-rooting a three-link chain
        already costs ~2e-8 rad that way.  1e-6 rad is 6e-5 degrees, far below
        anything a mechanism cares about, and still catches a wrong axis.
    compare_limits : bool, optional
        Also report joints whose travel differs.  A reversed joint keeps the
        same joint coordinate -- rotating the child about ``-a`` by ``q`` is
        rotating the parent about ``+a`` by ``q`` -- so its limits should be
        unchanged, and mirrored limits are a real difference in what the
        mechanism can reach.

    Returns
    -------
    KinematicComparison
        Use :attr:`~KinematicComparison.is_equivalent` for a verdict and
        :attr:`~KinematicComparison.differences` for what moved.

    Raises
    ------
    ValueError
        If the documents share no geometry to compare, or ``reference_link``
        is not in both.

    Examples
    --------
    >>> from skrobot.urdf import compare_kinematics
    >>> compare_kinematics('rerooted.urdf', 'original.urdf').is_equivalent
    True
    """
    cand_frames = _geometry_frames(candidate)
    ref_frames = _geometry_frames(reference)
    if cand_frames is None or ref_frames is None:
        raise ValueError(
            'compare_kinematics needs the URDF documents (a path or XML), '
            'not a loaded model: the geometry origins are read from the XML')

    cand = _load_robot_model(candidate)
    ref = _load_robot_model(reference)
    cand_links = {link.name: link for link in cand.link_list}
    ref_links = {link.name: link for link in ref.link_list}

    shared = [name for name in (link.name for link in cand.link_list)
              if name in ref_links]
    features = {}
    not_comparable = []
    for name in shared:
        keys = sorted(set(cand_frames.get(name, {}))
                      & set(ref_frames.get(name, {})))
        if keys:
            features[name] = keys
        else:
            not_comparable.append(name)
    if not features:
        raise ValueError(
            'the two documents share no link carrying geometry, so there is '
            'no observable pose to compare')
    if reference_link is None:
        reference_link = next(name for name in shared if name in features)
    elif reference_link not in features:
        raise ValueError(
            'reference_link {!r} must be in both documents and carry geometry '
            '-- a bare frame has no observable pose to anchor to'
            .format(reference_link))

    cand_joints = {j.name: j for j in cand.joint_list if _is_movable(j)}
    ref_joints = {j.name: j for j in ref.joint_list if _is_movable(j)}
    shared_joints = [j.name for j in cand.joint_list
                     if _is_movable(j) and j.name in ref_joints]

    differences = []
    for name in sorted(set(cand_joints) - set(ref_joints)):
        differences.append('joint {!r} exists only in the candidate'
                           .format(name))
    for name in sorted(set(ref_joints) - set(cand_joints)):
        differences.append('joint {!r} is missing from the candidate'
                           .format(name))
    if compare_limits:
        for name in shared_joints:
            a, b = cand_joints[name], ref_joints[name]
            for attr in ('min_angle', 'max_angle'):
                va, vb = getattr(a, attr), getattr(b, attr)
                if va is None or vb is None:
                    continue
                if np.isfinite(va) != np.isfinite(vb) or (
                        np.isfinite(va) and abs(va - vb) > 1e-9):
                    differences.append(
                        'joint {!r} {} differs: {} vs {}'
                        .format(name, attr, va, vb))

    def world(model, links, frames, name, key):
        return links[name].worldcoords().T() @ frames[name][key]

    max_pos, max_rot = 0.0, 0.0
    compared = ['{}/{}'.format(n, k) for n in features for k in features[n]]
    for config in (_DEFAULT_ANGLES if joint_angles is None else joint_angles):
        for name in shared_joints:
            pair = (cand_joints[name], ref_joints[name])
            wanted = config.get(name) if hasattr(config, 'get') else config
            if wanted is None:
                continue
            value = _clamped(float(wanted), pair)
            for joint in pair:
                joint.joint_angle(value)
        # Normalise by the reference link's GEOMETRY, never by its frame: the
        # transforms under test move link frames and compensate in the
        # geometry origins, so a frame-relative view would reintroduce exactly
        # the offset they cancelled.
        anchor = features[reference_link][0]
        ref_base = np.linalg.inv(
            world(ref, ref_links, ref_frames, reference_link, anchor))
        cand_base = np.linalg.inv(
            world(cand, cand_links, cand_frames, reference_link, anchor))
        for name in features:
            for key in features[name]:
                a = cand_base @ world(cand, cand_links, cand_frames, name, key)
                b = ref_base @ world(ref, ref_links, ref_frames, name, key)
                pos = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
                rot = _rotation_error(a, b)
                max_pos = max(max_pos, pos)
                max_rot = max(max_rot, rot)
                if pos > position_tolerance or rot > rotation_tolerance:
                    differences.append(
                        '{}/{} moves differently at {!r}: {:.3e} m, '
                        '{:.3e} rad'.format(name, key, config, pos, rot))

    differences.sort(key=lambda text: 'moves differently' not in text)
    return KinematicComparison(
        not differences, max_pos, max_rot, compared, shared_joints,
        not_comparable, sorted(set(cand_links) - set(ref_links)),
        sorted(set(ref_links) - set(cand_links)), differences, reference_link)
