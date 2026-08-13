"""Reverse the positive direction of a URDF joint.

Negating a joint's ``<axis>`` alone leaves the document inconsistent: the
travel range and any mimic coupling are still expressed in the old sense and
have to move with it.  :func:`flip_joint_axis` performs the whole edit -- axis,
limits, the joint's own mimic coefficients and the coefficients of every joint
that mimics it -- so the mechanism reaches the same poses afterwards and only
the sign of the commanded value changes.
"""

import xml.etree.ElementTree as ET


__all__ = [
    'flip_joint_axis',
    'flip_joint_axis_file',
]

# attribute pairs holding a lower/upper bound on the joint value, on <limit>
# and on <safety_controller>
_BOUND_PAIRS = (('lower', 'upper'),
                ('soft_lower_limit', 'soft_upper_limit'))


def _fmt(value):
    """Format a flipped number, folding ``-0.0`` back to ``0``."""
    return '0' if value == 0.0 else '{:.10g}'.format(value)


def _negate_attrib(element, name, default=None):
    """Negate one numeric attribute in place.

    ``default`` is the value URDF gives the attribute when it is absent; pass
    it for an attribute whose implied value is not its own negation, so the
    flipped value gets written out explicitly.  Unparsable values are left
    alone.
    """
    raw = element.get(name)
    if raw is None:
        if default is None:
            return
        raw = default
    try:
        element.set(name, _fmt(-float(raw)))
    except ValueError:
        pass


def _axis_of(joint):
    """The joint's axis as floats, or None when it has no usable one."""
    axis = joint.find('axis')
    if axis is None:
        return None
    try:
        values = [float(v) for v in axis.get('xyz', '').split()]
    except ValueError:
        return None
    if len(values) != 3 or not any(values):
        return None
    return values


def _negate_joint_axis(root, joint_name):
    """Negate one joint's ``<axis xyz>`` and nothing else, in place.

    On its own this is only half a reversal -- the joint value's sign changes
    while the limits and mimic coefficients still describe the old sense -- so
    it is deliberately private.  The one caller that wants exactly this is
    :func:`~skrobot.urdf.change_urdf_root_link`, where swapping a joint's
    ``<parent>`` and ``<child>`` supplies the other half.  Everyone else wants
    :func:`flip_joint_axis`.

    Returns the ``<joint>`` element, or None when there is no such joint or it
    has no usable axis.
    """
    for joint in root.findall('joint'):
        if joint.get('name') != joint_name:
            continue
        axis_values = _axis_of(joint)
        if axis_values is None:
            return None
        joint.find('axis').set(
            'xyz', ' '.join(_fmt(-v) for v in axis_values))
        return joint
    return None


def flip_joint_axis(root, joint_name):
    """Reverse one joint's positive direction, in place.

    The joint reaches the same physical range afterwards; only the sign of the
    value that commands it changes.  Four things move together:

    - ``<axis xyz>`` is negated;
    - ``<limit lower/upper>`` is swapped and negated, so ``[-0.5, 2.0]``
      becomes ``[-2.0, 0.5]``.  A bound changes side, not just sign: ``q <= U``
      is ``q' >= -U``, so a one-sided limit moves to the other attribute.  The
      soft bounds on ``<safety_controller>`` move with it;
    - this joint's own ``<mimic>`` has both ``multiplier`` and ``offset``
      negated, since its own value changed sign;
    - every OTHER joint mimicking this one has its ``multiplier`` negated and
      its ``offset`` left alone.  Those ``<mimic joint="...">`` tags live in
      the follower's element, not in this one, which is the part a hand-written
      flip usually misses: from ``q_f = m * q_d + b`` and ``q_d -> -q_d``, the
      follower needs ``-m`` to stay in phase.

    The joint's ``<origin>`` is deliberately untouched, so no frame moves, and
    the operation is its own inverse -- flipping twice restores the original
    values.

    Parameters
    ----------
    root : xml.etree.ElementTree.Element
        The ``<robot>`` element, modified in place.
    joint_name : str
        Name of the joint to reverse.

    Returns
    -------
    flipped : bool
        False when no such joint exists, or when it has no usable axis (a
        ``fixed`` joint, or one whose axis is the zero vector); nothing is
        modified in that case.

    Examples
    --------
    >>> import xml.etree.ElementTree as ET
    >>> from skrobot.urdf import flip_joint_axis
    >>> root = ET.fromstring('''<robot name="r">
    ...   <joint name="j" type="revolute">
    ...     <axis xyz="0 0 1"/>
    ...     <limit lower="-0.5" upper="2.0" effort="1" velocity="1"/>
    ...   </joint>
    ... </robot>''')
    >>> flip_joint_axis(root, 'j')
    True
    >>> joint = root.find('joint')
    >>> joint.find('axis').get('xyz')
    '0 0 -1'
    >>> joint.find('limit').get('lower'), joint.find('limit').get('upper')
    ('-2', '0.5')

    See Also
    --------
    skrobot.urdf.change_urdf_root_link : re-roots a URDF, reversing every joint
        along the path with this function.
    """
    joint = _negate_joint_axis(root, joint_name)
    if joint is None:
        return False

    # q <= U becomes q' >= -U, so a bound does not merely change sign: it
    # changes side.  A one-sided limit therefore moves to the other attribute
    # rather than staying put.
    for element in (joint.find('limit'), joint.find('safety_controller')):
        if element is None:
            continue
        for low_name, high_name in _BOUND_PAIRS:
            low, high = element.get(low_name), element.get(high_name)
            if low is None and high is None:
                continue
            try:
                flipped_low = None if high is None else _fmt(-float(high))
                flipped_high = None if low is None else _fmt(-float(low))
            except ValueError:
                continue
            for name, value in ((low_name, flipped_low),
                                (high_name, flipped_high)):
                if value is None:
                    element.attrib.pop(name, None)
                else:
                    element.set(name, value)

    mimic = joint.find('mimic')
    if mimic is not None:
        # multiplier defaults to 1 and offset to 0 when absent; the implied
        # multiplier has to be written out, since -1 is not the default
        _negate_attrib(mimic, 'multiplier', default='1')
        _negate_attrib(mimic, 'offset')

    for follower in root.findall('joint'):
        if follower is joint:
            continue
        follower_mimic = follower.find('mimic')
        if follower_mimic is not None \
                and follower_mimic.get('joint') == joint_name:
            _negate_attrib(follower_mimic, 'multiplier', default='1')
    return True


def flip_joint_axis_file(input_path, joint_name, output_path=None):
    """Apply :func:`flip_joint_axis` to a URDF file.

    Parameters
    ----------
    input_path : str
        URDF to read.
    joint_name : str
        Name of the joint to reverse.
    output_path : str or None
        Where to write.  Defaults to ``input_path``, editing it in place.

    Returns
    -------
    flipped : bool
        Whether the joint was reversed.  The output is written either way when
        ``output_path`` names a different file, so it can be used as a copy.
    """
    tree = ET.parse(input_path)
    flipped = flip_joint_axis(tree.getroot(), joint_name)
    if not flipped and output_path in (None, input_path):
        return False
    tree.write(output_path or input_path,
               encoding='utf-8', xml_declaration=True)
    return flipped
