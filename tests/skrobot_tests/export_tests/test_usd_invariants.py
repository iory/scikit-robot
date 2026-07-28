"""Structural USD physics invariants for exported articulations.

These tests validate authored-stage invariants that are known prerequisites for
stable PhysX execution (finite numeric data, valid rigid-body mass/inertia,
non-degenerate colliders, sane limits, articulation wiring). They do not run
Isaac Sim and do not validate runtime PhysX behavior.
"""

import numbers
import os
import shutil
import tempfile
import unittest

import numpy as np


try:
    from pxr import Sdf
    from pxr import UsdGeom
    from pxr import UsdPhysics
    _HAS_PXR = True
except ImportError:
    _HAS_PXR = False

import skrobot
from skrobot.export.usd import urdf_to_usd
from skrobot.urdf.sanitize import sanitize_name
from skrobot.utils.urdf import URDF


KUKA_URDF = os.path.join(
    os.path.dirname(skrobot.__file__),
    'data', 'kuka_description', 'kuka.urdf')


def _is_numeric_attr_type(type_name):
    """Return whether the USD value type name is numeric-like.

    Parameters
    ----------
    type_name : str
        USD attribute type name.

    Returns
    -------
    bool
        True when the type name encodes numeric scalars/vectors/matrices.
    """
    numeric_tokens = (
        'float',
        'double',
        'half',
        'int',
        'matrix',
        'point',
        'vector',
        'normal',
        'color',
        'quat',
    )
    return any(token in type_name for token in numeric_tokens)


def _iter_numeric_components(value):
    """Yield numeric scalar components from nested USD/Python values.

    Parameters
    ----------
    value : object
        Attribute value returned by pxr.

    Yields
    ------
    float
        Numeric scalar components.
    """
    if value is None:
        return
    if isinstance(value, (bool, str, bytes)):
        return
    if isinstance(value, numbers.Real):
        yield float(value)
        return
    if hasattr(value, 'GetReal') and hasattr(value, 'GetImaginary'):
        # Gf.Quat* types are not array-like but expose scalar/vector parts.
        yield float(value.GetReal())
        for scalar in _iter_numeric_components(value.GetImaginary()):
            yield scalar
        return
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        arr = None
    if arr is not None and arr.dtype != object:
        for scalar in arr.ravel():
            yield float(scalar)
        return
    try:
        iterator = iter(value)
    except TypeError:
        return
    for item in iterator:
        for scalar in _iter_numeric_components(item):
            yield scalar


@unittest.skipUnless(_HAS_PXR, 'usd-core (pxr) not installed')
class TestUsdInvariants(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.out = os.path.join(cls.tmpdir, 'kuka.usdc')
        cls.stage = urdf_to_usd(KUKA_URDF, cls.out)
        cls.urdf = URDF.load(KUKA_URDF)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _assert_finite_numeric_value(self, value, context):
        """Assert that every numeric component in ``value`` is finite."""
        found_numeric = False
        for scalar in _iter_numeric_components(value):
            found_numeric = True
            self.assertTrue(
                np.isfinite(scalar),
                '{}: non-finite component {!r}'.format(context, scalar))
        return found_numeric

    def test_no_non_finite_numbers_in_authored_attributes(self):
        for prim in self.stage.Traverse():
            for attr in prim.GetAttributes():
                if not attr.HasAuthoredValueOpinion():
                    continue
                type_name = str(attr.GetTypeName())
                is_numeric = _is_numeric_attr_type(type_name)
                contexts_and_values = [('default', attr.Get())]
                for sample_time in attr.GetTimeSamples():
                    contexts_and_values.append(
                        ('time {}'.format(sample_time), attr.Get(sample_time)))
                for sample_context, value in contexts_and_values:
                    context = '{} [{}] ({})'.format(
                        attr.GetPath(), type_name, sample_context)
                    found_numeric = self._assert_finite_numeric_value(
                        value, context)
                    if is_numeric:
                        self.assertTrue(
                            found_numeric,
                            '{}: numeric attribute had no numeric components'
                            .format(context))

    def test_rigid_bodies_have_positive_finite_mass_and_inertia(self):
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            mapi = UsdPhysics.MassAPI(prim)
            mass_attr = mapi.GetMassAttr()
            inertia_attr = mapi.GetDiagonalInertiaAttr()

            mass = mass_attr.Get()
            self.assertIsNotNone(mass, '{} missing physics:mass'.format(prim.GetPath()))
            mass = float(mass)
            self.assertTrue(np.isfinite(mass), '{} mass not finite'.format(prim.GetPath()))
            self.assertGreater(mass, 0.0, '{} mass must be > 0'.format(prim.GetPath()))

            diag = np.asarray(inertia_attr.Get(), dtype=float).reshape(-1)
            self.assertEqual(
                diag.size, 3,
                '{} diagonal inertia must have 3 components'.format(prim.GetPath()))
            self.assertTrue(
                np.all(np.isfinite(diag)),
                '{} diagonal inertia not finite: {}'.format(prim.GetPath(), diag))
            self.assertTrue(
                np.all(diag > 0.0),
                '{} diagonal inertia must be > 0: {}'.format(prim.GetPath(), diag))

    def test_mesh_colliders_are_non_degenerate(self):
        for prim in self.stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            self.assertIsNotNone(points, '{} has no authored points'.format(prim.GetPath()))
            self.assertGreaterEqual(
                len(points), 4,
                '{} collider has fewer than 4 points'.format(prim.GetPath()))
            found_numeric = self._assert_finite_numeric_value(
                points, '{}.points'.format(mesh.GetPath()))
            self.assertTrue(
                found_numeric,
                '{}.points had no numeric components'.format(mesh.GetPath()))

    def test_joint_limits_are_finite_and_ordered(self):
        for prim in self.stage.Traverse():
            if prim.GetTypeName() == 'PhysicsRevoluteJoint':
                joint = UsdPhysics.RevoluteJoint(prim)
            elif prim.GetTypeName() == 'PhysicsPrismaticJoint':
                joint = UsdPhysics.PrismaticJoint(prim)
            else:
                continue

            lower_attr = joint.GetLowerLimitAttr()
            upper_attr = joint.GetUpperLimitAttr()
            has_lower = lower_attr.HasAuthoredValueOpinion()
            has_upper = upper_attr.HasAuthoredValueOpinion()
            if not has_lower and not has_upper:
                continue

            self.assertTrue(
                has_lower and has_upper,
                '{} must author both lower and upper limits'.format(prim.GetPath()))

            lower = float(lower_attr.Get())
            upper = float(upper_attr.Get())
            self.assertTrue(
                np.isfinite(lower) and np.isfinite(upper),
                '{} has non-finite limits lower={} upper={}'.format(
                    prim.GetPath(), lower, upper))
            self.assertLessEqual(
                lower, upper,
                '{} has inverted limits lower={} upper={}'.format(
                    prim.GetPath(), lower, upper))

    def test_single_articulation_root_and_base_world_weld(self):
        articulation_roots = [
            prim for prim in self.stage.Traverse()
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        ]
        self.assertEqual(
            len(articulation_roots), 1,
            'expected exactly one articulation root, got {}'.format(
                [str(p.GetPath()) for p in articulation_roots]))
        self.assertEqual(str(articulation_roots[0].GetPath()), '/robot')

        joint_prim = self.stage.GetPrimAtPath('/robot/joints/base_fixed')
        joint = UsdPhysics.FixedJoint(joint_prim)
        self.assertTrue(joint.GetPrim().IsValid())
        self.assertEqual(joint.GetPrim().GetTypeName(), 'PhysicsFixedJoint')

        body0_targets = joint.GetBody0Rel().GetTargets()
        body1_targets = joint.GetBody1Rel().GetTargets()
        self.assertEqual(
            body0_targets, [],
            'base_fixed.body0 must target world (empty relation), got {}'.format(
                body0_targets))

        expected_base = Sdf.Path(
            '/robot/{}'.format(sanitize_name(self.urdf.base_link.name)))
        self.assertEqual(
            len(body1_targets), 1,
            'base_fixed.body1 must target exactly one base link, got {}'.format(
                body1_targets))
        self.assertEqual(
            body1_targets[0], expected_base,
            'base_fixed.body1 target mismatch: expected {}, got {}'.format(
                expected_base, body1_targets[0]))


if __name__ == '__main__':
    unittest.main()
