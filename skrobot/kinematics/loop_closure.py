"""Close declared kinematic loops on a RobotModel by numerical IK.

A URDF is a tree: building a closed linkage (four-bar, parallel
mechanism) cuts every loop, and the cut is exported as a closure
constraint -- :class:`skrobot.assembly.module_assembly.RobotAssembly`
writes it to a ``loop_closures.yaml`` sidecar next to the built URDF.
:class:`LoopClosureSolver` consumes that config and Gauss-Newton
solves the DEPENDENT joints so each cut hinge's witness points
coincide again: the same constraint a runtime relay node enforces on
a ROS robot, made available to plain skrobot users (viewers,
planners, analysis).
"""

import numpy as np


class LoopClosureSolver(object):
    """Solve a robot's declared loop closures in place.

    Set the driven (independent) joints to their targets, then call
    :meth:`solve`: the dependent joints are updated so every closure's
    two witness points -- two points on the cut hinge axis, one frame
    per side of the cut -- coincide.  Large driver motions are
    sub-stepped so the solution stays on the assembled branch of the
    mechanism instead of jumping to a mirror configuration.

    Parameters
    ----------
    robot_model : skrobot.model.RobotModel
        Robot built from the loop-cut URDF (e.g. via
        :meth:`RobotAssembly.build_robot_model`).
    config : dict
        Closure config ``{closures: [{link_a, link_b, point, axis}],
        dependent: [...], independent: [...]}`` with ``point`` and
        ``axis`` expressed in the root frame at the zero pose -- the
        ``loop_closures.yaml`` contract.

    Examples
    --------
    >>> robot = assembly.build_robot_model()
    >>> solver = LoopClosureSolver(robot, assembly.loop_closures)
    >>> robot.crank_hinge.joint_angle(0.4)
    >>> solver.solve()
    """

    def __init__(self, robot_model, config):
        if not config:
            raise ValueError(
                "no closure config (the assembly declares no loop "
                "connection, or the loop_closures.yaml is empty)")
        closures = list(config.get("closures", []))
        if not closures:
            raise ValueError("config declares no closures to solve")
        joints = {j.name: j for j in robot_model.joint_list}
        links = {link.name: link for link in robot_model.link_list}
        missing = [n for n in list(config.get("dependent", []))
                   + list(config.get("independent", []))
                   if n not in joints]
        if missing:
            raise ValueError(
                f"config joint(s) {missing} do not exist on the robot")
        self.robot_model = robot_model
        self.dependent = [joints[n] for n in config.get("dependent", [])]
        if not self.dependent:
            raise ValueError(
                "config marks no dependent joint; a closure with nothing "
                "to solve cannot be enforced")
        self.independent = [joints[n]
                            for n in config.get("independent", [])]
        multi_dof = [j.name for j in self.dependent + self.independent
                     if getattr(j, "joint_dof", 1) != 1]
        if multi_dof:
            raise ValueError(
                f"joint(s) {multi_dof} have more than one degree of "
                "freedom; the solver only handles scalar (revolute / "
                "continuous / prismatic) joints")

        # capture the witness pairs in link-local coordinates at the
        # zero pose, the frame the config's point/axis are expressed in.
        # The zero-pose transforms are composed analytically from the
        # joints' default_coords: actually zeroing and restoring the
        # robot would leave tiny incremental-rotation drift in the link
        # frames, which the solver would then bake into the dependents.
        self._witnesses = []
        for closure in closures:
            unknown = [n for n in (closure["link_a"], closure["link_b"])
                       if n not in links]
            if unknown:
                raise ValueError(
                    f"closure link(s) {unknown} do not exist on the robot")
            link_a = links[closure["link_a"]]
            link_b = links[closure["link_b"]]
            zero_a = self._zero_pose_transform(link_a)
            zero_b = self._zero_pose_transform(link_b)
            point = np.asarray(closure["point"], dtype=float)
            axis = np.asarray(closure["axis"], dtype=float)
            norm = np.linalg.norm(axis)
            if not np.isfinite(norm) or norm < 1e-12:
                raise ValueError(
                    f"closure {closure['link_a']} <-> "
                    f"{closure['link_b']} has a zero or non-finite "
                    "hinge axis")
            axis = axis / norm
            for witness in (point, point + 0.03 * axis):
                homogeneous = np.append(witness, 1.0)
                self._witnesses.append(
                    (link_a, np.linalg.solve(zero_a, homogeneous)[:3],
                     link_b, np.linalg.solve(zero_b, homogeneous)[:3]))
        self._last_independent = np.array(
            [j.joint_angle() for j in self.independent])

    @classmethod
    def from_yaml(cls, robot_model, path):
        """Build a solver from a ``loop_closures.yaml`` sidecar file."""
        import yaml
        with open(path) as f:
            return cls(robot_model, yaml.safe_load(f))

    @staticmethod
    def _zero_pose_transform(link):
        """4x4 world transform of ``link`` with every joint at zero.

        Composes each parent joint's ``default_coords`` (the child link
        frame relative to its parent, captured at model construction)
        down the chain from the root, so the current joint values never
        enter and the robot is not disturbed.
        """
        chain = []
        current = link
        while current.parent_link is not None and \
                getattr(current, "joint", None) is not None:
            chain.append(current.joint)
            current = current.parent_link
        coords = current.worldcoords()
        transform = np.eye(4)
        transform[:3, :3] = coords.worldrot()
        transform[:3, 3] = coords.worldpos()
        for joint in reversed(chain):
            local = np.eye(4)
            local[:3, :3] = joint.default_coords.rotation
            local[:3, 3] = joint.default_coords.translation
            transform = transform @ local
        return transform

    def _residual(self):
        """Stacked witness-point gaps (3 scalars per witness)."""
        out = np.empty(3 * len(self._witnesses))
        for i, (link_a, local_a, link_b, local_b) in \
                enumerate(self._witnesses):
            coords_a = link_a.worldcoords()
            coords_b = link_b.worldcoords()
            gap = (coords_a.worldrot() @ local_a + coords_a.worldpos()
                   - coords_b.worldrot() @ local_b - coords_b.worldpos())
            out[3 * i:3 * i + 3] = gap
        return out

    def _jacobian(self, residual, eps=1e-6):
        """Numeric Jacobian of the residual w.r.t. dependent joints."""
        jacobian = np.empty((residual.shape[0], len(self.dependent)))
        for k, joint in enumerate(self.dependent):
            q = joint.joint_angle()
            # perturb away from a limit so the clamped set is a no-op
            step = eps if q + eps <= joint.max_angle else -eps
            joint.joint_angle(q + step)
            jacobian[:, k] = (self._residual() - residual) / step
            joint.joint_angle(q)
        return jacobian

    def closure_error(self):
        """Largest witness-point gap (metres) over all closures.

        Unlike the stacked residual norm :meth:`solve` returns, this is
        a per-witness maximum, so the two differ by up to ``sqrt(2 *
        n_closures)`` on the same state.
        """
        residual = self._residual()
        return float(max(np.linalg.norm(residual[3 * i:3 * i + 3])
                         for i in range(len(self._witnesses))))

    def solve(self, tol=1e-10, max_iter=50, max_step=0.1, max_dq=0.5,
              raise_error=True):
        """Update the dependent joints so every loop closes.

        Reads the CURRENT independent joint values as the target,
        interpolates from the previously solved values in steps of at
        most ``max_step`` (radians / metres), and Gauss-Newton solves
        the dependent joints at each sub-step.  Each update is the
        minimum-norm least-squares step (SVD), which stays stable on
        the rank-deficient Jacobians every planar mechanism produces
        (the out-of-plane residual rows are identically zero).

        Parameters
        ----------
        tol : float
            Success threshold on the residual norm (metres).
        max_iter : int
            Gauss-Newton iterations per sub-step.
        max_step : float
            Largest independent-joint change per sub-step; keeps the
            solution on the assembled branch of the mechanism.
        max_dq : float
            Cap on a single Gauss-Newton update per joint (radians /
            metres), guarding against wild steps near singularities.
        raise_error : bool
            Raise ``ValueError`` when the final residual exceeds
            ``tol`` (e.g. the loop cannot close at these driver
            values).  Pass ``False`` to just return the residual.

        Returns
        -------
        float
            Final residual norm in metres.

        Notes
        -----
        On failure the robot is left at the unconverged pose, but the
        warm-start state is NOT advanced: the next call sub-steps from
        the last successfully solved independent values, so one
        unreachable target does not poison later reachable ones.
        """
        if max_step <= 0 or max_dq <= 0 or max_iter < 1 or tol < 0:
            raise ValueError(
                "max_step and max_dq must be positive, max_iter >= 1 "
                "and tol >= 0")
        target = np.array([j.joint_angle() for j in self.independent])
        if target.size and self._last_independent.size:
            delta = target - self._last_independent
            n_sub = max(1, int(np.ceil(np.max(np.abs(delta)) / max_step)))
        else:
            delta = target
            n_sub = 1
        for step in range(1, n_sub + 1):
            values = self._last_independent + delta * (step / n_sub)
            for joint, value in zip(self.independent, values):
                joint.joint_angle(value)
            residual = self._residual()
            for _ in range(max_iter):
                if np.linalg.norm(residual) < tol:
                    break
                jacobian = self._jacobian(residual)
                dq = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
                worst = np.max(np.abs(dq))
                if worst > max_dq:
                    dq *= max_dq / worst
                q = np.array([j.joint_angle() for j in self.dependent])
                for joint, value in zip(self.dependent, q - dq):
                    joint.joint_angle(value)
                residual = self._residual()
        error = float(np.linalg.norm(self._residual()))
        if error <= tol:
            # commit the values actually applied (joint_angle clamps to
            # limits), and only on success -- a failed target must not
            # become the warm start for the next call
            self._last_independent = np.array(
                [j.joint_angle() for j in self.independent])
        elif raise_error:
            raise ValueError(
                f"loop closure did not converge: residual {error:.3g} m "
                f"exceeds tol {tol:.3g} (the loop may not close at these "
                "driver values, e.g. a crank past its geometric limit)")
        return error
