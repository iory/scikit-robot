"""Residual specification for trajectory optimization."""


class ResidualSpec:
    """Specification for a residual function.

    This class holds the residual function and its parameters,
    allowing solvers to construct optimization problems.
    """

    def __init__(
        self,
        name,
        residual_fn,
        params,
        kind='soft',
        weight=1.0,
    ):
        """Initialize residual specification.

        Parameters
        ----------
        name : str
            Residual name for debugging.
        residual_fn : callable
            Function computing residual from variables and params.
        params : dict
            Parameters passed to residual function.
        kind : str
            'soft' for soft cost, 'eq' for equality constraint,
            'geq' for >= 0 constraint.
        weight : float
            Residual weight (for soft costs).
        """
        self.name = name
        self.residual_fn = residual_fn
        self.params = params
        self.kind = kind
        self.weight = weight
