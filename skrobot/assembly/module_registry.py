"""
Registry of URDF modules available for assembly.

Scans a directory of URDF files and turns each one into a
:class:`~skrobot.assembly.module_assembly.RobotModule`.
"""

import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from skrobot.assembly.module_assembly import RobotModule


logger = logging.getLogger(__name__)


class ModuleRegistry:
    """
    Registry for managing available URDF modules.

    This class scans a directory for URDF files and creates RobotModule
    instances that can be used in RobotAssembly.

    Parameters
    ----------
    assets_dir : str or Path
        Path to the directory containing URDF files.
    recursive : bool
        If True, scan subdirectories recursively.

    Examples
    --------
    >>> registry = ModuleRegistry("/path/to/reconfigurable_assets/urdf")
    >>> registry.scan()
    >>> module = registry.get_module("screw_100mmhinge_KRS6104_module")
    """

    def __init__(
        self,
        assets_dir: Union[str, Path],
        recursive: bool = False,
    ):
        self.assets_dir = Path(assets_dir)
        self.recursive = recursive
        self._modules: Dict[str, RobotModule] = {}

    def scan(self) -> int:
        """
        Scan the assets directory and register all URDF files.

        Returns
        -------
        int
            Number of modules registered.
        """
        if not self.assets_dir.exists():
            raise FileNotFoundError(f"Assets directory not found: {self.assets_dir}")

        pattern = "**/*.urdf" if self.recursive else "*.urdf"
        urdf_files = list(self.assets_dir.glob(pattern))

        for urdf_path in urdf_files:
            try:
                self._register_urdf(urdf_path)
            except Exception as e:
                logger.warning("Failed to register %s: %s", urdf_path, e)

        return len(self._modules)

    def _register_urdf(self, urdf_path: Path) -> None:
        """Register a single URDF file."""
        module_id = urdf_path.stem
        self._modules[module_id] = RobotModule.from_urdf(module_id, str(urdf_path))

    def get_module(self, module_id: str) -> Optional[RobotModule]:
        """
        Get a RobotModule by its ID.

        Parameters
        ----------
        module_id : str
            The module identifier.

        Returns
        -------
        RobotModule or None
            The module if found, None otherwise.
        """
        return self._modules.get(module_id)

    def list_modules(self) -> List[str]:
        """Return list of all registered module IDs."""
        return list(self._modules.keys())

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, module_id: str) -> bool:
        return module_id in self._modules

    def __iter__(self):
        return iter(self._modules.values())
