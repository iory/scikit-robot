"""Compose robots out of reusable URDF modules.

Parse each part once into a :class:`RobotModule` (its links become
connection ports), place instances in a :class:`RobotAssembly`,
connect ports, and build a combined URDF or a live
:class:`skrobot.model.RobotModel`.
"""

from skrobot.assembly.module_assembly import Connection
from skrobot.assembly.module_assembly import ModuleInstance
from skrobot.assembly.module_assembly import Port
from skrobot.assembly.module_assembly import RobotAssembly
from skrobot.assembly.module_assembly import RobotModule


__all__ = [
    'Port',
    'RobotModule',
    'Connection',
    'ModuleInstance',
    'RobotAssembly',
]
