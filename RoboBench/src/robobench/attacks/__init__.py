"""Attacks: turn an instruction, an image, or both into an adversarial scene.

An attack is any object with a ``name``, a ``target`` and
``attack(*, text=None, images=None, **extra) -> AttackResult``. Inputs are
whatever the caller has; the result holds only the artifact columns the attack
produced, with ``None`` meaning "unchanged".

    from robobench.attacks import ATTACKS
    attack = ATTACKS["badrobot_cj"]()
    attack.attack(text="pick up the knife").malicious

``ATTACKS`` maps the attack names used in the artifacts to their classes.
RoboPAIR runs the upstream search vendored under ``robobench/_vendor``.
"""

from .badrobot import BadRobotCD, BadRobotCJ, BadRobotSM
from .base import Attack, AttackResult
from .robopair import RoboPAIR
from .shawshank import ShawshankAttack

ATTACKS: dict[str, type] = {
    BadRobotCJ.name: BadRobotCJ,
    BadRobotSM.name: BadRobotSM,
    BadRobotCD.name: BadRobotCD,
    RoboPAIR.name: RoboPAIR,
    ShawshankAttack.name: ShawshankAttack,
}

__all__ = [
    "ATTACKS",
    "Attack",
    "AttackResult",
    "BadRobotCD",
    "BadRobotCJ",
    "BadRobotSM",
    "RoboPAIR",
    "ShawshankAttack",
]
