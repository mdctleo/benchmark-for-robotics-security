"""The attack interface shared by every attack in :mod:`robobench.attacks`."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class AttackResult:
    """Artifact columns an attack produced; ``None`` leaves the column unchanged."""

    malicious: str | None = None
    images: list[bytes] | None = None


class Attack(Protocol):
    name: str
    target: str | None

    def attack(self, *, text: str | None = None,
               images: list[bytes] | None = None, **extra) -> AttackResult: ...
