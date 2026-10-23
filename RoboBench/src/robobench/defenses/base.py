"""The defense interface shared by every defense in :mod:`robobench.defenses`."""

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Outcome:
    """What a defended system answered.

    ``response`` is the raw model text when a model was called. ``accepted`` is
    the verdict when the defense decides directly: True admits the instruction,
    False blocks it, None means no verdict was reached.
    """

    response: str | None = None
    accepted: bool | None = None


class Defense(Protocol):
    name: str

    def respond(self, *, text: str, images: list[bytes] | None = None,
                **extra) -> Outcome: ...
