"""Defenses: a defended system that answers an instruction about a scene.

A defense is any object with a ``name`` and
``respond(*, text, images=None, **extra) -> Outcome``. Defenses that call a
target model take it at construction (see :mod:`robobench.models`); defenses
that decide on their own, such as RoboGuard, take their own inputs.

    from robobench.defenses import DEFENSES
    from robobench.models import make_model
    defense = DEFENSES["google_prompt"](make_model("gpt-5.4-mini-2026-03-17"))
    defense.respond(text="pick up the knife", images=[frame_bytes]).response

``DEFENSES`` maps the defense names used in the experiments to their classes.
"""

from .base import Defense, Outcome
from .prompt import GooglePrompt, NoDefense
from .roboguard import RoboGuard

DEFENSES: dict[str, type] = {
    NoDefense.name: NoDefense,
    GooglePrompt.name: GooglePrompt,
    RoboGuard.name: RoboGuard,
}

__all__ = ["DEFENSES", "Defense", "GooglePrompt", "NoDefense", "Outcome", "RoboGuard"]
