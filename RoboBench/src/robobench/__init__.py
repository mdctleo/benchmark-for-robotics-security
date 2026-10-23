"""RoboBench downstream evaluation API.

    scenes = robobench.load_scenes("robopair_DROID", "mdctleo/RoboJailBench")
    labeled = robobench.label(responses)
    report = robobench.score(labeled)
"""

__version__ = "0.1.0"

from .labeling import label  # noqa: E402
from .scenes import Scenes, load_scenes  # noqa: E402
from .scoring import score  # noqa: E402

__all__ = ["Scenes", "label", "load_scenes", "score", "__version__"]
