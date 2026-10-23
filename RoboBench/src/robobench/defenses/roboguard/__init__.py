"""RoboGuard (Ravichandran et al., 2025): a guardrail that checks a planned
action sequence against safety specifications before the robot acts.

For one instruction RoboGuard (1) plans it over the scene graph with SPINE,
(2) derives safety specifications for that scene from a fixed rule set and
(3) model-checks the plan against them with spot. It never queries the target
model, so its verdict is the outcome: True admits the plan, False blocks it,
None means no usable verdict (no plan, or synthesis failed).

The scene graph can be passed directly, looked up under ``graph_dir`` by the
frame's key, or generated from the scene's images (see :mod:`.scene_graph`). ``model``
is the OpenAI model behind both the planner and the specification generator;
the default is the one used in the RoboBench experiments.

    from robobench.defenses.roboguard import RoboGuard
    guard = RoboGuard(graph_dir="scene_graphs")
    guard.respond(text="throw the bowl", key="DROID/frame_0.png").accepted

RoboGuard and SPINE are vendored under ``robobench/_vendor``; spot is a system
install (``_vendor/README.md``).
"""

import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path
from threading import Lock
from types import SimpleNamespace

from ..base import Outcome
from .scene_graph import generate_scene_graph

_VENDOR_DIR = Path(__file__).resolve().parents[2] / "_vendor"

SPOT_INSTALL_HINT = (
    "RoboGuard needs the spot LTL library with Python bindings: "
    "`conda install -c conda-forge spot`, `brew install spot` or "
    "`apt install python3-spot`; if spot is installed for another interpreter, "
    "set ROBOBENCH_SPOT_SITE_PACKAGES to that site-packages directory."
)

# spot's automata construction and model checking touch library-global state.
_spot_lock = Lock()

DEFAULT_MODEL = "gpt-5.4-nano"


def _import_backends():
    extra = os.environ.get("ROBOBENCH_SPOT_SITE_PACKAGES")
    if extra and extra not in sys.path:
        sys.path.append(extra)
    try:
        import spot  # noqa: F401
    except ImportError as error:
        raise ImportError(SPOT_INSTALL_HINT) from error

    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))
    from roboguard.generator import ContextualGrounding
    from roboguard.prompts.base import BASE_RULES
    from roboguard.synthesis import ControlSynthesis
    from spine.spine import SPINE, GraphHandler

    return SimpleNamespace(
        ContextualGrounding=ContextualGrounding,
        BASE_RULES=BASE_RULES,
        ControlSynthesis=ControlSynthesis,
        SPINE=SPINE,
        GraphHandler=GraphHandler,
    )


class RoboGuard:
    name = "roboguard"

    def __init__(self, graph_dir: str | os.PathLike | None = None,
                 model: str = DEFAULT_MODEL):
        self._backends = _import_backends()
        self.graph_dir = Path(graph_dir) if graph_dir is not None else None
        self.model = model
        self._grounding = None
        self._grounding_lock = Lock()
        self._spec_cache: dict[str, object] = {}
        self._spec_lock = Lock()
        self._graph_cache: dict[str, dict] = {}
        self._graph_lock = Lock()

    # -- inputs -------------------------------------------------------------

    def graph_path(self, source: str, basename: str) -> Path:
        """``graph_dir/<source>/<stem>.json``; keyed by stem since re-encoded
        artifacts carry a .jpg name for a .png frame."""
        if self.graph_dir is None:
            raise ValueError("RoboGuard(graph_dir=...) is needed to look up a graph by key")
        return self.graph_dir / source / f"{Path(basename).stem}.json"

    def _generate(self, images: list[bytes]) -> dict:
        digest = hashlib.sha256(b"".join(images)).hexdigest()
        with self._graph_lock:
            graph = self._graph_cache.get(digest)
        if graph is None:
            graph = generate_scene_graph(images)
            with self._graph_lock:
                self._graph_cache[digest] = graph
        return graph

    def resolve_graph(self, *, scene_graph=None, key=None, source=None,
                      basename=None, images=None) -> dict:
        if isinstance(scene_graph, dict):
            return scene_graph
        if scene_graph is not None:
            return json.loads(Path(scene_graph).read_text())

        if key is not None:
            source, basename = key.split("/", 1)
        if source is not None and basename is not None:
            path = self.graph_path(source, basename)
            if path.exists():
                return json.loads(path.read_text())
            if images:
                graph = self._generate(images)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(graph, indent=4))
                return graph
            raise FileNotFoundError(f"no scene graph at {path} and no images to generate one")

        if images:
            return self._generate(images)
        raise ValueError(
            "RoboGuard needs a scene graph: pass scene_graph=, key= (or source= and "
            "basename=) with graph_dir, or images="
        )

    # -- decision -----------------------------------------------------------

    def grounding(self):
        with self._grounding_lock:
            if self._grounding is None:
                self._grounding = self._backends.ContextualGrounding(
                    rules=self._backends.BASE_RULES, model=self.model)
            return self._grounding

    def aggregated_specs(self, graph_text: str):
        """Safety specifications for a scene graph (one LLM call, cached)."""
        with self._spec_lock:
            cached = self._spec_cache.get(graph_text)
        if cached is not None:
            return cached

        grounding = self.grounding()
        specs = grounding.get_specifications(f"Scene Graph: {graph_text}")
        aggregated = grounding.gather_specification_propositions(specs)

        with self._spec_lock:
            self._spec_cache[graph_text] = aggregated
        return aggregated

    def _decide(self, prompt: str, graph_json: dict):
        """True if RoboGuard admits the plan, False if it blocks it, None if no verdict."""
        b = self._backends
        start = graph_json.get("current_location", "ground_1")

        handler = b.GraphHandler("")
        handler.reset(str(graph_json).replace("'", '"'), start)

        planner = b.SPINE(graph=handler)
        planner.model = self.model
        try:
            response, success, _logs = planner.request(prompt)
        except Exception:  # noqa: BLE001 - planner failures vary
            return None
        if not success:
            return None

        plan = [list(action) for action in response.get("plan", [])]
        if not plan:
            return None

        try:
            aggregated = self.aggregated_specs(handler.to_json_str())
        except Exception:  # noqa: BLE001 - grounding failures vary
            return None

        try:
            with _spot_lock:
                synthesizer = b.ControlSynthesis(aggregated)
                is_safe, _results = synthesizer.validate_action_sequence(plan)
        except Exception:  # noqa: BLE001 - malformed LTL from the generator
            return None

        return bool(is_safe)

    def decide(self, prompt: str, graph_json: dict):
        """``_decide`` with the planner's tracing suppressed.

        SPINE and RoboGuard print their full planning and model-checking traces
        to stdout, which would run to gigabytes over a full evaluation.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            return self._decide(prompt, graph_json)

    def respond(self, *, text, images=None, scene_graph=None, key=None,
                source=None, basename=None, **extra) -> Outcome:
        graph = self.resolve_graph(scene_graph=scene_graph, key=key, source=source,
                                   basename=basename, images=images)
        return Outcome(accepted=self.decide(text, graph))


__all__ = ["DEFAULT_MODEL", "RoboGuard", "SPOT_INSTALL_HINT", "generate_scene_graph"]
