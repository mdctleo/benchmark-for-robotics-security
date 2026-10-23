import json
import sys
from types import SimpleNamespace

import pytest

import robobench.defenses.roboguard as roboguard_module
from robobench.defenses import DEFENSES, GooglePrompt, NoDefense, Outcome, RoboGuard
from robobench.defenses.prompt import GOOGLE_SYSTEM_PROMPT, INSTR


class FakeModel:
    def __init__(self, reply="okay"):
        self.calls = []
        self.reply = reply

    def __call__(self, images, prompt):
        self.calls.append((images, prompt))
        return self.reply


def test_registry_names_match_classes():
    assert set(DEFENSES) == {"no_defense", "google_prompt", "roboguard"}
    for name, cls in DEFENSES.items():
        assert cls.name == name


def test_no_defense_prompt_and_images():
    model = FakeModel("deny")
    outcome = NoDefense(model).respond(text="throw the bowl", images=[b"\x89PNG"])

    assert outcome == Outcome(response="deny", accepted=None)
    assert model.calls == [([b"\x89PNG"], INSTR + " user input:throw the bowl")]


def test_google_prompt_prepends_reminder():
    model = FakeModel()
    GooglePrompt(model).respond(text="throw the bowl")

    images, prompt = model.calls[0]
    assert images == []
    assert prompt == GOOGLE_SYSTEM_PROMPT + INSTR + " user input:throw the bowl"


def test_extra_keywords_are_ignored():
    model = FakeModel()
    NoDefense(model).respond(text="x", key="DROID/frame_0.png", scene_graph=None)
    assert len(model.calls) == 1


# -- RoboGuard ---------------------------------------------------------------

def test_roboguard_without_spot_names_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "spot", None)
    with pytest.raises(ImportError, match="spot"):
        RoboGuard()


@pytest.fixture
def guard(monkeypatch, tmp_path):
    """A RoboGuard whose backends are stubbed out, so graph lookup can be tested without spot."""
    monkeypatch.setattr(roboguard_module, "_import_backends", lambda: SimpleNamespace())
    graph = {"current_location": "ground_1", "objects": [], "regions": []}
    (tmp_path / "DROID").mkdir()
    (tmp_path / "DROID" / "frame_0.json").write_text(json.dumps(graph))
    return RoboGuard(graph_dir=tmp_path), graph


def test_roboguard_graph_lookup_by_key_uses_stem(guard):
    rg, graph = guard
    assert rg.resolve_graph(key="DROID/frame_0.jpg") == graph
    assert rg.resolve_graph(source="DROID", basename="frame_0.png") == graph


def test_roboguard_graph_given_directly(guard, tmp_path):
    rg, graph = guard
    assert rg.resolve_graph(scene_graph=graph) is graph
    assert rg.resolve_graph(scene_graph=tmp_path / "DROID" / "frame_0.json") == graph


def test_roboguard_missing_graph_and_no_image(guard):
    rg, _ = guard
    with pytest.raises(FileNotFoundError):
        rg.resolve_graph(key="DROID/frame_1.png")
    with pytest.raises(ValueError):
        rg.resolve_graph()


def test_roboguard_generates_and_saves_missing_graph(guard, monkeypatch, tmp_path):
    rg, _ = guard
    generated = {"current_location": "ground_1", "objects": [{"name": "bowl_1"}], "regions": []}
    calls = []
    monkeypatch.setattr(roboguard_module, "generate_scene_graph",
                        lambda images: calls.append(images) or generated)

    assert rg.resolve_graph(key="DROID/frame_1.png", images=[b"\x89PNG"]) == generated
    assert json.loads((tmp_path / "DROID" / "frame_1.json").read_text()) == generated
    assert rg.resolve_graph(images=[b"\x89PNG"]) == generated   # cached by image hash
    assert calls == [[b"\x89PNG"]]


def test_roboguard_generates_from_every_frame(guard, monkeypatch):
    rg, _ = guard
    calls = []
    monkeypatch.setattr(roboguard_module, "generate_scene_graph",
                        lambda images: calls.append(images) or {"regions": []})

    frames = [b"\x89PNG1", b"\x89PNG2"]
    rg.resolve_graph(images=frames)
    rg.resolve_graph(images=frames)                 # same sequence: cached
    rg.resolve_graph(images=[b"\x89PNG1"])         # one frame of it: a different scene
    assert calls == [frames, [b"\x89PNG1"]]


def test_roboguard_respond_returns_verdict(guard, monkeypatch):
    rg, _ = guard
    monkeypatch.setattr(rg, "decide", lambda text, graph: False)
    assert rg.respond(text="throw the bowl", key="DROID/frame_0.png") == Outcome(accepted=False)


def test_roboguard_default_model_is_the_experiment_model(guard):
    rg, _ = guard
    assert rg.model == roboguard_module.DEFAULT_MODEL == "gpt-5.4-nano"


def test_roboguard_model_reaches_grounding_and_planner(monkeypatch, tmp_path):
    """The chosen model is used by both the specification generator and the planner."""
    seen = {}

    class FakeGrounding:
        def __init__(self, rules, model):
            seen["grounding"] = model

        def get_specifications(self, text):
            return {}

        def gather_specification_propositions(self, specs):
            return "specs"

    class FakeHandler:
        def __init__(self, _):
            pass

        def reset(self, graph, start):
            pass

        def to_json_str(self):
            return "{}"

    class FakePlanner:
        def __init__(self, graph):
            self.model = "gpt-5.4-nano"

        def request(self, prompt):
            seen["planner"] = self.model
            return {"plan": [("goto", "ground_1")]}, True, None

    class FakeSynthesis:
        def __init__(self, specs):
            pass

        def validate_action_sequence(self, plan):
            return True, None

    backends = SimpleNamespace(ContextualGrounding=FakeGrounding, BASE_RULES="rules",
                               ControlSynthesis=FakeSynthesis, SPINE=FakePlanner,
                               GraphHandler=FakeHandler)
    monkeypatch.setattr(roboguard_module, "_import_backends", lambda: backends)

    rg = RoboGuard(model="gpt-5.4-mini")
    outcome = rg.respond(text="go to the ground", scene_graph={"current_location": "ground_1"})

    assert outcome == Outcome(accepted=True)
    assert seen == {"grounding": "gpt-5.4-mini", "planner": "gpt-5.4-mini"}


# -- scene-graph generator --------------------------------------------------

class _GraphClient:
    """Records the request and answers with a fixed graph."""

    def __init__(self):
        self.requests = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text='{"regions": [], "objects": []}')


@pytest.mark.parametrize("count", [1, 3])
def test_generate_scene_graph_sends_one_part_per_frame(count):
    from robobench.defenses.roboguard import scene_graph

    pytest.importorskip("google.genai")
    client = _GraphClient()
    frames = [b"\x89PNG\r\n\x1a\n" + bytes([i]) for i in range(count)]

    assert scene_graph.generate_scene_graph(frames, client=client) == {"regions": [], "objects": []}
    contents = client.requests[0]["contents"]
    assert len(contents) == count + 1
    assert all(part.inline_data.data == raw for part, raw in zip(contents, frames))
    prompt = contents[-1]
    assert prompt.startswith(scene_graph.generate_semantic_prompt)
    assert (scene_graph.sequence_note in prompt) == (count > 1)


def test_generate_scene_graph_rejects_no_frames():
    from robobench.defenses.roboguard import scene_graph

    with pytest.raises(ValueError):
        scene_graph.generate_scene_graph([], client=_GraphClient())
