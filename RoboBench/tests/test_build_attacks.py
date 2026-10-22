import io
import json
import sys
from pathlib import Path

from datasets import Dataset, Features, Image, List, Value
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_attacks import build_tree  # noqa: E402
from robobench.attacks import AttackResult  # noqa: E402
from robobench.index import Index  # noqa: E402
from robobench.scenes import Scenes  # noqa: E402


def png_bytes(color):
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2), color).save(buf, format="PNG")
    return buf.getvalue()


def make_base(name, sources):
    """A vanilla artifact plus its index; ``sources`` is the per-frame source folder."""
    n = len(sources)
    rows = {
        "id": [f"{name}-{i}" for i in range(n)],
        "category": [None if source == name else source for source in sources],
        "images": [[{"bytes": png_bytes(i), "path": f"frame_{i}.png"}] for i in range(n)],
        "benign": [f"benign {i}" for i in range(n)],
        "malicious": [f"malicious {i}" for i in range(n)],
    }
    features = Features({
        "id": Value("string"), "category": Value("string"),
        "images": List(Image()), "benign": Value("string"), "malicious": Value("string"),
    })
    scenes = Scenes(name, Dataset.from_dict(rows, features=features))
    index = Index([
        {"id": rows["id"][i], "dataset": name, "base": name, "attack": None,
         "target": None, "key": f"{sources[i]}/frame_{i}.png"}
        for i in range(n)
    ])
    return scenes, index


class FakeAttack:
    name = "badrobot_cj"
    target = None

    def __init__(self, texts=None, images=None):
        self.texts = texts or {}
        self.images = images or {}
        self.calls = 0

    def attack(self, *, text=None, images=None, **extra):
        self.calls += 1
        return AttackResult(
            malicious=self.texts.get(text, f"attacked {text}"),
            images=self.images.get(text),
        )


def test_single_source_tree(tmp_path):
    scenes, index = make_base("DROID", ["DROID", "DROID"])
    attack = FakeAttack(texts={"malicious 1": None}, images={"malicious 0": [b"new-frame"]})
    tree = build_tree(attack, "DROID", scenes, index, tmp_path)

    assert tree == tmp_path / "badrobot_cj_DROID"
    instr = json.loads((tree / "DROID_instr.json").read_text())
    frame0 = tree / "dataset" / "frame_0.png"
    frame1 = tree / "dataset" / "frame_1.png"
    assert instr == {
        frame0.as_posix(): ["attacked malicious 0", "benign 0"],
        frame1.as_posix(): ["", "benign 1"],           # no attack found -> empty text
    }
    assert frame0.read_bytes() == b"new-frame"        # the attack replaced the frame
    assert frame1.read_bytes() == png_bytes(1)        # untouched frame copied from the artifact


def test_categorized_tree(tmp_path):
    scenes, index = make_base("RJB-Instructions", ["Illegal_Activity", "Physical_Safety_Offenses"])
    tree = build_tree(FakeAttack(), "RJB-Instructions", scenes, index, tmp_path)

    for i, source in enumerate(["Illegal_Activity", "Physical_Safety_Offenses"]):
        instr = json.loads((tree / source / f"{source}_instr.json").read_text())
        frame = tree / source / "dataset" / f"frame_{i}.png"
        assert instr == {frame.as_posix(): [f"attacked malicious {i}", f"benign {i}"]}
        assert frame.exists()
    assert not (tree / "RJB-Instructions_instr.json").exists()


def test_rerun_skips_done_frames_and_limit(tmp_path):
    scenes, index = make_base("DROID", ["DROID"] * 3)
    first = FakeAttack()
    build_tree(first, "DROID", scenes, index, tmp_path, limit=2)
    assert first.calls == 2

    second = FakeAttack()
    tree = build_tree(second, "DROID", scenes, index, tmp_path)
    assert second.calls == 1                           # only the frame the limit left out
    assert len(json.loads((tree / "DROID_instr.json").read_text())) == 3
