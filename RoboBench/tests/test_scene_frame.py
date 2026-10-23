import io
import sys
from pathlib import Path

import pytest
from datasets import Dataset, Features, Image, List, Value
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import main_attack_defense as runner  # noqa: E402
from robobench.scenes import Scenes  # noqa: E402


def png_bytes():
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def make_scenes(name, categories):
    rows = {
        "id": [f"{name}-{i}" for i in range(len(categories))],
        "category": categories,
        "images": [[{"bytes": png_bytes(), "path": f"frame_{i}.png"}] for i in range(len(categories))],
        "benign": ["pick up the cup"] * len(categories),
        "malicious": ["throw the cup"] * len(categories),
    }
    features = Features({
        "id": Value("string"), "category": Value("string"),
        "images": List(Image()), "benign": Value("string"), "malicious": Value("string"),
    })
    return Scenes(name, Dataset.from_dict(rows, features=features))


@pytest.mark.parametrize("name, categories, expected", [
    ("DROID", [None, None], ["DROID", "DROID"]),
    ("RJB-Instructions", ["Illegal_Activity", "Illegal_Activity"], ["Illegal_Activity"] * 2),
])
def test_source_is_category_or_base(monkeypatch, name, categories, expected):
    monkeypatch.setattr(runner, "load_scenes", lambda artifact, repo: make_scenes(name, categories))
    frame, _ = runner.scene_frame(name, "benign", "unused")
    assert frame["source"].tolist() == expected
    assert frame["basename"].tolist() == ["frame_0.png", "frame_1.png"]
    assert frame["text"].tolist() == ["pick up the cup"] * 2
