import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_artifacts import build  # noqa: E402
from robobench.attacks import ATTACKS  # noqa: E402
from robobench.ids import row_id  # noqa: E402


def png_bytes():
    buf = io.BytesIO()
    PILImage.new("RGB", (2, 2)).save(buf, format="PNG")
    return buf.getvalue()


def write_source(directory: Path, source: str, texts: list):
    """``<directory>/<source>_instr.json`` and ``<directory>/dataset/`` with one frame per text."""
    instr = {}
    for i, malicious in enumerate(texts):
        frame = directory / "dataset" / f"frame_{i}.png"
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(png_bytes())
        instr[frame.as_posix()] = [malicious, f"benign {i}"]
    (directory / f"{source}_instr.json").write_text(json.dumps(instr))


@pytest.fixture
def out(tmp_path):
    return tmp_path / "artifacts", tmp_path / "index.json"


def read_parquet(out_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(out_dir / "data" / name / "train.parquet")


def test_single_source_artifact(tmp_path, out):
    out_dir, index_path = out
    write_source(tmp_path / "DROID", "DROID", ["malicious 0", "malicious 1"])
    build(tmp_path / "DROID", None, out_dir, index_path)

    table = read_parquet(out_dir, "DROID")
    assert list(table.columns) == ["id", "dataset", "category", "images", "benign", "malicious", "target"]
    assert table["id"].tolist() == [row_id("DROID", f"DROID/frame_{i}.png") for i in range(2)]
    assert table["category"].isna().all()
    assert table["target"].isna().all()
    assert table["malicious"].tolist() == ["malicious 0", "malicious 1"]
    assert table["images"][0][0]["path"] == "frame_0.png"


def test_attack_artifact_from_tree(tmp_path, out):
    out_dir, index_path = out
    write_source(tmp_path / "badrobot_cj_DROID", "DROID", ["attacked", ""])
    build(tmp_path / "badrobot_cj_DROID", None, out_dir, index_path)

    table = read_parquet(out_dir, "badrobot_cj_DROID")
    assert table["id"].tolist() == [row_id("badrobot_cj_DROID", f"DROID/frame_{i}.png") for i in range(2)]
    assert table["malicious"].tolist()[0] == "attacked"
    assert pd.isna(table["malicious"][1])              # empty text kept as a null row
    entry = json.loads(index_path.read_text())[0]
    assert (entry["base"], entry["attack"], entry["key"]) == ("DROID", "badrobot_cj", "DROID/frame_0.png")
    assert entry["target"] == ATTACKS["badrobot_cj"].target


def test_categorized_artifact(tmp_path, out):
    out_dir, index_path = out
    root = tmp_path / "RJB-X"
    write_source(root / "Illegal_Activity", "Illegal_Activity", ["a"])
    write_source(root / "Physical_Safety_Offenses", "Physical_Safety_Offenses", ["b"])
    build(root, None, out_dir, index_path)

    table = read_parquet(out_dir, "RJB-X")
    assert table["category"].tolist() == ["Illegal_Activity", "Physical_Safety_Offenses"]
    assert table["id"].tolist() == [
        row_id("RJB-X", "Illegal_Activity/frame_0.png"),
        row_id("RJB-X", "Physical_Safety_Offenses/frame_0.png"),
    ]


def test_index_and_card_merge_per_artifact(tmp_path, out):
    out_dir, index_path = out
    write_source(tmp_path / "DROID", "DROID", ["m0", "m1"])
    write_source(tmp_path / "badrobot_cj_DROID", "DROID", ["a0", "a1"])
    build(tmp_path / "DROID", None, out_dir, index_path)
    build(tmp_path / "badrobot_cj_DROID", None, out_dir, index_path)
    write_source(tmp_path / "DROID", "DROID", ["m0"])      # rebuild DROID smaller
    index = build(tmp_path / "DROID", None, out_dir, index_path)

    by_dataset = {}
    for entry in index:
        by_dataset.setdefault(entry["dataset"], []).append(entry)
    assert len(by_dataset["DROID"]) == 1                  # replaced, not appended
    assert len(by_dataset["badrobot_cj_DROID"]) == 2      # untouched
    assert json.loads(index_path.read_text()) == index
    assert (out_dir / "data" / "badrobot_cj_DROID" / "train.parquet").exists()
    card = (out_dir / "README.md").read_text()
    assert "- config_name: DROID\n" in card and "- config_name: badrobot_cj_DROID\n" in card


def test_limit_per_source(tmp_path, out):
    out_dir, index_path = out
    write_source(tmp_path / "DROID", "DROID", ["m0", "m1", "m2"])
    build(tmp_path / "DROID", 2, out_dir, index_path)
    assert len(read_parquet(out_dir, "DROID")) == 2
