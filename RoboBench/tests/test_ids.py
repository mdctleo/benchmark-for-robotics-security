import uuid

from robobench.ids import frame_key, row_id
from robobench.index import SHIPPED_INDEX, load_index


def test_frame_key_uses_source_and_basename_only():
    assert frame_key("DROID", "./data/external_datasets/DROID/dataset/frame_0.png") == "DROID/frame_0.png"
    assert frame_key("DROID", "/elsewhere/frame_0.png") == "DROID/frame_0.png"
    assert (
        frame_key("Illegal_Activity", "./data/main_dataset_50/Illegal_Activity/dataset/$TerrorismExample 1.png")
        == "Illegal_Activity/$TerrorismExample 1.png"
    )


def test_row_id_is_deterministic_and_distinct_per_artifact():
    a = row_id("DROID", "DROID/frame_0.png")
    assert a == row_id("DROID", "DROID/frame_0.png")
    assert uuid.UUID(a).version == 5
    assert a != row_id("robopair_DROID", "DROID/frame_0.png")
    assert a != row_id("DROID", "DROID/frame_1.png")


def test_shipped_index_is_consistent():
    if not SHIPPED_INDEX.exists():
        import pytest
        pytest.skip("index not built yet (run src/build_artifacts.py)")
    index = load_index()
    for entry in index.entries:
        assert entry["id"] == row_id(entry["dataset"], entry["key"])
        assert entry["dataset"] == (
            entry["base"] if entry["attack"] is None else f"{entry['attack']}_{entry['base']}"
        )
