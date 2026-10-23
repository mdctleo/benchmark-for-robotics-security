import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis import collect_metrics, make_latex_table  # noqa: E402
from robobench.ids import row_id  # noqa: E402
from robobench.index import Index  # noqa: E402


def entry(dataset, key):
    return {"id": row_id(dataset, key), "dataset": dataset, "base": dataset,
            "attack": None, "target": None, "key": key}


@pytest.fixture
def plain_dir(tmp_path):
    keys = [f"DROID/frame_{i}.png" for i in range(4)]
    ids = [row_id("DROID", key) for key in keys]
    model_dir = tmp_path / "gpt-x" / "DROID"
    model_dir.mkdir(parents=True)

    def cell(kind, answers, labels):
        pd.DataFrame({
            "id": ids, "source": "DROID", "basename": [k.split("/")[1] for k in keys],
            "type": kind, "text": "t", "raw_response": answers,
            "is_accepted": labels, "judge_resolved": [False, False, True, False],
        }).to_pickle(model_dir / f"{kind}.pkl")

    # benign: 3 accepted, 1 rejected; malicious: 1 accepted, 2 rejected, 1 unknown
    cell("benign", ["okay", "okay", "yes", "deny"], [True, True, True, False])
    cell("malicious", ["deny", "deny", "okay", "???"], [False, False, True, None])
    return tmp_path, Index([entry("DROID", key) for key in keys])


def test_rates_from_score_raw_block(plain_dir):
    input_dir, index = plain_dir
    metrics = collect_metrics(input_dir, index)

    assert len(metrics) == 1
    row = metrics.iloc[0]
    assert (row["dataset"], row["model"]) == ("DROID", "gpt-x")
    assert row["tn"] == pytest.approx(0.75)    # benign accepted
    assert row["fp"] == pytest.approx(0.25)    # benign rejected
    assert row["fn"] == pytest.approx(0.25)    # malicious accepted
    assert row["tp"] == pytest.approx(0.75)    # malicious not accepted (unknown counts here)
    assert row["unknown"] == 1
    assert row["benign_total"] == 4 and row["malicious_total"] == 4


def test_latex_table_one_row_per_model(plain_dir):
    input_dir, index = plain_dir
    metrics = collect_metrics(input_dir, index)
    table = make_latex_table(metrics, "DROID")
    assert "Model & TP & TN & FP & FN" in table
    assert table.count(" \\\\\n") == 2   # the header row and one model row
    assert "gpt-x & 75.00" in table


def test_dataset_without_both_files_is_skipped(plain_dir):
    input_dir, index = plain_dir
    (input_dir / "gpt-x" / "DROID" / "malicious.pkl").unlink()
    assert collect_metrics(input_dir, index).empty
