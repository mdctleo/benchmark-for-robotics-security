"""Plain baseline scorer.

Reads the response files ``main.py`` wrote and scores them with
``robobench.score``. For every model under ``--input-dir`` and every dataset
with both its files, the row is

    TP  malicious rejected    TN  benign accepted
    FN  malicious accepted    FP  benign rejected

as rates over all rows of that type, with binomial standard errors, plus the
number of responses that carry no label (``robobench.scoring`` counts those
as not accepted). Writes one LaTeX table per dataset, a row per model, to
``--output-dir`` and prints the same rows.

Usage:
    uv run python src/analysis.py
    uv run python src/analysis.py --input-dir workdir/plain --output-dir workdir/plain/analysis
"""

import argparse
from pathlib import Path

import pandas as pd

from analysis_attack_defense import DATASET_LABELS, DATASET_ORDER, format_cell
from robobench import score
from robobench.index import load_index

RATES = ["tp", "tn", "fp", "fn"]


def responses_from_model_dir(model_dir: Path) -> list[dict]:
    """Rows for ``robobench.score`` from one model's baseline files."""
    rows = []
    for dataset in DATASET_ORDER:
        for kind in ("benign", "malicious"):
            path = model_dir / dataset / f"{kind}.pkl"
            if not path.exists():
                continue
            for record in pd.read_pickle(path).itertuples(index=False):
                label = None if pd.isna(record.is_accepted) else bool(record.is_accepted)
                source = None
                if label is not None:
                    source = "judge" if bool(record.judge_resolved) else "keyword"
                rows.append({
                    "id": record.id,
                    "type": kind,
                    "response": record.raw_response,
                    "label": label,
                    "label_source": source,
                })
    return rows


def collect_metrics(input_dir: Path, index) -> pd.DataFrame:
    """One row per (dataset, model) with both files present."""
    rows = []
    for model_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        responses = responses_from_model_dir(model_dir)
        if not responses:
            continue
        report = score(responses, index)
        for dataset, entry in report["datasets"].items():
            if entry["coverage"]["benign"] == 0 or entry["coverage"]["malicious"] == 0:
                continue
            record = {"dataset": dataset, "model": model_dir.name}
            for rate in RATES:
                record[rate] = entry["raw"][rate]["value"]
                record[f"{rate}_se"] = entry["raw"][rate]["se"]
            record["malicious_total"] = entry["coverage"]["malicious"]
            record["benign_total"] = entry["coverage"]["benign"]
            record["unknown"] = sum(entry["counts"][kind]["unknown"] for kind in entry["counts"])
            rows.append(record)
    return pd.DataFrame(rows)


def make_latex_table(block: pd.DataFrame, dataset: str) -> str:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{TP/TN/FP/FN rates for {DATASET_LABELS.get(dataset, dataset)} "
        f"with the raw instructions and no defense (\\%).}}",
        f"\\label{{tab:plain_{dataset.lower().replace('-', '_')}}}",
        "\\small",
        "\\begin{tabular}{@{}lcccc@{}}",
        "\\toprule",
        "Model & TP & TN & FP & FN \\\\",
        "\\midrule",
    ]
    for _, row in block.iterrows():
        cells = [row["model"]] + [
            format_cell(row[rate], row[f"{rate}_se"], False) for rate in RATES
        ]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="workdir/plain")
    parser.add_argument("--output-dir", default="workdir/plain/analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = collect_metrics(Path(args.input_dir), load_index())
    if metrics.empty:
        raise SystemExit(f"no baseline results under {args.input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display = metrics.copy()
    for rate in RATES:
        display[rate] = (100 * display[rate]).round(2)
    print(display[["dataset", "model", *RATES, "unknown"]].to_string(index=False))

    for dataset in DATASET_ORDER:
        block = metrics.loc[metrics["dataset"].eq(dataset)]
        if block.empty:
            continue
        (output_dir / f"{dataset}.tex").write_text(
            make_latex_table(block, dataset), encoding="utf-8"
        )
    print(f"\nwrote tables to {output_dir}")


if __name__ == "__main__":
    main()
