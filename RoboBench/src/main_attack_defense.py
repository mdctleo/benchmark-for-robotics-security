"""Attack-defense evaluation runner.

Runs every defense over the pre-generated adversarial prompts (ASR) and the
clean benign goals (UR) and stores what the defended system answered plus a
parsed acceptance label. Defenses come from ``robobench.defenses``; the target
model from ``robobench.models``.

Inputs are the RoboBench artifacts (``robobench.load_scenes``): benign rows come
from the vanilla artifact ``<dataset>``, malicious rows from ``<attack>_<dataset>``.
They are read from the Hugging Face dataset ``ARTIFACTS``.

Output layout for defenses that query the target model:
    workdir/attack_defense/<model>/malicious/<dataset>/<attack>/<defense>.pkl
    workdir/attack_defense/<model>/benign/<dataset>/clean/<defense>.pkl
Each file carries ``id``, ``source``, ``basename``, ``type``, ``text``,
``raw_response``, ``is_accepted`` and ``judge_resolved``.

Defenses that decide without a target model (RoboGuard) run once, independent
of ``--model``:
    workdir/attack_defense/<defense>/malicious/<dataset>/<attack>.pkl
    workdir/attack_defense/<defense>/benign/<dataset>/clean.pkl
Each file carries ``id``, ``source``, ``basename``, ``type``, ``text``,
``is_accepted`` and ``decided``. RoboGuard reads scene graphs from
``--scene-graphs`` (``scene_graphs/<source>/<frame stem>.json``).

Re-running resumes by ``id``: rows that already have an answer keep it and
their labels; a cell with nothing left to run is not rewritten.

Usage:
    uv run python src/main_attack_defense.py --model gpt-5.4-mini-2026-03-17
    uv run python src/main_attack_defense.py --defense roboguard --workers 8
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import datasets
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from robobench.defenses import DEFENSES
from robobench.labeling import parse_label
from robobench.models import make_model
from robobench.scenes import Scenes, load_scenes

datasets.disable_progress_bars()

ATTACKS = ["badrobot_cd", "badrobot_cj", "badrobot_sm", "robopair", "shawshank"]

DEFAULT_DEFENSES = ["no_defense", "google_prompt"]

# Defenses that decide without querying the target model.
MODEL_FREE_DEFENSES = {"roboguard"}

# Base datasets = vanilla artifact names. Attack artifacts are "<attack>_<base>".
BASES = ["DROID", "ROBOVQA", "RH20T", "robo2vlm", "car", "RJB-Instructions"]

# The Hugging Face dataset the artifacts are published in.
ARTIFACTS = "mdctleo/RoboJailBench"


def make_defense(name: str, model: str, retry_count: int, scene_graphs: str):
    if name in MODEL_FREE_DEFENSES:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set")
        from robobench.defenses.roboguard import RoboGuard
        return RoboGuard(graph_dir=scene_graphs)
    return DEFENSES[name](make_model(model, retry_count=retry_count))


def cell_path(output_dir: Path, model: str, defense: str, kind: str,
              dataset: str, attack: str) -> Path:
    """Pickle for one (defense, dataset, attack) cell; ``attack`` is ``clean`` for benign."""
    if defense in MODEL_FREE_DEFENSES:
        return output_dir / defense / kind / dataset / f"{attack}.pkl"
    return output_dir / model.replace("/", "_") / kind / dataset / attack / f"{defense}.pkl"


def split_artifact(name: str) -> tuple[str | None, str]:
    """``"robopair_DROID" -> ("robopair", "DROID")``; ``"DROID" -> (None, "DROID")``."""
    for attack in ATTACKS:
        if name.startswith(attack + "_"):
            return attack, name[len(attack) + 1:]
    return None, name


def available_artifacts(artifacts: str) -> set[str]:
    return set(datasets.get_dataset_config_names(artifacts))


def scene_frame(artifact: str, kind: str, artifacts: str) -> tuple[pd.DataFrame, Scenes]:
    """One row per scene of ``artifact`` with its ``kind`` (benign/malicious) text.

    ``source`` is the RJB category or the base name; ``basename`` is the frame
    file name. Together they name the frame on disk for RoboGuard's scene graphs.
    """
    scenes = load_scenes(artifact, artifacts)
    _, base = split_artifact(artifact)
    table = scenes.dataset.remove_columns("images").to_pandas()
    rows = []
    for row in table.itertuples(index=False):
        rows.append({
            "id": row.id,
            "source": row.category if isinstance(row.category, str) else base,
            "basename": scenes.image_bytes(row.id)[0]["path"],
            "type": kind,
            "text": getattr(row, kind),
        })
    return pd.DataFrame(rows), scenes


def load_benign_frame(dataset: str, artifacts: str) -> tuple[pd.DataFrame, Scenes]:
    return scene_frame(dataset, "benign", artifacts)


def load_malicious_frame(dataset: str, attack: str, artifacts: str) -> tuple[pd.DataFrame, Scenes]:
    return scene_frame(f"{attack}_{dataset}", "malicious", artifacts)


def resume_cell(frame: pd.DataFrame, output_path: Path,
                model_free: bool = False) -> pd.DataFrame:
    """Carry answers and labels over from an existing cell file, matched by id.

    Model-free cells record ``decided`` rather than a response: a verdict may
    legitimately be None (no verdict), so ``is_accepted`` alone cannot mark a
    row as done.
    """
    frame = frame.reset_index(drop=True).copy()
    if model_free:
        frame["is_accepted"] = None
        frame["decided"] = False
        if output_path.exists():
            existing = pd.read_pickle(output_path)
            if {"id", "decided"} <= set(existing.columns):
                existing = existing.drop_duplicates("id").set_index("id")
                hit = frame["id"].isin(existing.index)
                matched = existing.loc[frame.loc[hit, "id"]]
                frame.loc[hit, "is_accepted"] = matched["is_accepted"].values
                frame.loc[hit, "decided"] = matched["decided"].values
        return frame

    frame["raw_response"] = pd.NA
    frame["is_accepted"] = None
    frame["judge_resolved"] = False

    if output_path.exists():
        existing = pd.read_pickle(output_path)
        if {"id", "raw_response"} <= set(existing.columns):
            existing = existing.drop_duplicates("id").set_index("id")
            hit = frame["id"].isin(existing.index)
            matched = existing.loc[frame.loc[hit, "id"]]
            frame.loc[hit, "raw_response"] = matched["raw_response"].values
            if "is_accepted" in matched.columns:
                frame.loc[hit, "is_accepted"] = matched["is_accepted"].values
            if "judge_resolved" in matched.columns:
                frame.loc[hit, "judge_resolved"] = matched["judge_resolved"].values
    return frame


def run_cell(defense, frame: pd.DataFrame, scenes: Scenes, output_path: Path,
             workers: int, label: str, model_free: bool = False) -> pd.DataFrame:
    frame = resume_cell(frame, output_path, model_free)

    if model_free:
        todo = [index for index in frame.index if not frame.at[index, "decided"]]
    else:
        todo = [
            index
            for index in frame.index
            if pd.isna(frame.at[index, "raw_response"])
        ]
    if not todo:
        return frame  # nothing to run; leave the file as it is

    progress = tqdm(total=len(todo), desc=label, leave=False)
    lock = Lock()

    def work(index: int):
        row_id = frame.at[index, "id"]
        outcome = defense.respond(
            text=frame.at[index, "text"],
            images=[image["bytes"] for image in scenes.image_bytes(row_id)],
            key=f"{frame.at[index, 'source']}/{frame.at[index, 'basename']}",
        )
        with lock:
            if model_free:
                frame.at[index, "is_accepted"] = outcome.accepted
                frame.at[index, "decided"] = True
            else:
                frame.at[index, "raw_response"] = outcome.response
                frame.at[index, "is_accepted"] = parse_label(outcome.response)
                frame.at[index, "judge_resolved"] = False
            progress.update(1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    progress.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(output_path)
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--output-dir", default="workdir/attack_defense")
    parser.add_argument("--scene-graphs", default="scene_graphs",
                        help="scene-graph directory for RoboGuard")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--defense", action="append", dest="defenses",
                        choices=sorted(DEFENSES))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retry-count", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    bases = args.datasets or BASES
    defenses = {
        name: make_defense(name, args.model, args.retry_count, args.scene_graphs)
        for name in (args.defenses or DEFAULT_DEFENSES)
    }
    output_dir = Path(args.output_dir)
    available = available_artifacts(ARTIFACTS)

    for dataset in bases:
        benign_frame, benign_scenes = load_benign_frame(dataset, ARTIFACTS)
        if args.limit is not None:
            benign_frame = benign_frame.head(args.limit)

        for name, defense in defenses.items():
            run_cell(
                defense=defense,
                frame=benign_frame,
                scenes=benign_scenes,
                output_path=cell_path(output_dir, args.model, name, "benign", dataset, "clean"),
                workers=args.workers,
                label=f"{dataset}/benign/{name}",
                model_free=name in MODEL_FREE_DEFENSES,
            )

        for attack in ATTACKS:
            if f"{attack}_{dataset}" not in available:
                print(f"no artifact {attack}_{dataset}; skipped")
                continue
            malicious_frame, malicious_scenes = load_malicious_frame(
                dataset, attack, ARTIFACTS
            )
            if args.limit is not None:
                malicious_frame = malicious_frame.head(args.limit)

            for name, defense in defenses.items():
                run_cell(
                    defense=defense,
                    frame=malicious_frame,
                    scenes=malicious_scenes,
                    output_path=cell_path(output_dir, args.model, name, "malicious", dataset, attack),
                    workers=args.workers,
                    label=f"{dataset}/{attack}/{name}",
                    model_free=name in MODEL_FREE_DEFENSES,
                )

        print(f"done: {dataset}")

    print(f"saved under {output_dir}")


if __name__ == "__main__":
    main()
