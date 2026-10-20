"""RoboGuard defense runner.

RoboGuard replaces the target model with the SPINE planner and validates the
resulting plan against LTL safety specifications, so its accept/reject decision
is a function of the instruction and the scene graph only -- it does not depend
on the target model. One run therefore serves every target model; the scorer
combines these decisions with each model's no-defense decisions.

Safety specifications are derived from the scene graph alone and are cached per
graph. ``ControlSynthesis`` carries mutable state across validations, so a fresh
synthesizer is built per sample from the cached specifications.

Output layout:
    workdir/attack_defense/roboguard/malicious/<dataset>/<attack>.pkl
    workdir/attack_defense/roboguard/benign/<dataset>/clean.pkl
"""

import argparse
import contextlib
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# spot's Python bindings come from the system install; RoboGuard and SPINE are
# checked out under src/defenses/roboguard/RoboGuard.
SPOT_SITE_PACKAGES = "/opt/homebrew/lib/python3.14/site-packages"
HERE = Path(__file__).resolve().parent
ROBOGUARD_DIR = HERE / "defenses/roboguard/RoboGuard"
for path in (
    SPOT_SITE_PACKAGES,
    str(ROBOGUARD_DIR / "src"),
    str(ROBOGUARD_DIR / "src/SPINE/src"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import spot  # noqa: E402

spot.setup()

from roboguard.generator import ContextualGrounding  # noqa: E402
from roboguard.prompts.base import BASE_RULES  # noqa: E402
from roboguard.synthesis import ControlSynthesis  # noqa: E402
from spine.spine import SPINE, GraphHandler  # noqa: E402

sys.path.insert(0, str(HERE))
from build_artifacts import DATASETS as REGISTRY, ROOT  # noqa: E402
from main_attack_defense import (  # noqa: E402
    ATTACKS,
    BASES,
    DEFAULT_ARTIFACTS,
    available_artifacts,
    load_benign_frame,
    load_malicious_frame,
    resume_cell,
)

# spot's automata construction and model checking touch library-global state.
_spot_lock = Lock()

_spec_cache: dict[str, object] = {}
_spec_lock = Lock()

_grounding = None
_grounding_lock = Lock()


def get_grounding() -> ContextualGrounding:
    global _grounding
    with _grounding_lock:
        if _grounding is None:
            _grounding = ContextualGrounding(rules=BASE_RULES)
        return _grounding


_frame_paths: dict[tuple[str, str], Path] | None = None


def graph_path_for(source: str, basename: str) -> str:
    """Scene-graph file next to the frame on disk.

    Artifact rows name a frame by ``source`` (RJB category or base) and file
    name; the on-disk path comes from the builder's registry, since the graphs
    live beside the original frames, not in the artifacts.
    """
    global _frame_paths
    if _frame_paths is None:
        _frame_paths = {}
        for sources in REGISTRY.values():
            for src, instr_path in sources:
                for filename in json.loads((ROOT / instr_path).read_text(encoding="utf-8")):
                    _frame_paths[(src, Path(filename).stem)] = ROOT / filename
    # Keyed by stem: re-encoded artifacts carry a .jpg name for a .png frame.
    frame = _frame_paths[(source, Path(basename).stem)]
    return str(frame.with_suffix(".json"))


def get_aggregated_specs(graph_path: str, graph_text: str):
    """Safety specifications for a scene graph (one LLM call, cached)."""
    with _spec_lock:
        cached = _spec_cache.get(graph_path)
    if cached is not None:
        return cached

    grounding = get_grounding()
    specs = grounding.get_specifications(f"Scene Graph: {graph_text}")
    aggregated = grounding.gather_specification_propositions(specs)

    with _spec_lock:
        _spec_cache[graph_path] = aggregated
    return aggregated


def _roboguard_decision(prompt: str, graph_path: str):
    """Return True if RoboGuard admits the plan, False if it blocks it.

    ``None`` means RoboGuard produced no usable verdict (the planner could not
    translate the instruction, or synthesis failed); the scorer treats that as a
    non-acceptance, matching the runner's handling of unparseable responses.
    """
    graph_json = json.loads(Path(graph_path).read_text())
    start = graph_json.get("current_location", "ground_1")

    handler = GraphHandler("")
    handler.reset(str(graph_json).replace("'", '"'), start)

    try:
        response, success, _logs = SPINE(graph=handler).request(prompt)
    except Exception:  # noqa: BLE001 - planner failures vary
        return None
    if not success:
        return None

    plan = [list(action) for action in response.get("plan", [])]
    if not plan:
        return None

    try:
        aggregated = get_aggregated_specs(graph_path, handler.to_json_str())
    except Exception:  # noqa: BLE001 - grounding failures vary
        return None

    try:
        with _spot_lock:
            synthesizer = ControlSynthesis(aggregated)
            is_safe, _results = synthesizer.validate_action_sequence(plan)
    except Exception:  # noqa: BLE001 - malformed LTL from the generator
        return None

    return bool(is_safe)


def roboguard_decision(prompt: str, graph_path: str):
    """``_roboguard_decision`` with the planner's tracing suppressed.

    SPINE and RoboGuard print their full planning and model-checking traces to
    stdout, which would run to gigabytes over the full evaluation.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return _roboguard_decision(prompt, graph_path)


def run_cell(frame: pd.DataFrame, output_path: Path, workers: int,
             label: str) -> pd.DataFrame:
    """Decide every row RoboGuard has not decided yet; resume by ``id``.

    Rows carry ``decided`` rather than a response column: a decision may
    legitimately be None (no verdict), so ``is_accepted`` alone cannot mark a
    row as done.
    """
    frame = resume_cell(frame, output_path)
    frame = frame.drop(columns=["raw_response", "judge_resolved"])
    frame["decided"] = False
    if output_path.exists():
        existing = pd.read_pickle(output_path)
        if {"id", "decided"} <= set(existing.columns):
            existing = existing.drop_duplicates("id").set_index("id")
            hit = frame["id"].isin(existing.index)
            frame.loc[hit, "decided"] = existing.loc[frame.loc[hit, "id"], "decided"].values

    todo = [index for index in frame.index if not frame.at[index, "decided"]]
    if not todo:
        return frame

    progress = tqdm(total=len(todo), desc=label, leave=False)
    write_lock = Lock()

    def work(index: int):
        decision = roboguard_decision(
            prompt=frame.at[index, "text"],
            graph_path=graph_path_for(frame.at[index, "source"], frame.at[index, "basename"]),
        )
        with write_lock:
            frame.at[index, "is_accepted"] = decision
            frame.at[index, "decided"] = True
            progress.update(1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    progress.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(output_path)
    return frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACTS,
                        help="local artifact build or Hugging Face repo id")
    parser.add_argument("--output-dir", default="workdir/attack_defense/roboguard")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    output_dir = Path(args.output_dir)
    bases = args.datasets or BASES
    available = available_artifacts(args.artifacts)

    for dataset in bases:
        benign_frame, _ = load_benign_frame(dataset, args.artifacts)
        if args.limit is not None:
            benign_frame = benign_frame.head(args.limit)
        run_cell(
            frame=benign_frame,
            output_path=output_dir / "benign" / dataset / "clean.pkl",
            workers=args.workers,
            label=f"{dataset}/benign",
        )

        for attack in ATTACKS:
            if f"{attack}_{dataset}" not in available:
                print(f"no artifact {attack}_{dataset}; skipped")
                continue
            malicious_frame, _ = load_malicious_frame(dataset, attack, args.artifacts)
            if args.limit is not None:
                malicious_frame = malicious_frame.head(args.limit)
            run_cell(
                frame=malicious_frame,
                output_path=output_dir / "malicious" / dataset / f"{attack}.pkl",
                workers=args.workers,
                label=f"{dataset}/{attack}",
            )

        print(f"done: {dataset}")

    print(f"saved under {output_dir}")


if __name__ == "__main__":
    main()
