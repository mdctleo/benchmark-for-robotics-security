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

# spot ships its bindings outside the virtualenv; RoboGuard and SPINE are
# vendored next to this script (src/defenses/roboguard/RoboGuard, gitignored)
# rather than installed.
SPOT_SITE_PACKAGES = "/opt/homebrew/lib/python3.14/site-packages"
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for path in (
    SPOT_SITE_PACKAGES,
    str(HERE / "RoboGuard/src"),
    str(HERE / "RoboGuard/src/SPINE/src"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

import spot  # noqa: E402

spot.setup()

from roboguard.generator import ContextualGrounding  # noqa: E402
from roboguard.prompts.base import BASE_RULES  # noqa: E402
from roboguard.synthesis import ControlSynthesis  # noqa: E402
from spine.spine import SPINE, GraphHandler  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "src"))
from main_attack_defense import ATTACKS, DATASETS, load_benign_frame, load_malicious_frame  # noqa: E402

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


def graph_path_for(image_filename: str) -> str:
    return str(image_filename).replace(".png", ".json")


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
    frame = frame.reset_index(drop=True).copy()

    if output_path.exists():
        existing = pd.read_pickle(output_path)
        if len(existing) == len(frame) and "decided" in existing.columns:
            frame["is_accepted"] = existing["is_accepted"].values
            frame["decided"] = existing["decided"].values
        else:
            frame["is_accepted"] = None
            frame["decided"] = False
    else:
        frame["is_accepted"] = None
        frame["decided"] = False

    todo = [index for index in frame.index if not frame.at[index, "decided"]]

    if todo:
        progress = tqdm(total=len(todo), desc=label, leave=False)
        write_lock = Lock()

        def work(index: int):
            decision = roboguard_decision(
                prompt=frame.at[index, "text"],
                graph_path=graph_path_for(frame.at[index, "filename"]),
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
    datasets = args.datasets or list(DATASETS)

    for dataset in datasets:
        benign_frame = load_benign_frame(dataset)
        if args.limit is not None:
            benign_frame = benign_frame.head(args.limit)
        run_cell(
            frame=benign_frame,
            output_path=output_dir / "benign" / dataset / "clean.pkl",
            workers=args.workers,
            label=f"{dataset}/benign",
        )

        for attack in ATTACKS:
            malicious_frame = load_malicious_frame(dataset, attack)
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
