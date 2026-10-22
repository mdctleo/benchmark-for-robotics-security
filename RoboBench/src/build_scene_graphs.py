"""Build the scene graphs RoboGuard needs.

RoboGuard does not look at pixels. SPINE plans over a scene graph and the
safety specifications are derived from that graph, so every frame needs one.
For every frame of the vanilla artifacts this asks a model for the graph
(``robobench.defenses.roboguard.scene_graph.generate_scene_graph``) and saves
it as ``<scene-graphs>/<source>/<frame stem>.json``, where the runner's
RoboGuard reads it. Frames that already have a graph are skipped.

Frames come from the Hugging Face dataset ``main_attack_defense.ARTIFACTS``;
``source`` and ``basename`` are the runner's, so the graphs land where it looks.

Usage:
    uv run python src/build_scene_graphs.py
    uv run python src/build_scene_graphs.py --dataset DROID --scene-graphs scene_graphs
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from main_attack_defense import ARTIFACTS, BASES, scene_frame
from robobench.defenses.roboguard.scene_graph import generate_scene_graph, make_client


def build_graph(client, image: bytes, output: Path) -> None:
    while True:
        try:
            graph = generate_scene_graph(image, client=client)
            break
        except Exception as e:  # noqa: BLE001 - retry on any API failure
            print(f"failed, retrying: {e}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, indent=4), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--scene-graphs", default="scene_graphs")
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    graph_dir = Path(args.scene_graphs)
    client = make_client()

    for dataset in args.datasets or BASES:
        frame, scenes = scene_frame(dataset, "benign", ARTIFACTS)
        for row in tqdm(frame.itertuples(index=False), total=len(frame), desc=dataset):
            output = graph_dir / row.source / (Path(row.basename).stem + ".json")
            if output.exists():
                continue
            build_graph(client, scenes.image_bytes(row.id)[0]["bytes"], output)

    print(f"scene graphs under {graph_dir}")


if __name__ == "__main__":
    main()
