"""Generate an attack over a vanilla artifact, as a reviewable raw-layout tree.

For every frame of the base artifact the attack
(``robobench.attacks.ATTACKS[<attack>]``) rewrites the malicious instruction
and, for attacks that touch pixels, the frame. The result is a copy of the
base's raw tree with the malicious text replaced, written as
``<output-dir>/<attack>_<base>/`` in the layout ``build_artifacts.py`` reads:

    <attack>_<base>/<base>_instr.json + <attack>_<base>/dataset/<frame>          one source
    <attack>_<base>/<source>/<source>_instr.json + .../<source>/dataset/<frame>  categorized base

The instruction file maps each frame path to ``[malicious, benign]``, benign
copied from the artifact unchanged; an attack that found nothing leaves the
malicious text empty. Frames keep the names recorded in the package index.
Review the file and the frames, then build the artifact:

    uv run python src/build_artifacts.py generated_attacks/<attack>_<base>

Frames already in the instruction file are skipped, so a run can be stopped and
restarted. The base artifact is read from ``--artifacts``, the Hugging Face
dataset by default or a local artifact directory.

Usage:
    uv run python src/build_attacks.py --attack robopair --dataset DROID
    uv run python src/build_attacks.py --attack badrobot_cj
    uv run python src/build_attacks.py --attack badrobot_cd --dataset DROID --limit 3
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from main_attack_defense import ARTIFACTS, BASES
from robobench.attacks import ATTACKS
from robobench.index import load_index
from robobench.scenes import Scenes, load_scenes


def build_tree(attack, base: str, scenes: Scenes, index, output_dir: Path,
               limit=None) -> Path:
    """Run ``attack`` over ``scenes`` into ``<output_dir>/<attack.name>_<base>``."""
    tree = output_dir / f"{attack.name}_{base}"
    table = scenes.dataset.remove_columns("images").to_pandas()
    if limit is not None:
        table = table.head(limit)

    instructions: dict[Path, dict] = {}  # instruction file -> its entries

    def instruction_file(source: str) -> Path:
        source_dir = tree if source == base else tree / source
        path = source_dir / f"{source}_instr.json"
        if path not in instructions:
            instructions[path] = (
                json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            )
        return path

    for row in tqdm(table.itertuples(index=False), total=len(table), desc=tree.name):
        source, filename = index[row.id]["key"].split("/", 1)
        instr_path = instruction_file(source)
        frame_path = instr_path.parent / "dataset" / filename
        if frame_path.as_posix() in instructions[instr_path]:
            continue

        raw = scenes.image_bytes(row.id)[0]["bytes"]
        result = attack.attack(text=row.malicious, images=[raw])

        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame_path.write_bytes(result.images[0] if result.images else raw)
        instructions[instr_path][frame_path.as_posix()] = [result.malicious or "", row.benign]
        instr_path.write_text(json.dumps(instructions[instr_path], indent=4), encoding="utf-8")

    return tree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", required=True, choices=sorted(ATTACKS))
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--artifacts", default=ARTIFACTS,
                        help="Hugging Face dataset or local artifact directory")
    parser.add_argument("--output-dir", default="generated_attacks")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    attack = ATTACKS[args.attack]()
    index = load_index()
    output_dir = Path(args.output_dir)

    for base in args.datasets or BASES:
        scenes = load_scenes(base, args.artifacts)
        tree = build_tree(attack, base, scenes, index, output_dir, args.limit)
        print(f"done: {tree}")

    print(f"review the instruction files, then build with src/build_artifacts.py {output_dir}/<attack>_<base>")


if __name__ == "__main__":
    main()
