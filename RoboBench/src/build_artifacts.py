"""Build one RoboBench artifact from a directory in the raw layout.

The directory's name is the artifact name: a base (``DROID``,
``RJB-Instructions``) or ``<attack>_<base>`` (``robopair_DROID``). Two layouts
are read, chosen by what is in the directory:

    <dir>/<base>_instr.json + <dir>/dataset/<frame>              one source, the base
    <dir>/<source>/<source>_instr.json + <dir>/<source>/dataset/  one source per subfolder;
                                                                   ``source`` fills ``category``

An instruction file maps a frame path (relative to the repo root) to
``[malicious, benign]``; ``generate_prompt_from_img.py`` writes it for raw data
and ``build_attacks.py`` for attacks. Every frame listed becomes one row. An
empty malicious text is kept as null and reported, so an attack artifact has
one row per frame of its base. The ``target`` column of an attack artifact
comes from the attack class in ``robobench.attacks``.

Output:
    artifacts/data/<name>/train.parquet   rewritten from scratch
    artifacts/index.json                  [{id, dataset, base, attack, target, key}] of every artifact
    src/robobench/index.json              copy of the index, shipped in the package
    artifacts/README.md                   dataset card, one config per artifact in the index

The index keeps the rows of every other artifact and replaces this one's.

Frames of bases listed in ``REENCODE_JPEG`` (data no model has been run
against yet) are re-encoded to JPEG; every other frame is copied byte for byte.

Usage:
    uv run python src/build_artifacts.py data/external_datasets/DROID
    uv run python src/build_artifacts.py data/RJB-Instructions
    uv run python src/build_artifacts.py generated_attacks/robopair_DROID
    uv run python src/build_artifacts.py data/external_datasets/DROID --limit 3   # smoke test
"""

import argparse
import io
import json
import math
import shutil
from pathlib import Path

import datasets
import numpy as np
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

datasets.disable_progress_bars()

try:  # datasets >= 4 names the list feature ``List``; older versions ``Sequence``
    from datasets import List as ListFeature
except ImportError:  # pragma: no cover
    from datasets import Sequence as ListFeature

from main_attack_defense import split_artifact
from robobench.attacks import ATTACKS
from robobench.ids import frame_key, row_id
from robobench.index import Index

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts"
PACKAGE_INDEX = ROOT / "src/robobench/index.json"

# Bases whose frames are re-encoded to JPEG when written. Only data that no
# model has been queried on yet may go here.
REENCODE_JPEG = set()
# JPEG settings for re-encoded frames. Every frame must stay at or above
# PSNR_FLOOR_DB against its original; the build aborts otherwise.
JPEG_QUALITY = 97
JPEG_SUBSAMPLING = 0  # 4:4:4, full-resolution colour
PSNR_FLOOR_DB = 40.0

FEATURES = Features({
    "id": Value("string"),
    "dataset": Value("string"),
    "category": Value("string"),
    "images": ListFeature(Image()),
    "benign": Value("string"),
    "malicious": Value("string"),
    "target": Value("string"),
})


def psnr(original: np.ndarray, encoded: np.ndarray) -> float:
    mse = float(np.mean((original.astype(np.float64) - encoded.astype(np.float64)) ** 2))
    if mse == 0:
        return math.inf
    return 10.0 * math.log10(255.0 ** 2 / mse)


def reencode_jpeg(raw: bytes) -> tuple[bytes, float]:
    image = PILImage.open(io.BytesIO(raw)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=JPEG_SUBSAMPLING)
    encoded = buffer.getvalue()
    decoded = PILImage.open(io.BytesIO(encoded)).convert("RGB")
    return encoded, psnr(np.asarray(image), np.asarray(decoded))


def find_sources(directory: Path, base: str) -> list[tuple[str, Path, bool]]:
    """``(source, instruction file, categorized)`` for each source of ``directory``."""
    single = directory / f"{base}_instr.json"
    if single.exists():
        return [(base, single, False)]
    sources = [
        (sub.name, sub / f"{sub.name}_instr.json", True)
        for sub in sorted(directory.iterdir())
        if sub.is_dir() and (sub / f"{sub.name}_instr.json").exists()
    ]
    if not sources:
        raise SystemExit(f"{directory}: no instruction file for {base} and no source subfolders")
    return sources


def load_frames(directory: Path, name: str, limit) -> list[dict]:
    """One record per frame of the artifact: key, source, texts, bytes."""
    _, base = split_artifact(name)
    frames = []
    psnrs = []
    empty = []
    for source, instr_path, categorized in find_sources(directory, base):
        data = json.loads(instr_path.read_text(encoding="utf-8"))
        items = list(data.items())
        if limit is not None:
            items = items[:limit]
        for filename, pair in items:
            raw = (ROOT / filename).read_bytes()
            frame_name = Path(filename).name
            if base in REENCODE_JPEG:
                raw, quality = reencode_jpeg(raw)
                psnrs.append((quality, frame_key(source, filename)))
                frame_name = Path(frame_name).with_suffix(".jpg").name
            malicious = pair[0]
            if malicious is None or not str(malicious).strip():
                malicious = None
                empty.append(frame_name)
            frames.append({
                "key": frame_key(source, filename),
                "category": source if categorized else None,
                "malicious": malicious,
                "benign": pair[1],
                "image": {"bytes": raw, "path": frame_name},
            })

    keys = [frame["key"] for frame in frames]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{name}: duplicate frame keys")
    if empty:
        # Frames with no malicious text keep their row, with a null instruction,
        # so an attack artifact has one row per frame of its base.
        print(f"  WARNING {name}: {len(empty)} empty malicious texts kept as null: "
              + ", ".join(empty))
    if psnrs:
        worst = min(psnrs)
        mean = sum(q for q, _ in psnrs) / len(psnrs)
        print(f"  {name}: re-encoded {len(psnrs)} frames to JPEG q{JPEG_QUALITY}; "
              f"PSNR mean {mean:.2f} dB, min {worst[0]:.2f} dB ({worst[1]})")
        if worst[0] < PSNR_FLOOR_DB:
            raise SystemExit(f"{name}: PSNR below {PSNR_FLOOR_DB} dB for {worst[1]}")
    return frames


def write_artifact(name: str, frames: list[dict], out_dir: Path) -> list[dict]:
    """Write ``data/<name>/train.parquet`` from scratch; return the index entries."""
    attack, base = split_artifact(name)
    target = ATTACKS[attack].target if attack is not None else None
    rows = []
    entries = []
    for frame in frames:
        id_ = row_id(name, frame["key"])
        rows.append({
            "id": id_,
            "dataset": name,
            "category": frame["category"],
            "images": [frame["image"]],
            "benign": frame["benign"],
            "malicious": frame["malicious"],
            "target": target,
        })
        entries.append({
            "id": id_,
            "dataset": name,
            "base": base,
            "attack": attack,
            "target": target,
            "key": frame["key"],
        })

    artifact_dir = out_dir / "data" / name
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)
    path = artifact_dir / "train.parquet"
    Dataset.from_list(rows, features=FEATURES).to_parquet(str(path))
    print(f"  {name}: {len(rows)} rows -> {path}")
    return entries


def merge_index(entries: list[dict], name: str, index_path: Path) -> list[dict]:
    """The index with ``name``'s rows replaced by ``entries``; duplicate ids abort."""
    existing = []
    if index_path.exists():
        existing = json.loads(index_path.read_text(encoding="utf-8"))
    merged = [entry for entry in existing if entry["dataset"] != name] + entries
    Index(merged)  # raises on duplicate ids
    return merged


def write_readme(names: list[str], out_dir: Path) -> None:
    lines = ["---", "configs:"]
    for name in names:
        lines += [
            f"- config_name: {name}",
            "  data_files:",
            "  - split: train",
            f"    path: data/{name}/train.parquet",
        ]
    lines += [
        "---",
        "",
        "# RoboBench",
        "",
        "Benchmark artifacts for jailbreak attacks on vision-language robot planners.",
        "One config per artifact; every artifact is self-contained (images included).",
        "Vanilla artifacts carry the original malicious and benign instruction per",
        "frame; `<attack>_<base>` artifacts carry the attack's rewritten malicious",
        "instruction with the same frames and benign instruction.",
        "",
        "Columns: `id` (uuid, unique across the repo), `dataset`, `category`,",
        "`images`, `benign`, `malicious`, `target`.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build(directory: Path, limit, out_dir: Path, index_path: Path) -> list[dict]:
    """Build the artifact named after ``directory``; return the merged index."""
    name = directory.resolve().name
    print(name)
    frames = load_frames(directory, name, limit)
    entries = write_artifact(name, frames, out_dir)
    index = merge_index(entries, name, index_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "index.json", index_path)
    write_readme(sorted({entry["dataset"] for entry in index}), out_dir)
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path,
                        help="dataset directory in the raw layout; its name is the artifact name")
    parser.add_argument("--limit", type=int, default=None,
                        help="frames per source folder (smoke test)")
    args = parser.parse_args()

    index = build(args.directory, args.limit, OUT_DIR, PACKAGE_INDEX)
    names = {entry["dataset"] for entry in index}
    print(f"{len(names)} artifacts, {len(index)} index rows -> {OUT_DIR.relative_to(ROOT)}; "
          f"index copied to {PACKAGE_INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
