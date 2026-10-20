"""Build the RoboBench artifacts in Hugging Face repo layout.

Reads the instruction files and the attack files for every base in ``DATASETS``
and writes one self-contained parquet artifact per (base, attack), plus the
dataset card and the id index:

    artifacts/README.md                      configs yaml, one entry per artifact
    artifacts/index.json                     [{id, dataset, base, attack, target, key}]
    artifacts/data/<artifact>/train.parquet  one per artifact
    src/robobench/index.json                 copy of the index, shipped in the package

Artifacts are ``<base>`` (vanilla) and ``<attack>_<base>``. An attack artifact is
built only when the attack file exists for every source folder of the base and
its ``filename`` column matches the base's instruction keys exactly; a file that
matches none of them belongs to another build of the data and is skipped with a
note. Partial matches, duplicate ids and frame-count mismatches abort the build.
Frames whose attack text is empty are kept with a null ``malicious`` and listed
in a warning, so every attack artifact has one row per frame of its base.

Frames of bases listed in ``REENCODE_JPEG`` (data no model has been run
against yet) are re-encoded to JPEG; every other frame is copied byte for byte.

Usage:
    uv run python src/build_artifacts.py            # full build
    uv run python src/build_artifacts.py --limit 3  # smoke test, 3 frames per source
"""

import argparse
import io
import json
import math
import shutil
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage

datasets.disable_progress_bars()

try:  # datasets >= 4 names the list feature ``List``; older versions ``Sequence``
    from datasets import List as ListFeature
except ImportError:  # pragma: no cover
    from datasets import Sequence as ListFeature

from robobench.ids import frame_key, row_id

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts"
ATTACK_DIR = ROOT / "generated_attacks_external"
PACKAGE_INDEX = ROOT / "src/robobench/index.json"

ATTACKS = ["badrobot_cd", "badrobot_cj", "badrobot_sm", "robopair"]

# Model each attack was optimized against, or None for template attacks.
# Written to the ``target`` column and to the index.
ATTACK_TARGETS = {
    "badrobot_cd": None,
    "badrobot_cj": None,
    "badrobot_sm": None,
    "robopair": None,
}

# Logical base -> [(source folder, instruction file)]. The source folder is also
# the attack-file folder under generated_attacks_external/ and the first half
# of every row key. RJB-Instructions is the union of the four taxonomy folders.
DATASETS = {
    "DROID": [("DROID", "data/external_datasets/DROID/DROID_instr.json")],
    "ROBOVQA": [("ROBOVQA", "data/external_datasets/ROBOVQA/ROBOVQA_instr.json")],
    "RH20T": [("RH20T", "data/external_datasets/RH20T/RH20T_instr.json")],
    "robo2vlm": [("robo2vlm", "data/external_datasets/robo2vlm/robo2vlm_instr.json")],
    "car": [("car", "data/cross_embodiment/car/car_instr.json")],
    "RJB-Instructions": [
        (category, f"data/main_dataset_50/{category}/{category}_instr.json")
        for category in (
            "Physical_Safety_Offenses",
            "Permission_Related_Offense",
            "Ethical_Social_Offenses",
            "Illegal_Activity",
        )
    ],
}

# Bases whose category column is the source folder.
CATEGORIZED = {"RJB-Instructions"}

# Bases whose frames are re-encoded to JPEG when written. Only data that no
# model has been queried on yet may go here.
REENCODE_JPEG = {"RJB-Instructions"}
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


def load_frames(base: str, limit) -> list[dict]:
    """One record per frame of ``base``: key, source, instruction key, texts, bytes."""
    frames = []
    psnrs = []
    for source, instr_path in DATASETS[base]:
        data = json.loads((ROOT / instr_path).read_text(encoding="utf-8"))
        items = list(data.items())
        if limit is not None:
            items = items[:limit]
        for filename, pair in items:
            raw = (ROOT / filename).read_bytes()
            name = Path(filename).name
            if base in REENCODE_JPEG:
                raw, quality = reencode_jpeg(raw)
                psnrs.append((quality, frame_key(source, filename)))
                name = Path(name).with_suffix(".jpg").name
            frames.append({
                "key": frame_key(source, filename),
                "source": source,
                "filename": filename,
                "category": source if base in CATEGORIZED else None,
                "malicious": pair[0],
                "benign": pair[1],
                "image": {"bytes": raw, "path": name},
            })

    keys = [frame["key"] for frame in frames]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"{base}: duplicate frame keys")
    if psnrs:
        worst = min(psnrs)
        mean = sum(q for q, _ in psnrs) / len(psnrs)
        print(f"  {base}: re-encoded {len(psnrs)} frames to JPEG q{JPEG_QUALITY}; "
              f"PSNR mean {mean:.2f} dB, min {worst[0]:.2f} dB ({worst[1]})")
        if worst[0] < PSNR_FLOOR_DB:
            raise SystemExit(f"{base}: PSNR below {PSNR_FLOOR_DB} dB for {worst[1]}")
    return frames


def load_attack_texts(base: str, attack: str, frames: list[dict]):
    """``{instruction key: attack text}`` for ``base``, or None when not applicable."""
    wanted = {frame["filename"] for frame in frames}
    texts = {}
    for source, _ in DATASETS[base]:
        path = ATTACK_DIR / source / f"{attack}.json"
        if not path.exists():
            print(f"  {attack}_{base}: no attack file for {source}; skipped")
            return None
        table = pd.read_csv(path)
        table = table.loc[table["type"].eq("malicious")]
        matching = table.loc[table["filename"].isin(wanted)]
        if matching.empty:
            print(f"  {attack}_{base}: {path} names none of this data's frames; skipped")
            return None
        if matching["filename"].duplicated().any():
            raise SystemExit(f"{attack}_{base}: duplicate filenames in {path}")
        is_empty = matching["text"].isna() | matching["text"].astype(str).str.strip().eq("")
        if is_empty.any():
            # Frames with no attack text keep their row, with a null instruction,
            # so every attack artifact has one row per frame of its base.
            listing = ", ".join(Path(f).name for f in matching.loc[is_empty, "filename"])
            print(f"  WARNING {attack}_{base}: {int(is_empty.sum())} empty attack texts in "
                  f"{path.relative_to(ROOT)} kept as null: {listing}")
        texts.update({
            filename: (None if empty else text)
            for filename, text, empty in zip(matching["filename"], matching["text"], is_empty)
        })

    missing = wanted - set(texts)
    if missing:
        raise SystemExit(
            f"{attack}_{base}: {len(missing)} frames have no attack text, e.g. {sorted(missing)[:3]}"
        )
    return texts


def write_artifact(name: str, base: str, attack, frames: list[dict],
                   malicious_texts, out_dir: Path) -> list[dict]:
    """Write ``data/<name>/train.parquet``; return the index entries."""
    target = ATTACK_TARGETS[attack] if attack is not None else None
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
            "malicious": (
                malicious_texts[frame["filename"]] if malicious_texts is not None
                else frame["malicious"]
            ),
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

    path = out_dir / "data" / name / "train.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows, features=FEATURES).to_parquet(str(path))
    print(f"  {name}: {len(rows)} rows -> {path.relative_to(ROOT)}")
    return entries


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="frames per source folder (smoke test)")
    args = parser.parse_args()

    out_dir = OUT_DIR
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    index = []
    names = []
    for base in DATASETS:
        print(base)
        frames = load_frames(base, args.limit)
        index += write_artifact(base, base, None, frames, None, out_dir)
        names.append(base)
        for attack in ATTACKS:
            texts = load_attack_texts(base, attack, frames)
            if texts is None:
                continue
            name = f"{attack}_{base}"
            index += write_artifact(name, base, attack, frames, texts, out_dir)
            names.append(name)

    ids = [entry["id"] for entry in index]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate ids across artifacts")

    write_readme(names, out_dir)
    (out_dir / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    shutil.copyfile(out_dir / "index.json", PACKAGE_INDEX)
    print(f"{len(names)} artifacts, {len(index)} index rows -> {out_dir.relative_to(ROOT)}; "
          f"index copied to {PACKAGE_INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
