"""Add the artifact row ``id`` to the existing response pickles.

The pickles under ``workdir/attack_defense/`` predate the artifacts and identify
rows by ``source`` + ``filename`` and by their path
(``<model>/malicious/<dataset>/<attack>/<defense>.pkl``,
``<model>/benign/<dataset>/clean/<defense>.pkl``, and the RoboGuard equivalents
without the defense level). The id is a pure function of those, so this adds an
``id`` column with no API calls and no change to any response or label.

Pickles whose frames come from the earlier ``data/main_dataset_5`` build are
skipped and listed: their keys match the current RJB-Instructions frames but
the images and instructions differ.

Usage:
    uv run python src/backfill_ids.py
"""

from pathlib import Path

import pandas as pd

from robobench.ids import frame_key, row_id
from robobench.index import load_index

ROOT = Path(__file__).resolve().parent.parent
WORKDIR = ROOT / "workdir/attack_defense"
EARLIER_BUILD_MARKER = "main_dataset_5/"


def artifact_for(path: Path) -> str:
    """Artifact name encoded in a pickle's path."""
    parts = path.relative_to(WORKDIR).parts  # model, kind, dataset, [attack|clean], file
    kind, dataset = parts[1], parts[2]
    if kind == "benign":
        return dataset
    attack = parts[3] if len(parts) == 5 else path.stem  # roboguard: malicious/<ds>/<attack>.pkl
    return f"{attack}_{dataset}"


def main():
    index = load_index()
    paths = sorted(WORKDIR.rglob("*.pkl"))
    done = refused = 0
    for path in paths:
        frame = pd.read_pickle(path)
        if frame["filename"].astype(str).str.contains(EARLIER_BUILD_MARKER, regex=False).any():
            print(f"skipped (earlier data): {path.relative_to(ROOT)}")
            refused += 1
            continue

        artifact = artifact_for(path)
        ids = [
            row_id(artifact, frame_key(source, filename))
            for source, filename in zip(frame["source"], frame["filename"])
        ]
        unknown = [id_ for id_ in ids if id_ not in index]
        if unknown:
            raise SystemExit(
                f"{path.relative_to(ROOT)}: {len(unknown)} ids not in the index "
                f"(artifact {artifact}); is the index built?"
            )
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{path.relative_to(ROOT)}: duplicate ids")

        frame["id"] = ids
        frame.to_pickle(path)
        done += 1

    print(f"back-filled {done} pickles, refused {refused}")


if __name__ == "__main__":
    main()
