"""The id index: every row of every artifact, without the images.

``index.json`` is a list of ``{id, dataset, base, attack, target, key}`` built by
``build_artifacts.py`` and shipped with the package, so ``score`` can resolve
an id to its artifact without loading any artifact.
"""

import json
from collections import defaultdict
from pathlib import Path

SHIPPED_INDEX = Path(__file__).with_name("index.json")


class Index:
    def __init__(self, entries: list[dict]):
        self.entries = entries
        self.by_id = {entry["id"]: entry for entry in entries}
        if len(self.by_id) != len(entries):
            raise ValueError("index contains duplicate ids")

        self._by_dataset: dict[str, list[dict]] = defaultdict(list)
        for entry in entries:
            self._by_dataset[entry["dataset"]].append(entry)

    def __contains__(self, id_: str) -> bool:
        return id_ in self.by_id

    def __getitem__(self, id_: str) -> dict:
        return self.by_id[id_]

    def datasets(self) -> list[str]:
        return list(self._by_dataset)

    def rows(self, dataset: str) -> list[dict]:
        return self._by_dataset[dataset]

    def n_rows(self, dataset: str) -> int:
        return len(self._by_dataset[dataset])

    def target(self, dataset: str):
        rows = self._by_dataset[dataset]
        return rows[0]["target"] if rows else None


def load_index(path=None) -> Index:
    path = Path(path) if path is not None else SHIPPED_INDEX
    if not path.exists():
        raise FileNotFoundError(
            f"no index at {path}; run src/build_artifacts.py to build the artifacts"
        )
    return Index(json.loads(path.read_text(encoding="utf-8")))
