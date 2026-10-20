"""Load one artifact.

    scenes = robobench.load_scenes("robopair_DROID")
    scenes.rows      # sequence of dicts, keys as in the schema; images are PIL
    scenes[id]       # one row by id
    scenes.name      # the artifact name

Text columns are read when the artifact is loaded; an image is decoded only
when its row is accessed. ``repo_or_path`` is the Hugging Face repo id or a
local directory in the same layout (the output of ``build_artifacts.py``).
"""

from datasets import Image, load_dataset

try:  # datasets >= 4 names the list feature ``List``; older versions ``Sequence``
    from datasets import List as ListFeature
except ImportError:  # pragma: no cover
    from datasets import Sequence as ListFeature

# Set to the hub repo id at upload time.
DEFAULT_REPO = "robobench/RoboBench"


class _Rows:
    """Sequence view over the dataset; decodes images per accessed row."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, position):
        return self._dataset[position]

    def __iter__(self):
        return iter(self._dataset)


class Scenes:
    def __init__(self, name: str, dataset):
        self.name = name
        self.dataset = dataset
        self.rows = _Rows(dataset)
        self._position = {id_: i for i, id_ in enumerate(dataset["id"])}
        self._raw = None

    def __len__(self):
        return len(self.dataset)

    def __contains__(self, id_):
        return id_ in self._position

    def __getitem__(self, id_: str) -> dict:
        return self.dataset[self._position[id_]]

    @property
    def ids(self) -> list[str]:
        return list(self._position)

    def image_bytes(self, id_: str) -> list[dict]:
        """``[{"bytes", "path"}, ...]`` for a row, without decoding to PIL."""
        if self._raw is None:
            self._raw = self.dataset.cast_column("images", ListFeature(Image(decode=False)))
        return self._raw[self._position[id_]]["images"]


def load_scenes(name: str, repo_or_path: str = DEFAULT_REPO) -> Scenes:
    return Scenes(name, load_dataset(repo_or_path, name, split="train"))
