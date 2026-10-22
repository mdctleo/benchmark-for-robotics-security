"""Load one artifact.

    scenes = robobench.load_scenes("robopair_DROID", "mdctleo/RoboJailBench")
    scenes.rows      # sequence of dicts, keys as in the schema; images are PIL
    scenes[id]       # one row by id
    scenes.name      # the artifact name

Text columns are read when the artifact is loaded; an image is decoded only
when its row is accessed. ``repo`` is the Hugging Face dataset the artifacts
are published in; ``datasets`` downloads it once and reads its cache after.
"""

from datasets import Image, load_dataset

try:  # datasets >= 4 names the list feature ``List``; older versions ``Sequence``
    from datasets import List as ListFeature
except ImportError:  # pragma: no cover
    from datasets import Sequence as ListFeature

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


def load_scenes(name: str, repo: str) -> Scenes:
    return Scenes(name, load_dataset(repo, name, split="train"))
