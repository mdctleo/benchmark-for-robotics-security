"""Row ids.

An id is ``uuid5(ROBOBENCH_NS, "<dataset>:<key>")`` where ``dataset`` is the
artifact name (``DROID``, ``robopair_DROID``, ...) and ``key`` is
``"<source folder>/<frame basename>"`` (``DROID/frame_0.png``,
``Illegal_Activity/$TerrorismExample 1.png``). The key is independent of where
the data lives on disk, so a rebuild over the same frames yields the same ids,
and the same scene in another artifact is the same key under another dataset
name.
"""

import uuid
from pathlib import PurePath

ROBOBENCH_NS = uuid.UUID("7a6b8c1e-3f2d-4e5a-9b0c-1d2e3f4a5b6c")


def frame_key(source: str, filename: str) -> str:
    """``"<source>/<basename>"`` for a frame path as recorded in the inputs."""
    return f"{source}/{PurePath(str(filename)).name}"


def row_id(dataset: str, key: str) -> str:
    return str(uuid.uuid5(ROBOBENCH_NS, f"{dataset}:{key}"))
