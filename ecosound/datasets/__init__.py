"""
ecosound.datasets — download and load curated bioacoustic datasets.

Quick start
-----------
>>> import ecosound
>>> ecosound.datasets.list_datasets()          # see available datasets
>>> annots = ecosound.datasets.load("haddock-stellwagen-2023")

To change where datasets are cached (run once, persists across sessions):
>>> ecosound.datasets.init("/path/to/my/data")

To also download audio files:
>>> annots, audio_paths = ecosound.datasets.load("haddock-stellwagen-2023", audio=True)

To add a new dataset to the registry:
>>> ecosound.datasets.make_registry_entry(
...     dataset_id="my-dataset-2024",
...     annotation_base_urls=["https://zenodo.org/record/1234567/files/"],
...     audio_base_urls=["https://zenodo.org/record/1234567/files/"],
...     title="My dataset",
...     description="...",
...     doi="10.5281/zenodo.1234567",
... )
"""

import json
from pathlib import Path

from ._loader import _load_dataset, save_settings
from ._tools import make_registry_entry

_REGISTRY_DIR = Path(__file__).parent / "registry"


def _load_registry():
    registry = {}
    for json_file in sorted(_REGISTRY_DIR.glob("*.json")):
        with open(json_file) as f:
            registry[json_file.stem] = json.load(f)
    return registry


_REGISTRY = _load_registry()


def init(cache_dir):
    """
    Set the local directory used to cache downloaded datasets.

    Saved to ``~/.ecosound/settings.json`` and used automatically by
    :func:`load` in all future sessions.

    Parameters
    ----------
    cache_dir : str or Path
        Path to the directory where datasets should be stored.

    Examples
    --------
    >>> ecosound.datasets.init("/data/bioacoustics/datasets")
    """
    save_settings(cache_dir)


def list_datasets():
    """
    Print all available datasets.

    Examples
    --------
    >>> ecosound.datasets.list_datasets()
    haddock-stellwagen-2023: Haddock calls at Stellwagen Bank 2022-2023  (DOI: 10.5281/zenodo.XXXXXXX)
    """
    for dataset_id, meta in _REGISTRY.items():
        print(f"{dataset_id}: {meta['title']}  (DOI: {meta['doi']})")


def load(dataset_id, audio=True, cache_dir=None):
    """
    Download (if needed) and load a dataset as an Annotation object.

    Files are downloaded once and cached locally. Subsequent calls return
    immediately from cache without re-downloading.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier. Use :func:`list_datasets` to see available options.
    audio : bool, optional
        If True, also download audio files and update the annotation's
        ``audio_file_dir`` column to point to the local cached files.
        Default True. Already-cached files are not re-downloaded.
        Set to False to load annotations only (faster, no large downloads).
    cache_dir : str or Path, optional
        Override the cache directory for this call only. If not provided, the
        directory set by :func:`init` is used; if that has not been configured
        the OS default (``pooch.os_cache("ecosound")``) is used.

    Returns
    -------
    annots : ecosound.core.annotation.Annotation
        Loaded annotations with ``audio_file_dir`` pointing to the local
        cache directory.

    Raises
    ------
    ValueError
        If *dataset_id* is not found in the registry.

    Examples
    --------
    >>> annots = ecosound.datasets.load("minke-whale-mouy-2026")
    >>> annots = ecosound.datasets.load("minke-whale-mouy-2026", audio=False)
    """
    if dataset_id not in _REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_id}'. "
            "Call ecosound.datasets.list_datasets() to see available datasets."
        )
    return _load_dataset(dataset_id, _REGISTRY[dataset_id], audio=audio, cache_dir=cache_dir)
