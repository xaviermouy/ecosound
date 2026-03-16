"""
Download and loading logic for ecosound.datasets.

Uses pooch for download, checksum verification, and local caching.
Each dataset is cached in its own subdirectory: {cache_dir}/{dataset_id}/
"""

import json
from pathlib import Path

import pooch

from ecosound.core.annotation import Annotation

# Path of the user-level config file written by ecosound.datasets.init()
_SETTINGS_FILE = Path.home() / ".ecosound" / "settings.json"


def _get_cache_dir():
    """
    Return the configured cache directory.

    Reads from ``~/.ecosound/settings.json`` if it exists; otherwise falls
    back to the OS-appropriate default (``pooch.os_cache("ecosound")``).
    """
    if _SETTINGS_FILE.exists():
        with open(_SETTINGS_FILE) as f:
            settings = json.load(f)
        cache_dir = settings.get("cache_dir")
        if cache_dir:
            return Path(cache_dir)

    default = pooch.os_cache("ecosound")
    print(
        f"[ecosound.datasets] Caching data in: {default}\n"
        "  To use a different location call: ecosound.datasets.init(cache_dir)"
    )
    return Path(default)


def save_settings(cache_dir):
    """
    Persist *cache_dir* to ``~/.ecosound/settings.json``.

    Creates the ``~/.ecosound/`` directory if it does not exist.
    """
    _SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    settings = {}
    if _SETTINGS_FILE.exists():
        with open(_SETTINGS_FILE) as f:
            settings = json.load(f)
    settings["cache_dir"] = str(Path(cache_dir).expanduser().resolve())
    with open(_SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"[ecosound.datasets] Cache directory set to: {settings['cache_dir']}")


def _fetch_file_group(base_url, files_dict, dest_path):
    """
    Create a pooch retriever for one base_url group and fetch all files.

    Parameters
    ----------
    base_url : str
        Base URL prepended to each filename key for download.
    files_dict : dict
        {relative_filename: checksum} for all files in this group.
    dest_path : Path
        Local directory where files are cached.

    Returns
    -------
    list of str
        Local paths of all downloaded (or already cached) files.
    """
    retriever = pooch.create(
        path=dest_path,
        base_url=base_url,
        registry=files_dict,
    )
    paths = []
    for fname in files_dict:
        if fname.lower().endswith(".zip"):
            # Extract ZIP in-place; pooch returns list of extracted file paths.
            # Extracted files land in {dest_path}/{fname}.unzip/ and are found
            # automatically by update_audio_dir() via its recursive search.
            paths.extend(retriever.fetch(fname, processor=pooch.Unzip()))
        else:
            paths.append(retriever.fetch(fname))
    return paths


def _load_dataset(dataset_id, entry, audio=False, cache_dir=None):
    """
    Download (if needed) and load a dataset.

    Parameters
    ----------
    dataset_id : str
        Dataset identifier, used to namespace the cache subdirectory.
    entry : dict
        A single dataset entry loaded from its JSON registry file.
    audio : bool
        If True, also download audio files and update the annotation's
        ``audio_file_dir`` column to point to the downloaded files.
        Default False.
    cache_dir : str or Path, optional
        Override the cache directory for this call only.

    Returns
    -------
    annots : ecosound.core.annotation.Annotation
        Loaded annotations.
    audio_paths : list of str
        Paths of downloaded audio files. Only returned when ``audio=True``,
        i.e. return value is ``(annots, audio_paths)`` when ``audio=True``,
        otherwise just ``annots``.
    """
    root = Path(cache_dir) if cache_dir is not None else _get_cache_dir()
    # Each dataset gets its own subdirectory to avoid filename collisions
    path = root / dataset_id
    path.mkdir(parents=True, exist_ok=True)

    # --- Annotation files (always downloaded; usually small) ---
    annot_paths = []
    for base_url, files in entry["annotation_files"].items():
        annot_paths.extend(_fetch_file_group(base_url, files, path))

    annots = Annotation()
    annots.from_netcdf(annot_paths)

    if not audio:
        return annots

    # --- Audio files (opt-in; can be large) ---
    audio_paths = []
    for base_url, files in entry["audio_files"].items():
        audio_paths.extend(_fetch_file_group(base_url, files, path))

    # Update audio_file_dir in annotations to point at the cache directory.
    # update_audio_dir() recursively searches path for matching filenames.
    if audio_paths:
        annots.update_audio_dir(str(path))

    return annots, audio_paths
