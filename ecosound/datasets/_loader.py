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


def _is_dataverse_url(base_url):
    """Return True if *base_url* is a Dataverse datafile access endpoint."""
    return "/api/access/datafile/" in base_url


def _fetch_file_group(base_url, files_dict, dest_path):
    """
    Download all files in one base_url group and return their local paths.

    Dispatches to a Dataverse-specific path when *base_url* is a Dataverse
    datafile endpoint (``…/api/access/datafile/``), otherwise uses a standard
    pooch retriever where ``url = base_url + filename``.

    Parameters
    ----------
    base_url : str
        For Zenodo/GCS: base URL prepended to each filename key.
        For Dataverse: ``https://{host}/api/access/datafile/``.
    files_dict : dict
        For Zenodo/GCS: ``{relative_filename: checksum}``.
        For Dataverse: ``{"fileId:relative_filename": checksum}``.
    dest_path : Path
        Local directory where files are cached.

    Returns
    -------
    list of str
        Local paths of all downloaded (or already cached) files.
    """
    if _is_dataverse_url(base_url):
        return _fetch_file_group_dataverse(base_url, files_dict, dest_path)

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


def _fetch_file_group_dataverse(base_url, files_dict, dest_path):
    """
    Download Dataverse files using per-file URLs constructed from file IDs.

    Dataverse registry keys have the form ``"{fileId}:{relative_filename}"``.
    The download URL is ``base_url + fileId``; the file is saved locally as
    ``relative_filename`` (with any directory component flattened).

    Parameters
    ----------
    base_url : str
        ``https://{host}/api/access/datafile/``
    files_dict : dict
        ``{"fileId:relative_filename": checksum}``
    dest_path : Path
        Local directory where files are cached.

    Returns
    -------
    list of str
        Local paths of all downloaded (or already cached) files.
    """
    paths = []
    for key, checksum in files_dict.items():
        # key format: "12345:path/to/filename.wav"
        file_id, rel_name = key.split(":", 1)
        rel_path = Path(rel_name)
        # Preserve subfolder structure: save into dest_path / directoryLabel /
        file_dir = dest_path / rel_path.parent
        file_dir.mkdir(parents=True, exist_ok=True)
        url = base_url + file_id
        local_path = pooch.retrieve(
            url=url,
            known_hash=checksum,
            fname=rel_path.name,
            path=file_dir,
        )
        paths.append(local_path)
    return paths


def _load_dataset(dataset_id, entry, audio=True, cache_dir=None):
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
        Default True. Already-cached files are not re-downloaded.
    cache_dir : str or Path, optional
        Override the cache directory for this call only.

    Returns
    -------
    annots : ecosound.core.annotation.Annotation
        Loaded annotations with ``audio_file_dir`` pointing to the local
        cache directory.
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
        # Update audio_file_dir to the cache dir so that audio files already
        # present from a previous call are found automatically.
        annots.update_audio_dir(str(path))
        return annots

    # --- Audio files (already-cached files are skipped by pooch) ---
    for base_url, files in entry["audio_files"].items():
        _fetch_file_group(base_url, files, path)

    # Update audio_file_dir to point to the local cache.
    annots.update_audio_dir(str(path))

    return annots
