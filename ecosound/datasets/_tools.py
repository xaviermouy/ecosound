"""
Maintainer tools for ecosound.datasets.

make_registry_entry() queries a repository's public API to build a dataset
registry JSON file automatically, without having to list every file manually.

Supported repositories
----------------------
- Zenodo       : https://zenodo.org
- NCEI / GCS   : https://storage.googleapis.com  (public buckets only)
"""

import base64
import binascii
import json
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

# Default output directory — the registry/ subfolder alongside this file.
# Works out of the box with editable installs (pip install -e .).
_REGISTRY_DIR = Path(__file__).parent / "registry"

# File extensions used to classify files automatically.
_ANNOTATION_EXTENSIONS = {".nc"}
_AUDIO_EXTENSIONS = {".wav", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".bwf", ".zip"}


# ---------------------------------------------------------------------------
# Repository-specific file listers
# ---------------------------------------------------------------------------

def _list_files_zenodo(base_url):
    """Return {relative_filename: checksum} for all files in a Zenodo record."""
    match = re.search(r"((?:sandbox\.)?zenodo\.org)/(?:record|records)/(\d+)", base_url)
    if not match:
        raise ValueError(
            f"Cannot extract Zenodo record ID from URL: {base_url!r}\n"
            "Expected format: https://zenodo.org/record/{{record_id}}/files/\n"
            "         or      https://sandbox.zenodo.org/record/{{record_id}}/files/"
        )
    host = match.group(1)
    record_id = match.group(2)
    api_url = f"https://{host}/api/records/{record_id}"
    with urlopen(api_url) as resp:
        data = json.loads(resp.read())
    files = {}
    for f in data.get("files", []):
        # Zenodo returns checksum as "md5:hexvalue" — directly usable by pooch
        files[f["key"]] = f["checksum"]
    return files


def _list_files_gcs(base_url):
    """Return {relative_filename: checksum} for all objects under a GCS prefix."""
    # URL: https://storage.googleapis.com/{bucket}/{prefix}/
    match = re.match(r"https://storage\.googleapis\.com/([^/]+)/?(.*)$", base_url)
    if not match:
        raise ValueError(
            f"Cannot parse GCS URL: {base_url!r}\n"
            "Expected format: https://storage.googleapis.com/{{bucket}}/{{prefix}}/"
        )
    bucket = match.group(1)
    prefix = match.group(2).strip("/")
    search_prefix = (prefix + "/") if prefix else ""

    files = {}
    page_token = None
    while True:
        params = {"prefix": search_prefix}
        if page_token:
            params["pageToken"] = page_token
        api_url = (
            f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?"
            + urlencode(params)
        )
        with urlopen(api_url) as resp:
            data = json.loads(resp.read())
        for item in data.get("items", []):
            name = item["name"]
            # Make path relative to the requested prefix
            rel_name = name[len(search_prefix):] if search_prefix else name
            if not rel_name or rel_name.endswith("/"):
                continue  # skip prefix placeholder entries
            # GCS returns MD5 as base64 — convert to hex for pooch
            md5_hex = binascii.hexlify(base64.b64decode(item["md5Hash"])).decode()
            files[rel_name] = f"md5:{md5_hex}"
        page_token = data.get("nextPageToken")
        if not page_token:
            break
    return files


def _list_files(base_url):
    """Auto-detect repository type and return {filename: checksum} dict."""
    if "zenodo.org" in base_url:
        return _list_files_zenodo(base_url)
    elif "storage.googleapis.com" in base_url:
        return _list_files_gcs(base_url)
    else:
        raise ValueError(
            f"Unsupported repository URL: {base_url!r}\n"
            "Supported: Zenodo (zenodo.org), NCEI/GCS (storage.googleapis.com)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_registry_entry(
    dataset_id,
    annotation_base_urls,
    audio_base_urls,
    title,
    description,
    doi="",
    license="CC-BY-4.0",
    source="builtin",
    output_dir=None,
):
    """
    Query a repository's API to build a dataset registry entry and save it as JSON.

    File lists and checksums are fetched automatically — you do not need to
    list individual files by hand. Files are classified by extension:
    ``.nc`` files go to ``annotation_files``; audio files (``.wav``, ``.aif``,
    ``.aiff``, ``.flac``, ``.mp3``, ``.ogg``, ``.bwf``) go to ``audio_files``.

    Supported repositories: **Zenodo** and **NCEI / Google Cloud Storage**
    (public buckets only).

    After the JSON is written, reload the ecosound.datasets module (or restart
    Python) for the new dataset to appear in :func:`list_datasets`.

    Parameters
    ----------
    dataset_id : str
        Unique dataset identifier used as the JSON filename and as the key
        passed to ``ecosound.datasets.load()``.
        E.g. ``"haddock-stellwagen-2023"``.
    annotation_base_urls : list of str
        Base URLs of the folders/buckets containing annotation ``.nc`` files.
        For Zenodo this is typically the same URL as *audio_base_urls*.
        For NCEI/GCS this may point to a different bucket than the audio.
    audio_base_urls : list of str
        Base URLs of the folders/buckets containing audio files.
    title : str
        Human-readable dataset title.
    description : str
        Short description of the dataset content.
    doi : str, optional
        Dataset DOI (e.g. ``"10.5281/zenodo.1234567"``). Default ``""``.
    license : str, optional
        Dataset license identifier. Default ``"CC-BY-4.0"``.
    source : str, optional
        ``"builtin"`` for maintainer-curated entries. Default ``"builtin"``.
    output_dir : str or Path, optional
        Directory where the JSON file is written. Defaults to
        ``ecosound/datasets/registry/`` in the source tree. Specify an
        alternative path if the package is not installed in editable mode.

    Returns
    -------
    dict
        The registry entry (also written to disk as ``{dataset_id}.json``).

    Examples
    --------
    Zenodo dataset (annotations and audio in the same record):

    >>> import ecosound
    >>> ecosound.datasets.make_registry_entry(
    ...     dataset_id="haddock-stellwagen-2023",
    ...     annotation_base_urls=["https://zenodo.org/record/1234567/files/"],
    ...     audio_base_urls=["https://zenodo.org/record/1234567/files/"],
    ...     title="Haddock calls at Stellwagen Bank 2022-2023",
    ...     description="Manually annotated haddock knocks.",
    ...     doi="10.5281/zenodo.1234567",
    ... )

    NCEI dataset (annotations and audio in different GCS buckets):

    >>> ecosound.datasets.make_registry_entry(
    ...     dataset_id="minke-whale-ncei-2022",
    ...     annotation_base_urls=[
    ...         "https://storage.googleapis.com/noaa-passive-bioacoustic/project/annotations/"
    ...     ],
    ...     audio_base_urls=[
    ...         "https://storage.googleapis.com/noaa-passive-bioacoustic/project/audio-A/",
    ...         "https://storage.googleapis.com/noaa-passive-bioacoustic/project/audio-B/",
    ...     ],
    ...     title="Minke whale calls NCEI 2022",
    ...     description="...",
    ... )
    """
    out_dir = Path(output_dir) if output_dir else _REGISTRY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_files = {}
    for base_url in annotation_base_urls:
        print(f"[ecosound.datasets] Querying {base_url} ...")
        all_files = _list_files(base_url)
        nc_files = {
            k: v for k, v in all_files.items()
            if Path(k).suffix.lower() in _ANNOTATION_EXTENSIONS
        }
        if nc_files:
            annotation_files[base_url] = nc_files
        else:
            print(f"  Warning: no .nc annotation files found at {base_url!r}")

    audio_files = {}
    for base_url in audio_base_urls:
        # Avoid re-querying a URL already fetched for annotations
        if base_url in annotation_files:
            all_files_for_audio = {
                **_list_files(base_url)
            } if base_url not in annotation_files else _list_files(base_url)
        else:
            print(f"[ecosound.datasets] Querying {base_url} ...")
            all_files_for_audio = _list_files(base_url)
        audio = {
            k: v for k, v in all_files_for_audio.items()
            if Path(k).suffix.lower() in _AUDIO_EXTENSIONS
        }
        if audio:
            audio_files[base_url] = audio
        else:
            print(f"  Warning: no audio files found at {base_url!r}")

    entry = {
        "title": title,
        "description": description,
        "doi": doi,
        "source": source,
        "license": license,
        "annotation_files": annotation_files,
        "audio_files": audio_files,
    }

    out_path = out_dir / f"{dataset_id}.json"
    with open(out_path, "w") as f:
        json.dump(entry, f, indent=2)

    n_annot = sum(len(v) for v in annotation_files.values())
    n_audio = sum(len(v) for v in audio_files.values())
    print(
        f"[ecosound.datasets] Registry entry written to: {out_path}\n"
        f"  {n_annot} annotation file(s), {n_audio} audio file(s)\n"
        "  Restart Python (or reload ecosound.datasets) to use the new dataset."
    )
    return entry
