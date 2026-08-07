"""
Maintainer tools for ecosound.datasets.

make_registry_entry() queries a repository's public API to build a dataset
registry JSON file automatically, without having to list every file manually.

Supported repositories
----------------------
- Zenodo       : https://zenodo.org
- NCEI / GCS   : https://storage.googleapis.com  (public buckets only)
- Dataverse    : https://dataverse.harvard.edu  (any Dataverse installation)
"""

import base64
import binascii
import json
import re
from pathlib import Path
from urllib.parse import urlencode, urlparse, parse_qs
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


def _list_files_dataverse(dataset_url):
    """
    Return ``(download_base_url, {file_key: checksum})`` for all files in a
    Dataverse dataset.

    ``file_key`` has the form ``"{fileId}:{filename}"`` so the loader can
    reconstruct both the download URL (``download_base_url + fileId``) and
    the local filename to save as.

    Parameters
    ----------
    dataset_url : str
        Any of:
        - Dataset page URL:
          ``https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XXX``
        - API URL:
          ``https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:10.7910/DVN/XXX``
        - Bare DOI with server prefix:
          ``https://dataverse.harvard.edu doi:10.7910/DVN/XXX``
    """
    # --- Parse server and persistent ID (DOI) from the URL ---
    parsed = urlparse(dataset_url)
    server = f"{parsed.scheme}://{parsed.netloc}"

    # Extract DOI from ?persistentId=doi:... query parameter
    qs = parse_qs(parsed.query)
    if "persistentId" in qs:
        doi = qs["persistentId"][0]
    else:
        # Try to extract inline doi: reference from the path/query string
        doi_match = re.search(r"(doi:[^\s&]+)", dataset_url)
        if not doi_match:
            raise ValueError(
                f"Cannot extract persistent ID (DOI) from Dataverse URL: {dataset_url!r}\n"
                "Expected format: https://dataverse.harvard.edu/dataset.xhtml"
                "?persistentId=doi:10.7910/DVN/XXXXX"
            )
        doi = doi_match.group(1)

    # --- Fetch file list via Dataverse native API (paginated) ---
    files = {}
    limit = 1000
    offset = 0
    while True:
        params = urlencode({"persistentId": doi, "limit": limit, "offset": offset})
        api_url = (
            f"{server}/api/datasets/:persistentId/versions/:latest/files?{params}"
        )
        with urlopen(api_url) as resp:
            data = json.loads(resp.read())

        batch = data.get("data", [])
        for item in batch:
            file_info = item.get("dataFile", {})
            file_id   = file_info.get("id")
            filename  = item.get("label") or file_info.get("filename", "")
            directory = item.get("directoryLabel", "")

            # Preserve folder structure in the filename key if present
            rel_name = f"{directory}/{filename}" if directory else filename

            checksum_info = file_info.get("checksum", {})
            ctype = checksum_info.get("type", "").upper()
            cval  = checksum_info.get("value", "")
            if ctype == "MD5":
                checksum = f"md5:{cval}"
            elif ctype == "SHA-1":
                checksum = f"sha1:{cval}"
            elif ctype in ("SHA-256", "SHA256"):
                checksum = f"sha256:{cval}"
            else:
                checksum = f"md5:{cval}"  # fallback

            # Encode as "fileId:rel_name" so the loader can split them apart
            files[f"{file_id}:{rel_name}"] = checksum

        if len(batch) < limit:
            break
        offset += limit

    if not files:
        raise ValueError(
            f"No files found in Dataverse dataset: {doi!r}\n"
            f"  Server: {server}\n"
            "  Check that the dataset is published and publicly accessible."
        )

    download_base_url = f"{server}/api/access/datafile/"
    return download_base_url, files


def _list_files(base_url):
    """
    Auto-detect repository type and return ``(download_base_url, {filename: checksum})``.

    For Zenodo and GCS the ``download_base_url`` is ``None`` (the caller's
    ``base_url`` is used as-is).  For Dataverse it is the computed datafile
    access endpoint.
    """
    if "zenodo.org" in base_url:
        return None, _list_files_zenodo(base_url)
    elif "storage.googleapis.com" in base_url:
        return None, _list_files_gcs(base_url)
    elif "dataverse" in base_url.lower() or "/dataset.xhtml" in base_url:
        return _list_files_dataverse(base_url)
    else:
        raise ValueError(
            f"Unsupported repository URL: {base_url!r}\n"
            "Supported: Zenodo (zenodo.org), NCEI/GCS (storage.googleapis.com), "
            "Dataverse (any host with /dataset.xhtml or 'dataverse' in URL)"
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

    Supported repositories: **Zenodo**, **NCEI / Google Cloud Storage**
    (public buckets only), and **Dataverse** (any public installation).

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

    Dataverse dataset (Harvard Dataverse or any Dataverse installation):

    >>> ecosound.datasets.make_registry_entry(
    ...     dataset_id="minke-whale-mouy-2026",
    ...     annotation_base_urls=[
    ...         "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XXXXX"
    ...     ],
    ...     audio_base_urls=[
    ...         "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/XXXXX"
    ...     ],
    ...     title="Minke whale passive acoustic annotation dataset",
    ...     description="...",
    ...     doi="10.7910/DVN/XXXXX",
    ... )
    """
    out_dir = Path(output_dir) if output_dir else _REGISTRY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache query results to avoid hitting the same URL twice
    _query_cache = {}

    def _query(url):
        if url not in _query_cache:
            print(f"[ecosound.datasets] Querying {url} ...")
            dl_base, files = _list_files(url)
            # dl_base is None for Zenodo/GCS (use the original url as registry key)
            registry_key = dl_base if dl_base is not None else url
            _query_cache[url] = (registry_key, files)
        return _query_cache[url]

    annotation_files = {}
    for base_url in annotation_base_urls:
        registry_key, all_files = _query(base_url)
        # For Dataverse keys are "fileId:path" — extract the path part for extension check
        nc_files = {
            k: v for k, v in all_files.items()
            if Path(k.split(":", 1)[-1]).suffix.lower() in _ANNOTATION_EXTENSIONS
        }
        if nc_files:
            annotation_files[registry_key] = nc_files
        else:
            print(f"  Warning: no .nc annotation files found at {base_url!r}")

    audio_files = {}
    for base_url in audio_base_urls:
        registry_key, all_files = _query(base_url)
        audio = {
            k: v for k, v in all_files.items()
            if Path(k.split(":", 1)[-1]).suffix.lower() in _AUDIO_EXTENSIONS
        }
        if audio:
            audio_files[registry_key] = audio
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
