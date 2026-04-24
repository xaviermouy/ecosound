# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ERDDAP Data Fetcher
Class-based interface for retrieving data from ERDDAP servers.

"""

from __future__ import annotations
from typing import Tuple, Iterable, List, Optional, Dict
import xarray as xr
import pandas as pd
import numpy as np
import os
import tempfile
from urllib.request import urlretrieve
import xml.etree.ElementTree as ET


class ERDDAPDataFetcher:
    """
    Class for fetching data from ERDDAP servers.

    Handles data retrieval, dateline crossing, quality masking, and spatial/temporal striding.
    """

    def __init__(self, server: str, dataset_id: Optional[str] = None):
        """
        Initialize the ERDDAP data fetcher.

        Args:
            server: ERDDAP server URL (e.g., "https://comet.nefsc.noaa.gov/erddap")
            dataset_id: Dataset ID (optional, can be set later for fetch operations)
        """
        self.server = server.rstrip('/')  # Remove trailing slash if present
        self.dataset_id = dataset_id

    def list_datasets(self, search_text: Optional[str] = None) -> pd.DataFrame:
        """
        List all datasets available on the ERDDAP server.

        Args:
            search_text: Optional text to filter datasets (searches in title and summary)

        Returns:
            DataFrame with columns: dataset_id, title, summary, institution
        """
        # Use ERDDAP's allDatasets endpoint with CSV response
        url = f"{self.server}/info/index.csv"

        try:
            # Try using requests if available
            try:
                import requests
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
            except ImportError:
                # Fallback to pandas read_csv directly
                df = pd.read_csv(url)

            # Filter columns of interest
            columns_to_keep = ['Dataset ID', 'Title', 'Summary', 'Institution']
            available_columns = [col for col in columns_to_keep if col in df.columns]
            df = df[available_columns]

            # Rename for consistency
            df = df.rename(columns={
                'Dataset ID': 'dataset_id',
                'Title': 'title',
                'Summary': 'summary',
                'Institution': 'institution'
            })

            # Filter by search text if provided
            if search_text:
                mask = (
                    df['title'].str.contains(search_text, case=False, na=False) |
                    df['summary'].str.contains(search_text, case=False, na=False)
                )
                df = df[mask]

            return df

        except Exception as e:
            print(f"Error fetching dataset list: {e}")
            print(f"Attempted URL: {url}")
            return pd.DataFrame()

    def get_dataset_info(self, dataset_id: Optional[str] = None) -> Dict:
        """
        Get detailed information about a specific dataset.

        Args:
            dataset_id: Dataset ID (uses instance dataset_id if not provided)

        Returns:
            Dictionary with dataset metadata
        """
        ds_id = dataset_id or self.dataset_id
        if not ds_id:
            raise ValueError("No dataset_id specified")

        url = f"{self.server}/info/{ds_id}/index.csv"

        try:
            df = pd.read_csv(url)

            # Convert to dictionary
            info = {}
            for _, row in df.iterrows():
                if 'Variable Name' in df.columns and 'Attribute Name' in df.columns:
                    var_name = row.get('Variable Name', '')
                    attr_name = row.get('Attribute Name', '')
                    value = row.get('Value', '')

                    if var_name not in info:
                        info[var_name] = {}
                    info[var_name][attr_name] = value

            return info

        except Exception as e:
            print(f"Error fetching dataset info: {e}")
            return {}

    def list_variables(self, dataset_id: Optional[str] = None) -> List[str]:
        """
        List all variables available in a dataset.

        Args:
            dataset_id: Dataset ID (uses instance dataset_id if not provided)

        Returns:
            List of variable names
        """
        info = self.get_dataset_info(dataset_id)

        # Filter out coordinate variables and metadata
        variables = [
            var for var in info.keys()
            if var and var not in ['NC_GLOBAL', '', 'time', 'latitude', 'longitude', 'altitude', 'depth']
        ]

        return variables

    @staticmethod
    def _wrap_to_180(lon: float) -> float:
        """Wrap longitude to [-180, 180] range."""
        x = ((lon + 180) % 360) - 180
        return 180.0 if abs(x - (-180.0)) < 1e-9 else x

    @staticmethod
    def _iso_noon(d) -> str:
        """Convert date to ISO format at noon UTC."""
        return pd.to_datetime(d).strftime("%Y-%m-%dT12:00:00Z")

    def _constraint_index_range(
        self,
        t0_iso: str,
        t1_iso: str,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time_stride: int = 1,
        spatial_stride: int = 1
    ) -> str:
        """
        Build ERDDAP value-style constraint indices for [time][lat][lon].
        """
        return (
            f"[({t0_iso}):{time_stride}:({t1_iso})]"
            f"[({lat_min}):{spatial_stride}:({lat_max})]"
            f"[({lon_min}):{spatial_stride}:({lon_max})]"
        )

    def _build_urls(
        self,
        variables: Iterable[str],
        t0_iso: str,
        t1_iso: str,
        lat_bounds: Tuple[float, float],
        lon_bounds: Tuple[float, float],
        time_stride: int = 1,
        spatial_stride: int = 1,
        dataset_id: Optional[str] = None
    ) -> List[str]:
        """
        Create one or two ERDDAP .nc URLs.
        If the box crosses the dateline, split into [lon_min, 180] + [-180, lon_max].
        """
        ds_id = dataset_id or self.dataset_id
        if not ds_id:
            raise ValueError("No dataset_id specified")

        lat1, lat2 = sorted(lat_bounds)
        lo1, lo2 = self._wrap_to_180(lon_bounds[0]), self._wrap_to_180(lon_bounds[1])
        base = f"{self.server}/griddap/{ds_id}.nc?"

        def url_for(lmin, lmax):
            idx = self._constraint_index_range(
                t0_iso, t1_iso, lat1, lat2, lmin, lmax,
                time_stride=time_stride, spatial_stride=spatial_stride
            )
            vars_clause = ",".join(f"{v}{idx}" for v in variables)
            return base + vars_clause

        if lo1 <= lo2:
            return [url_for(lo1, lo2)]
        else:
            # Dateline crossing
            return [url_for(lo1, 180.0), url_for(-180.0, lo2)]

    @staticmethod
    def _open_erddap_nc(url: str) -> xr.Dataset:
        """Download ERDDAP .nc subset to a temp file, then open with xarray."""
        tmp_path = None
        try:
            try:
                import requests
                r = requests.get(url, stream=True, timeout=180)
                r.raise_for_status()
                fd, tmp_path = tempfile.mkstemp(suffix=".nc")
                os.close(fd)
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if chunk:
                            f.write(chunk)
            except (ImportError, Exception):
                # Fallback to stdlib if requests not available
                tmp_path, _ = urlretrieve(url)

            ds = xr.open_dataset(tmp_path)
            return ds.load()  # Load so we can safely delete the temp file
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def fetch_data(
        self,
        variables: str | List[str],
        *,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        dataset_id: Optional[str] = None,
        include_quality: bool = False,
        quality_variable: str = "quality_level",
        quality_mask_value: Optional[int] = None,
        spatial_stride: int = 1,
        time_stride: int = 1,
        max_request_duration_days: int = 31
    ) -> xr.Dataset:
        """
        Fetch gridded data from ERDDAP for a single day or date range.
        Automatically splits large time ranges into chunks to avoid overwhelming the server.

        Args:
            variables: Variable name(s) to fetch (string or list of strings)
            date: Single date (YYYY-MM-DD) - mutually exclusive with start_date/end_date
            start_date: Start date (YYYY-MM-DD) for range
            end_date: End date (YYYY-MM-DD) for range
            lat_min: Minimum latitude
            lat_max: Maximum latitude
            lon_min: Minimum longitude
            lon_max: Maximum longitude
            dataset_id: Dataset ID (uses instance dataset_id if not provided)
            include_quality: Whether to include quality variable
            quality_variable: Name of quality variable (default: "quality_level")
            quality_mask_value: Mask data by quality value (e.g., 5 for best quality)
            spatial_stride: Spatial stride for thinning data (1=all, 2=every other point, etc.)
            time_stride: Temporal stride for thinning data (1=all, 2=every other day, etc.)
            max_request_duration_days: Maximum time range per request (default: 31 days/~1 month)
                                      Large ranges are split into multiple requests automatically.

        Returns:
            xarray.Dataset with requested data (seamlessly combined from multiple requests if needed)

        Examples:
            Single day::

                ds = fetcher.fetch_data(
                    "sea_surface_temperature", date="2018-01-01",
                    lat_min=42, lat_max=43, lon_min=-71, lon_max=-69)

            Large date range (automatically chunked into monthly requests)::

                ds = fetcher.fetch_data(
                    ["sst", "chlorophyll"], start_date="2018-01-01",
                    end_date="2018-06-30", lat_min=42, lat_max=43,
                    lon_min=-71, lon_max=-69, max_request_duration_days=31)
        """
        # Normalize time inputs
        if date is not None:
            t0 = t1 = pd.to_datetime(date)
        else:
            if start_date is None or end_date is None:
                raise ValueError("Provide either `date` OR both `start_date` and `end_date`.")
            t0, t1 = sorted([pd.to_datetime(start_date), pd.to_datetime(end_date)])

        # Normalize variables to list
        if isinstance(variables, str):
            variables = [variables]

        vars_to_request = list(variables)
        if include_quality or quality_mask_value is not None:
            if quality_variable not in vars_to_request:
                vars_to_request.append(quality_variable)

        # Check if we need to split the request into chunks
        time_range_days = (t1 - t0).days + 1  # +1 to include both endpoints

        if time_range_days <= max_request_duration_days:
            # Single request - no chunking needed
            print(f"Fetching {time_range_days} days of data in 1 request...")
            ds = self._fetch_single_timerange(
                vars_to_request, t0, t1,
                lat_min, lat_max, lon_min, lon_max,
                time_stride, spatial_stride, dataset_id
            )
        else:
            # Split into chunks
            num_chunks = int(np.ceil(time_range_days / max_request_duration_days))
            print(f"Fetching {time_range_days} days of data in {num_chunks} requests (max {max_request_duration_days} days each)...")

            # Create time chunks
            time_chunks = []
            current_start = t0
            while current_start <= t1:
                current_end = min(current_start + pd.Timedelta(days=max_request_duration_days - 1), t1)
                time_chunks.append((current_start, current_end))
                current_start = current_end + pd.Timedelta(days=1)

            # Fetch each chunk
            chunk_datasets = []
            for i, (chunk_start, chunk_end) in enumerate(time_chunks, 1):
                chunk_days = (chunk_end - chunk_start).days + 1
                print(f"  Request {i}/{num_chunks}: {chunk_start.date()} to {chunk_end.date()} ({chunk_days} days)")

                chunk_ds = self._fetch_single_timerange(
                    vars_to_request, chunk_start, chunk_end,
                    lat_min, lat_max, lon_min, lon_max,
                    time_stride, spatial_stride, dataset_id
                )
                chunk_datasets.append(chunk_ds)

            # Concatenate all chunks along time dimension
            print(f"  Combining {len(chunk_datasets)} chunks...")
            ds = xr.concat(chunk_datasets, dim="time")

        # Optional: mask by quality level
        if quality_variable in ds and quality_mask_value is not None:
            for var in variables:
                if var in ds:
                    ds[var] = ds[var].where(ds[quality_variable] == quality_mask_value)

        # Sort coordinates (helps with plotting and time operations)
        for coord in ("time", "latitude", "longitude"):
            if coord in ds.coords:
                ds = ds.sortby(coord)

        print(f"Fetched data with shape: {ds.dims}")
        return ds

    def _fetch_single_timerange(
        self,
        variables: List[str],
        t0: pd.Timestamp,
        t1: pd.Timestamp,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        time_stride: int,
        spatial_stride: int,
        dataset_id: Optional[str]
    ) -> xr.Dataset:
        """
        Internal method to fetch data for a single time range.
        Handles dateline crossing by splitting into multiple spatial requests if needed.
        """
        t0_iso = self._iso_noon(t0)
        t1_iso = self._iso_noon(t1)

        # Build URLs (may split if crossing dateline)
        urls = self._build_urls(
            variables, t0_iso, t1_iso,
            (lat_min, lat_max), (lon_min, lon_max),
            time_stride=time_stride, spatial_stride=spatial_stride,
            dataset_id=dataset_id
        )

        # Download and open (stitch if crossing dateline)
        parts = [self._open_erddap_nc(u) for u in urls]
        ds = xr.concat(parts, dim="longitude") if len(parts) > 1 else parts[0]

        return ds


# Example usage
if __name__ == "__main__":
    # ERDDAP server and datasets
    ERDDAP_SERVER = "https://comet.nefsc.noaa.gov/erddap"
    ACSPO_REANALYSIS = "noaa_coastwatch_acspo_v2_reanalysis"
    ACSPO_NRT = "noaa_coastwatch_acspo_v2_nrt"

    # Initialize fetcher
    fetcher = ERDDAPDataFetcher(server=ERDDAP_SERVER, dataset_id=ACSPO_REANALYSIS)

    # Example 1: List all datasets on the server
    print("Example 1: List all datasets")
    print("=" * 80)
    datasets = fetcher.list_datasets()
    print(f"Found {len(datasets)} datasets")
    print(datasets.head(10))
    print()

    # Example 2: Search for specific datasets
    print("Example 2: Search for SST datasets")
    print("=" * 80)
    sst_datasets = fetcher.list_datasets(search_text="temperature")
    print(f"Found {len(sst_datasets)} datasets containing 'temperature'")
    print(sst_datasets[['dataset_id', 'title']].head())
    print()

    # Example 3: Get dataset info
    print("Example 3: Get dataset information")
    print("=" * 80)
    info = fetcher.get_dataset_info()
    print(f"Dataset: {fetcher.dataset_id}")
    if 'NC_GLOBAL' in info:
        print("Global attributes:")
        for key, value in list(info['NC_GLOBAL'].items())[:5]:
            print(f"  {key}: {value}")
    print()

    # Example 4: List variables
    print("Example 4: List available variables")
    print("=" * 80)
    variables = fetcher.list_variables()
    print(f"Available variables: {variables}")
    print()

    # Example 5: Fetch SST data
    print("Example 5: Fetch SST data")
    print("=" * 80)
    ds = fetcher.fetch_data(
        "sea_surface_temperature",
        start_date="2018-01-01",
        end_date="2018-01-03",
        lat_min=42.0,
        lat_max=43.0,
        lon_min=-71.0,
        lon_max=-69.0,
        include_quality=True,
        quality_mask_value=5,  # Best quality only
        spatial_stride=1,
        time_stride=1
    )
    print(ds)
    print()

    # Example 6: Fetch multiple variables
    print("Example 6: Fetch with different dataset")
    print("=" * 80)
    fetcher2 = ERDDAPDataFetcher(server=ERDDAP_SERVER, dataset_id=ACSPO_REANALYSIS)
    ds2 = fetcher2.fetch_data(
        "sea_surface_temperature",
        date="2018-01-01",
        lat_min=42.0,
        lat_max=43.0,
        lon_min=-71.0,
        lon_max=-69.0
    )
    print(ds2)
