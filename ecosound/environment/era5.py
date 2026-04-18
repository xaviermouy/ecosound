# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
ERA5 Atmospheric Data Fetcher
Class-based interface for retrieving ERA5 reanalysis 10-m wind and precipitation
data at a given location and time range.

Two data backends are supported:
    - "open_meteo" (default): free, no registration, no API key required.
      Uses the Open-Meteo historical weather API, which serves ERA5 reanalysis
      data via simple HTTP requests.  Requires only the 'requests' package.
    - "cds": direct Copernicus Climate Data Store access via the cdsapi package.
      Requires a free CDS account and a configured ~/.cdsapirc file.
      See https://cds.climate.copernicus.eu/api-how-to

Both backends return identical xr.Dataset structures with the same coordinates
(time, lat, lon) as NECOFS and HMD outputs, enabling straightforward merging.
"""

from __future__ import annotations
from typing import Optional, Union
from datetime import datetime
import os
import json
import tempfile
from urllib.request import urlopen
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import xarray as xr


class ERA5:
    """
    Class for fetching ERA5 reanalysis 10-m wind and precipitation data at a point location.

    ERA5 provides hourly wind data at 0.25° (~28 km) spatial resolution from
    1940 to present.  The nearest grid point to the requested location is used.

    By default, data is fetched via the Open-Meteo API (no registration needed).
    Set source="cds" to use the Copernicus CDS API directly (requires a free
    account and configured ~/.cdsapirc).

    Args:
        source: Data backend — "open_meteo" (default) or "cds".
        cds_url: CDS API URL override (only used when source="cds").
        cds_key: CDS API key as "UID:key" override (only used when source="cds").
        verbose: Print progress messages (default: True).

    Attributes:
        wind_timeseries (xr.Dataset | None): Most recent wind time series from
            get_wind_timeseries().  Dimension: time.  Coordinates: time, lat, lon.
            None until first call.
        precipitation_timeseries (xr.Dataset | None): Most recent precipitation
            time series from get_precipitation_timeseries().  Dimension: time.
            Coordinates: time, lat, lon.  None until first call.

    Examples:
        # No registration required (Open-Meteo backend)
        >>> from ERA5 import ERA5
        >>> era5 = ERA5()
        >>> era5.get_wind_timeseries(lat=42.5, lon=-70.0,
        ...     start_dt="2015-08-01", end_dt="2015-08-07")
        >>> era5.plot_wind_timeseries()

        # CDS backend (requires ~/.cdsapirc)
        >>> era5_cds = ERA5(source="cds")
        >>> era5_cds.get_wind_timeseries(lat=42.5, lon=-70.0,
        ...     start_dt="2015-08-01", end_dt="2015-08-07")

        # Combine with NECOFS (shared time/lat/lon coordinates)
        >>> ds_wind_aligned = era5.wind_timeseries.sel(
        ...     time=ds_ocean.time, method="nearest")
        >>> ds_combined = xr.merge([ds_ocean, ds_wind_aligned])
    """

    # Open-Meteo historical archive endpoint (ERA5 reanalysis)
    OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

    # CDS dataset identifier
    CDS_DATASET = "reanalysis-era5-single-levels"
    CDS_WIND_VARIABLES = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]
    CDS_PRECIP_VARIABLES = [
        "total_precipitation",
        "snowfall",
    ]

    GRID_RESOLUTION = 0.25  # degrees (ERA5 native resolution)

    def __init__(
        self,
        source: str = "open_meteo",
        cds_url: Optional[str] = None,
        cds_key: Optional[str] = None,
        verbose: bool = True,
    ):
        if source not in ("open_meteo", "cds"):
            raise ValueError("source must be 'open_meteo' or 'cds'.")
        self.source = source
        self.cds_url = cds_url
        self.cds_key = cds_key
        self.verbose = verbose
        self.wind_timeseries: Optional[xr.Dataset] = None           # set by get_wind_timeseries()
        self.precipitation_timeseries: Optional[xr.Dataset] = None  # set by get_precipitation_timeseries()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _wind_direction(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute meteorological wind direction (direction FROM which wind blows).

        Convention: 0° = from North, 90° = from East, 180° = from South,
        270° = from West.

        Args:
            u: Eastward wind component (m/s)
            v: Northward wind component (m/s)

        Returns:
            Wind direction in degrees [0, 360)
        """
        return (270.0 - np.degrees(np.arctan2(v, u))) % 360.0

    @staticmethod
    def _uv_from_speed_direction(
        speed: np.ndarray, wdir: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Derive U (eastward) and V (northward) components from wind speed and
        meteorological direction (FROM convention).

        Args:
            speed: Wind speed (m/s)
            wdir:  Meteorological wind direction (degrees, FROM)

        Returns:
            (u, v) tuple of numpy arrays (m/s)
        """
        rad = np.radians(wdir)
        u = -speed * np.sin(rad)
        v = -speed * np.cos(rad)
        return u, v

    def _build_output_dataset(
        self,
        times: np.ndarray,
        u10: np.ndarray,
        v10: np.ndarray,
        nearest_lat: float,
        nearest_lon: float,
        lat: float,
        lon: float,
    ) -> xr.Dataset:
        """Assemble the standard output xr.Dataset from raw arrays."""
        speed = np.sqrt(u10 ** 2 + v10 ** 2)
        wdir  = self._wind_direction(u10, v10)

        return xr.Dataset(
            data_vars={
                "u10_ms":        ("time", u10,   {"units": "m s-1",  "long_name": "10m eastward wind component"}),
                "v10_ms":        ("time", v10,   {"units": "m s-1",  "long_name": "10m northward wind component"}),
                "wind_speed_ms": ("time", speed, {"units": "m s-1",  "long_name": "10m wind speed"}),
                "wind_dir_deg":  ("time", wdir,  {"units": "degrees","long_name": "10m wind direction (FROM, meteorological)"}),
            },
            coords={
                "time": times,
                "lat":  nearest_lat,
                "lon":  nearest_lon,
            },
            attrs={
                "requested_lat":       lat,
                "requested_lon":       lon,
                "nearest_grid_lat":    nearest_lat,
                "nearest_grid_lon":    nearest_lon,
                "source":              f"ERA5 reanalysis (ECMWF) via {self.source}",
                "temporal_resolution": "hourly",
                "spatial_resolution":  f"{self.GRID_RESOLUTION}° (~28 km)",
                "wind_height":         "10 m above surface",
                "wind_dir_convention": "meteorological: direction FROM which wind blows; 0=N, 90=E, 180=S, 270=W",
            },
        )

    # ------------------------------------------------------------------
    # Open-Meteo backend (no registration required)
    # ------------------------------------------------------------------

    def _query_open_meteo(
        self,
        lat: float,
        lon: float,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
        hourly_vars: list,
        extra_params: Optional[dict] = None,
    ) -> dict:
        """
        Execute a single Open-Meteo historical API request and return the JSON
        response.  Shared by wind and precipitation fetchers.

        Args:
            lat, lon: Target coordinates.
            t0, t1:   Time range.
            hourly_vars: List of Open-Meteo hourly variable names to request.
            extra_params: Additional query parameters (e.g. wind_speed_unit).

        Returns:
            Parsed JSON response dict from Open-Meteo.
        """
        params: dict = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": t0.strftime("%Y-%m-%d"),
            "end_date":   t1.strftime("%Y-%m-%d"),
            "hourly":     ",".join(hourly_vars),
            "timezone":   "UTC",
        }
        if extra_params:
            params.update(extra_params)

        url = f"{self.OPEN_METEO_URL}?{urlencode(params)}"
        self._log("  Querying Open-Meteo API...")

        try:
            import requests
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return response.json()
        except ImportError:
            with urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode())

    def _fetch_open_meteo(
        self,
        lat: float,
        lon: float,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
    ) -> xr.Dataset:
        """Fetch wind data via Open-Meteo (ERA5 reanalysis, no registration)."""
        data        = self._query_open_meteo(
            lat, lon, t0, t1,
            hourly_vars=["wind_speed_10m", "wind_direction_10m"],
            extra_params={"wind_speed_unit": "ms"},
        )
        hourly      = data["hourly"]
        nearest_lat = float(data["latitude"])
        nearest_lon = float(data["longitude"])

        times = pd.to_datetime(hourly["time"]).tz_localize(None)
        mask  = (times >= t0) & (times <= t1)
        times = times[mask]

        speed = np.array(hourly["wind_speed_10m"],     dtype=float)[mask]
        wdir  = np.array(hourly["wind_direction_10m"], dtype=float)[mask]
        u10, v10 = self._uv_from_speed_direction(speed, wdir)

        return self._build_output_dataset(
            times.values, u10, v10, nearest_lat, nearest_lon, lat, lon
        )

    def _fetch_open_meteo_precip(
        self,
        lat: float,
        lon: float,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
    ) -> xr.Dataset:
        """Fetch precipitation data via Open-Meteo (ERA5 reanalysis, no registration)."""
        data        = self._query_open_meteo(
            lat, lon, t0, t1,
            hourly_vars=["precipitation", "rain", "showers", "snowfall"],
        )
        hourly      = data["hourly"]
        nearest_lat = float(data["latitude"])
        nearest_lon = float(data["longitude"])

        times = pd.to_datetime(hourly["time"]).tz_localize(None)
        mask  = (times >= t0) & (times <= t1)
        times = times[mask]

        precip  = np.array(hourly["precipitation"], dtype=float)[mask]   # mm/h
        rain_ls = np.array(hourly["rain"],          dtype=float)[mask]   # mm/h (large-scale)
        showers = np.array(hourly["showers"],       dtype=float)[mask]   # mm/h (convective)
        snow_cm = np.array(hourly["snowfall"],      dtype=float)[mask]   # cm/h
        snow_mm = snow_cm * 10.0                                         # convert to mm/h
        rain    = rain_ls + showers                                       # total rain

        return self._build_precip_dataset(
            times.values, precip, rain, snow_mm, nearest_lat, nearest_lon, lat, lon
        )

    def _build_precip_dataset(
        self,
        times: np.ndarray,
        precip: np.ndarray,
        rain: np.ndarray,
        snow_mm: np.ndarray,
        nearest_lat: float,
        nearest_lon: float,
        lat: float,
        lon: float,
    ) -> xr.Dataset:
        """Assemble the standard precipitation xr.Dataset from raw arrays."""
        return xr.Dataset(
            data_vars={
                "precipitation_mmh": ("time", precip,   {"units": "mm h-1", "long_name": "Total precipitation rate"}),
                "rain_mmh":          ("time", rain,     {"units": "mm h-1", "long_name": "Rainfall rate (large-scale + convective)"}),
                "snowfall_mmh":      ("time", snow_mm,  {"units": "mm h-1", "long_name": "Snowfall rate (liquid water equivalent)"}),
            },
            coords={
                "time": times,
                "lat":  nearest_lat,
                "lon":  nearest_lon,
            },
            attrs={
                "requested_lat":       lat,
                "requested_lon":       lon,
                "nearest_grid_lat":    nearest_lat,
                "nearest_grid_lon":    nearest_lon,
                "source":              f"ERA5 reanalysis (ECMWF) via {self.source}",
                "temporal_resolution": "hourly",
                "spatial_resolution":  f"{self.GRID_RESOLUTION}° (~28 km)",
                "units_note":          "snowfall converted from cm/h to mm/h (liquid water equivalent)",
            },
        )

    # ------------------------------------------------------------------
    # CDS backend (requires cdsapi + ~/.cdsapirc)
    # ------------------------------------------------------------------

    def _get_cds_client(self):
        """Create and return a CDS API client, with a helpful error if not configured."""
        try:
            import cdsapi
        except ImportError:
            raise ImportError(
                "The 'cdsapi' package is required when source='cds'.\n"
                "  Install:   pip install cdsapi\n"
                "  Configure: create ~/.cdsapirc with your CDS API credentials.\n"
                "  Details:   https://cds.climate.copernicus.eu/api-how-to\n"
                "  Tip:       Use the default source='open_meteo' to skip registration."
            )
        kwargs = {"quiet": not self.verbose}
        if self.cds_url is not None:
            kwargs["url"] = self.cds_url
        if self.cds_key is not None:
            kwargs["key"] = self.cds_key
        return cdsapi.Client(**kwargs)

    def _fetch_cds(
        self,
        lat: float,
        lon: float,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
    ) -> xr.Dataset:
        """
        Fetch wind data month-by-month from the Copernicus CDS (ERA5).

        Requires a CDS account and a configured ~/.cdsapirc file.
        Data is fetched one request per calendar month, then concatenated.
        """
        client = self._get_cds_client()

        half = self.GRID_RESOLUTION
        area = [lat + half, lon - half, lat - half, lon + half]  # N, W, S, E

        all_dates = pd.date_range(t0.floor("D"), t1.ceil("D"), freq="D")
        monthly_groups: dict = {}
        for d in all_dates:
            monthly_groups.setdefault((d.year, d.month), set()).add(d.day)

        monthly_datasets = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for (year, month), days in sorted(monthly_groups.items()):
                tmp_file = os.path.join(tmp_dir, f"era5_{year}{month:02d}.nc")
                self._log(f"  Downloading CDS {year}-{month:02d} ({len(days)} day(s))...")
                client.retrieve(
                    self.CDS_DATASET,
                    {
                        "product_type": "reanalysis",
                        "variable":     self.CDS_WIND_VARIABLES,
                        "year":         str(year),
                        "month":        f"{month:02d}",
                        "day":          [f"{d:02d}" for d in sorted(days)],
                        "time":         [f"{h:02d}:00" for h in range(24)],
                        "area":         area,
                        "format":       "netcdf",
                    },
                    tmp_file,
                )
                monthly_datasets.append(xr.open_dataset(tmp_file).load())

        ds_raw = (
            xr.concat(monthly_datasets, dim="time")
            if len(monthly_datasets) > 1
            else monthly_datasets[0]
        )
        ds_raw = ds_raw.sel(time=slice(t0, t1))

        lat_idx     = int(np.argmin(np.abs(ds_raw["latitude"].values  - lat)))
        lon_idx     = int(np.argmin(np.abs(ds_raw["longitude"].values - lon)))
        nearest_lat = float(ds_raw["latitude"].values[lat_idx])
        nearest_lon = float(ds_raw["longitude"].values[lon_idx])
        ds_pt       = ds_raw.isel(latitude=lat_idx, longitude=lon_idx)

        u10 = ds_pt["u10"].values.astype(float)
        v10 = ds_pt["v10"].values.astype(float)

        return self._build_output_dataset(
            ds_pt["time"].values, u10, v10, nearest_lat, nearest_lon, lat, lon
        )

    def _fetch_cds_precip(
        self,
        lat: float,
        lon: float,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
    ) -> xr.Dataset:
        """
        Fetch precipitation data month-by-month from the Copernicus CDS (ERA5).

        Requires a CDS account and a configured ~/.cdsapirc file.
        Data is fetched one request per calendar month, then concatenated.
        """
        client = self._get_cds_client()

        half = self.GRID_RESOLUTION
        area = [lat + half, lon - half, lat - half, lon + half]  # N, W, S, E

        all_dates = pd.date_range(t0.floor("D"), t1.ceil("D"), freq="D")
        monthly_groups: dict = {}
        for d in all_dates:
            monthly_groups.setdefault((d.year, d.month), set()).add(d.day)

        monthly_datasets = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            for (year, month), days in sorted(monthly_groups.items()):
                tmp_file = os.path.join(tmp_dir, f"era5_precip_{year}{month:02d}.nc")
                self._log(f"  Downloading CDS precip {year}-{month:02d} ({len(days)} day(s))...")
                client.retrieve(
                    self.CDS_DATASET,
                    {
                        "product_type": "reanalysis",
                        "variable":     self.CDS_PRECIP_VARIABLES,
                        "year":         str(year),
                        "month":        f"{month:02d}",
                        "day":          [f"{d:02d}" for d in sorted(days)],
                        "time":         [f"{h:02d}:00" for h in range(24)],
                        "area":         area,
                        "format":       "netcdf",
                    },
                    tmp_file,
                )
                monthly_datasets.append(xr.open_dataset(tmp_file).load())

        ds_raw = (
            xr.concat(monthly_datasets, dim="time")
            if len(monthly_datasets) > 1
            else monthly_datasets[0]
        )
        ds_raw = ds_raw.sel(time=slice(t0, t1))

        lat_idx     = int(np.argmin(np.abs(ds_raw["latitude"].values  - lat)))
        lon_idx     = int(np.argmin(np.abs(ds_raw["longitude"].values - lon)))
        nearest_lat = float(ds_raw["latitude"].values[lat_idx])
        nearest_lon = float(ds_raw["longitude"].values[lon_idx])
        ds_pt       = ds_raw.isel(latitude=lat_idx, longitude=lon_idx)

        # CDS ERA5 precipitation variables: tp (m/h accumulated), sf (m/h)
        # Convert from m to mm  (*1000) and from accumulation to rate (already per-hour)
        precip_m  = ds_pt["tp"].values.astype(float)       # total precipitation  (m/h)
        snow_m    = ds_pt["sf"].values.astype(float)       # snowfall (m water eq, /h)
        precip_mm = precip_m  * 1000.0                     # → mm/h
        snow_mm   = snow_m    * 1000.0                     # → mm/h
        rain_mm   = np.maximum(precip_mm - snow_mm, 0.0)   # rainfall = total − snow

        return self._build_precip_dataset(
            ds_pt["time"].values, precip_mm, rain_mm, snow_mm,
            nearest_lat, nearest_lon, lat, lon
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_wind_timeseries(
        self,
        lat: float,
        lon: float,
        start_dt: Union[datetime, pd.Timestamp, str],
        end_dt: Union[datetime, pd.Timestamp, str],
    ) -> xr.Dataset:
        """
        Fetch ERA5 10-m wind data at the nearest grid point for a given
        location and time range.

        The backend used depends on self.source:
            - "open_meteo" (default): no registration needed, uses HTTP GET.
            - "cds": Copernicus CDS API, requires ~/.cdsapirc.

        Both backends return the same xr.Dataset structure.  The result is
        always stored in self.wind_timeseries and always returned, so the
        method can be used with or without assignment:

            era5.get_wind_timeseries(lat, lon, start_dt, end_dt)
            ds = era5.get_wind_timeseries(lat, lon, start_dt, end_dt)

        Args:
            lat: Latitude (decimal degrees, WGS84).
            lon: Longitude (decimal degrees, WGS84, negative west).
            start_dt: Start of the time range (inclusive).  Accepts datetime,
                      pd.Timestamp, or ISO 8601 string (e.g. "2015-08-01").
            end_dt: End of the time range (inclusive).

        Returns:
            xr.Dataset with dimension time, also stored in self.wind_timeseries:
                Data variables:
                    - u10_ms        (time) : 10m eastward wind component (m/s)
                    - v10_ms        (time) : 10m northward wind component (m/s)
                    - wind_speed_ms (time) : 10m wind speed (m/s)
                    - wind_dir_deg  (time) : 10m wind direction, FROM, met. convention (°)
                Coordinates:
                    - time : hourly datetime64 (UTC)
                    - lat  : nearest grid latitude (scalar)
                    - lon  : nearest grid longitude (scalar)
                Dataset.attrs:
                    - requested_lat/lon, nearest_grid_lat/lon, source,
                      temporal_resolution, spatial_resolution, wind_height,
                      wind_dir_convention

        Raises:
            ImportError: If cdsapi is not installed (source="cds" only).
            Exception:   If the API request fails (network, quota, etc.).
        """
        t0 = pd.Timestamp(start_dt)
        t1 = pd.Timestamp(end_dt)

        self._log(
            f"Requesting ERA5 wind [{self.source}] at ({lat}°N, {lon}°E) "
            f"from {t0.date()} to {t1.date()}"
        )

        if self.source == "open_meteo":
            ds = self._fetch_open_meteo(lat, lon, t0, t1)
        else:
            ds = self._fetch_cds(lat, lon, t0, t1)

        n = len(ds["time"])
        self._log(
            f"Nearest grid point: ({float(ds.coords['lat']):.2f}°N, "
            f"{float(ds.coords['lon']):.2f}°E) — {n} hourly records."
        )

        self.wind_timeseries = ds
        return ds

    def get_precipitation_timeseries(
        self,
        lat: float,
        lon: float,
        start_dt: Union[datetime, pd.Timestamp, str],
        end_dt: Union[datetime, pd.Timestamp, str],
    ) -> xr.Dataset:
        """
        Fetch ERA5 hourly precipitation at the nearest grid point for a given
        location and time range.

        The backend used depends on self.source:
            - "open_meteo" (default): no registration needed, uses HTTP GET.
            - "cds": Copernicus CDS API, requires ~/.cdsapirc.

        Both backends return the same xr.Dataset structure.  The result is
        always stored in self.precipitation_timeseries and always returned:

            era5.get_precipitation_timeseries(lat, lon, start_dt, end_dt)
            ds = era5.get_precipitation_timeseries(lat, lon, start_dt, end_dt)

        Args:
            lat: Latitude (decimal degrees, WGS84).
            lon: Longitude (decimal degrees, WGS84, negative west).
            start_dt: Start of the time range (inclusive).  Accepts datetime,
                      pd.Timestamp, or ISO 8601 string (e.g. "2015-08-01").
            end_dt: End of the time range (inclusive).

        Returns:
            xr.Dataset with dimension time, also stored in self.precipitation_timeseries:
                Data variables:
                    - precipitation_mmh (time): Total precipitation rate (mm/h)
                    - rain_mmh          (time): Rainfall rate, large-scale + convective (mm/h)
                    - snowfall_mmh      (time): Snowfall rate, liquid water equivalent (mm/h)
                Coordinates:
                    - time : hourly datetime64 (UTC)
                    - lat  : nearest grid latitude (scalar)
                    - lon  : nearest grid longitude (scalar)
                Dataset.attrs:
                    - requested_lat/lon, nearest_grid_lat/lon, source,
                      temporal_resolution, spatial_resolution, units_note

        Raises:
            ImportError: If cdsapi is not installed (source="cds" only).
            Exception:   If the API request fails (network, quota, etc.).
        """
        t0 = pd.Timestamp(start_dt)
        t1 = pd.Timestamp(end_dt)

        self._log(
            f"Requesting ERA5 precipitation [{self.source}] at ({lat}°N, {lon}°E) "
            f"from {t0.date()} to {t1.date()}"
        )

        if self.source == "open_meteo":
            ds = self._fetch_open_meteo_precip(lat, lon, t0, t1)
        else:
            ds = self._fetch_cds_precip(lat, lon, t0, t1)

        n = len(ds["time"])
        self._log(
            f"Nearest grid point: ({float(ds.coords['lat']):.2f}°N, "
            f"{float(ds.coords['lon']):.2f}°E) — {n} hourly records."
        )

        self.precipitation_timeseries = ds
        return ds

    def plot_precipitation_timeseries(
        self,
        figsize: tuple = (12, 6),
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the precipitation time series stored in self.precipitation_timeseries.

        Produces a figure with two stacked subplots sharing the time axis:
            1. Total precipitation rate (mm/h), filled area
            2. Rainfall and snowfall rates (mm/h), stacked filled areas

        Each panel is auto-scaled to the data range of that variable.

        Args:
            figsize: Figure size as (width, height) in inches (default: (12, 6)).
            display: Show the figure on screen (default: True).
            filename: If provided, save the figure to this path (e.g. "precip.png").
                      Any format supported by matplotlib is accepted.  Default: None.

        Raises:
            RuntimeError: If get_precipitation_timeseries() has not been called yet.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.precipitation_timeseries is None:
            raise RuntimeError(
                "No precipitation data available. "
                "Call get_precipitation_timeseries() first."
            )

        ds    = self.precipitation_timeseries
        times = pd.to_datetime(ds["time"].values)

        lat_str = f"{float(ds.coords['lat']):.2f}°N"
        lon_str = f"{float(ds.coords['lon']):.2f}°E"
        title = (
            f"ERA5 Precipitation  |  ({lat_str}, {lon_str})  "
            f"|  {times[0].date()} – {times[-1].date()}"
        )

        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

        precip  = ds["precipitation_mmh"].values
        rain    = ds["rain_mmh"].values
        snow    = ds["snowfall_mmh"].values

        # --- Panel 1: Total precipitation ---
        axes[0].fill_between(times, precip, alpha=0.7, color="tab:blue", linewidth=0)
        axes[0].plot(times, precip, color="tab:blue", linewidth=0.6)
        axes[0].set_ylabel("Precipitation (mm/h)", fontsize=9)
        axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[0].tick_params(labelsize=8)
        ymax = np.nanmax(precip) if np.nanmax(precip) > 0 else 1.0
        axes[0].set_ylim(0, ymax * 1.15)

        # --- Panel 2: Rain vs Snowfall ---
        axes[1].fill_between(times, rain, alpha=0.65, color="tab:green",  linewidth=0, label="Rain")
        axes[1].fill_between(times, snow, alpha=0.65, color="tab:cyan",   linewidth=0, label="Snowfall")
        axes[1].plot(times, rain, color="tab:green", linewidth=0.6)
        axes[1].plot(times, snow, color="tab:cyan",  linewidth=0.6)
        axes[1].set_ylabel("Rate (mm/h)", fontsize=9)
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[1].tick_params(labelsize=8)
        ymax2 = np.nanmax(np.concatenate([rain, snow]))
        ymax2 = ymax2 if ymax2 > 0 else 1.0
        axes[1].set_ylim(0, ymax2 * 1.15)

        # --- Time axis format ---
        span_days = (times[-1] - times[0]).days
        if span_days <= 3:
            fmt     = mdates.DateFormatter("%b %d %H:%M")
            locator = mdates.HourLocator(interval=6)
        elif span_days <= 60:
            fmt     = mdates.DateFormatter("%b %d")
            locator = mdates.DayLocator()
        else:
            fmt     = mdates.DateFormatter("%Y-%m")
            locator = mdates.MonthLocator()

        axes[1].xaxis.set_major_formatter(fmt)
        axes[1].xaxis.set_major_locator(locator)
        axes[1].set_xlabel("Time (UTC)", fontsize=9)
        plt.setp(axes[1].xaxis.get_ticklabels(), rotation=30, ha="right", fontsize=8)

        fig.suptitle(title, fontsize=10)
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved to: {filename}")

        if display:
            plt.show()
        else:
            plt.close(fig)

    def plot_wind_timeseries(
        self,
        figsize: tuple = (12, 7),
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the wind time series stored in self.wind_timeseries.

        Produces a figure with three stacked subplots sharing the time axis:
            1. Wind speed (m/s)
            2. U (eastward) and V (northward) components (m/s), on the same axes
            3. Wind direction (degrees, meteorological convention), as scatter to
               avoid misleading line artefacts across the 0°/360° wrap

        Each panel is auto-scaled to the data range of that variable.

        Args:
            figsize: Figure size as (width, height) in inches (default: (12, 7)).
            display: Show the figure on screen (default: True).
            filename: If provided, save the figure to this path (e.g. "wind.png").
                      Any format supported by matplotlib is accepted.  Default: None.

        Raises:
            RuntimeError: If get_wind_timeseries() has not been called yet.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.wind_timeseries is None:
            raise RuntimeError(
                "No wind data available. Call get_wind_timeseries() first."
            )

        ds    = self.wind_timeseries
        times = pd.to_datetime(ds["time"].values)

        lat_str = f"{float(ds.coords['lat']):.2f}°N"
        lon_str = f"{float(ds.coords['lon']):.2f}°E"
        title = (
            f"ERA5 10-m Wind  |  ({lat_str}, {lon_str})  "
            f"|  {times[0].date()} – {times[-1].date()}"
        )

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # --- Panel 1: Wind speed ---
        speed  = ds["wind_speed_ms"].values
        axes[0].plot(times, speed, color="tab:blue", linewidth=0.8)
        axes[0].set_ylabel("Wind speed (m/s)", fontsize=9)
        axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[0].tick_params(labelsize=8)
        axes[0].set_ylim(0, np.nanmax(speed) * 1.1)

        # --- Panel 2: U and V components ---
        u = ds["u10_ms"].values
        v = ds["v10_ms"].values
        axes[1].plot(times, u, color="tab:green",  linewidth=0.8, label="U (eastward)")
        axes[1].plot(times, v, color="tab:orange", linewidth=0.8, label="V (northward)")
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
        axes[1].set_ylabel("Wind component (m/s)", fontsize=9)
        axes[1].legend(fontsize=8, loc="upper right")
        axes[1].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[1].tick_params(labelsize=8)
        uv_abs = np.nanmax(np.abs(np.concatenate([u, v])))
        axes[1].set_ylim(-uv_abs * 1.1, uv_abs * 1.1)

        # --- Panel 3: Wind direction (scatter to avoid wrap artefacts) ---
        axes[2].scatter(times, ds["wind_dir_deg"].values,
                        s=1, color="tab:red", alpha=0.6)
        axes[2].set_ylabel("Wind direction (°)", fontsize=9)
        axes[2].set_ylim(-5, 365)
        axes[2].set_yticks([0, 90, 180, 270, 360])
        axes[2].set_yticklabels(["N (0°)", "E (90°)", "S (180°)", "W (270°)", "N (360°)"])
        axes[2].grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        axes[2].tick_params(labelsize=8)

        # --- Time axis format (adapts to span) ---
        span_days = (times[-1] - times[0]).days
        if span_days <= 3:
            fmt    = mdates.DateFormatter("%b %d %H:%M")
            locator = mdates.HourLocator(interval=6)
        elif span_days <= 60:
            fmt    = mdates.DateFormatter("%b %d")
            locator = mdates.DayLocator()
        else:
            fmt    = mdates.DateFormatter("%Y-%m")
            locator = mdates.MonthLocator()

        axes[2].xaxis.set_major_formatter(fmt)
        axes[2].xaxis.set_major_locator(locator)
        axes[2].set_xlabel("Time (UTC)", fontsize=9)
        plt.setp(axes[2].xaxis.get_ticklabels(), rotation=30, ha="right", fontsize=8)

        fig.suptitle(title, fontsize=10)
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved to: {filename}")

        if display:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime

    lat, lon = 42.40382, -70.12225

    # ---- Open-Meteo backend (default, no registration) --------------------
    era5 = ERA5(verbose=True)   # source="open_meteo" by default

    ds_wind = era5.get_wind_timeseries(
        lat=lat, lon=lon,
        start_dt=datetime(2026, 1, 1),
        end_dt=datetime(2026, 3, 20),
    )
    print("\nWind dataset:")
    print(ds_wind)
    era5.plot_wind_timeseries()

    # ---- Precipitation (same ERA5 object, same backend) -------------------
    ds_precip = era5.get_precipitation_timeseries(
        lat=lat, lon=lon,
        start_dt=datetime(2026, 1, 1),
        end_dt=datetime(2026, 3, 20),
    )
    print("\nPrecipitation dataset:")
    print(ds_precip)
    era5.plot_precipitation_timeseries()

    # ---- CDS backend (requires ~/.cdsapirc) --------------------------------
    # era5_cds = ERA5(source="cds")
    # era5_cds.get_wind_timeseries(lat=lat, lon=lon,
    #     start_dt=datetime(2015, 8, 1), end_dt=datetime(2015, 8, 7))
    # era5_cds.plot_wind_timeseries()

    # ---- Combine with NECOFS (hourly ocean profiles) -----------------------
    # from NECOFS import NECOFS
    # necofs = NECOFS()
    # ds_ocean = necofs.get_vertical_profiles(lat=lat, lon=lon,
    #     start_dt=datetime(2015, 8, 1), end_dt=datetime(2015, 8, 7))
    #
    # # Align ERA5 times to NECOFS times (both hourly, nearest neighbour)
    # ds_wind_aligned = era5.wind_timeseries.sel(
    #     time=ds_ocean.time, method="nearest")
    # ds_combined = xr.merge([ds_ocean, ds_wind_aligned])
    # print(ds_combined)

    # ---- Combine with HMD (1-minute acoustic noise) ------------------------
    # hmd = HMD()
    # hmd.load_nc_files("path/to/deployment")
    # ds_wind_1min = era5.wind_timeseries.reindex(
    #     time=hmd.ds.time, method="nearest", tolerance="1h")
    # ds_combined = xr.merge([hmd.ds, ds_wind_1min])
