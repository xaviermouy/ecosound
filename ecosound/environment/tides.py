# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
NOAA CO-OPS Tide Data Fetcher
Class-based interface for retrieving water level and tidal index data at a
point location from the NOAA Center for Operational Oceanographic Products
and Services (CO-OPS) API.

No registration or API key is required.  Data is fetched via simple HTTP GET
requests and returned as xr.Dataset with the same (time, lat, lon) coordinate
structure as NECOFS, ERA5, and HMD outputs, enabling straightforward merging.

Two products are supported:
    - "predictions" (default): astronomical tide predictions (clean, no surge).
      Best for computing tidal phase and time-since-high-tide.
    - "water_level": observed water levels (includes meteorological surge).
      Best for comparing with actual sea surface conditions.

The tidal index (time elapsed since last high tide, tidal phase) is computed
automatically by get_water_level() and merged directly into the returned
xr.Dataset.  Detected high- and low-tide times are stored separately as
class attributes (self.high_tide_times, self.low_tide_times) because they
have variable length and cannot share the water-level time dimension.
"""

from __future__ import annotations
from typing import Optional, Union
from datetime import datetime
import json
from urllib.request import urlopen
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import xarray as xr


class Tides:
    """
    Class for fetching NOAA CO-OPS water level and computing tidal indices.

    Data is fetched at 6-minute resolution from the nearest NOAA tide gauge
    station to the requested location.  Results are returned as xr.Dataset
    compatible with NECOFS, ERA5, and HMD outputs.

    By default, tidal index variables (time_since_high_tide_h, tidal_phase)
    are computed automatically and included in the returned dataset.
    Detected high- and low-tide event times are stored as class attributes.

    Args:
        verbose: Print progress messages (default: True).

    Attributes:
        water_level (xr.Dataset | None): Most recent water level time series
            from get_water_level().  Dimension: time.  Coordinates: time, lat,
            lon.  Includes tidal index variables when compute_tidal_index=True.
            None until first call.
        high_tide_times (np.ndarray | None): datetime64 array of detected
            high-tide peak times.  None until get_water_level() is called with
            compute_tidal_index=True.
        high_tide_levels (np.ndarray | None): Water level (m) at each
            high-tide peak (same order as high_tide_times).
        low_tide_times (np.ndarray | None): datetime64 array of detected
            low-tide trough times.
        low_tide_levels (np.ndarray | None): Water level (m) at each
            low-tide trough (same order as low_tide_times).

    Examples:
        >>> from ecosound.environment import Tides
        >>> tides = Tides()

        # Fetch by explicit time range (tidal index included by default)
        >>> ds = tides.get_water_level(lat=42.5, lon=-70.0,
        ...     start_dt="2015-08-01", end_dt="2015-08-31")
        >>> print(ds)                      # contains water_level_m, time_since_high_tide_h, tidal_phase
        >>> print(tides.high_tide_times)   # numpy array of high-tide datetimes
        >>> tides.plot_water_level()

        # Fetch using another xr.Dataset's time axis (e.g. from HMD or NECOFS)
        >>> tides.get_water_level(lat=42.5, lon=-70.0, time_ref=hmd.ds)

        # Merge with HMD acoustic data (reindex to 1-minute HMD time axis)
        >>> ds_aligned = tides.water_level.reindex(
        ...     time=hmd.ds.time, method="nearest", tolerance="10min")
        >>> ds_combined = xr.merge([hmd.ds, ds_aligned])
    """

    COOPS_API_URL   = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
    COOPS_MDAPI_URL = "https://api.tidesandcurrents.noaa.gov/mdapi/prod/webapi"

    # Dominant M2 semidiurnal tidal period (hours)
    TIDAL_PERIOD_H = 12.42

    # NOAA CO-OPS maximum date range per request for 6-minute data (days)
    _MAX_DAYS_PER_REQUEST = 31

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.water_level:     Optional[xr.Dataset]  = None
        self.high_tide_times: Optional[np.ndarray]  = None
        self.high_tide_levels: Optional[np.ndarray] = None
        self.low_tide_times:  Optional[np.ndarray]  = None
        self.low_tide_levels: Optional[np.ndarray]  = None
        self._stations: Optional[pd.DataFrame] = None  # cached station list

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _fetch_json(self, url: str) -> dict:
        """HTTP GET and parse JSON.  Uses requests if available, else urllib."""
        try:
            import requests
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except ImportError:
            with urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode())

    def _load_stations(self) -> pd.DataFrame:
        """Fetch and cache the NOAA water-level station list."""
        if self._stations is not None:
            return self._stations
        self._log("  Fetching NOAA tide gauge station list...")
        url  = f"{self.COOPS_MDAPI_URL}/stations.json?type=waterlevels"
        data = self._fetch_json(url)
        self._stations = pd.DataFrame([
            {
                "id":   s["id"],
                "name": s["name"],
                "lat":  float(s["lat"]),
                "lon":  float(s["lng"]),
            }
            for s in data["stations"]
        ])
        self._log(f"  {len(self._stations)} water-level stations loaded.")
        return self._stations

    def _fetch_chunk(
        self,
        station_id: str,
        t0: pd.Timestamp,
        t1: pd.Timestamp,
        product: str,
        datum: str,
    ) -> list[dict]:
        """Fetch one ≤31-day chunk from NOAA CO-OPS API and return records list."""
        params = {
            "station":    station_id,
            "begin_date": t0.strftime("%Y%m%d"),
            "end_date":   t1.strftime("%Y%m%d"),
            "product":    product,
            "datum":      datum,
            "time_zone":  "GMT",
            "units":      "metric",
            "format":     "json",
            "interval":   "6",    # 6-minute resolution
        }
        url  = f"{self.COOPS_API_URL}?{urlencode(params)}"
        data = self._fetch_json(url)

        if "error" in data:
            raise RuntimeError(
                f"NOAA CO-OPS API error for {t0.date()}–{t1.date()}: "
                f"{data['error']['message']}"
            )
        key = "predictions" if product == "predictions" else "data"
        return data.get(key, [])

    def _detect_peaks(
        self, wl: np.ndarray, times: pd.DatetimeIndex
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect high-tide and low-tide indices using scipy.signal.find_peaks.

        Args:
            wl:    Water level array.
            times: Corresponding DatetimeIndex.

        Returns:
            (high_idx, low_idx) arrays of integer indices into wl/times.
        """
        from scipy.signal import find_peaks

        dt_h     = (times[1] - times[0]).total_seconds() / 3600.0
        min_dist = int(self.TIDAL_PERIOD_H * 0.7 / dt_h)

        high_idx, _ = find_peaks( wl, distance=min_dist, prominence=0.05)
        low_idx,  _ = find_peaks(-wl, distance=min_dist, prominence=0.05)
        return high_idx, low_idx

    def _get_tidal_index(
        self, wl: np.ndarray, times: pd.DatetimeIndex
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute tidal index arrays and detect high/low tide events.

        Args:
            wl:    Water level array.
            times: Corresponding DatetimeIndex (same length as wl).

        Returns:
            Tuple of:
                time_since_high_tide_h (np.ndarray): Hours since last high tide.
                tidal_phase            (np.ndarray): Fractional tidal cycle [0, 1).
                high_tide_times        (np.ndarray): datetime64 of high-tide peaks.
                high_tide_levels       (np.ndarray): Water level (m) at each peak.
                low_tide_times         (np.ndarray): datetime64 of low-tide troughs.
                low_tide_levels        (np.ndarray): Water level (m) at each trough.
        """
        high_idx, low_idx = self._detect_peaks(wl, times)

        high_tide_times  = times[high_idx].values
        high_tide_levels = wl[high_idx]
        low_tide_times   = times[low_idx].values
        low_tide_levels  = wl[low_idx]

        self._log(
            f"  Detected {len(high_idx)} high tides and "
            f"{len(low_idx)} low tides."
        )

        # For each timestamp, find the most recent preceding high tide
        time_arr    = times.values
        time_since  = np.full(len(time_arr), np.nan)
        tidal_phase = np.full(len(time_arr), np.nan)

        for i, t in enumerate(time_arr):
            preceding = high_tide_times[high_tide_times <= t]
            if len(preceding) == 0:
                continue
            last_peak    = preceding[-1]
            dt_h_val     = (t - last_peak) / np.timedelta64(1, "h")
            time_since[i]  = dt_h_val
            tidal_phase[i] = (dt_h_val % self.TIDAL_PERIOD_H) / self.TIDAL_PERIOD_H

        return (
            time_since, tidal_phase,
            high_tide_times, high_tide_levels,
            low_tide_times, low_tide_levels,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def find_nearest_station(
        self, lat: float, lon: float
    ) -> tuple[str, str, float, float]:
        """
        Find the nearest NOAA water-level gauge to a given location.

        Args:
            lat: Latitude (decimal degrees).
            lon: Longitude (decimal degrees, negative west).

        Returns:
            Tuple (station_id, station_name, station_lat, station_lon).
        """
        df   = self._load_stations()
        dist = np.sqrt((df["lat"].values - lat) ** 2 + (df["lon"].values - lon) ** 2)
        idx  = int(np.argmin(dist))
        row  = df.iloc[idx]
        return row["id"], row["name"], float(row["lat"]), float(row["lon"])

    def get_water_level(
        self,
        lat: float,
        lon: float,
        start_dt: Union[datetime, pd.Timestamp, str, None] = None,
        end_dt: Union[datetime, pd.Timestamp, str, None] = None,
        time_ref: Optional[xr.Dataset] = None,
        product: str = "predictions",
        datum: str = "MLLW",
        compute_tidal_index: bool = True,
    ) -> xr.Dataset:
        """
        Fetch NOAA CO-OPS water level at the nearest tide gauge.

        The time range can be specified either explicitly with start_dt/end_dt,
        or implicitly by passing another xr.Dataset via time_ref (its first and
        last time coordinate values are used as the range).

        Data is fetched in ≤31-day chunks (NOAA API limit for 6-minute data)
        and concatenated automatically.

        When compute_tidal_index=True (default), the tidal index variables
        time_since_high_tide_h and tidal_phase are computed and added directly
        to the returned dataset.  Detected high- and low-tide event times and
        levels are stored in self.high_tide_times, self.high_tide_levels,
        self.low_tide_times, and self.low_tide_levels.

        The result is always stored in self.water_level and always returned:

            tides.get_water_level(lat, lon, start_dt="2015-08-01", end_dt="2015-08-31")
            ds = tides.get_water_level(lat, lon, time_ref=hmd.ds)

        Args:
            lat: Latitude (decimal degrees, WGS84).
            lon: Longitude (decimal degrees, WGS84, negative west).
            start_dt: Start of the time range (inclusive).  Accepts datetime,
                      pd.Timestamp, or ISO 8601 string.  Ignored if time_ref
                      is provided.
            end_dt: End of the time range (inclusive).  Ignored if time_ref
                    is provided.
            time_ref: Optional xr.Dataset whose time coordinate defines the
                      fetch range.  When provided, start_dt and end_dt are
                      derived from time_ref.time.values[0] and [-1].
            product: NOAA product — "predictions" (default, astronomical tides)
                     or "water_level" (observed, includes surge).
            datum: Vertical datum (default "MLLW").  Other options: "MSL",
                   "MHW", "NAVD", etc.
            compute_tidal_index: If True (default), compute and merge tidal
                                 index variables into the returned dataset.
                                 Requires scipy.  Set to False to skip.

        Returns:
            xr.Dataset with dimension time, also stored in self.water_level:
                Data variables (always present):
                    - water_level_m          (time): Water level (m above datum)
                Data variables (when compute_tidal_index=True):
                    - time_since_high_tide_h (time): Hours since last high tide.
                      NaN before the first detected high tide in the record.
                    - tidal_phase            (time): Fractional tidal cycle [0,1).
                      0 = high tide, ~0.5 = low tide.  NaN before first peak.
                Coordinates:
                    - time : datetime64 UTC, 6-minute resolution
                    - lat  : station latitude (scalar)
                    - lon  : station longitude (scalar)
                Dataset.attrs:
                    - requested_lat/lon, station_id, station_name,
                      station_lat, station_lon, product, datum, source,
                      temporal_resolution, tidal_period_h (if tidal index),
                      n_high_tides, n_low_tides (if tidal index)

        Side effects (when compute_tidal_index=True):
            - self.high_tide_times  set to numpy datetime64 array of high-tide peaks
            - self.high_tide_levels set to numpy float array of water levels at peaks
            - self.low_tide_times   set to numpy datetime64 array of low-tide troughs
            - self.low_tide_levels  set to numpy float array of water levels at troughs

        Raises:
            ValueError: If neither time_ref nor both start_dt/end_dt are given.
            ImportError: If scipy is not installed and compute_tidal_index=True.
            RuntimeError: If the NOAA API returns an error.
        """
        # --- Check scipy early if needed ---
        if compute_tidal_index:
            try:
                import scipy.signal  # noqa: F401
            except ImportError:
                raise ImportError(
                    "scipy is required for compute_tidal_index=True.\n"
                    "Install with: pip install scipy\n"
                    "Or call get_water_level(..., compute_tidal_index=False)."
                )

        # --- Resolve time range ---
        if time_ref is not None:
            t0 = pd.Timestamp(time_ref.time.values[0])
            t1 = pd.Timestamp(time_ref.time.values[-1])
        elif start_dt is not None and end_dt is not None:
            t0 = pd.Timestamp(start_dt)
            t1 = pd.Timestamp(end_dt)
        else:
            raise ValueError(
                "Provide either time_ref or both start_dt and end_dt."
            )

        # --- Find nearest gauge ---
        station_id, station_name, station_lat, station_lon = (
            self.find_nearest_station(lat, lon)
        )
        dist_deg = np.sqrt((station_lat - lat) ** 2 + (station_lon - lon) ** 2)
        self._log(
            f"Nearest gauge: {station_name} (ID {station_id})  "
            f"{station_lat:.3f}°N, {station_lon:.3f}°E  "
            f"(dist ≈ {dist_deg * 111:.1f} km)"
        )
        self._log(
            f"Fetching {product} [{datum}] from {t0.date()} to {t1.date()}..."
        )

        # --- Chunk into ≤31-day blocks ---
        chunk_starts = pd.date_range(
            t0.floor("D"), t1.ceil("D"),
            freq=f"{self._MAX_DAYS_PER_REQUEST}D",
        )
        all_records: list[dict] = []
        for cs in chunk_starts:
            ce = min(cs + pd.Timedelta(days=self._MAX_DAYS_PER_REQUEST - 1), t1.ceil("D"))
            self._log(f"  Chunk {cs.date()} → {ce.date()}...")
            all_records.extend(self._fetch_chunk(station_id, cs, ce, product, datum))

        if not all_records:
            raise RuntimeError("No data returned by NOAA CO-OPS API.")

        # --- Parse and trim to exact requested range ---
        times = pd.to_datetime([r["t"] for r in all_records])
        wl    = np.array([float(r["v"]) for r in all_records], dtype=float)
        mask  = (times >= t0) & (times <= t1)
        times = times[mask]
        wl    = wl[mask]

        self._log(f"  {len(times)} records retrieved.")

        # --- Assemble base dataset ---
        attrs = {
            "requested_lat":       lat,
            "requested_lon":       lon,
            "station_id":          station_id,
            "station_name":        station_name,
            "station_lat":         station_lat,
            "station_lon":         station_lon,
            "product":             product,
            "datum":               datum,
            "source":              "NOAA CO-OPS",
            "temporal_resolution": "6-minute",
        }

        data_vars: dict = {
            "water_level_m": (
                "time", wl,
                {"units": "m", "long_name": f"Water level — {product}", "datum": datum},
            ),
        }

        # --- Optionally compute and merge tidal index ---
        if compute_tidal_index:
            (
                time_since, tidal_phase,
                high_tide_times, high_tide_levels,
                low_tide_times, low_tide_levels,
            ) = self._get_tidal_index(wl, times)

            data_vars["time_since_high_tide_h"] = (
                "time", time_since,
                {
                    "units":     "hours",
                    "long_name": "Time elapsed since last high tide",
                },
            )
            data_vars["tidal_phase"] = (
                "time", tidal_phase,
                {
                    "units":     "1",
                    "long_name": (
                        "Tidal phase (0 = high tide, "
                        "~0.5 = low tide, 1 = next high tide)"
                    ),
                },
            )

            # Store event times/levels as class attributes
            self.high_tide_times  = high_tide_times
            self.high_tide_levels = high_tide_levels
            self.low_tide_times   = low_tide_times
            self.low_tide_levels  = low_tide_levels

            attrs["tidal_period_h"] = self.TIDAL_PERIOD_H
            attrs["n_high_tides"]   = int(len(high_tide_times))
            attrs["n_low_tides"]    = int(len(low_tide_times))

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": times.values,
                "lat":  station_lat,
                "lon":  station_lon,
            },
            attrs=attrs,
        )

        self.water_level = ds
        return ds

    def plot_water_level(
        self,
        figsize: tuple = (12, 4),
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the water level time series with high and low tide markers.

        If get_water_level() was called with compute_tidal_index=True, the
        already-detected high/low tide times stored in self.high_tide_times
        and self.low_tide_times are used directly.  Otherwise, peak detection
        is run on the fly (requires scipy).

        Args:
            figsize: Figure size as (width, height) in inches (default: (12, 4)).
            display: Show the figure on screen (default: True).
            filename: If provided, save the figure to this path.  Default: None.

        Raises:
            ImportError: If scipy is not installed and peaks were not pre-detected.
            RuntimeError: If get_water_level() has not been called yet.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.water_level is None:
            raise RuntimeError(
                "No water level data. Call get_water_level() first."
            )

        ds    = self.water_level
        wl    = ds["water_level_m"].values
        times = pd.to_datetime(ds["time"].values)

        # Use cached peaks if available, otherwise detect on the fly
        if self.high_tide_times is not None:
            high_times  = pd.to_datetime(self.high_tide_times)
            high_levels = self.high_tide_levels
            low_times   = pd.to_datetime(self.low_tide_times)
            low_levels  = self.low_tide_levels
        else:
            try:
                high_idx, low_idx = self._detect_peaks(wl, times)
            except ImportError:
                raise ImportError(
                    "scipy is required for plot_water_level() when "
                    "compute_tidal_index=False was used.\n"
                    "Install with: pip install scipy"
                )
            high_times  = times[high_idx]
            high_levels = wl[high_idx]
            low_times   = times[low_idx]
            low_levels  = wl[low_idx]

        lat_str = f"{float(ds.coords['lat']):.3f}°N"
        lon_str = f"{float(ds.coords['lon']):.3f}°E"
        title = (
            f"NOAA CO-OPS Water Level — {ds.attrs.get('product', '')}  |  "
            f"{ds.attrs.get('station_name', '')} ({lat_str}, {lon_str})  |  "
            f"{times[0].date()} – {times[-1].date()}"
        )

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(times, wl, color="tab:blue", linewidth=0.8, label="Water level")
        ax.scatter(high_times, high_levels, color="tab:red",   s=15, zorder=5,
                   label=f"High tide (n={len(high_times)})")
        ax.scatter(low_times,  low_levels,  color="tab:green", s=15, zorder=5,
                   label=f"Low tide (n={len(low_times)})")
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

        ax.set_ylabel(
            f"Water level (m, {ds.attrs.get('datum', 'MLLW')})", fontsize=9
        )
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=8)

        wl_range = np.nanmax(wl) - np.nanmin(wl)
        margin   = wl_range * 0.1
        ax.set_ylim(np.nanmin(wl) - margin, np.nanmax(wl) + margin)

        # Adaptive time axis
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

        ax.xaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_locator(locator)
        ax.set_xlabel("Time (UTC)", fontsize=9)
        plt.setp(ax.xaxis.get_ticklabels(), rotation=30, ha="right", fontsize=8)

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

    tides = Tides(verbose=True)

    # ---- Fetch (tidal index included by default) -------------------------
    ds = tides.get_water_level(
        lat=lat, lon=lon,
        start_dt=datetime(2026, 1, 1),
        end_dt=datetime(2026, 3, 20),
        product="predictions",   # astronomical tide (clean signal)
        datum="MLLW",
    )
    print("\nWater level + tidal index dataset:")
    print(ds)

    print(f"\nHigh tides detected: {len(tides.high_tide_times)}")
    print("First 5 high tides:")
    for t, h in zip(tides.high_tide_times[:5], tides.high_tide_levels[:5]):
        print(f"  {pd.Timestamp(t)}  {h:.3f} m")

    tides.plot_water_level()

    # ---- Merge with ERA5 wind (aligned to tide 6-min time axis) ----------
    # from ERA5 import ERA5
    # era5 = ERA5()
    # era5.get_wind_timeseries(lat=lat, lon=lon,
    #     start_dt=datetime(2026, 1, 1), end_dt=datetime(2026, 3, 20))
    # ds_wind_aligned = era5.wind_timeseries.reindex(
    #     time=ds.time, method="nearest", tolerance="1h")
    # ds_combined = xr.merge([ds, ds_wind_aligned])
    # print(ds_combined)

    # ---- Use HMD time axis to define fetch range, then merge -------------
    # from HMD import HMD
    # hmd = HMD()
    # hmd.load_nc_files("path/to/deployment")
    # tides.get_water_level(lat=lat, lon=lon, time_ref=hmd.ds)
    # ds_aligned = tides.water_level.reindex(
    #     time=hmd.ds.time, method="nearest", tolerance="10min")
    # ds_combined = xr.merge([hmd.ds, ds_aligned])
    # print(ds_combined)
