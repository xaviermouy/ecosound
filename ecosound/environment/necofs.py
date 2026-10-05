# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
NECOFS (Northeast Coastal Ocean Forecast System) Data Fetcher
Class-based interface for retrieving FVCOM-GOM (GOM3) ocean model data via OPeNDAP.

Provides vertical profiles of temperature, salinity, currents, and sound speed
at a given geographic location and time from the NECOFS GOM3 unstructured-grid
ocean model.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Union
from datetime import datetime
import json
import os

import numpy as np
import pandas as pd
import xarray as xr


class NECOFS:
    """
    Class for fetching FVCOM-GOM (NECOFS GOM3) ocean model data via OPeNDAP.

    NECOFS GOM3 is an unstructured-grid (FVCOM) ocean model covering the
    Gulf of Maine and surrounding waters.  Because FVCOM uses a triangular
    unstructured grid, spatial queries find the nearest grid node (for scalar
    fields) or element center (for velocity fields) using a fast distance search.

    Grid coordinates and bathymetry are downloaded once on the first query and
    cached for subsequent calls.

    Args:
        url: Explicit OPeNDAP URL (overrides *source* if given).
        source: ``"archive"`` (default) for the Seaplan 33 Hindcast v1
             monthly files (1978-2024), ``"hindcast_30yr"`` for the legacy
             aggregated 30-year hindcast (1978-2016), or ``"forecast"``
             for the NECOFS GOM3 operational forecast.
        verbose: Print progress messages (default: True).

    Attributes:
        vertical_profile (xr.Dataset | None): Most recent single vertical profile
            from get_vertical_profile().  Dimensions: sigma_layer.
            Coordinates: time (scalar), lat, lon.  None until first call.
        vertical_profiles (xr.Dataset | None): Collection of vertical profiles
            from get_vertical_profiles().  Dimensions: time, sigma_layer.
            Coordinates: time, lat, lon.  None until first call.

    Examples:
        >>> from ecosound.environment import NECOFS
        >>> from datetime import datetime
        >>> necofs = NECOFS()
        >>> necofs.get_vertical_profile(lat=42.5, lon=-70.0, dt=datetime(2015, 8, 1, 12))
        >>> necofs.plot_vertical_profile()
    """

    # Known OPeNDAP endpoints for NECOFS GOM3
    GOM3_HINDCAST_URL = (
        "http://www.smast.umassd.edu:8080/thredds/dodsC/fvcom/hindcasts/30yr_gom3"
    )
    GOM3_FORECAST_URL = (
        "http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/"
        "Forecasts/NECOFS_GOM3_FORECAST.nc"
    )
    # Seaplan 33 Hindcast Archive — monthly files, 1978-2024
    SEAPLAN_ARCHIVE_BASE = (
        "http://www.smast.umassd.edu:8080/thredds/dodsC/models/fvcom/NECOFS/"
        "Archive/Seaplan_33_Hindcast_v1"
    )

    # Default spatial bounds for the Gulf of Maine region
    GOM_BOUNDS = {
        "lat_min": 40.5, "lat_max": 45.5,
        "lon_min": -71.5, "lon_max": -65.0,
    }

    def __init__(self, url: Optional[str] = None, source: str = "archive",
                 verbose: bool = True):
        """
        Args:
            url: Explicit OPeNDAP URL. Overrides *source* if given.
            source: Which dataset to use when *url* is not provided.
                ``"archive"`` — Seaplan 33 Hindcast v1, monthly files,
                1978-2024 (default).
                ``"hindcast_30yr"`` — legacy 30-year aggregated hindcast
                (1978-2016, lower-resolution grid).
                ``"forecast"`` — NECOFS GOM3 operational forecast.
            verbose: Print progress messages (default True).
        """
        if url is not None:
            self.url = url
            self._use_archive = False
        elif source == "archive":
            self.url = self.SEAPLAN_ARCHIVE_BASE
            self._use_archive = True
        elif source == "hindcast_30yr":
            self.url = self.GOM3_HINDCAST_URL
            self._use_archive = False
        elif source == "forecast":
            self.url = self.GOM3_FORECAST_URL
            self._use_archive = False
        else:
            raise ValueError(
                f"Unknown source {source!r}. "
                "Use 'archive', 'hindcast_30yr', or 'forecast'."
            )
        self.verbose = verbose
        self.vertical_profile: Optional[xr.Dataset] = None   # set by get_vertical_profile()
        self.vertical_profiles: Optional[xr.Dataset] = None   # set by get_vertical_profiles()
        self.current_field: Optional[xr.Dataset] = None        # set by get_current_field()
        self.current_fields: Optional[xr.Dataset] = None       # set by get_current_fields()

        # Cached grid/dataset per URL (populated by _open_dataset)
        self._cache = {}          # url -> {ds, lon_node, lat_node, ...}
        # Active dataset pointers (set by _open_dataset)
        self._ds = None           # xarray Dataset handle (lazy OPeNDAP)
        self._lon_node = None     # Node longitudes  (node,)
        self._lat_node = None     # Node latitudes   (node,)
        self._lon_elem = None     # Element-center longitudes  (nele,)
        self._lat_elem = None     # Element-center latitudes   (nele,)
        self._h = None            # Bathymetric depth at nodes (node,), m positive down
        self._siglay = None       # Sigma-layer centers (siglay, node), range [0, -1]
        self._times = None        # Model time steps as pd.DatetimeIndex
        self._active_url = None   # URL of the currently loaded dataset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _archive_url_for_month(base: str, year: int, month: int) -> str:
        """Build the OPeNDAP URL for a specific month in the Seaplan archive.

        Pre-2017 files:  ``<base>/gom3_YYYYMM.nc``
        2017+    files:  ``<base>/necofs_YYYY/NECOFS_YYYYMM.nc``
        """
        ym = f"{year:04d}{month:02d}"
        if year < 2017:
            return f"{base}/gom3_{ym}.nc"
        return f"{base}/necofs_{year:04d}/NECOFS_{ym}.nc"

    def _open_dataset(self, target_dt: Optional[Union[datetime, pd.Timestamp, str]] = None) -> None:
        """Open the OPeNDAP dataset and cache grid data.

        For archive mode, *target_dt* selects the correct monthly file.
        Grid coordinates are cached per URL so switching between months
        on the same grid is cheap.
        """
        if self._use_archive:
            if target_dt is None:
                raise ValueError(
                    "Archive mode requires a target datetime to select the "
                    "correct monthly file.  Pass target_dt to _open_dataset."
                )
            ts = pd.Timestamp(target_dt)
            url = self._archive_url_for_month(self.url, ts.year, ts.month)
        else:
            url = self.url

        # Already pointing at this URL — nothing to do
        if url == self._active_url:
            return

        # Check cache first
        if url in self._cache:
            c = self._cache[url]
            self._ds = c["ds"]
            self._lon_node = c["lon_node"]
            self._lat_node = c["lat_node"]
            self._lon_elem = c["lon_elem"]
            self._lat_elem = c["lat_elem"]
            self._h = c["h"]
            self._siglay = c["siglay"]
            self._times = c["times"]
            self._active_url = url
            self._log(f"Switched to cached dataset: {url}")
            return

        self._log(f"Connecting to NECOFS GOM3: {url}")
        try:
            ds = xr.open_dataset(
                url, engine="netcdf4", mask_and_scale=True, decode_times=False
            )
        except Exception as exc:
            raise ConnectionError(
                f"Could not open NECOFS dataset at:\n  {url}\n"
                "Check the URL, your network connection, and that netCDF4 is installed.\n"
                f"Original error: {exc}"
            ) from exc

        self._ds = ds
        self._log("Loading grid coordinates...")
        self._lon_node = ds["lon"].values
        self._lat_node = ds["lat"].values
        self._lon_elem = ds["lonc"].values
        self._lat_elem = ds["latc"].values
        self._h = ds["h"].values
        self._siglay = ds["siglay"].values
        self._times = self._decode_fvcom_times()
        self._active_url = url

        # Cache for later reuse
        self._cache[url] = {
            "ds": self._ds,
            "lon_node": self._lon_node,
            "lat_node": self._lat_node,
            "lon_elem": self._lon_elem,
            "lat_elem": self._lat_elem,
            "h": self._h,
            "siglay": self._siglay,
            "times": self._times,
        }

        self._log(
            f"Grid loaded: {len(self._lon_node):,} nodes, "
            f"{len(self._lon_elem):,} elements, "
            f"{len(self._times):,} time steps "
            f"({self._times[0].date()} – {self._times[-1].date()})"
        )

    def _decode_fvcom_times(self) -> pd.DatetimeIndex:
        """
        Decode FVCOM time to a pd.DatetimeIndex.

        FVCOM stores time in a float 'time' variable (days since an epoch given
        in its 'units' attribute, typically MJD: 'days since 1858-11-17').
        A secondary integer variable 'Itime2' stores milliseconds and carries
        the non-standard unit string 'msec since 00:00:00', which xarray cannot
        parse.  We therefore open the dataset with decode_times=False and decode
        only the primary 'time' variable here using cftime.

        Falls back to reconstructing timestamps from Itime (integer days) +
        Itime2 (milliseconds) if cftime decoding also fails.
        """
        time_var = self._ds["time"]
        units = time_var.attrs.get("units", "days since 1858-11-17 00:00:00")
        calendar = time_var.attrs.get("calendar", "gregorian")

        try:
            import cftime
            dates = cftime.num2date(
                time_var.values, units=units, calendar=calendar,
                only_use_cftime_datetimes=False
            )
            return pd.DatetimeIndex([pd.Timestamp(str(d)) for d in dates])
        except Exception:
            pass

        # Fallback: Itime = integer days since MJD epoch, Itime2 = milliseconds
        self._log("Warning: cftime decode failed; reconstructing time from Itime/Itime2.")
        mjd_epoch = pd.Timestamp("1858-11-17")
        itime = self._ds["Itime"].values.astype(int)
        itime2 = self._ds["Itime2"].values.astype(int)
        times = [
            mjd_epoch
            + pd.Timedelta(days=int(d))
            + pd.Timedelta(milliseconds=int(ms))
            for d, ms in zip(itime, itime2)
        ]
        return pd.DatetimeIndex(times)

    def _find_nearest_node(self, lat: float, lon: float) -> int:
        """Return the index of the grid node nearest to (lat, lon)."""
        dist2 = (self._lat_node - lat) ** 2 + (self._lon_node - lon) ** 2
        return int(np.argmin(dist2))

    def _find_nearest_elem(self, lat: float, lon: float) -> int:
        """Return the index of the element center nearest to (lat, lon)."""
        dist2 = (self._lat_elem - lat) ** 2 + (self._lon_elem - lon) ** 2
        return int(np.argmin(dist2))

    def _find_nearest_time(self, dt: Union[datetime, pd.Timestamp, str]) -> int:
        """Return the index of the model time step nearest to dt."""
        target = pd.Timestamp(dt)
        return int(np.argmin(np.abs(self._times - target)))

    @staticmethod
    def _sound_speed_mackenzie(
        T: np.ndarray, S: np.ndarray, D: np.ndarray
    ) -> np.ndarray:
        """
        Compute sound speed using the Mackenzie (1981) empirical formula.

        Valid for: T in [-2, 30] °C, S in [25, 40] PSU, D in [0, 8000] m.

        Args:
            T: Temperature (°C)
            S: Salinity (PSU)
            D: Depth (m, positive down)

        Returns:
            Sound speed (m/s)

        Reference:
            Mackenzie, K.V. (1981). Nine-term equation for the sound speed in
            the oceans. J. Acoust. Soc. Am. 70(3), 807–812.
        """
        return (
            1448.96
            + 4.591 * T
            - 5.304e-2 * T ** 2
            + 2.374e-4 * T ** 3
            + 1.340 * (S - 35)
            + 1.630e-2 * D
            + 1.675e-7 * D ** 2
            - 1.025e-2 * T * (S - 35)
            - 7.139e-13 * T * D ** 3
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_vertical_profile(
        self,
        lat: float,
        lon: float,
        dt: Union[datetime, pd.Timestamp, str],
    ) -> xr.Dataset:
        """
        Extract a vertical profile of temperature, salinity, currents, and
        sound speed at the nearest FVCOM grid node for a given location and time.

        The result is always stored in self.vertical_profile and always returned,
        so the method can be used with or without assignment:

            necofs.get_vertical_profile(lat, lon, dt)            # stored in self.vertical_profile
            ds = necofs.get_vertical_profile(lat, lon, dt)       # also captured locally

        Temperature, salinity, and sound speed are returned at FVCOM sigma-layer
        centers converted to physical depths.  Current components (u, v) come
        from the nearest element center (FVCOM stores velocities at element
        centroids, not nodes).  Sound speed is derived from temperature, salinity,
        and depth using the Mackenzie (1981) empirical formula.

        Args:
            lat: Latitude of the query point (decimal degrees, WGS84).
            lon: Longitude of the query point (decimal degrees, WGS84, negative west).
            dt: Date and time for the profile.  Accepts datetime, pd.Timestamp,
                or ISO 8601 string (e.g. "2015-08-01T12:00").
                The nearest available model time step is used.

        Returns:
            xr.Dataset with dimension sigma_layer (sorted surface to bottom):
                Data variables:
                    - temperature_C (sigma_layer)  : in-situ temperature (°C)
                    - salinity_PSU  (sigma_layer)  : practical salinity (PSU)
                    - u_ms          (sigma_layer)  : eastward current (m/s)
                    - v_ms          (sigma_layer)  : northward current (m/s)
                    - sound_speed_ms(sigma_layer)  : sound speed, Mackenzie 1981 (m/s)
                    - depth_m       (sigma_layer)  : depth below surface (m, positive down)
                    - zeta_m        ()             : sea surface elevation (m)
                Coordinates:
                    - sigma_layer                  : integer layer index (0 = surface)
                    - time                         : model time step (scalar datetime64)
                    - lat                          : nearest node latitude
                    - lon                          : nearest node longitude
                Dataset.attrs:
                    - requested_lat/lon/time, nearest_node_lat/lon/idx,
                      nearest_elem_idx, bathymetry_m, total_depth_m, url

        Raises:
            ConnectionError: If the OPeNDAP dataset cannot be opened.
        """
        self._open_dataset(target_dt=dt)

        # Locate nearest grid node (scalars), element center (velocities), and time
        node_idx = self._find_nearest_node(lat, lon)
        elem_idx = self._find_nearest_elem(lat, lon)
        time_idx = self._find_nearest_time(dt)

        nearest_lat = float(self._lat_node[node_idx])
        nearest_lon = float(self._lon_node[node_idx])
        model_time = self._times[time_idx]

        self._log(
            f"Nearest node: ({nearest_lat:.4f}°N, {nearest_lon:.4f}°E)  "
            f"Model time: {model_time}"
        )

        zeta = float(self._ds["zeta"][time_idx, node_idx].values)
        h = float(self._h[node_idx])
        total_depth = h + zeta

        siglay_node = self._siglay[:, node_idx]
        depths = np.abs(siglay_node) * total_depth
        sort_idx = np.argsort(depths)   # surface → bottom

        temp = self._ds["temp"][time_idx, :, node_idx].values.astype(float)
        salt = self._ds["salinity"][time_idx, :, node_idx].values.astype(float)
        u = self._ds["u"][time_idx, :, elem_idx].values.astype(float)
        v = self._ds["v"][time_idx, :, elem_idx].values.astype(float)
        sound_speed = self._sound_speed_mackenzie(temp, salt, depths)

        n_layers = len(depths)
        ds = xr.Dataset(
            data_vars={
                "temperature_C":  ("sigma_layer", temp[sort_idx],        {"units": "degC",  "long_name": "In-situ temperature"}),
                "salinity_PSU":   ("sigma_layer", salt[sort_idx],        {"units": "PSU",   "long_name": "Practical salinity"}),
                "u_ms":           ("sigma_layer", u[sort_idx],           {"units": "m s-1", "long_name": "Eastward current velocity"}),
                "v_ms":           ("sigma_layer", v[sort_idx],           {"units": "m s-1", "long_name": "Northward current velocity"}),
                "sound_speed_ms": ("sigma_layer", sound_speed[sort_idx], {"units": "m s-1", "long_name": "Sound speed (Mackenzie 1981)"}),
                "depth_m":        ("sigma_layer", depths[sort_idx],      {"units": "m",     "long_name": "Depth below sea surface", "positive": "down"}),
                "zeta_m":         ([], zeta,                             {"units": "m",     "long_name": "Sea surface elevation"}),
            },
            coords={
                "sigma_layer": np.arange(n_layers),
                "time":        np.datetime64(model_time, "ns"),
                "lat":         nearest_lat,
                "lon":         nearest_lon,
            },
            attrs={
                "requested_lat":    lat,
                "requested_lon":    lon,
                "requested_time":   str(pd.Timestamp(dt)),
                "nearest_node_lat": nearest_lat,
                "nearest_node_lon": nearest_lon,
                "nearest_node_idx": int(node_idx),
                "nearest_elem_idx": int(elem_idx),
                "bathymetry_m":     h,
                "total_depth_m":    total_depth,
                "url":              self._active_url or self.url,
                "model":            "NECOFS GOM3 (FVCOM)",
                "sound_speed_ref":  "Mackenzie (1981)",
            },
        )

        self.vertical_profile = ds
        return ds

    def get_vertical_profiles(
        self,
        lat: float,
        lon: float,
        dt: Optional[List[Union[datetime, pd.Timestamp, str]]] = None,
        start_dt: Optional[Union[datetime, pd.Timestamp, str]] = None,
        end_dt: Optional[Union[datetime, pd.Timestamp, str]] = None,
    ) -> xr.Dataset:
        """
        Extract vertical profiles at multiple time steps in a single efficient
        OPeNDAP request.

        All variables (temp, salinity, u, v, zeta) for the target node are
        fetched in one contiguous block per variable — regardless of whether
        discrete times or a range are requested — keeping network round trips
        to a minimum (5 requests total, one per variable, vs. 5×N for N
        repeated calls to get_vertical_profile).

        Provide either a list of discrete datetimes (dt) OR a time range
        (start_dt + end_dt), not both.

        Args:
            lat: Latitude of the query point (decimal degrees, WGS84).
            lon: Longitude of the query point (decimal degrees, WGS84, negative west).
            dt: List of datetimes to extract.  Each is snapped to the nearest
                model time step.  Accepts datetime, pd.Timestamp, or ISO strings.
            start_dt: Start of a time range (inclusive).  Must be paired with end_dt.
            end_dt: End of a time range (inclusive).  Must be paired with start_dt.

        Returns:
            xr.Dataset stored in ``self.vertical_profiles`` with dimensions
            ``(time, sigma_layer)``. Data variables: ``temperature_C``,
            ``salinity_PSU``, ``u_ms``, ``v_ms``, ``sound_speed_ms``,
            ``depth_m`` (varies with time due to sea-surface elevation),
            and ``zeta_m``. Coordinates: ``time``, ``sigma_layer``,
            ``lat``, ``lon``.

        Raises:
            ValueError: If neither dt nor start_dt/end_dt are provided, or both are.
            ConnectionError: If the OPeNDAP dataset cannot be opened.

        Examples:
            Discrete time list::

                ds = necofs.get_vertical_profiles(
                    lat=42.5, lon=-70.0,
                    dt=["2015-08-01T00:00", "2015-08-01T06:00", "2015-08-01T12:00"])

            Time range (all hourly steps between start and end)::

                ds = necofs.get_vertical_profiles(
                    lat=42.5, lon=-70.0,
                    start_dt="2015-08-01", end_dt="2015-08-07")

            Access a single time step::

                ds.sel(time="2015-08-01T06:00", method="nearest")
        """
        if dt is not None and (start_dt is not None or end_dt is not None):
            raise ValueError("Provide either 'dt' or 'start_dt'/'end_dt', not both.")
        if dt is None and (start_dt is None or end_dt is None):
            raise ValueError("Provide either 'dt' (list) or both 'start_dt' and 'end_dt'.")

        # In archive mode, split the request by month so each monthly file
        # is opened separately.  For non-archive mode, fetch everything at once.
        if self._use_archive:
            ds = self._get_vertical_profiles_archive(lat, lon, dt, start_dt, end_dt)
        else:
            # Resolve target_dt for opening the dataset (non-archive ignores it)
            ref_dt = (dt[0] if dt else start_dt)
            self._open_dataset(target_dt=ref_dt)
            ds = self._fetch_profiles_from_open_dataset(lat, lon, dt, start_dt, end_dt)

        self._log(f"Done — {ds.sizes['time']} profiles extracted.")
        self.vertical_profiles = ds
        return ds

    def _get_vertical_profiles_archive(
        self,
        lat: float,
        lon: float,
        dt: Optional[List[Union[datetime, pd.Timestamp, str]]],
        start_dt, end_dt,
    ) -> xr.Dataset:
        """Fetch profiles across monthly archive files, concatenating results."""
        from itertools import groupby

        if dt is not None:
            timestamps = sorted(pd.Timestamp(t) for t in dt)
        else:
            # Generate monthly boundaries between start and end
            t0 = pd.Timestamp(start_dt)
            t1 = pd.Timestamp(end_dt)
            if t1 < t0:
                t0, t1 = t1, t0
            timestamps = None  # will use start/end per month

        chunks = []

        if timestamps is not None:
            # Group discrete timestamps by (year, month)
            for (yr, mo), grp in groupby(timestamps, key=lambda t: (t.year, t.month)):
                month_dts = list(grp)
                self._open_dataset(target_dt=month_dts[0])
                chunk = self._fetch_profiles_from_open_dataset(
                    lat, lon, dt=[str(t) for t in month_dts],
                    start_dt=None, end_dt=None,
                )
                chunks.append(chunk)
        else:
            # Walk month by month through the range
            cur = pd.Timestamp(start_dt).to_period("M")
            end_period = pd.Timestamp(end_dt).to_period("M")
            t0_ts = pd.Timestamp(start_dt)
            t1_ts = pd.Timestamp(end_dt)
            while cur <= end_period:
                month_start = max(t0_ts, cur.start_time)
                month_end = min(t1_ts, cur.end_time)
                self._open_dataset(target_dt=month_start)
                chunk = self._fetch_profiles_from_open_dataset(
                    lat, lon, dt=None,
                    start_dt=str(month_start), end_dt=str(month_end),
                )
                chunks.append(chunk)
                cur += 1

        if len(chunks) == 1:
            return chunks[0]
        return xr.concat(chunks, dim="time")

    def _fetch_profiles_from_open_dataset(
        self,
        lat: float,
        lon: float,
        dt: Optional[List[Union[datetime, pd.Timestamp, str]]],
        start_dt, end_dt,
    ) -> xr.Dataset:
        """Core profile extraction from the currently open dataset."""
        node_idx = self._find_nearest_node(lat, lon)
        elem_idx = self._find_nearest_elem(lat, lon)
        nearest_lat = float(self._lat_node[node_idx])
        nearest_lon = float(self._lon_node[node_idx])

        # Resolve requested time steps to model time indices
        if dt is not None:
            time_indices = sorted({self._find_nearest_time(t) for t in dt})
        else:
            t0_idx = self._find_nearest_time(start_dt)
            t1_idx = self._find_nearest_time(end_dt)
            if t1_idx < t0_idx:
                t0_idx, t1_idx = t1_idx, t0_idx
            time_indices = list(range(t0_idx, t1_idx + 1))

        n_steps = len(time_indices)
        self._log(
            f"Fetching {n_steps} profiles at nearest node "
            f"({nearest_lat:.4f}°N, {nearest_lon:.4f}°E) "
            f"[{self._times[time_indices[0]]} – {self._times[time_indices[-1]]}]"
        )

        # Fetch a contiguous block per variable — one OPeNDAP request each
        t_min, t_max = time_indices[0], time_indices[-1]
        t_slice = slice(t_min, t_max + 1)
        local_idx = [i - t_min for i in time_indices]

        zeta_block = self._ds["zeta"][t_slice, node_idx].values       # (block,)
        temp_block = self._ds["temp"][t_slice, :, node_idx].values     # (block, siglay)
        salt_block = self._ds["salinity"][t_slice, :, node_idx].values
        u_block    = self._ds["u"][t_slice, :, elem_idx].values
        v_block    = self._ds["v"][t_slice, :, elem_idx].values

        siglay_node = self._siglay[:, node_idx]
        h = float(self._h[node_idx])
        n_siglay = len(siglay_node)

        # Pre-allocate output arrays (time, sigma_layer)
        temp_arr  = np.empty((n_steps, n_siglay))
        salt_arr  = np.empty_like(temp_arr)
        u_arr     = np.empty_like(temp_arr)
        v_arr     = np.empty_like(temp_arr)
        ss_arr    = np.empty_like(temp_arr)
        depth_arr = np.empty_like(temp_arr)
        zeta_arr  = np.empty(n_steps)
        model_times = []

        for k, (abs_idx, loc_idx) in enumerate(zip(time_indices, local_idx)):
            model_times.append(self._times[abs_idx])
            zeta = float(zeta_block[loc_idx])
            total_depth = h + zeta
            depths = np.abs(siglay_node) * total_depth
            sort_idx = np.argsort(depths)

            temp = temp_block[loc_idx].astype(float)
            salt = salt_block[loc_idx].astype(float)
            u    = u_block[loc_idx].astype(float)
            v    = v_block[loc_idx].astype(float)
            ss   = self._sound_speed_mackenzie(temp, salt, depths)

            temp_arr[k]  = temp[sort_idx]
            salt_arr[k]  = salt[sort_idx]
            u_arr[k]     = u[sort_idx]
            v_arr[k]     = v[sort_idx]
            ss_arr[k]    = ss[sort_idx]
            depth_arr[k] = depths[sort_idx]
            zeta_arr[k]  = zeta

        return xr.Dataset(
            data_vars={
                "temperature_C":  (["time", "sigma_layer"], temp_arr,  {"units": "degC",  "long_name": "In-situ temperature"}),
                "salinity_PSU":   (["time", "sigma_layer"], salt_arr,  {"units": "PSU",   "long_name": "Practical salinity"}),
                "u_ms":           (["time", "sigma_layer"], u_arr,     {"units": "m s-1", "long_name": "Eastward current velocity"}),
                "v_ms":           (["time", "sigma_layer"], v_arr,     {"units": "m s-1", "long_name": "Northward current velocity"}),
                "sound_speed_ms": (["time", "sigma_layer"], ss_arr,    {"units": "m s-1", "long_name": "Sound speed (Mackenzie 1981)"}),
                "depth_m":        (["time", "sigma_layer"], depth_arr, {"units": "m",     "long_name": "Depth below sea surface", "positive": "down"}),
                "zeta_m":         (["time"],                zeta_arr,  {"units": "m",     "long_name": "Sea surface elevation"}),
            },
            coords={
                "time":        pd.DatetimeIndex(model_times),
                "sigma_layer": np.arange(n_siglay),
                "lat":         nearest_lat,
                "lon":         nearest_lon,
            },
            attrs={
                "requested_lat":    lat,
                "requested_lon":    lon,
                "nearest_node_lat": nearest_lat,
                "nearest_node_lon": nearest_lon,
                "nearest_node_idx": int(node_idx),
                "nearest_elem_idx": int(elem_idx),
                "bathymetry_m":     h,
                "url":              self._active_url or self.url,
                "model":            "NECOFS GOM3 (FVCOM)",
                "sound_speed_ref":  "Mackenzie (1981)",
            },
        )

    def plot_vertical_profile(
        self,
        figsize: tuple = (14, 6),
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the single vertical profile stored in self.vertical_profile.

        Produces a figure with five subplots (one per variable): temperature,
        salinity, eastward current (u), northward current (v), and sound speed —
        all versus depth (surface at top, seafloor at bottom).  Each x-axis is
        auto-scaled to the data range of that variable.

        Args:
            figsize: Figure size as (width, height) in inches (default: (14, 6)).
            display: Show the figure on screen (default: True).
            filename: If provided, save the figure to this path (e.g. "profile.png").
                      Any format supported by matplotlib is accepted.  Default: None.

        Raises:
            RuntimeError: If get_vertical_profile() has not been called yet.
        """
        import matplotlib.pyplot as plt

        if self.vertical_profile is None:
            raise RuntimeError(
                "No vertical profile available. Call get_vertical_profile() first."
            )

        ds = self.vertical_profile
        depth = ds["depth_m"].values

        lat_str  = f"{float(ds.coords['lat']):.4f}°N"
        lon_str  = f"{float(ds.coords['lon']):.4f}°E"
        time_str = str(pd.Timestamp(ds.coords["time"].values))
        title = f"NECOFS GOM3 Vertical Profile  |  ({lat_str}, {lon_str})  |  {time_str}"

        variables = [
            ("temperature_C",  "Temperature (°C)",       "tab:red"),
            ("salinity_PSU",   "Salinity (PSU)",          "tab:blue"),
            ("u_ms",           "Eastward current (m/s)",  "tab:green"),
            ("v_ms",           "Northward current (m/s)", "tab:orange"),
            ("sound_speed_ms", "Sound speed (m/s)",       "tab:purple"),
        ]

        fig, axes = plt.subplots(1, len(variables), figsize=figsize, sharey=True)

        for ax, (var, xlabel, color) in zip(axes, variables):
            vals = ds[var].values
            ax.plot(vals, depth, color=color, linewidth=1.5)
            ax.set_xlabel(xlabel, fontsize=9)
            ax.invert_yaxis()
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
            ax.tick_params(labelsize=8)
            xmin, xmax = vals.min(), vals.max()
            margin = (xmax - xmin) * 0.05 if xmax != xmin else 0.5
            ax.set_xlim(xmin - margin, xmax + margin)

        axes[0].set_ylabel("Depth (m)", fontsize=9)
        fig.suptitle(title, fontsize=10)
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved to: {filename}")

        if display:
            plt.show()
        else:
            plt.close(fig)

    def plot_vertical_profiles(
        self,
        figsize: tuple = (16, 6),
        cmap: str = "viridis",
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Overlay all vertical profiles stored in self.vertical_profiles.

        Produces a figure with five subplots (one per variable).  Each time
        step is drawn as a separate line; line colour encodes model time using
        the chosen colormap and a shared colorbar on the right.

        Args:
            figsize: Figure size as (width, height) in inches (default: (16, 6)).
            cmap: Matplotlib colormap name for the time axis (default: "viridis").
            display: Show the figure on screen (default: True).
            filename: If provided, save the figure to this path.  Default: None.

        Raises:
            RuntimeError: If get_vertical_profiles() has not been called yet.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import matplotlib.dates as mdates

        if self.vertical_profiles is None:
            raise RuntimeError(
                "No profiles available. Call get_vertical_profiles() first."
            )

        ds = self.vertical_profiles
        times_np = ds["time"].values                          # numpy datetime64 array
        t_nums   = mdates.date2num(pd.to_datetime(times_np)) # float for colormap
        norm     = mcolors.Normalize(vmin=t_nums.min(), vmax=t_nums.max())
        colormap = plt.get_cmap(cmap)

        lat_str = f"{float(ds.coords['lat']):.4f}°N"
        lon_str = f"{float(ds.coords['lon']):.4f}°E"
        t0_str  = str(pd.Timestamp(times_np[0]).date())
        t1_str  = str(pd.Timestamp(times_np[-1]).date())
        title = (
            f"NECOFS GOM3 Vertical Profiles  |  ({lat_str}, {lon_str})  "
            f"|  {t0_str} – {t1_str}  ({len(times_np)} steps)"
        )

        variables = [
            ("temperature_C",  "Temperature (°C)"),
            ("salinity_PSU",   "Salinity (PSU)"),
            ("u_ms",           "Eastward current (m/s)"),
            ("v_ms",           "Northward current (m/s)"),
            ("sound_speed_ms", "Sound speed (m/s)"),
        ]

        fig, axes = plt.subplots(1, len(variables), figsize=figsize, sharey=True)

        for i, t in enumerate(times_np):
            color = colormap(norm(t_nums[i]))
            depth = ds["depth_m"].isel(time=i).values
            sort_idx = np.argsort(depth)
            depth_sorted = depth[sort_idx]
            for ax, (var, _) in zip(axes, variables):
                vals = ds[var].isel(time=i).values[sort_idx]
                ax.plot(vals, depth_sorted, color=color, linewidth=0.8, alpha=0.7)

        for ax, (_, xlabel) in zip(axes, variables):
            ax.set_xlabel(xlabel, fontsize=9)
            ax.invert_yaxis()
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
            ax.tick_params(labelsize=8)

        # Per-panel x limits using all time steps
        for ax, (var, xlabel) in zip(axes, variables):
            all_vals = ds[var].values
            xmin, xmax = np.nanmin(all_vals), np.nanmax(all_vals)
            margin = (xmax - xmin) * 0.05 if xmax != xmin else 0.5
            ax.set_xlim(xmin - margin, xmax + margin)

        axes[0].set_ylabel("Depth (m)", fontsize=9)

        # Colorbar encoding time
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.tolist(), orientation="vertical",
                            pad=0.01, shrink=0.85, aspect=30)
        cbar.set_label("Model time", fontsize=9)
        # Choose date format based on time span
        span_days = (pd.Timestamp(times_np[-1]) - pd.Timestamp(times_np[0])).days
        date_fmt = "%Y-%m-%d" if span_days >= 1 else "%H:%M"
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
        cbar.ax.yaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(cbar.ax.yaxis.get_ticklabels(), fontsize=7, rotation=30, ha="right")

        fig.suptitle(title, fontsize=10)
        plt.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved to: {filename}")

        if display:
            plt.show()
        else:
            plt.close(fig)


    # ------------------------------------------------------------------
    # Horizontal current field methods
    # ------------------------------------------------------------------

    def _find_sigma_layer_for_depth(self, depth_m: float) -> int:
        """
        Find the sigma layer index closest to a target physical depth.

        Uses domain-mean bathymetry so that a single layer index can be fetched
        for the whole grid in one OPeNDAP request.  The actual physical depth
        of the returned layer varies with local bathymetry.

        Args:
            depth_m: Target depth in metres (positive down).

        Returns:
            Integer sigma layer index (0-based).
        """
        mean_h = float(np.mean(self._h))
        target_sigma = min(depth_m / max(mean_h, 0.1), 1.0)
        mean_siglay_abs = np.abs(self._siglay.mean(axis=1))
        layer_idx = int(np.argmin(np.abs(mean_siglay_abs - target_sigma)))
        approx_depth = mean_siglay_abs[layer_idx] * mean_h
        self._log(
            f"Target depth {depth_m:.1f} m -> sigma layer {layer_idx} "
            f"(~{approx_depth:.1f} m at mean bathymetry {mean_h:.0f} m)"
        )
        return layer_idx

    def _interpolate_to_regular_grid(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        u: np.ndarray,
        v: np.ndarray,
        resolution: float,
        bounds: Dict[str, float],
    ) -> tuple:
        """
        Interpolate u, v from unstructured element centres to a regular grid.

        Uses scipy.interpolate.griddata with linear interpolation.  Grid points
        outside the convex hull of the input points are set to NaN.

        Args:
            lons, lats: 1-D arrays of element-centre coordinates.
            u, v: 1-D arrays of current components at element centres.
            resolution: Grid spacing in degrees.
            bounds: dict with lat_min, lat_max, lon_min, lon_max.

        Returns:
            (lon_grid, lat_grid, u_grid, v_grid) — 1-D coord arrays and
            2-D (ny, nx) interpolated fields.
        """
        from scipy.interpolate import griddata

        lon_grid = np.arange(
            bounds["lon_min"], bounds["lon_max"] + resolution / 2, resolution
        )
        lat_grid = np.arange(
            bounds["lat_min"], bounds["lat_max"] + resolution / 2, resolution
        )
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)

        points = np.column_stack([lons, lats])
        u_grid = griddata(points, u, (lon_mesh, lat_mesh), method="linear")
        v_grid = griddata(points, v, (lon_mesh, lat_mesh), method="linear")

        return lon_grid, lat_grid, u_grid, v_grid

    @staticmethod
    def _build_leaflet_velocity_json(
        lon_grid: np.ndarray,
        lat_grid: np.ndarray,
        u_grid: np.ndarray,
        v_grid: np.ndarray,
        ref_time: str = "",
    ) -> list:
        """
        Package one time step of gridded u, v into the GRIB2-JSON structure
        expected by the leaflet-velocity plugin.

        Data is ordered north-to-south, west-to-east (row-major).
        """
        nx = len(lon_grid)
        ny = len(lat_grid)
        dx = round(float(lon_grid[1] - lon_grid[0]), 6)
        dy = round(float(lat_grid[1] - lat_grid[0]), 6)

        u_ns = np.flipud(u_grid)
        v_ns = np.flipud(v_grid)

        def _flat(arr):
            return [
                round(float(x), 4) if np.isfinite(x) else None
                for x in arr.flatten()
            ]

        hdr = {
            "lo1": round(float(lon_grid[0]), 6),
            "la1": round(float(lat_grid[-1]), 6),
            "lo2": round(float(lon_grid[-1]), 6),
            "la2": round(float(lat_grid[0]), 6),
            "dx": dx,
            "dy": dy,
            "nx": nx,
            "ny": ny,
            "refTime": ref_time,
        }
        return [
            {
                "header": {
                    **hdr,
                    "parameterCategory": 2,
                    "parameterNumber": 2,
                    "parameterNumberName": "eastward_current",
                },
                "data": _flat(u_ns),
            },
            {
                "header": {
                    **hdr,
                    "parameterCategory": 2,
                    "parameterNumber": 3,
                    "parameterNumberName": "northward_current",
                },
                "data": _flat(v_ns),
            },
        ]

    def _build_leaflet_html(
        self,
        all_json_data: list,
        times: List[str],
        config: dict,
    ) -> str:
        """
        Generate a self-contained HTML page with Leaflet + leaflet-velocity.

        All time-step data is embedded as JavaScript literals so the file can
        be opened directly from disk without a web server.

        Args:
            all_json_data: List of leaflet-velocity JSON objects (one per frame).
            times: List of human-readable time labels.
            config: dict with center_lat, center_lon, zoom, depth_m,
                    max_velocity, velocity_scale, particle_multiplier.

        Returns:
            Complete HTML string.
        """
        data_js = json.dumps(all_json_data, separators=(",", ":"))
        times_js = json.dumps(times)
        config_js = json.dumps(config)

        html = _LEAFLET_HTML_TEMPLATE
        html = html.replace("__ALL_DATA__", data_js)
        html = html.replace("__TIMES__", times_js)
        html = html.replace("__CONFIG__", config_js)
        return html

    # ------------------------------------------------------------------
    # Public current-field methods
    # ------------------------------------------------------------------

    def get_current_field(
        self,
        depth_m: float,
        dt: Union[datetime, pd.Timestamp, str],
        bounds: Optional[Dict[str, float]] = None,
    ) -> xr.Dataset:
        """
        Extract a 2-D horizontal current field at one time step.

        Fetches u and v at the sigma layer nearest to the requested depth
        for all FVCOM element centres (optionally filtered by spatial bounds).

        The result is stored in ``self.current_field`` and returned.

        Args:
            depth_m: Target depth in metres (positive down).  The nearest
                     sigma layer is selected using domain-mean bathymetry.
            dt: Date/time of the snapshot.  The nearest model time step is used.
            bounds: Optional spatial subset — dict with ``lat_min``, ``lat_max``,
                    ``lon_min``, ``lon_max``.  Defaults to the full domain.

        Returns:
            xr.Dataset with dimension ``nele``:
                - u_ms, v_ms, speed_ms  (nele)
                - lon_elem, lat_elem    (nele, as coordinates)
                - time                  (scalar coordinate)
        """
        self._open_dataset(target_dt=dt)

        time_idx = self._find_nearest_time(dt)
        model_time = self._times[time_idx]
        layer_idx = self._find_sigma_layer_for_depth(depth_m)

        self._log(f"Fetching current field at {model_time}...")

        u = self._ds["u"][time_idx, layer_idx, :].values.astype(float)
        v = self._ds["v"][time_idx, layer_idx, :].values.astype(float)
        lons = self._lon_elem.copy()
        lats = self._lat_elem.copy()

        if bounds is not None:
            mask = (
                (lats >= bounds["lat_min"])
                & (lats <= bounds["lat_max"])
                & (lons >= bounds["lon_min"])
                & (lons <= bounds["lon_max"])
            )
            u, v, lons, lats = u[mask], v[mask], lons[mask], lats[mask]

        speed = np.sqrt(u ** 2 + v ** 2)

        ds = xr.Dataset(
            data_vars={
                "u_ms": ("nele", u, {"units": "m s-1", "long_name": "Eastward current velocity"}),
                "v_ms": ("nele", v, {"units": "m s-1", "long_name": "Northward current velocity"}),
                "speed_ms": ("nele", speed, {"units": "m s-1", "long_name": "Current speed"}),
            },
            coords={
                "lon_elem": ("nele", lons),
                "lat_elem": ("nele", lats),
                "time": np.datetime64(model_time, "ns"),
            },
            attrs={
                "depth_m": depth_m,
                "sigma_layer_idx": layer_idx,
                "model_time": str(model_time),
                "url": self._active_url or self.url,
                "model": "NECOFS GOM3 (FVCOM)",
            },
        )

        self.current_field = ds
        return ds

    def get_current_fields(
        self,
        depth_m: float,
        start_dt: Union[datetime, pd.Timestamp, str],
        end_dt: Union[datetime, pd.Timestamp, str],
        bounds: Optional[Dict[str, float]] = None,
        stride: int = 1,
    ) -> xr.Dataset:
        """
        Extract 2-D current fields over a time range.

        Fetches u and v at the sigma layer nearest to the requested depth
        for all element centres and all time steps in the range (with optional
        stride).  Data is downloaded in a single contiguous OPeNDAP request
        per variable for efficiency.

        The result is stored in ``self.current_fields`` and returned.

        Args:
            depth_m: Target depth in metres (positive down).
            start_dt: Start of the time range (inclusive, snapped to nearest step).
            end_dt: End of the time range (inclusive, snapped to nearest step).
            bounds: Optional spatial subset dict.
            stride: Take every *stride*-th time step (default 1 = all steps).

        Returns:
            xr.Dataset with dimensions ``(time, nele)``:
                - u_ms, v_ms, speed_ms  (time, nele)
                - lon_elem, lat_elem    (nele, as coordinates)
                - time                  (coordinate)
        """
        if self._use_archive:
            ds = self._get_current_fields_archive(
                depth_m, start_dt, end_dt, bounds, stride,
            )
        else:
            self._open_dataset(target_dt=start_dt)
            ds = self._fetch_current_fields_from_open_dataset(
                depth_m, start_dt, end_dt, bounds, stride,
            )

        self._log(f"Done — {ds.sizes['time']} current fields extracted.")
        self.current_fields = ds
        return ds

    def _get_current_fields_archive(
        self, depth_m, start_dt, end_dt, bounds, stride,
    ) -> xr.Dataset:
        """Fetch current fields across monthly archive files."""
        t0_ts = pd.Timestamp(start_dt)
        t1_ts = pd.Timestamp(end_dt)
        if t1_ts < t0_ts:
            t0_ts, t1_ts = t1_ts, t0_ts

        chunks = []
        cur = t0_ts.to_period("M")
        end_period = t1_ts.to_period("M")
        while cur <= end_period:
            month_start = max(t0_ts, cur.start_time)
            month_end = min(t1_ts, cur.end_time)
            self._open_dataset(target_dt=month_start)
            chunk = self._fetch_current_fields_from_open_dataset(
                depth_m, str(month_start), str(month_end), bounds, stride,
            )
            chunks.append(chunk)
            cur += 1

        if len(chunks) == 1:
            return chunks[0]
        return xr.concat(chunks, dim="time")

    def _fetch_current_fields_from_open_dataset(
        self, depth_m, start_dt, end_dt, bounds, stride,
    ) -> xr.Dataset:
        """Core current field extraction from the currently open dataset."""
        t0 = self._find_nearest_time(start_dt)
        t1 = self._find_nearest_time(end_dt)
        if t1 < t0:
            t0, t1 = t1, t0

        layer_idx = self._find_sigma_layer_for_depth(depth_m)
        time_indices = list(range(t0, t1 + 1, stride))
        n_steps = len(time_indices)

        self._log(
            f"Fetching {n_steps} current fields "
            f"[{self._times[t0]} to {self._times[t1]}, stride={stride}]..."
        )

        # Two OPeNDAP requests total (one per variable, contiguous with stride)
        u_block = self._ds["u"][t0 : t1 + 1 : stride, layer_idx, :].values.astype(float)
        v_block = self._ds["v"][t0 : t1 + 1 : stride, layer_idx, :].values.astype(float)
        lons = self._lon_elem.copy()
        lats = self._lat_elem.copy()

        if bounds is not None:
            mask = (
                (lats >= bounds["lat_min"])
                & (lats <= bounds["lat_max"])
                & (lons >= bounds["lon_min"])
                & (lons <= bounds["lon_max"])
            )
            u_block = u_block[:, mask]
            v_block = v_block[:, mask]
            lons = lons[mask]
            lats = lats[mask]

        speed_block = np.sqrt(u_block ** 2 + v_block ** 2)
        model_times = self._times[time_indices]

        return xr.Dataset(
            data_vars={
                "u_ms": (["time", "nele"], u_block, {"units": "m s-1", "long_name": "Eastward current velocity"}),
                "v_ms": (["time", "nele"], v_block, {"units": "m s-1", "long_name": "Northward current velocity"}),
                "speed_ms": (["time", "nele"], speed_block, {"units": "m s-1", "long_name": "Current speed"}),
            },
            coords={
                "time": pd.DatetimeIndex(model_times),
                "lon_elem": ("nele", lons),
                "lat_elem": ("nele", lats),
            },
            attrs={
                "depth_m": depth_m,
                "sigma_layer_idx": layer_idx,
                "stride": stride,
                "url": self._active_url or self.url,
                "model": "NECOFS GOM3 (FVCOM)",
            },
        )

    def export_currents_html(
        self,
        depth_m: float,
        start_dt: Union[datetime, pd.Timestamp, str],
        end_dt: Union[datetime, pd.Timestamp, str],
        output_path: str,
        resolution: float = 0.05,
        bounds: Optional[Dict[str, float]] = None,
        stride: int = 1,
    ) -> str:
        """
        Full pipeline: fetch currents, interpolate to regular grid, and write
        a standalone HTML file with an animated Windy-style visualisation.

        The HTML uses Leaflet + leaflet-velocity to render animated particles
        that trace the current flow field.  All data is embedded in the file
        so it can be opened directly in a browser from disk.

        Args:
            depth_m: Target depth in metres (positive down).
            start_dt: Start of the time range.
            end_dt: End of the time range.
            output_path: File path for the HTML output (e.g. "currents.html").
            resolution: Regular-grid spacing in degrees (default 0.05 ~ 5 km).
                        Smaller values give finer detail but larger files.
            bounds: Spatial extent dict.  Defaults to ``NECOFS.GOM_BOUNDS``.
            stride: Take every *stride*-th model time step (default 1).

        Returns:
            The *output_path* string (for convenience).
        """
        if bounds is None:
            bounds = dict(self.GOM_BOUNDS)

        # Step 1 — Fetch raw unstructured data
        ds = self.get_current_fields(
            depth_m, start_dt, end_dt, bounds=bounds, stride=stride
        )

        n_steps = ds.sizes["time"]
        lons = ds["lon_elem"].values
        lats = ds["lat_elem"].values

        all_json: list = []
        time_labels: List[str] = []
        max_speed = 0.0

        # Step 2 — Interpolate each frame to a regular grid
        for i in range(n_steps):
            u = ds["u_ms"].isel(time=i).values
            v = ds["v_ms"].isel(time=i).values
            t_str = str(pd.Timestamp(ds["time"].values[i]))

            self._log(f"  Interpolating frame {i + 1}/{n_steps}: {t_str}")

            lon_g, lat_g, u_g, v_g = self._interpolate_to_regular_grid(
                lons, lats, u, v, resolution, bounds
            )

            frame_json = self._build_leaflet_velocity_json(
                lon_g, lat_g, u_g, v_g, ref_time=t_str
            )
            all_json.append(frame_json)
            time_labels.append(t_str)

            spd = float(
                np.nanmax(
                    np.sqrt(np.nan_to_num(u_g) ** 2 + np.nan_to_num(v_g) ** 2)
                )
            )
            if spd > max_speed:
                max_speed = spd

        # Step 3 — Build HTML
        config = {
            "center_lat": (bounds["lat_min"] + bounds["lat_max"]) / 2,
            "center_lon": (bounds["lon_min"] + bounds["lon_max"]) / 2,
            "zoom": 7,
            "depth_m": depth_m,
            "max_velocity": round(float(max_speed) * 1.2, 2) or 0.5,
            "velocity_scale": 0.01,
            "particle_multiplier": 1 / 200,
        }

        html = self._build_leaflet_html(all_json, time_labels, config)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        self._log(f"Written to {output_path} ({size_mb:.1f} MB, {n_steps} frames)")
        return output_path

    def plot_current_field(
        self,
        figsize: tuple = (12, 8),
        quiver_step: int = 5,
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Quick-look matplotlib quiver plot of the current field stored in
        ``self.current_field``.

        Args:
            figsize: Figure size in inches (default (12, 8)).
            quiver_step: Plot every *quiver_step*-th element for readability.
            display: Show the figure interactively (default True).
            filename: If provided, save the figure to this path.

        Raises:
            RuntimeError: If ``get_current_field()`` has not been called yet.
        """
        import matplotlib.pyplot as plt

        if self.current_field is None:
            raise RuntimeError(
                "No current field available. Call get_current_field() first."
            )

        ds = self.current_field
        lons = ds["lon_elem"].values
        lats = ds["lat_elem"].values
        u = ds["u_ms"].values
        v = ds["v_ms"].values
        speed = ds["speed_ms"].values

        fig, ax = plt.subplots(figsize=figsize)

        s = quiver_step
        q = ax.quiver(
            lons[::s], lats[::s], u[::s], v[::s], speed[::s],
            cmap="cividis", scale=3, scale_units="inches", alpha=0.85,
        )
        plt.colorbar(q, ax=ax, label="Current speed (m/s)", shrink=0.8)

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"NECOFS GOM3 Currents | ~{ds.attrs['depth_m']:.0f} m depth | "
            f"{ds.attrs['model_time']}"
        )
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.5)
        plt.tight_layout()

        if filename:
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            self._log(f"Figure saved to: {filename}")
        if display:
            plt.show()
        else:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Custom particle renderer HTML template (no leaflet-velocity dependency)
# ---------------------------------------------------------------------------

_LEAFLET_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NECOFS Ocean Currents</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body { width: 100%; height: 100%; }
  #map { width: 100%; height: 100%; }
  #controls {
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: rgba(255, 255, 255, 0.95); padding: 14px 16px;
    border-radius: 8px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 13px; min-width: 360px; max-width: 400px;
  }
  #time-label {
    font-size: 15px; font-weight: 600; margin-bottom: 6px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  #frame-counter { font-size: 11px; color: #666; margin-bottom: 4px; }
  #time-slider { width: 100%; margin: 2px 0 8px; }
  .btn-row { display: flex; gap: 4px; align-items: center; }
  .btn-row button {
    padding: 5px 12px; border: 1px solid #ccc; border-radius: 4px;
    background: #f8f8f8; cursor: pointer; font-size: 13px;
  }
  .btn-row button:hover { background: #e8e8e8; }
  .btn-row button.active { background: #4a90d9; color: #fff; border-color: #3a7bc8; }
  .section-label {
    font-size: 11px; font-weight: 600; color: #555;
    margin: 10px 0 4px; text-transform: uppercase; letter-spacing: 0.5px;
    border-top: 1px solid #e0e0e0; padding-top: 8px;
  }
  .slider-row { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
  .slider-row label { font-size: 12px; color: #444; min-width: 90px; }
  .slider-row input[type="range"] { flex: 1; }
  .slider-row .val { font-size: 11px; color: #666; min-width: 40px; text-align: right; }
  .palette-row { display: flex; align-items: center; gap: 6px; margin: 4px 0; }
  .palette-row label { font-size: 12px; color: #444; min-width: 90px; }
  .palette-row select { flex: 1; font-size: 12px; padding: 2px 4px; }
  #info { font-size: 11px; color: #888; margin-top: 8px; }
  #map-time {
    position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(0, 0, 0, 0.65); color: #fff;
    padding: 6px 18px; border-radius: 6px; font-size: 18px; font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    pointer-events: none; white-space: nowrap;
    text-shadow: 0 1px 3px rgba(0,0,0,0.5);
  }
  #colorbar {
    position: absolute; bottom: 70px; left: 20px; z-index: 1000;
    background: rgba(255, 255, 255, 0.9); padding: 8px 12px;
    border-radius: 6px; box-shadow: 0 1px 5px rgba(0,0,0,0.2);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 11px; pointer-events: none;
  }
  #colorbar-title { font-weight: 600; margin-bottom: 4px; color: #333; text-align: center; }
  #colorbar-gradient { width: 200px; height: 14px; border: 1px solid #aaa; border-radius: 2px; }
  #colorbar-labels { display: flex; justify-content: space-between; margin-top: 2px; color: #555; }
  #cursor-speed {
    position: absolute; bottom: 30px; right: 20px; z-index: 1000;
    background: rgba(0, 0, 0, 0.65); color: #fff;
    padding: 4px 12px; border-radius: 4px; font-size: 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    pointer-events: none; display: none;
  }
</style>
</head>
<body>
<div id="map"></div>
<div id="map-time">--</div>
<div id="colorbar">
  <div id="colorbar-title">Current speed (m/s)</div>
  <canvas id="colorbar-gradient" width="200" height="14"></canvas>
  <div id="colorbar-labels">
    <span id="cb-min">0</span><span id="cb-mid"></span><span id="cb-max">1.0</span>
  </div>
</div>
<div id="cursor-speed"></div>

<div id="controls">
  <div id="time-label">Loading&hellip;</div>
  <div id="frame-counter"></div>
  <input type="range" id="time-slider" min="0" max="0" value="0" />
  <div class="btn-row">
    <button id="btn-prev" title="Previous time step">&#9664;</button>
    <button id="btn-play" class="active" title="Play / Pause">&#9208; Pause</button>
    <button id="btn-next" title="Next time step">&#9654;</button>
    <button id="btn-speed">1x</button>
  </div>
  <div class="section-label">Display settings</div>
  <div class="slider-row">
    <label>Particles</label>
    <input type="range" id="sl-particles" min="500" max="15000" step="500" value="4000" />
    <span class="val" id="val-particles">4000</span>
  </div>
  <div class="slider-row">
    <label>Trace speed</label>
    <input type="range" id="sl-vscale" min="0.05" max="2.0" step="0.05" value="0.5" />
    <span class="val" id="val-vscale">0.50</span>
  </div>
  <div class="slider-row">
    <label>Max velocity</label>
    <input type="range" id="sl-maxvel" min="0.1" max="3.0" step="0.1" value="1.0" />
    <span class="val" id="val-maxvel">1.0</span>
  </div>
  <div class="slider-row">
    <label>Line width</label>
    <input type="range" id="sl-linewidth" min="0.5" max="4" step="0.5" value="1.5" />
    <span class="val" id="val-linewidth">1.5</span>
  </div>
  <div class="slider-row">
    <label>Trail length</label>
    <input type="range" id="sl-fade" min="0.90" max="0.99" step="0.01" value="0.96" />
    <span class="val" id="val-fade">0.96</span>
  </div>
  <div class="palette-row">
    <label>Color palette</label>
    <select id="sel-palette">
      <option value="ocean">Ocean (blue-cyan-green)</option>
      <option value="thermal">Thermal (blue-red)</option>
      <option value="viridis">Viridis</option>
      <option value="plasma">Plasma</option>
      <option value="white">White (dark bg)</option>
    </select>
  </div>
  <div id="info"></div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// =====================================================================
// Data injected by Python
// =====================================================================
var ALL_DATA = __ALL_DATA__;
var TIMES    = __TIMES__;
var CONFIG   = __CONFIG__;

// =====================================================================
// Color palettes
// =====================================================================
var PALETTES = {
  ocean:   [[0,0,51],[0,51,136],[0,102,204],[0,153,204],[0,204,170],[51,221,119],[136,238,68],[204,255,51]],
  thermal: [[0,0,68],[0,0,170],[0,68,255],[0,170,204],[68,204,68],[170,221,0],[255,204,0],[255,102,0],[255,0,0]],
  viridis: [[68,1,84],[72,39,119],[62,73,137],[49,104,142],[38,130,142],[31,158,137],[108,206,90],[182,222,43],[254,232,37]],
  plasma:  [[13,8,135],[75,3,161],[125,3,168],[168,34,150],[203,70,121],[229,107,93],[248,148,65],[253,195,40],[240,249,33]],
  white:   [[255,255,255],[238,238,238],[221,221,221],[204,204,204],[187,187,187],[170,170,170],[153,153,153]]
};
var currentPalette = "ocean";

// =====================================================================
// Display settings
// =====================================================================
var opts = {
  numParticles: 4000,
  velocityScale: 0.5,
  maxVelocity: CONFIG.max_velocity,
  lineWidth: 1.5,
  fadeOpacity: 0.96,
  maxAge: 200
};

// =====================================================================
// Custom CurrentsLayer — canvas-based particle system on Leaflet
// Particles persist across frame (data) changes.
// =====================================================================
var CurrentsLayer = L.Layer.extend({
  initialize: function (options) {
    this._data = null;     // current GRIB2-JSON data
    this._grid = null;     // parsed { nx, ny, lo1, la1, dx, dy, uArr, vArr }
    this._particles = [];
    this._animId = null;
    this._canvas = null;
    this._ctx = null;
  },

  onAdd: function (map) {
    this._map = map;
    this._canvas = L.DomUtil.create("canvas", "currents-canvas");
    this._canvas.style.position = "absolute";
    this._canvas.style.top = "0";
    this._canvas.style.left = "0";
    this._canvas.style.pointerEvents = "none";
    this._canvas.style.zIndex = "450";
    map.getPanes().overlayPane.appendChild(this._canvas);
    this._ctx = this._canvas.getContext("2d");
    this._resize();
    map.on("move zoom resize", this._reset, this);
    this._animate();
  },

  onRemove: function (map) {
    if (this._animId) cancelAnimationFrame(this._animId);
    map.getPanes().overlayPane.removeChild(this._canvas);
    map.off("move zoom resize", this._reset, this);
  },

  setData: function (data) {
    // Swap velocity field — particles are NOT reset
    this._data = data;
    this._parseGrid(data);
  },

  resetParticles: function () {
    this._particles = [];
  },

  _parseGrid: function (data) {
    if (!data || data.length < 2) { this._grid = null; return; }
    var hU = data[0].header, hV = data[1].header;
    this._grid = {
      nx: hU.nx, ny: hU.ny,
      lo1: hU.lo1, la1: hU.la1,
      dx: hU.dx, dy: hU.dy,
      uArr: data[0].data,
      vArr: data[1].data
    };
  },

  _resize: function () {
    var size = this._map.getSize();
    this._canvas.width = size.x;
    this._canvas.height = size.y;
    var topLeft = this._map.containerPointToLayerPoint([0, 0]);
    L.DomUtil.setPosition(this._canvas, topLeft);
  },

  _reset: function () {
    this._resize();
  },

  // Bilinear interpolation of velocity at (lon, lat)
  _interpolate: function (lon, lat) {
    var g = this._grid;
    if (!g) return null;
    // Fractional grid indices
    var fi = (lon - g.lo1) / g.dx;
    var fj = (g.la1 - lat) / g.dy;  // la1 is north edge, data goes south
    var i0 = Math.floor(fi), j0 = Math.floor(fj);
    if (i0 < 0 || i0 >= g.nx - 1 || j0 < 0 || j0 >= g.ny - 1) return null;
    var fx = fi - i0, fy = fj - j0;
    var idx00 = j0 * g.nx + i0;
    var idx10 = idx00 + 1;
    var idx01 = idx00 + g.nx;
    var idx11 = idx01 + 1;
    var u00 = g.uArr[idx00], u10 = g.uArr[idx10], u01 = g.uArr[idx01], u11 = g.uArr[idx11];
    var v00 = g.vArr[idx00], v10 = g.vArr[idx10], v01 = g.vArr[idx01], v11 = g.vArr[idx11];
    if (u00 == null || u10 == null || u01 == null || u11 == null) return null;
    if (v00 == null || v10 == null || v01 == null || v11 == null) return null;
    var u = (1 - fx) * (1 - fy) * u00 + fx * (1 - fy) * u10 + (1 - fx) * fy * u01 + fx * fy * u11;
    var v = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 + (1 - fx) * fy * v01 + fx * fy * v11;
    return [u, v];
  },

  // Color from speed
  _colorForSpeed: function (speed) {
    var t = Math.min(speed / opts.maxVelocity, 1.0);
    var pal = PALETTES[currentPalette];
    var idx = t * (pal.length - 1);
    var i0 = Math.floor(idx);
    var i1 = Math.min(i0 + 1, pal.length - 1);
    var f = idx - i0;
    var c0 = pal[i0], c1 = pal[i1];
    return "rgb(" +
      Math.round(c0[0] + f * (c1[0] - c0[0])) + "," +
      Math.round(c0[1] + f * (c1[1] - c0[1])) + "," +
      Math.round(c0[2] + f * (c1[2] - c0[2])) + ")";
  },

  _spawnParticle: function () {
    var g = this._grid;
    if (!g) return null;
    var lon = g.lo1 + Math.random() * g.dx * (g.nx - 1);
    var lat = g.la1 - Math.random() * g.dy * (g.ny - 1);
    return { lon: lon, lat: lat, age: Math.floor(Math.random() * opts.maxAge) };
  },

  _animate: function () {
    var self = this;
    function frame() {
      self._animId = requestAnimationFrame(frame);
      self._drawFrame();
    }
    frame();
  },

  _drawFrame: function () {
    var ctx = this._ctx;
    var w = this._canvas.width, h = this._canvas.height;
    if (!this._grid || !this._map) return;

    // Fade existing trails
    ctx.globalCompositeOperation = "destination-in";
    ctx.fillStyle = "rgba(0, 0, 0, " + opts.fadeOpacity + ")";
    ctx.fillRect(0, 0, w, h);
    ctx.globalCompositeOperation = "source-over";

    // Ensure particle pool is full
    while (this._particles.length < opts.numParticles) {
      var p = this._spawnParticle();
      if (p) this._particles.push(p);
      else break;
    }
    // Trim if user reduced count
    if (this._particles.length > opts.numParticles) {
      this._particles.length = opts.numParticles;
    }

    ctx.lineWidth = opts.lineWidth;

    for (var i = 0; i < this._particles.length; i++) {
      var p = this._particles[i];
      var vel = this._interpolate(p.lon, p.lat);
      if (!vel || p.age >= opts.maxAge) {
        // Respawn
        this._particles[i] = this._spawnParticle();
        if (!this._particles[i]) { this._particles.splice(i, 1); i--; }
        continue;
      }
      var u = vel[0], v = vel[1];
      var speed = Math.sqrt(u * u + v * v);

      // Move particle in geo coordinates
      var newLon = p.lon + u * opts.velocityScale * 0.01;
      var newLat = p.lat + v * opts.velocityScale * 0.01;

      // Project old and new positions to pixel
      var pt0 = this._map.latLngToContainerPoint([p.lat, p.lon]);
      var pt1 = this._map.latLngToContainerPoint([newLat, newLon]);

      // Draw line segment
      ctx.beginPath();
      ctx.moveTo(pt0.x, pt0.y);
      ctx.lineTo(pt1.x, pt1.y);
      ctx.strokeStyle = this._colorForSpeed(speed);
      ctx.stroke();

      // Update particle
      p.lon = newLon;
      p.lat = newLat;
      p.age++;
    }
  }
});

// =====================================================================
// Map setup
// =====================================================================
var map = L.map("map").setView([CONFIG.center_lat, CONFIG.center_lon], CONFIG.zoom);
L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}", {
  attribution: "Esri, GEBCO, NOAA, Garmin, HERE, UNEP-WCMC",
  maxZoom: 13
}).addTo(map);

// =====================================================================
// Create the currents layer (single instance — never destroyed)
// =====================================================================
var currentsLayer = new CurrentsLayer();
currentsLayer.addTo(map);

// =====================================================================
// Animation state
// =====================================================================
var currentIdx = 0;
var playing = true;
var playTimer = null;
var speeds = [0.5, 1, 2, 4];
var speedIdx = 1;

// =====================================================================
// Colorbar
// =====================================================================
function drawColorbar() {
  var canvas = document.getElementById("colorbar-gradient");
  var ctx = canvas.getContext("2d");
  var w = canvas.width, h = canvas.height;
  var pal = PALETTES[currentPalette];
  var grad = ctx.createLinearGradient(0, 0, w, 0);
  for (var i = 0; i < pal.length; i++) {
    var c = pal[i];
    grad.addColorStop(i / (pal.length - 1), "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")");
  }
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, h);
  var maxV = opts.maxVelocity;
  document.getElementById("cb-min").textContent = "0";
  document.getElementById("cb-mid").textContent = (maxV / 2).toFixed(2);
  document.getElementById("cb-max").textContent = maxV.toFixed(2);
}

// =====================================================================
// Frame management — setData swaps velocity, particles persist
// =====================================================================
function showFrame(idx) {
  currentIdx = idx;
  currentsLayer.setData(ALL_DATA[idx]);
  document.getElementById("time-slider").value = idx;
  document.getElementById("time-label").textContent = TIMES[idx];
  document.getElementById("frame-counter").textContent =
    "Frame " + (idx + 1) + " / " + ALL_DATA.length;
  document.getElementById("map-time").textContent = TIMES[idx];
}

function startPlaying() {
  if (playTimer) clearInterval(playTimer);
  playTimer = setInterval(function () {
    showFrame((currentIdx + 1) % ALL_DATA.length);
  }, 4000 / speeds[speedIdx]);
}

function togglePlay() {
  playing = !playing;
  var btn = document.getElementById("btn-play");
  btn.textContent = playing ? "\u23F8 Pause" : "\u25B6 Play";
  btn.classList.toggle("active", playing);
  if (playing) { startPlaying(); }
  else if (playTimer) { clearInterval(playTimer); playTimer = null; }
}

// =====================================================================
// Time controls
// =====================================================================
document.getElementById("time-slider").max = ALL_DATA.length - 1;
document.getElementById("time-slider").addEventListener("input", function (e) {
  showFrame(parseInt(e.target.value, 10));
});
document.getElementById("btn-prev").addEventListener("click", function () {
  showFrame((currentIdx - 1 + ALL_DATA.length) % ALL_DATA.length);
});
document.getElementById("btn-next").addEventListener("click", function () {
  showFrame((currentIdx + 1) % ALL_DATA.length);
});
document.getElementById("btn-play").addEventListener("click", togglePlay);
document.getElementById("btn-speed").addEventListener("click", function () {
  speedIdx = (speedIdx + 1) % speeds.length;
  this.textContent = speeds[speedIdx] + "x";
  if (playing) { startPlaying(); }
});

// =====================================================================
// Display setting controls
// =====================================================================
document.getElementById("sl-particles").addEventListener("input", function () {
  opts.numParticles = parseInt(this.value, 10);
  document.getElementById("val-particles").textContent = this.value;
});
document.getElementById("sl-vscale").addEventListener("input", function () {
  opts.velocityScale = parseFloat(this.value);
  document.getElementById("val-vscale").textContent = opts.velocityScale.toFixed(2);
});
document.getElementById("sl-maxvel").addEventListener("input", function () {
  opts.maxVelocity = parseFloat(this.value);
  document.getElementById("val-maxvel").textContent = opts.maxVelocity.toFixed(1);
  drawColorbar();
});
document.getElementById("sl-linewidth").addEventListener("input", function () {
  opts.lineWidth = parseFloat(this.value);
  document.getElementById("val-linewidth").textContent = opts.lineWidth.toFixed(1);
});
document.getElementById("sl-fade").addEventListener("input", function () {
  opts.fadeOpacity = parseFloat(this.value);
  document.getElementById("val-fade").textContent = opts.fadeOpacity.toFixed(2);
});
document.getElementById("sel-palette").addEventListener("change", function () {
  currentPalette = this.value;
  drawColorbar();
});

// Set initial slider values from config
document.getElementById("sl-maxvel").value = opts.maxVelocity;
document.getElementById("val-maxvel").textContent = opts.maxVelocity.toFixed(1);

// Cursor speed readout
var cursorEl = document.getElementById("cursor-speed");
map.on("mousemove", function (e) {
  var vel = currentsLayer._interpolate(e.latlng.lng, e.latlng.lat);
  if (vel) {
    var spd = Math.sqrt(vel[0] * vel[0] + vel[1] * vel[1]);
    cursorEl.textContent = spd.toFixed(3) + " m/s";
    cursorEl.style.display = "block";
  } else {
    cursorEl.style.display = "none";
  }
});
map.on("mouseout", function () { cursorEl.style.display = "none"; });

document.getElementById("info").innerHTML =
  "Depth: " + CONFIG.depth_m + " m &nbsp;|&nbsp; " +
  ALL_DATA.length + " frames &nbsp;|&nbsp; NECOFS GOM3 (FVCOM)";

// =====================================================================
// Auto-start
// =====================================================================
showFrame(0);
drawColorbar();
startPlaying();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime

    # ---- Initialize -------------------------------------------------------
    # Default uses the Seaplan archive (1978-2024 monthly files).
    # For the legacy 30-year aggregated hindcast: NECOFS(source="hindcast_30yr")
    # For operational forecasts: NECOFS(source="forecast")
    necofs = NECOFS(verbose=True)

    # ---- Vertical profile at a Gulf of Maine location ---------------------
    lat, lon = 42.40382, -70.12225
    dt = datetime(2025, 8, 1, 12, 0, 0)

    print(f"\nRequesting vertical profile at ({lat}°N, {lon}°E) on {dt}")
    print("=" * 70)

    # Single profile — stored in necofs.vertical_profile (xr.Dataset)
    ds = necofs.get_vertical_profile(lat=lat, lon=lon, dt=dt)
    print("\nVertical profile dataset:")
    print(ds)

    # Plot single profile
    necofs.plot_vertical_profile()

    # ---- Multiple profiles over a time range ----------------------------
    ds_multi = necofs.get_vertical_profiles(
        lat=lat, lon=lon,
        start_dt=datetime(2015, 8, 1, 0, 0, 0),
        end_dt=datetime(2015, 8, 3, 0, 0, 0),
    )
    print("\nMulti-profile dataset:")
    print(ds_multi)

    # Overlay all profiles coloured by time
    necofs.plot_vertical_profiles()

