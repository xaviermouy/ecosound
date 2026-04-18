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
        url: OPeNDAP URL for the NECOFS GOM3 dataset.
             Defaults to the 30-year hindcast endpoint.
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

    def __init__(self, url: Optional[str] = None, verbose: bool = True):
        self.url = url or self.GOM3_HINDCAST_URL
        self.verbose = verbose
        self.vertical_profile: Optional[xr.Dataset] = None   # set by get_vertical_profile()
        self.vertical_profiles: Optional[xr.Dataset] = None   # set by get_vertical_profiles()

        # Cached grid data (populated on first call to _open_dataset)
        self._ds = None           # xarray Dataset handle (lazy OPeNDAP)
        self._lon_node = None     # Node longitudes  (node,)
        self._lat_node = None     # Node latitudes   (node,)
        self._lon_elem = None     # Element-center longitudes  (nele,)
        self._lat_elem = None     # Element-center latitudes   (nele,)
        self._h = None            # Bathymetric depth at nodes (node,), m positive down
        self._siglay = None       # Sigma-layer centers (siglay, node), range [0, -1]
        self._times = None        # Model time steps as pd.DatetimeIndex

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _open_dataset(self) -> None:
        """Open the OPeNDAP dataset and cache time-invariant grid data (runs once)."""
        if self._ds is not None:
            return

        self._log(f"Connecting to NECOFS GOM3: {self.url}")
        try:
            # decode_times=False avoids a crash on FVCOM's non-standard Itime2
            # units ('msec since 00:00:00') that xarray cannot parse.
            # Time is decoded manually below.
            self._ds = xr.open_dataset(
                self.url, engine="netcdf4", mask_and_scale=True, decode_times=False
            )
        except Exception as exc:
            raise ConnectionError(
                f"Could not open NECOFS dataset at:\n  {self.url}\n"
                "Check the URL, your network connection, and that netCDF4 is installed.\n"
                f"Original error: {exc}"
            ) from exc

        self._log("Loading grid coordinates (cached for subsequent calls)...")
        self._lon_node = self._ds["lon"].values    # (node,)
        self._lat_node = self._ds["lat"].values    # (node,)
        self._lon_elem = self._ds["lonc"].values   # (nele,)
        self._lat_elem = self._ds["latc"].values   # (nele,)
        self._h = self._ds["h"].values             # (node,)
        self._siglay = self._ds["siglay"].values   # (siglay, node)
        self._times = self._decode_fvcom_times()

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
        self._open_dataset()

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
                "url":              self.url,
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
            xr.Dataset with dimensions (time, sigma_layer), also stored in
            self.vertical_profiles:
                Data variables:
                    - temperature_C (time, sigma_layer)  : in-situ temperature (°C)
                    - salinity_PSU  (time, sigma_layer)  : practical salinity (PSU)
                    - u_ms          (time, sigma_layer)  : eastward current (m/s)
                    - v_ms          (time, sigma_layer)  : northward current (m/s)
                    - sound_speed_ms(time, sigma_layer)  : sound speed, Mackenzie 1981 (m/s)
                    - depth_m       (time, sigma_layer)  : depth below surface (m); varies
                                                           with time due to sea-surface elevation
                    - zeta_m        (time)               : sea surface elevation (m)
                Coordinates:
                    - time                  : model time steps (datetime64)
                    - sigma_layer           : integer layer index (0 = surface)
                    - lat                   : nearest node latitude
                    - lon                   : nearest node longitude

        Raises:
            ValueError: If neither dt nor start_dt/end_dt are provided, or both are.
            ConnectionError: If the OPeNDAP dataset cannot be opened.

        Examples:
            # Discrete time list
            ds = necofs.get_vertical_profiles(
                lat=42.5, lon=-70.0,
                dt=["2015-08-01T00:00", "2015-08-01T06:00", "2015-08-01T12:00"]
            )

            # Time range (all hourly steps between start and end)
            ds = necofs.get_vertical_profiles(
                lat=42.5, lon=-70.0,
                start_dt="2015-08-01", end_dt="2015-08-07"
            )

            # Access a single time step
            ds.sel(time="2015-08-01T06:00", method="nearest")
        """
        if dt is not None and (start_dt is not None or end_dt is not None):
            raise ValueError("Provide either 'dt' or 'start_dt'/'end_dt', not both.")
        if dt is None and (start_dt is None or end_dt is None):
            raise ValueError("Provide either 'dt' (list) or both 'start_dt' and 'end_dt'.")

        self._open_dataset()

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

        ds = xr.Dataset(
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
                "url":              self.url,
                "model":            "NECOFS GOM3 (FVCOM)",
                "sound_speed_ref":  "Mackenzie (1981)",
            },
        )

        self._log(f"Done — {n_steps} profiles extracted.")
        self.vertical_profiles = ds
        return ds

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


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime

    # ---- Initialize -------------------------------------------------------
    # Use default hindcast URL; swap in GOM3_FORECAST_URL for operational forecasts
    necofs = NECOFS(verbose=True)

    # ---- Vertical profile at a Gulf of Maine location ---------------------
    lat, lon = 42.40382, -70.12225
    dt = datetime(2015, 8, 1, 12, 0, 0)

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

