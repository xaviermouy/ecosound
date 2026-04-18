# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Lunar and Solar Ephemeris
Class-based interface for computing lunar and solar positions, illumination,
and derived day/night/twilight flags at a fixed point location and time range.

All quantities are computed locally using the PyEphem library — no internet
connection or API key is required.  Computation is fast even for multi-year,
sub-hourly time series (pure C extensions under the hood).

Returns an xr.Dataset with the same (time, lat, lon) coordinate structure as
NECOFS, ERA5, Tides, and HMD outputs, enabling straightforward merging.

Requires:
    pip install ephem
"""

from __future__ import annotations
from typing import Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr


class LunarSolar:
    """
    Class for computing lunar and solar ephemeris at a fixed location.

    All quantities are derived geometrically from the observer's position and
    time using PyEphem.  Atmospheric refraction is disabled (pressure=0) for
    clean, deterministic altitude values.

    Args:
        verbose: Print progress messages (default: True).

    Attributes:
        timeseries (xr.Dataset | None): Most recent ephemeris time series from
            get_timeseries().  Dimension: time.  Coordinates: time, lat, lon.
            None until first call.

    Examples:
        >>> from ecosound.environment import LunarSolar
        >>> ls = LunarSolar()

        # Fetch by explicit time range at hourly resolution
        >>> ds = ls.get_timeseries(lat=42.5, lon=-70.0,
        ...     start_dt="2015-08-01", end_dt="2015-08-31", freq="1h")
        >>> ls.plot_timeseries()

        # Match the time axis of another dataset (e.g. HMD 1-minute data)
        >>> ls.get_timeseries(lat=42.5, lon=-70.0, time_ref=hmd.ds)

        # Merge with HMD acoustic data
        >>> ds_combined = xr.merge([hmd.ds, ls.timeseries])
    """

    # Lunar synodic (phase) period in days
    SYNODIC_PERIOD_DAYS = 29.53059

    # Reference new moon (used for phase computation)
    # Jan 6, 2000 18:14 UTC — well-known new moon in the J2000 epoch
    _REF_NEW_MOON_STR = "2000/1/6 18:14:00"

    # Solar altitude thresholds (degrees) for twilight classification
    CIVIL_TWILIGHT_ALT     = -6.0
    NAUTICAL_TWILIGHT_ALT  = -12.0
    ASTRONOMICAL_TWILIGHT_ALT = -18.0

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.timeseries: Optional[xr.Dataset] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    @staticmethod
    def _check_ephem():
        """Import and return the ephem module, with a helpful error if missing."""
        try:
            import ephem
            return ephem
        except ImportError:
            raise ImportError(
                "The 'ephem' package is required for LunarSolar.\n"
                "Install with: pip install ephem"
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_timeseries(
        self,
        lat: float,
        lon: float,
        start_dt: Union[datetime, pd.Timestamp, str, None] = None,
        end_dt: Union[datetime, pd.Timestamp, str, None] = None,
        freq: str = "1h",
        time_ref: Optional[xr.Dataset] = None,
    ) -> xr.Dataset:
        """
        Compute lunar and solar ephemeris at a fixed location for a time range.

        The time axis can be specified either as a regular grid (start_dt,
        end_dt, freq) or by passing another xr.Dataset via time_ref, in which
        case the exact timestamps of that dataset are used (freq is ignored).
        This allows direct merging without reindexing when working with HMD,
        NECOFS, or ERA5 datasets.

        The result is always stored in self.timeseries and always returned:

            ls.get_timeseries(lat, lon, start_dt="2015-08-01", end_dt="2015-08-31")
            ds = ls.get_timeseries(lat, lon, time_ref=hmd.ds)

        Args:
            lat: Latitude (decimal degrees, WGS84).
            lon: Longitude (decimal degrees, WGS84, negative west).
            start_dt: Start of the time range (inclusive).  Accepts datetime,
                      pd.Timestamp, or ISO 8601 string.  Ignored if time_ref
                      is provided.
            end_dt: End of the time range (inclusive).  Ignored if time_ref
                    is provided.
            freq: Time step for the regular grid (default "1h").  Any pandas
                  frequency string is accepted (e.g. "6min", "30min", "1D").
                  Ignored when time_ref is provided.
            time_ref: Optional xr.Dataset whose time coordinate defines the
                      timestamps to compute.  When provided, start_dt, end_dt,
                      and freq are ignored.

        Returns:
            xr.Dataset with dimension time, also stored in self.timeseries:
                Solar data variables:
                    - sun_altitude_deg      (time): Solar altitude above horizon (°).
                      Negative = below horizon.
                    - sun_azimuth_deg       (time): Solar azimuth (°, 0=N, 90=E).
                    - is_day                (time): Bool — sun altitude > 0°.
                    - is_civil_twilight     (time): Bool — -6° < sun altitude ≤ 0°.
                    - is_nautical_twilight  (time): Bool — -12° < sun altitude ≤ -6°.
                    - is_night              (time): Bool — sun altitude ≤ -12°.
                Lunar data variables:
                    - moon_altitude_deg     (time): Lunar altitude above horizon (°).
                    - moon_azimuth_deg      (time): Lunar azimuth (°, 0=N, 90=E).
                    - moon_illumination_pct (time): Fraction of lunar surface lit (%).
                      0 = new moon, 100 = full moon.
                    - moon_phase            (time): Fractional lunar cycle [0, 1).
                      0 = new moon, 0.25 = first quarter,
                      0.5 = full moon, 0.75 = last quarter.
                    - is_moon_up            (time): Bool — moon altitude > 0°.
                Coordinates:
                    - time : datetime64 UTC
                    - lat  : observer latitude (scalar)
                    - lon  : observer longitude (scalar)

        Raises:
            ImportError: If PyEphem is not installed.
            ValueError: If neither time_ref nor both start_dt/end_dt are given.
        """
        ephem = self._check_ephem()

        # --- Resolve timestamps ---
        if time_ref is not None:
            times = pd.to_datetime(time_ref.time.values)
        elif start_dt is not None and end_dt is not None:
            times = pd.date_range(
                pd.Timestamp(start_dt), pd.Timestamp(end_dt), freq=freq
            )
        else:
            raise ValueError(
                "Provide either time_ref or both start_dt and end_dt."
            )

        n = len(times)
        self._log(
            f"Computing lunar/solar ephemeris at ({lat}°N, {lon}°E) "
            f"for {n} timestamps..."
        )

        # --- Pre-allocate output arrays ---
        sun_alt    = np.empty(n, dtype=float)
        sun_az     = np.empty(n, dtype=float)
        moon_alt   = np.empty(n, dtype=float)
        moon_az    = np.empty(n, dtype=float)
        moon_ill   = np.empty(n, dtype=float)
        moon_phase = np.empty(n, dtype=float)

        # --- Set up fixed observer (position doesn't change) ---
        obs = ephem.Observer()
        obs.lat       = str(lat)
        obs.lon       = str(lon)
        obs.elevation = 0.0
        obs.pressure  = 0.0  # disable atmospheric refraction for clean altitudes

        sun  = ephem.Sun()
        moon = ephem.Moon()
        ref_new_moon = ephem.Date(self._REF_NEW_MOON_STR)

        # --- Compute per timestamp ---
        for i, t in enumerate(times):
            obs.date = ephem.Date(t.to_pydatetime())

            sun.compute(obs)
            moon.compute(obs)

            sun_alt[i]  = np.degrees(float(sun.alt))
            sun_az[i]   = np.degrees(float(sun.az))
            moon_alt[i] = np.degrees(float(moon.alt))
            moon_az[i]  = np.degrees(float(moon.az))
            moon_ill[i] = float(moon.phase)  # % illuminated (0–100)

            # Lunar phase: days since reference new moon, normalised to [0, 1)
            days_since     = float(obs.date - ref_new_moon) % self.SYNODIC_PERIOD_DAYS
            moon_phase[i]  = days_since / self.SYNODIC_PERIOD_DAYS

        # --- Derived boolean flags ---
        is_day               = sun_alt > 0.0
        is_civil_twilight    = (sun_alt > self.CIVIL_TWILIGHT_ALT)    & ~is_day
        is_nautical_twilight = (sun_alt > self.NAUTICAL_TWILIGHT_ALT) & (sun_alt <= self.CIVIL_TWILIGHT_ALT)
        is_night             = sun_alt <= self.NAUTICAL_TWILIGHT_ALT
        is_moon_up           = moon_alt > 0.0

        self._log(f"  Done.")

        ds = xr.Dataset(
            data_vars={
                # Solar
                "sun_altitude_deg": (
                    "time", sun_alt,
                    {"units": "degrees", "long_name": "Solar altitude above horizon (negative = below)"},
                ),
                "sun_azimuth_deg": (
                    "time", sun_az,
                    {"units": "degrees", "long_name": "Solar azimuth (0=N, 90=E, 180=S, 270=W)"},
                ),
                "is_day": (
                    "time", is_day,
                    {"long_name": "Sun above horizon (altitude > 0°)"},
                ),
                "is_civil_twilight": (
                    "time", is_civil_twilight,
                    {"long_name": "Civil twilight (−6° < sun altitude ≤ 0°)"},
                ),
                "is_nautical_twilight": (
                    "time", is_nautical_twilight,
                    {"long_name": "Nautical twilight (−12° < sun altitude ≤ −6°)"},
                ),
                "is_night": (
                    "time", is_night,
                    {"long_name": "Night (sun altitude ≤ −12°)"},
                ),
                # Lunar
                "moon_altitude_deg": (
                    "time", moon_alt,
                    {"units": "degrees", "long_name": "Lunar altitude above horizon (negative = below)"},
                ),
                "moon_azimuth_deg": (
                    "time", moon_az,
                    {"units": "degrees", "long_name": "Lunar azimuth (0=N, 90=E, 180=S, 270=W)"},
                ),
                "moon_illumination_pct": (
                    "time", moon_ill,
                    {"units": "%", "long_name": "Fraction of lunar surface illuminated (0=new, 100=full)"},
                ),
                "moon_phase": (
                    "time", moon_phase,
                    {
                        "units":     "1",
                        "long_name": (
                            "Fractional lunar cycle "
                            "(0=new moon, 0.25=first quarter, "
                            "0.5=full moon, 0.75=last quarter)"
                        ),
                    },
                ),
                "is_moon_up": (
                    "time", is_moon_up,
                    {"long_name": "Moon above horizon (altitude > 0°)"},
                ),
            },
            coords={
                "time": times.values,
                "lat":  lat,
                "lon":  lon,
            },
            attrs={
                "lat":                     lat,
                "lon":                     lon,
                "source":                  "Computed via PyEphem",
                "atmospheric_refraction":  "disabled (pressure=0)",
                "twilight_convention":     (
                    "civil > -6°, nautical -12° to -6°, "
                    "astronomical -18° to -12°, night ≤ -12°"
                ),
                "lunar_phase_convention":  (
                    "0=new moon, 0.25=first quarter, "
                    "0.5=full moon, 0.75=last quarter"
                ),
                "lunar_phase_reference":   self._REF_NEW_MOON_STR,
                "synodic_period_days":     self.SYNODIC_PERIOD_DAYS,
            },
        )

        self.timeseries = ds
        return ds

    def plot_timeseries(
        self,
        figsize: tuple = (12, 8),
        display: bool = True,
        filename: Optional[str] = None,
    ) -> None:
        """
        Plot the lunar and solar ephemeris stored in self.timeseries.

        Produces a figure with three stacked subplots sharing the time axis:
            1. Solar altitude with colour-coded day/twilight/night background
            2. Lunar altitude with above-horizon shading
            3. Lunar illumination (%) with new/full moon markers

        Args:
            figsize: Figure size as (width, height) in inches (default: (12, 8)).
            display: Show the figure on screen (default: True).
            filename: If provided, save the figure to this path.  Default: None.

        Raises:
            RuntimeError: If get_timeseries() has not been called yet.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if self.timeseries is None:
            raise RuntimeError(
                "No data available. Call get_timeseries() first."
            )

        ds    = self.timeseries
        times = pd.to_datetime(ds["time"].values)

        lat_str = f"{float(ds.coords['lat']):.2f}°N"
        lon_str = f"{float(ds.coords['lon']):.2f}°E"
        title = (
            f"Lunar & Solar Ephemeris  |  ({lat_str}, {lon_str})  "
            f"|  {times[0].date()} – {times[-1].date()}"
        )

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # --- Panel 1: Solar altitude + shaded zones ---
        sun_alt = ds["sun_altitude_deg"].values
        ymin_sun = min(np.nanmin(sun_alt) * 1.1, -20)

        axes[0].fill_between(times, sun_alt, 0,
            where=sun_alt >= 0,
            color="lightyellow", alpha=0.8, zorder=1)
        axes[0].fill_between(times, np.maximum(sun_alt, self.CIVIL_TWILIGHT_ALT), self.CIVIL_TWILIGHT_ALT,
            where=(sun_alt > self.CIVIL_TWILIGHT_ALT) & (sun_alt < 0),
            color="peachpuff", alpha=0.8, zorder=1)
        axes[0].fill_between(times, np.maximum(sun_alt, self.NAUTICAL_TWILIGHT_ALT), self.NAUTICAL_TWILIGHT_ALT,
            where=(sun_alt > self.NAUTICAL_TWILIGHT_ALT) & (sun_alt <= self.CIVIL_TWILIGHT_ALT),
            color="lightsalmon", alpha=0.7, zorder=1)
        axes[0].fill_between(times, sun_alt, ymin_sun,
            where=sun_alt <= self.NAUTICAL_TWILIGHT_ALT,
            color="midnightblue", alpha=0.2, zorder=1)

        axes[0].plot(times, sun_alt, color="goldenrod", linewidth=0.8, zorder=3)
        axes[0].axhline(0,                            color="gray", linewidth=0.5, linestyle="--", zorder=2)
        axes[0].axhline(self.CIVIL_TWILIGHT_ALT,      color="gray", linewidth=0.4, linestyle=":",  zorder=2)
        axes[0].axhline(self.NAUTICAL_TWILIGHT_ALT,   color="gray", linewidth=0.4, linestyle=":",  zorder=2)

        # Legend patches
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="lightyellow",  alpha=0.8, label="Day (alt > 0°)"),
            Patch(facecolor="peachpuff",    alpha=0.8, label="Civil twilight (0° to −6°)"),
            Patch(facecolor="lightsalmon",  alpha=0.7, label="Nautical twilight (−6° to −12°)"),
            Patch(facecolor="midnightblue", alpha=0.3, label="Night (alt ≤ −12°)"),
        ]
        axes[0].legend(handles=legend_elements, fontsize=7, loc="upper right", ncol=2)
        axes[0].set_ylabel("Sun altitude (°)", fontsize=9)
        axes[0].set_ylim(ymin_sun, np.nanmax(sun_alt) * 1.1)
        axes[0].grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        axes[0].tick_params(labelsize=8)

        # --- Panel 2: Moon altitude ---
        moon_alt = ds["moon_altitude_deg"].values
        axes[1].plot(times, moon_alt, color="slategray", linewidth=0.8, zorder=3)
        axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--", zorder=2)
        axes[1].fill_between(times, moon_alt, 0,
            where=moon_alt >= 0,
            color="lightsteelblue", alpha=0.5, zorder=1, label="Moon above horizon")
        axes[1].set_ylabel("Moon altitude (°)", fontsize=9)
        axes[1].legend(fontsize=7, loc="upper right")
        axes[1].grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        axes[1].tick_params(labelsize=8)

        # --- Panel 3: Moon illumination + new/full moon markers ---
        moon_ill   = ds["moon_illumination_pct"].values
        moon_phase = ds["moon_phase"].values

        axes[2].fill_between(times, moon_ill, alpha=0.35, color="mediumpurple", linewidth=0)
        axes[2].plot(times, moon_ill, color="mediumpurple", linewidth=0.8, label="Illumination (%)")

        # Mark new moons (phase near 0) and full moons (phase near 0.5)
        new_moon_mask  = (moon_phase < 0.03) | (moon_phase > 0.97)
        full_moon_mask = (moon_phase > 0.47) & (moon_phase < 0.53)
        if np.any(new_moon_mask):
            axes[2].scatter(times[new_moon_mask], moon_ill[new_moon_mask],
                color="black", s=25, zorder=5, label="New moon")
        if np.any(full_moon_mask):
            axes[2].scatter(times[full_moon_mask], moon_ill[full_moon_mask],
                color="gold", edgecolors="darkorange", s=25, zorder=5, label="Full moon")

        axes[2].set_ylabel("Lunar illumination (%)", fontsize=9)
        axes[2].set_ylim(-2, 112)
        axes[2].legend(fontsize=7, loc="upper right")
        axes[2].grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        axes[2].tick_params(labelsize=8)

        # --- Adaptive time axis ---
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

    ls = LunarSolar(verbose=True)

    # ---- Hourly ephemeris for a time range ----------------------------------
    ds = ls.get_timeseries(
        lat=lat, lon=lon,
        start_dt=datetime(2026, 1, 1),
        end_dt=datetime(2026, 3, 20),
        freq="1h",
    )
    print("\nEphemeris dataset:")
    print(ds)
    ls.plot_timeseries()

    # ---- Merge with HMD (exact 1-minute time axis) -------------------------
    # from HMD import HMD
    # hmd = HMD()
    # hmd.load_nc_files("path/to/deployment")
    # ls.get_timeseries(lat=lat, lon=lon, time_ref=hmd.ds)
    # ds_combined = xr.merge([hmd.ds, ls.timeseries])
    # print(ds_combined)

    # ---- Merge with ERA5 wind + tides + ephemeris --------------------------
    # from ERA5 import ERA5
    # from Tides import Tides
    # era5 = ERA5()
    # tides = Tides()
    # era5.get_wind_timeseries(lat=lat, lon=lon, start_dt="2026-01-01", end_dt="2026-03-20")
    # tides.get_water_level(lat=lat, lon=lon, start_dt="2026-01-01", end_dt="2026-03-20")
    # ds_wind  = era5.wind_timeseries.reindex(time=ds.time, method="nearest", tolerance="1h")
    # ds_tide  = tides.water_level.reindex(time=ds.time, method="nearest", tolerance="10min")
    # ds_combined = xr.merge([ds, ds_wind, ds_tide])
    # print(ds_combined)
