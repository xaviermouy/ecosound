# -*- coding: utf-8 -*-
"""
Example: Fetch environmental data and merge with HMD acoustic data.

Demonstrates how to use the ecosound.environment module to retrieve
environmental co-variates (wind, tides, lunar/solar ephemeris, SST,
chlorophyll, ocean profiles) at a hydrophone location and time range,
then merge them onto the HMD acoustic time axis.

Each section is independent — comment out any you don't need.
"""

import numpy as np
import pandas as pd
import xarray as xr

from ecosound.environment import (
    ERA5,
    Tides,
    LunarSolar,
    ERDDAPDataFetcher,
    AISQueryHelper,
    NECOFS,
)
from ecosound.soundscape.hmd import HMD

# =====================================================================
# Configuration
# =====================================================================
# Site location and time range
SITE_NAME = "NRS09"
LAT = 42.40382
LON = -70.12225
START_DT = "2023-08-01"
END_DT = "2023-08-03"

# Path to HMD NetCDF files (set to None to skip HMD loading)
HMD_PATH = r'C:\Users\xavier.mouy\Documents\Projects\2024_NERACOOS_Soundscape_GOM\Analysis\data\HMD_pyPAM\NRS09\PMEL_SBNMS\PMEL_SBNMS_201801_NRS09\nc'

# Path to AIS DuckDB database (set to None to skip AIS)
AIS_DB = r'C:\Users\xavier.mouy\Documents\GitHub\NERACOOS_processing_scripts\non_acoustic_data\ais_db\gulf_of_maine_ais.duckdb'
AIS_RADIUS_KM = 50  # radius around the site for AIS queries

# Bounding box padding for gridded data (SST, chlorophyll)
BBOX_PAD_LAT = 2.0  # degrees
BBOX_PAD_LON = 3.0  # degrees

# =====================================================================
# 1. Load HMD data and extract location/time from the files
# =====================================================================
print(f"Site: {SITE_NAME} ({LAT}°N, {LON}°E)")
print(f"Time range: {START_DT} → {END_DT}\n")

if HMD_PATH is not None:
    print("=" * 70)
    print("Loading HMD acoustic data...")
    print("=" * 70)
    hmd = HMD(n_workers=4)
    hmd.load_nc_files(HMD_PATH)
    ds_hmd = hmd.ds
    print(ds_hmd)
    print()

    # Plot spectrogram (LTSA)
    hmd.plot_ltsa(
        bin="1H",                    # 1-hour time bins
        freq_range=(10, 2000),       # Hz
        db_range=(32, 108),          # color scale limits
        scale="log",                 # log frequency axis
        cmap="rainbow",
        statistic="median",
    )
else:
    ds_hmd = None
    print("No HMD_PATH set — skipping acoustic data.\n")

# Derived bounding box for gridded data
LAT_MIN, LAT_MAX = LAT - BBOX_PAD_LAT, LAT + BBOX_PAD_LAT
LON_MIN, LON_MAX = LON - BBOX_PAD_LON, LON + BBOX_PAD_LON


# =====================================================================
# 2. ERA5 — Wind and Precipitation (hourly, ~28 km, free)
# =====================================================================
print("=" * 70)
print("Fetching ERA5 wind and precipitation...")
print("=" * 70)

era5 = ERA5(source="open_meteo", verbose=True)

# Wind
ds_wind = era5.get_wind_timeseries(
    lat=LAT, lon=LON,
    start_dt=START_DT, end_dt=END_DT,
)
print(ds_wind)
print()

# Precipitation
ds_precip = era5.get_precipitation_timeseries(
    lat=LAT, lon=LON,
    start_dt=START_DT, end_dt=END_DT,
)
print(ds_precip)
print()

# Plot
era5.plot_wind_timeseries(display=True)
era5.plot_precipitation_timeseries(display=True)


# =====================================================================
# 3. Tides — Water level and tidal index (6-min, NOAA CO-OPS, free)
# =====================================================================
print("=" * 70)
print("Fetching tidal data...")
print("=" * 70)

tides = Tides(verbose=True)

ds_tide = tides.get_water_level(
    lat=LAT, lon=LON,
    start_dt=START_DT, end_dt=END_DT,
    product="predictions",      # astronomical tide (clean signal)
    datum="MLLW",
    compute_tidal_index=True,   # adds time_since_high_tide_h & tidal_phase
)
print(ds_tide)
print(f"High tides detected: {len(tides.high_tide_times)}")
print(f"Low  tides detected: {len(tides.low_tide_times)}")
print()

tides.plot_water_level(display=True)


# =====================================================================
# 4. Lunar & Solar ephemeris (computed locally, no internet needed)
# =====================================================================
print("=" * 70)
print("Computing lunar/solar ephemeris...")
print("=" * 70)

ls = LunarSolar(verbose=True)

ds_ephem = ls.get_timeseries(
    lat=LAT, lon=LON,
    start_dt=START_DT, end_dt=END_DT,
    freq="1h",
)
print(ds_ephem)
print()

ls.plot_timeseries(display=True)


# =====================================================================
# 5. Sea Surface Temperature (ERDDAP, daily ~2 km)
# =====================================================================
print("=" * 70)
print("Fetching SST from ERDDAP...")
print("=" * 70)

fetcher_sst = ERDDAPDataFetcher(
    server="https://comet.nefsc.noaa.gov/erddap",
    dataset_id="noaa_coastwatch_acspo_v2_reanalysis",
)

ds_sst = fetcher_sst.fetch_data(
    "sea_surface_temperature",
    start_date=START_DT,
    end_date=END_DT,
    lat_min=LAT_MIN, lat_max=LAT_MAX,
    lon_min=LON_MIN, lon_max=LON_MAX,
    quality_mask_value=5,          # best quality only
    max_request_duration_days=7,   # chunk into weekly requests to avoid server errors
    spatial_stride=2,              # thin spatial resolution to reduce request size
)

# Extract time series at the site (nearest grid cell)
sst_site = ds_sst.sea_surface_temperature.sel(
    latitude=LAT, longitude=LON, method="nearest"
)
print(f"SST — mean: {float(sst_site.mean(skipna=True)):.2f} °C")
print()


# =====================================================================
# 6. Chlorophyll-a (ERDDAP, daily ~1 km)
# =====================================================================
print("=" * 70)
print("Fetching Chlorophyll-a from ERDDAP...")
print("=" * 70)

fetcher_chla = ERDDAPDataFetcher(
    server="https://comet.nefsc.noaa.gov/erddap",
    dataset_id="occci_v6_daily_1km",
)

ds_chla = fetcher_chla.fetch_data(
    "chlor_a",
    start_date=START_DT,
    end_date=END_DT,
    lat_min=LAT_MIN, lat_max=LAT_MAX,
    lon_min=LON_MIN, lon_max=LON_MAX,
    max_request_duration_days=7,   # chunk into weekly requests to avoid server errors
    spatial_stride=2,              # thin spatial resolution to reduce request size
)

chla_site = ds_chla.chlor_a.sel(
    latitude=LAT, longitude=LON, method="nearest"
)
print(f"Chla — mean: {float(chla_site.mean(skipna=True)):.3f} mg/m³")
print()


# =====================================================================
# 7. NECOFS — Ocean vertical profiles (OPeNDAP, Seaplan hindcast 1978-2024)
# =====================================================================
from datetime import datetime

print("=" * 70)
print("Fetching NECOFS ocean profiles...")
print("=" * 70)

necofs = NECOFS(verbose=True)

# Single snapshot profile (mid-month at noon)
ds_profile = necofs.get_vertical_profile(
    lat=LAT, lon=LON,
    dt=datetime(int(START_DT[:4]), int(START_DT[5:7]), 15, 12, 0, 0),
)
necofs.plot_vertical_profile(display=True)

# Time series of profiles over a week
ds_profiles = necofs.get_vertical_profiles(
    lat=LAT, lon=LON,
    start_dt=START_DT,
    end_dt=END_DT,
)
necofs.plot_vertical_profiles(display=True)


# =====================================================================
# 8. AIS — Vessel traffic (local DuckDB database)
# =====================================================================
import os
if AIS_DB is not None and os.path.exists(AIS_DB):
    print("=" * 70)
    print("Querying AIS vessel traffic...")
    print("=" * 70)

    with AISQueryHelper(AIS_DB) as ais:
        # --- Statistics for the area ---
        bbox = AISQueryHelper.calculate_bounding_box(LAT, LON, AIS_RADIUS_KM)
        stats = ais.get_statistics(
            start_date=START_DT,
            end_date=END_DT,
            min_lat=bbox["min_lat"], max_lat=bbox["max_lat"],
            min_lon=bbox["min_lon"], max_lon=bbox["max_lon"],
        )
        print("AIS Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print()

        # --- Query AIS points within radius ---
        gdf_ais = ais.query_radius(
            start_date=START_DT,
            end_date=END_DT,
            center_lat=LAT, center_lon=LON,
            radius_km=AIS_RADIUS_KM,
        )
        print(f"Found {len(gdf_ais)} AIS records within {AIS_RADIUS_KM} km")
        if len(gdf_ais) > 0:
            print(gdf_ais[["mmsi", "vessel_name", "vessel_category",
                           "sog", "distance_km"]].head(10))
        print()

        # --- Unique vessels ---
        unique_vessels = ais.get_unique_vessels(
            start_date=START_DT, end_date=END_DT,
            min_lat=bbox["min_lat"], max_lat=bbox["max_lat"],
            min_lon=bbox["min_lon"], max_lon=bbox["max_lon"],
        )
        print(f"Unique vessels: {len(unique_vessels)}")
        if len(unique_vessels) > 0:
            print(unique_vessels[["mmsi", "vessel_name", "vessel_category",
                                  "num_positions"]].head(10))
        print()

        # --- Gridded vessel counts (hourly, 1 km cells) ---
        vessel_grid = ais.create_gridded_vessel_counts(
            start_date=START_DT, end_date=END_DT,
            min_lat=bbox["min_lat"], max_lat=bbox["max_lat"],
            min_lon=bbox["min_lon"], max_lon=bbox["max_lon"],
            width_km=1, height_km=1,
            time_resolution_hours=1,
        )

    # --- Plot gridded vessel counts (cumulative map) ---
    if vessel_grid is not None and vessel_grid.size > 0:
        from ecosound.visualization import GridPlotter
        vessel_grid_sum = vessel_grid.sum(dim='time', skipna=True, min_count=1)
        plotter_ais = GridPlotter()
        fig_ais = plotter_ais.plot_static_map(
            vessel_grid_sum,
            timestamp=START_DT,
            label='Cumulative Vessel Count',
            colormap='YlOrRd',
            vmin=0,
            vmax=500,
            markers=[(LAT, LON)],
            marker_labels=[SITE_NAME],
        )
        import matplotlib.pyplot as plt
        plt.show()

    # --- Vessel category bar chart ---
    if len(gdf_ais) > 0:
        import matplotlib.pyplot as plt
        cat_counts = (
            gdf_ais.groupby("vessel_category")
            .size()
            .sort_values(ascending=False)
            .reset_index(name="n_records")
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(cat_counts.vessel_category, cat_counts.n_records,
                color="steelblue", edgecolor="white")
        ax.set_xlabel("Number of AIS records")
        ax.set_title(f"Vessel categories within {AIS_RADIUS_KM} km of {SITE_NAME}\n"
                     f"{START_DT} to {END_DT}")
        ax.grid(True, axis="x", ls=":", alpha=0.6)
        plt.tight_layout()
        plt.show()

else:
    if AIS_DB is not None:
        print(f"AIS database not found at: {AIS_DB}")
    print("Skipping AIS section.\n")
