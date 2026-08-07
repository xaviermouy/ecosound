# ecosound.environment — Environmental Data Module

This module provides classes to fetch and compute environmental data relevant to
passive-acoustic deployments.  All classes return `xarray.Dataset` or
`GeoDataFrame` objects and share a consistent `(time, lat, lon)` coordinate
structure, making it straightforward to merge datasets from different sources.

Visualization of fetched data is handled by `ecosound.visualization`:

| Plotter class | Import | Purpose |
|---|---|---|
| `GridPlotter` | `from ecosound.visualization import GridPlotter` | Static (Cartopy/PNG) and interactive (Folium/HTML) maps from `xr.DataArray` grids — used for SST, Chla, and gridded AIS vessel counts |
| `AISMapPlotter` | `from ecosound.visualization import AISMapPlotter` | Interactive Folium maps of raw AIS point data with clickable markers |

---

## Table of Contents

1. [ERA5 — Atmospheric Wind & Precipitation](#1-era5--atmospheric-wind--precipitation)
2. [NECOFS — Ocean Water-Column Profiles](#2-necofs--ocean-water-column-profiles)
3. [Tides — Water Level & Tidal Index](#3-tides--water-level--tidal-index)
4. [LunarSolar — Sun & Moon Ephemeris](#4-lunarsolar--sun--moon-ephemeris)
5. [ERDDAPDataFetcher — Generic Gridded Data (SST, Chlorophyll)](#5-erddapdatafetcher--generic-gridded-data)
6. [AISDataDownloaderDuckDB — Download AIS Data](#6-aisdatadownloaderduckdb--download-ais-data)
7. [AISQueryHelper — Query AIS Vessel Traffic](#7-aisqueryhelper--query-ais-vessel-traffic)
8. [Merging Multiple Sources](#8-merging-multiple-sources)

---

## 1. ERA5 — Atmospheric Wind & Precipitation

**Class**: `ecosound.environment.ERA5`  
**Source**: [Open-Meteo](https://open-meteo.com/) (default, free, no registration) or [Copernicus CDS](https://cds.climate.copernicus.eu/) (`source="cds"`, requires account)  
**Spatial resolution**: 0.25° (~28 km) | **Temporal resolution**: Hourly  
**Coverage**: 1940 – present, global

### Measurements

| Variable | Units | Description |
|---|---|---|
| `u10_ms` | m/s | 10 m eastward wind component |
| `v10_ms` | m/s | 10 m northward wind component |
| `wind_speed_ms` | m/s | 10 m wind speed |
| `wind_dir_deg` | ° | Wind direction (FROM convention: 0=N, 90=E, 180=S, 270=W) |
| `precipitation_mmh` | mm/h | Total precipitation rate |
| `rain_mmh` | mm/h | Rainfall rate (large-scale + convective) |
| `snowfall_mmh` | mm/h | Snowfall rate (liquid water equivalent) |

### Methods

| Method | Description |
|---|---|
| `get_wind_timeseries(lat, lon, start_dt, end_dt)` | Fetch ERA5 10 m wind; result stored in `self.wind_timeseries` |
| `get_precipitation_timeseries(lat, lon, start_dt, end_dt)` | Fetch ERA5 hourly precipitation; result stored in `self.precipitation_timeseries` |
| `plot_wind_timeseries(figsize, display, filename)` | 3-panel plot: speed, U/V components, direction |
| `plot_precipitation_timeseries(figsize, display, filename)` | 2-panel plot: total precipitation, rain vs. snowfall |

### Example

```python
from ecosound.environment import ERA5
from datetime import datetime

lat, lon = 42.40, -70.12

# --- Open-Meteo backend (default, no registration) ---
era5 = ERA5(source="open_meteo", verbose=True)

# Fetch wind
ds_wind = era5.get_wind_timeseries(
    lat=lat, lon=lon,
    start_dt=datetime(2015, 8, 1),
    end_dt=datetime(2015, 8, 31),
)
print(ds_wind)
# Coordinates: time (hourly), lat, lon
# Variables:   u10_ms, v10_ms, wind_speed_ms, wind_dir_deg

era5.plot_wind_timeseries(filename="wind_august2015.png")

# Fetch precipitation (same object, same backend)
ds_precip = era5.get_precipitation_timeseries(
    lat=lat, lon=lon,
    start_dt=datetime(2015, 8, 1),
    end_dt=datetime(2015, 8, 31),
)
print(ds_precip)
era5.plot_precipitation_timeseries()

# --- CDS backend (requires ~/.cdsapirc configuration) ---
# era5_cds = ERA5(source="cds")
# era5_cds.get_wind_timeseries(lat=lat, lon=lon,
#     start_dt="2015-08-01", end_dt="2015-08-31")
# era5_cds.plot_wind_timeseries()
```

> **Dependencies**: `numpy`, `pandas`, `xarray`.  
> Optional: `requests` (faster HTTP); `cdsapi` (for CDS backend only).

---

## 2. NECOFS — Ocean Water-Column Profiles

**Class**: `ecosound.environment.NECOFS`  
**Source**: [NECOFS GOM3](http://www.smast.umassd.edu:8080/thredds/catalog.html) FVCOM ocean model via OPeNDAP  
**Grid**: Unstructured triangular (FVCOM); nearest-node lookup  
**Coverage**: Gulf of Maine and surrounding waters | Hourly hindcast (30-year) or operational forecast

### Measurements

| Variable | Units | Dimension | Description |
|---|---|---|---|
| `temperature_C` | °C | sigma_layer | In-situ temperature |
| `salinity_PSU` | PSU | sigma_layer | Practical salinity |
| `u_ms` | m/s | sigma_layer | Eastward current velocity |
| `v_ms` | m/s | sigma_layer | Northward current velocity |
| `sound_speed_ms` | m/s | sigma_layer | Sound speed (Mackenzie 1981 formula) |
| `depth_m` | m | sigma_layer | Depth below sea surface (positive down) |
| `zeta_m` | m | scalar | Sea surface elevation |

### Methods

| Method | Description |
|---|---|
| `get_vertical_profile(lat, lon, dt)` | Extract one profile at a single datetime; result in `self.vertical_profile` |
| `get_vertical_profiles(lat, lon, dt=None, start_dt=None, end_dt=None)` | Extract profiles over a list of datetimes or a time range; result in `self.vertical_profiles` |
| `plot_vertical_profile(figsize, display, filename)` | 5-panel plot (one per variable) vs. depth |
| `plot_vertical_profiles(figsize, cmap, display, filename)` | Overlay all profiles, time-coloured by colormap |

### Example

```python
from ecosound.environment import NECOFS
from datetime import datetime

lat, lon = 42.40, -70.12

# --- Use default 30-year hindcast (swap URL for operational forecast) ---
necofs = NECOFS(verbose=True)
# necofs = NECOFS(url=NECOFS.GOM3_FORECAST_URL)  # operational forecast

# --- Single profile ---
ds_profile = necofs.get_vertical_profile(
    lat=lat, lon=lon,
    dt=datetime(2015, 8, 1, 12, 0, 0),
)
print(ds_profile)
# Dimensions: sigma_layer (surface → bottom)
# Variables:  temperature_C, salinity_PSU, u_ms, v_ms, sound_speed_ms, depth_m, zeta_m

necofs.plot_vertical_profile(filename="profile_20150801.png")

# --- Multiple profiles over a time range ---
ds_multi = necofs.get_vertical_profiles(
    lat=lat, lon=lon,
    start_dt=datetime(2015, 8, 1),
    end_dt=datetime(2015, 8, 7),
)
print(ds_multi)
# Dimensions: (time, sigma_layer)

necofs.plot_vertical_profiles(cmap="plasma")

# --- Discrete list of datetimes ---
ds_discrete = necofs.get_vertical_profiles(
    lat=lat, lon=lon,
    dt=["2015-08-01T00:00", "2015-08-01T06:00", "2015-08-01T12:00"],
)

# --- Access a specific time step ---
ds_noon = ds_multi.sel(time="2015-08-01T12:00", method="nearest")
print(ds_noon["sound_speed_ms"].values)  # sound speed profile at noon
```

> **Dependencies**: `numpy`, `pandas`, `xarray`, `netCDF4` (for OPeNDAP).  
> Optional: `cftime` (for time decoding; falls back to Itime/Itime2 if missing).

---

## 3. Tides — Water Level & Tidal Index

**Class**: `ecosound.environment.Tides`  
**Source**: [NOAA CO-OPS](https://tidesandcurrents.noaa.gov/) REST API (no registration required)  
**Resolution**: 6-minute | **Coverage**: ~500 NOAA tide gauge stations (US coastal)

### Measurements

| Variable | Units | Description |
|---|---|---|
| `water_level_m` | m | Water level above datum |
| `time_since_high_tide_h` | h | Hours since last high tide (NaN before first detected peak) |
| `tidal_phase` | — | Fractional tidal cycle [0, 1): 0 = high tide, ~0.5 = low tide |
| `high_tide_times` | datetime64 | Class attribute: detected high-tide peak datetimes |
| `high_tide_levels` | m | Class attribute: water level at each high-tide peak |
| `low_tide_times` | datetime64 | Class attribute: detected low-tide trough datetimes |
| `low_tide_levels` | m | Class attribute: water level at each low-tide trough |

### Methods

| Method | Description |
|---|---|
| `find_nearest_station(lat, lon)` | Returns `(id, name, lat, lon)` of the nearest NOAA gauge |
| `get_water_level(lat, lon, start_dt, end_dt, time_ref, product, datum, compute_tidal_index)` | Fetch water level; result in `self.water_level` |
| `plot_water_level(figsize, display, filename)` | Plot water level with high/low tide markers |

**Key parameters for `get_water_level`**:
- `product`: `"predictions"` (astronomical, default) or `"water_level"` (observed, includes surge)
- `datum`: `"MLLW"` (default), `"MSL"`, `"MHW"`, `"NAVD"`, etc.
- `compute_tidal_index`: `True` (default) — adds `time_since_high_tide_h` and `tidal_phase` to dataset (requires `scipy`)
- `time_ref`: pass another `xr.Dataset` to derive the time range from its `.time` coordinate

### Example

```python
from ecosound.environment import Tides
from datetime import datetime

lat, lon = 42.40, -70.12

tides = Tides(verbose=True)

# --- Find nearest gauge before fetching ---
station_id, name, slat, slon = tides.find_nearest_station(lat, lon)
print(f"Nearest station: {name} (ID {station_id})")

# --- Fetch astronomical tide predictions with tidal index ---
ds = tides.get_water_level(
    lat=lat, lon=lon,
    start_dt=datetime(2015, 8, 1),
    end_dt=datetime(2015, 8, 31),
    product="predictions",    # clean astronomical signal
    datum="MLLW",
    compute_tidal_index=True, # adds time_since_high_tide_h and tidal_phase
)
print(ds)
# Variables: water_level_m, time_since_high_tide_h, tidal_phase

# --- Inspect detected tides ---
print(f"High tides: {len(tides.high_tide_times)}")
for t, h in zip(tides.high_tide_times[:5], tides.high_tide_levels[:5]):
    print(f"  {t}  {h:.3f} m")

# --- Plot ---
tides.plot_water_level(filename="tides_august2015.png")

# --- Observed water level (includes meteorological surge) ---
ds_obs = tides.get_water_level(
    lat=lat, lon=lon,
    start_dt="2015-08-01", end_dt="2015-08-31",
    product="water_level",
    datum="MSL",
)

# --- Derive time range from another dataset ---
# tides.get_water_level(lat=lat, lon=lon, time_ref=necofs.vertical_profiles)

# --- Align to a higher-resolution dataset ---
# ds_aligned = tides.water_level.reindex(
#     time=hmd.ds.time, method="nearest", tolerance="10min")
```

> **Dependencies**: `numpy`, `pandas`, `xarray`.  
> Optional: `scipy` (required for `compute_tidal_index=True`); `requests`.

---

## 4. LunarSolar — Sun & Moon Ephemeris

**Class**: `ecosound.environment.LunarSolar`  
**Source**: Computed locally using [PyEphem](https://rhodesmill.org/pyephem/) — no internet required  
**Coverage**: Any date and location

### Measurements

| Variable | Units | Description |
|---|---|---|
| `sun_altitude_deg` | ° | Solar altitude above horizon (negative = below) |
| `sun_azimuth_deg` | ° | Solar azimuth (0=N, 90=E, 180=S, 270=W) |
| `is_day` | bool | Sun altitude > 0° |
| `is_civil_twilight` | bool | −6° < sun altitude ≤ 0° |
| `is_nautical_twilight` | bool | −12° < sun altitude ≤ −6° |
| `is_night` | bool | Sun altitude ≤ −12° |
| `moon_altitude_deg` | ° | Lunar altitude above horizon (negative = below) |
| `moon_azimuth_deg` | ° | Lunar azimuth (0=N, 90=E, 180=S, 270=W) |
| `moon_illumination_pct` | % | Fraction of lunar surface illuminated (0=new, 100=full) |
| `moon_phase` | — | Fractional lunar cycle [0, 1): 0=new, 0.25=first quarter, 0.5=full, 0.75=last quarter |
| `is_moon_up` | bool | Moon altitude > 0° |

### Methods

| Method | Description |
|---|---|
| `get_timeseries(lat, lon, start_dt, end_dt, freq, time_ref)` | Compute ephemeris time series; result in `self.timeseries` |
| `plot_timeseries(figsize, display, filename)` | 3-panel plot: solar altitude, lunar altitude, lunar illumination |

**Key parameters for `get_timeseries`**:
- `freq`: any pandas frequency string (e.g., `"1h"`, `"6min"`, `"30min"`, `"1D"`) — default `"1h"`
- `time_ref`: pass another `xr.Dataset` to compute ephemeris at its exact timestamps (ignores `freq`)

### Example

```python
from ecosound.environment import LunarSolar
from datetime import datetime

lat, lon = 42.40, -70.12

ls = LunarSolar(verbose=True)

# --- Hourly ephemeris ---
ds = ls.get_timeseries(
    lat=lat, lon=lon,
    start_dt=datetime(2015, 8, 1),
    end_dt=datetime(2015, 8, 31),
    freq="1h",
)
print(ds)
# Variables: sun_altitude_deg, sun_azimuth_deg, is_day, is_civil_twilight,
#            is_nautical_twilight, is_night, moon_altitude_deg, moon_azimuth_deg,
#            moon_illumination_pct, moon_phase, is_moon_up

ls.plot_timeseries(filename="ephemeris_august2015.png")

# --- Sub-hourly to match tide gauge (6-minute) ---
ds_6min = ls.get_timeseries(
    lat=lat, lon=lon,
    start_dt="2015-08-01", end_dt="2015-08-31",
    freq="6min",
)

# --- Match exact timestamps of another dataset ---
# ls.get_timeseries(lat=lat, lon=lon, time_ref=tides.water_level)
# ds_combined = xr.merge([tides.water_level, ls.timeseries])

# --- Count nighttime records ---
n_night = int(ds["is_night"].sum())
print(f"Nighttime hours in August: {n_night}")

# --- Filter to full-moon periods ---
full_moon = ds.where(ds["moon_phase"].between(0.47, 0.53), drop=True)
print(f"Full-moon timestamps: {full_moon.time.values}")
```

> **Dependencies**: `numpy`, `pandas`, `xarray`, `ephem` (`pip install ephem`).

---

## 5. ERDDAPDataFetcher — Generic Gridded Data

**Class**: `ecosound.environment.ERDDAPDataFetcher`  
**Source**: Any [ERDDAP](https://www.ncei.noaa.gov/erddap/) server (e.g., NOAA CoastWatch, NEFSC)  
**Coverage**: Dataset-dependent; handles dateline crossing and auto-chunking of large requests

### Measurements

Variables are dataset-specific. Common examples on NOAA ERDDAP servers:

| Variable | Description |
|---|---|
| `sea_surface_temperature` | SST (°C) |
| `chlorophyll` | Chlorophyll-a concentration |
| `quality_level` | Quality flag (e.g., 5 = best quality) |

### Methods

| Method | Description |
|---|---|
| `list_datasets(search_text=None)` | List datasets on the server; optionally filter by keyword |
| `get_dataset_info(dataset_id=None)` | Return metadata dictionary for a dataset |
| `list_variables(dataset_id=None)` | List variable names in a dataset |
| `fetch_data(variables, *, date=None, start_date=None, end_date=None, lat_min, lat_max, lon_min, lon_max, ...)` | Fetch gridded data; auto-chunks large time ranges |

**Key parameters for `fetch_data`**:
- `variables`: variable name (string) or list of names
- `date`: single date `"YYYY-MM-DD"` (alternative to `start_date`/`end_date`)
- `quality_mask_value`: keep only records where `quality_level == value` (e.g., `5` for best)
- `spatial_stride`: spatial thinning (1 = all points, 2 = every other point)
- `time_stride`: temporal thinning
- `max_request_duration_days`: split large ranges into chunks (default: 31 days)

### Example

```python
from ecosound.environment import ERDDAPDataFetcher

ERDDAP_SERVER  = "https://comet.nefsc.noaa.gov/erddap"
ACSPO_DATASET  = "noaa_coastwatch_acspo_v2_reanalysis"

fetcher = ERDDAPDataFetcher(server=ERDDAP_SERVER, dataset_id=ACSPO_DATASET)

# --- Explore available datasets ---
all_datasets = fetcher.list_datasets()
print(f"Total datasets: {len(all_datasets)}")
print(all_datasets.head())

sst_datasets = fetcher.list_datasets(search_text="temperature")
print(sst_datasets[["dataset_id", "title"]])

# --- Inspect a dataset ---
info = fetcher.get_dataset_info()
variables = fetcher.list_variables()
print(f"Variables: {variables}")

# --- Fetch SST for a single day (best quality only) ---
ds_sst = fetcher.fetch_data(
    "sea_surface_temperature",
    date="2018-06-15",
    lat_min=42.0, lat_max=44.0,
    lon_min=-71.0, lon_max=-68.0,
    include_quality=True,
    quality_mask_value=5,   # 5 = best quality
)
print(ds_sst)

# --- Fetch SST over a date range (auto-chunked into monthly requests) ---
ds_range = fetcher.fetch_data(
    "sea_surface_temperature",
    start_date="2018-01-01",
    end_date="2018-06-30",
    lat_min=42.0, lat_max=44.0,
    lon_min=-71.0, lon_max=-68.0,
    quality_mask_value=5,
    spatial_stride=2,               # take every other grid point
    max_request_duration_days=31,   # request one month at a time
)

# --- Fetch multiple variables ---
ds_multi = fetcher.fetch_data(
    ["sea_surface_temperature", "quality_level"],
    start_date="2018-06-01", end_date="2018-06-30",
    lat_min=42.0, lat_max=44.0,
    lon_min=-71.0, lon_max=-68.0,
)

# --- Use a different dataset or server ---
fetcher2 = ERDDAPDataFetcher(
    server="https://coastwatch.pfeg.noaa.gov/erddap",
    dataset_id="erdMH1chla8day",
)
ds_chl = fetcher2.fetch_data(
    "chlorophyll",
    start_date="2018-06-01", end_date="2018-06-30",
    lat_min=42.0, lat_max=44.0,
    lon_min=-71.0, lon_max=-68.0,
)
```

> **Dependencies**: `numpy`, `pandas`, `xarray`.  
> Optional: `requests` (faster downloads; falls back to `urllib`).

### Visualization with GridPlotter

`GridPlotter` (`ecosound.visualization.GridPlotter`) creates publication-quality
static maps (Cartopy/matplotlib) and interactive HTML maps (Folium) from any
`xr.DataArray` with `(time, latitude, longitude)` dimensions — exactly the
format returned by ERDDAP.

#### Gulf of Maine ERDDAP datasets used below

| Variable | Dataset ID | Server | Notes |
|---|---|---|---|
| `sea_surface_temperature` | `noaa_coastwatch_acspo_v2_reanalysis` | `https://comet.nefsc.noaa.gov/erddap` | Daily, ~2 km |
| `chlor_a` | `occci_v6_daily_1km` | `https://comet.nefsc.noaa.gov/erddap` | Daily, 1 km OC-CCI v6 |

#### Static map — Sea Surface Temperature

```python
from ecosound.environment import ERDDAPDataFetcher
from ecosound.visualization import GridPlotter
import pandas as pd
import os

# --- Study area (Gulf of Maine) ---
lat_min, lat_max = 41.0, 44.5
lon_min, lon_max = -71.3, -64.0

# --- Hydrophone recorder locations ---
recorder_data = pd.DataFrame({
    'name':      ['SB03',    'NRS09'],
    'latitude':  [42.2554,   42.40382],
    'longitude': [-70.1786, -70.12225],
    'depth_m':   [45,        78],
})

outdir = r"C:\my_output\sst"
os.makedirs(outdir, exist_ok=True)

# --- Fetch daily SST (best-quality only) ---
fetcher = ERDDAPDataFetcher(
    server="https://comet.nefsc.noaa.gov/erddap",
    dataset_id="noaa_coastwatch_acspo_v2_reanalysis",
)
ds_sst = fetcher.fetch_data(
    "sea_surface_temperature",
    start_date="2018-06-01",
    end_date="2018-08-31",
    lat_min=lat_min, lat_max=lat_max,
    lon_min=lon_min, lon_max=lon_max,
    quality_mask_value=5,            # keep only best-quality pixels
    max_request_duration_days=31,
)

# --- Weekly mean ---
ds_weekly = ds_sst.resample(time="1W").mean(skipna=True)

# --- Plot each week ---
plotter = GridPlotter()
for t in ds_weekly.time:
    date_str = str(t.values)[:10]
    fig = plotter.plot_static_map(
        ds_weekly.sea_surface_temperature,
        timestamp=date_str,
        label="Weekly Mean Sea Surface Temperature (°C)",
        colormap="RdYlBu_r",
        vmin=-1,   vmax=30,
        cbar_min=-1, cbar_max=30,
        axes_fontsize=8,
        marker_type="o",
        marker_size=2,
        colorbar_fontsize=10,
        title_fontsize=10,
        recorder_df=recorder_data,
        show_recorder_names=True,
        bathymetry_contours=[-200],  # 200 m isobath
        bathymetry_color="black",
        bathymetry_linewidth=0.3,
        bathymetry_linestyle="-",
        bathymetry_fontsize=5,
        recorder_name_fontsize=8,
        dpi=300,
        save_path=os.path.join(outdir, f"{date_str}_weekly_sst.png"),
        show=False,
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

print(f"Figures saved to: {outdir}")
```

#### Static map — Chlorophyll-a

```python
from ecosound.environment import ERDDAPDataFetcher
from ecosound.visualization import GridPlotter
import pandas as pd, os, matplotlib.pyplot as plt

lat_min, lat_max = 41.0, 44.5
lon_min, lon_max = -71.3, -64.0

recorder_data = pd.DataFrame({
    'name':      ['SB03',    'NRS09'],
    'latitude':  [42.2554,   42.40382],
    'longitude': [-70.1786, -70.12225],
    'depth_m':   [45,        78],
})

outdir = r"C:\my_output\chla"
os.makedirs(outdir, exist_ok=True)

# --- Fetch daily chlorophyll (OC-CCI v6) ---
fetcher = ERDDAPDataFetcher(
    server="https://comet.nefsc.noaa.gov/erddap",
    dataset_id="occci_v6_daily_1km",
)
ds_chla = fetcher.fetch_data(
    "chlor_a",
    start_date="2023-05-01",
    end_date="2023-08-31",
    lat_min=lat_min, lat_max=lat_max,
    lon_min=lon_min, lon_max=lon_max,
    max_request_duration_days=31,
)

# --- Weekly mean ---
ds_weekly = ds_chla.resample(time="1W").mean(skipna=True)

# --- Plot each week ---
plotter = GridPlotter()
for t in ds_weekly.time:
    date_str = str(t.values)[:10]
    fig = plotter.plot_static_map(
        ds_weekly.chlor_a,
        timestamp=date_str,
        label="Weekly Mean Chlorophyll-a (mg m⁻³)",
        colormap="cmo.tempo",        # requires cmocean: pip install cmocean
        vmin=-0.1,  vmax=10,
        cbar_min=0.1, cbar_max=10,
        axes_fontsize=8,
        marker_type="o",
        marker_size=2,
        colorbar_fontsize=10,
        title_fontsize=10,
        recorder_df=recorder_data,
        show_recorder_names=True,
        bathymetry_contours=[-200],
        bathymetry_color="black",
        bathymetry_linewidth=0.3,
        bathymetry_linestyle="-",
        bathymetry_fontsize=5,
        recorder_name_fontsize=8,
        dpi=300,
        save_path=os.path.join(outdir, f"{date_str}_weekly_chla.png"),
        show=False,
    )
    plt.close(fig)

print(f"Figures saved to: {outdir}")
```

#### Interactive HTML map — SST time slider (Folium)

```python
from ecosound.environment import ERDDAPDataFetcher
from ecosound.visualization import GridPlotter
import pandas as pd

lat_min, lat_max = 41.0, 44.5
lon_min, lon_max = -71.3, -64.0

recorder_data = pd.DataFrame({
    'name':      ['SB03',    'NRS09'],
    'latitude':  [42.2554,   42.40382],
    'longitude': [-70.1786, -70.12225],
})

fetcher = ERDDAPDataFetcher(
    server="https://comet.nefsc.noaa.gov/erddap",
    dataset_id="noaa_coastwatch_acspo_v2_reanalysis",
)
ds_sst = fetcher.fetch_data(
    "sea_surface_temperature",
    start_date="2018-06-01", end_date="2018-08-31",
    lat_min=lat_min, lat_max=lat_max,
    lon_min=lon_min, lon_max=lon_max,
    quality_mask_value=5,
    max_request_duration_days=31,
)
ds_weekly = ds_sst.resample(time="1W").mean(skipna=True)

# --- Interactive map with time slider ---
plotter = GridPlotter(basemap="Esri Ocean", zoom_start=7)
plotter.plot_timeseries_grid(
    ds_weekly.sea_surface_temperature,
    label="Weekly Mean SST (°C)",
    colormap="RdYlBu_r",
    opacity=0.7,
    playback_speed_ms=800,
)
plotter.add_recorder_locations(recorder_data)
plotter.save("sst_timeseries_map.html")   # open in any browser
# plotter.display()                        # opens immediately in default browser
```

> **GridPlotter dependencies**: `cartopy`, `matplotlib-scalebar` (static maps); `folium`, `branca` (interactive maps).  
> Optional for bathymetry contours: `pooch` (downloads ETOPO1 on first call).  
> Optional for chlorophyll colormap: `cmocean`.

---

## 6. AISDataDownloaderDuckDB — Download AIS Data

**Class**: `ecosound.environment.AISDataDownloaderDuckDB`  
**Source**: [Marine Cadastre (NOAA)](https://marinecadastre.gov/ais/) — daily ZIP/CSV files  
**Purpose**: Download raw AIS data and ingest it into a local **DuckDB + Parquet** database for fast queries with `AISQueryHelper`.

> **Run this first** to build the local database, then use `AISQueryHelper` to query it.

### Methods

| Method | Description |
|---|---|
| `__init__(db_path, parquet_dir, temp_dir=None)` | Initialize, create DB schema and Parquet directory |
| `setup_database()` | Create metadata and vessel-type-lookup tables |
| `generate_date_urls(start_date, end_date)` | List download URLs for a date range |
| `is_date_in_database(date_str)` | Check if a date is already ingested |
| `download_files(start_date, end_date, max_concurrent, force_download)` | Async parallel download of ZIP files |
| `extract_and_process_file(zip_path, min_lat, max_lat, min_lon, max_lon, force_process)` | Extract ZIP, apply geographic filter, write Parquet |
| `process_all_files(downloaded_files, min_lat, max_lat, min_lon, max_lon, max_workers, force_process)` | Process downloaded files in parallel |
| `create_ais_view()` | Create `ais_data` view over all Parquet files |
| `create_vessel_type_lookup_table()` | Populate vessel-type code lookup table (AIS standard) |
| `optimize_database()` | Create view + optimize (run after ingestion) |
| `get_stats()` | Print total records, unique vessels, date range, geographic bounds, storage size |
| `cleanup_temp_files()` | Delete downloaded ZIP files from temp directory |
| `example_queries()` | Print sample DuckDB SQL queries |

### Example

```python
import asyncio
from ecosound.environment import AISDataDownloaderDuckDB

# --- Initialize (creates DB schema and Parquet directory) ---
downloader = AISDataDownloaderDuckDB(
    db_path="./ais_gom.duckdb",
    parquet_dir="./ais_parquet",
    temp_dir="./ais_temp",    # optional; defaults to system temp
)

# --- Download with optional geographic pre-filter ---
# (filtering here reduces storage significantly)
downloaded_files = asyncio.run(
    downloader.download_files(
        start_date="2018-01-01",
        end_date="2018-01-31",
        max_concurrent=5,        # parallel downloads
        force_download=False,    # skip dates already in DB
    )
)

# --- Process (extract ZIP, filter, convert to Parquet) ---
downloader.process_all_files(
    downloaded_files,
    min_lat=41.65, max_lat=46.02,   # Gulf of Maine bounding box
    min_lon=-71.1,  max_lon=-65.0,
    max_workers=4,
    force_process=False,
)

# --- Finalize ---
downloader.optimize_database()              # create ais_data view
downloader.create_vessel_type_lookup_table()  # populate vessel type codes
downloader.get_stats()                      # print DB summary
downloader.cleanup_temp_files()             # delete temp ZIPs

# --- Check if a date is already downloaded ---
if downloader.is_date_in_database("2018-01-15"):
    print("Jan 15 already in database")
```

**Command-line usage** (equivalent):

```bash
python -m ecosound.environment.ais_downloader \
    --start 2018-01-01 --end 2018-01-31 \
    --min-lat 41.65 --max-lat 46.02 \
    --min-lon -71.1  --max-lon -65.0 \
    --db ./ais_gom.duckdb

# View statistics only
python -m ecosound.environment.ais_downloader --stats-only --db ./ais_gom.duckdb
```

> **Dependencies**: `duckdb`, `pandas`, `aiohttp`, `pyarrow`.

---

## 7. AISQueryHelper — Query AIS Vessel Traffic

**Class**: `ecosound.environment.AISQueryHelper`  
**Source**: Local DuckDB + Parquet database (created by `AISDataDownloaderDuckDB`)  
**Returns**: `GeoDataFrame` (WGS84/EPSG:4326 Point geometries) or `xarray.DataArray`

### AIS Fields

| Field | Description |
|---|---|
| `mmsi` | Maritime Mobile Service Identity (unique vessel ID) |
| `base_datetime` | Timestamp of AIS report |
| `latitude`, `longitude` | Position (WGS84) |
| `sog` | Speed over ground (knots) |
| `cog` | Course over ground (°) |
| `heading` | Vessel heading (°) |
| `vessel_name`, `imo`, `call_sign` | Vessel identity |
| `vessel_type` | AIS numeric type code |
| `vessel_type_name` | Human-readable type (from lookup) |
| `vessel_category` | Category: Cargo, Tanker, Fishing, Passenger, etc. |
| `status` | Navigation status code |
| `length`, `width`, `draft` | Vessel dimensions (m) |
| `cargo` | Cargo type code |
| `distance_km` | Distance from query center (radius queries only) |

### Methods

| Method | Returns | Description |
|---|---|---|
| `query_rectangle(start_date, end_date, min_lat, max_lat, min_lon, max_lon, mmsi=None, vessel_type=None, vessel_name=None, limit=None)` | GeoDataFrame | Bounding-box query |
| `query_radius(start_date, end_date, center_lat, center_lon, radius_km, mmsi=None, vessel_type=None, limit=None)` | GeoDataFrame | Circular query with `distance_km` column |
| `query_vessel_track(mmsi, start_date, end_date, min_lat=None, max_lat=None, min_lon=None, max_lon=None)` | GeoDataFrame | All positions for a single vessel, sorted by time |
| `query_by_vessel_category(start_date, end_date, min_lat, max_lat, min_lon, max_lon, categories, limit=None)` | GeoDataFrame | Filter by vessel category (e.g., `"Cargo"`, `["Cargo", "Tanker"]`) |
| `get_unique_vessels(start_date, end_date, min_lat, max_lat, min_lon, max_lon)` | DataFrame | Unique vessels with count, first/last seen |
| `get_statistics(start_date, end_date, min_lat=None, ...)` | dict | Aggregate stats: total records, unique vessels, speed stats, geographic bounds |
| `create_gridded_vessel_counts(start_date, end_date, min_lat, max_lat, min_lon, max_lon, width_km, height_km, time_resolution_hours)` | xr.DataArray | Gridded unique-vessel counts `(time, lat, lon)` |
| `calculate_bounding_box(center_lat, center_lon, radius_km)` | dict | Utility: compute bounding box for a radius |

All query methods accept `return_gdf=False` to return a plain `DataFrame`.

### Example

```python
from ecosound.environment import AISQueryHelper

DB_PATH = "./ais_gom.duckdb"

# Use context manager (recommended — ensures connection is closed)
with AISQueryHelper(DB_PATH) as ais:

    # --- 1. Bounding-box query ---
    gdf = ais.query_rectangle(
        start_date="2018-01-01",
        end_date="2018-01-07",
        min_lat=41.65, max_lat=46.02,
        min_lon=-71.1,  max_lon=-65.0,
        limit=5000,
    )
    print(f"Records: {len(gdf)}")
    print(gdf[["mmsi", "vessel_name", "sog", "vessel_category"]].head())

    # --- 2. Radius query (adds distance_km column) ---
    gdf_r = ais.query_radius(
        start_date="2018-01-01",
        end_date="2018-01-07",
        center_lat=42.5, center_lon=-68.0,
        radius_km=100,
    )
    print(gdf_r[["vessel_name", "vessel_category", "distance_km"]].head())

    # --- 3. Track a specific vessel ---
    gdf_track = ais.query_vessel_track(
        mmsi=367123456,
        start_date="2018-01-01",
        end_date="2018-01-07",
    )
    print(f"Track positions: {len(gdf_track)}")

    # --- 4. Filter by vessel category ---
    gdf_cargo = ais.query_by_vessel_category(
        start_date="2018-01-01",
        end_date="2018-01-07",
        min_lat=41.65, max_lat=46.02,
        min_lon=-71.1,  max_lon=-65.0,
        categories=["Cargo", "Tanker"],  # or single string: "Fishing"
    )

    # --- 5. Unique vessel summary ---
    vessels = ais.get_unique_vessels(
        start_date="2018-01-01",
        end_date="2018-01-07",
        min_lat=41.65, max_lat=46.02,
        min_lon=-71.1,  max_lon=-65.0,
    )
    print(f"Unique vessels: {len(vessels)}")
    print(vessels[["mmsi", "vessel_name", "vessel_category", "num_positions"]].head(10))

    # --- 6. Aggregate statistics ---
    stats = ais.get_statistics(
        start_date="2018-01-01",
        end_date="2018-01-31",
        min_lat=41.65, max_lat=46.02,
        min_lon=-71.1,  max_lon=-65.0,
    )
    print(f"Total records:  {stats['total_records']:,}")
    print(f"Unique vessels: {stats['unique_vessels']:,}")
    print(f"Avg speed:      {stats['avg_speed']:.1f} knots")

    # --- 7. Gridded vessel density (xarray DataArray) ---
    da = ais.create_gridded_vessel_counts(
        start_date="2018-01-01",
        end_date="2018-01-07",
        min_lat=41.65, max_lat=46.02,
        min_lon=-71.1,  max_lon=-65.0,
        width_km=10, height_km=10,
        time_resolution_hours=1,
    )
    print(da)  # dims: (time, latitude, longitude), values: unique vessel count or NaN

    # --- 8. Utility: bounding box from center + radius ---
    bbox = AISQueryHelper.calculate_bounding_box(
        center_lat=42.5, center_lon=-68.0, radius_km=50
    )
    print(bbox)  # {'min_lat': ..., 'max_lat': ..., 'min_lon': ..., 'max_lon': ...}

    # --- 9. Export GeoDataFrame to file ---
    gdf.to_file("ais_january2018.geojson", driver="GeoJSON")
    # gdf.to_file("ais_january2018.shp")      # shapefile
    # gdf.to_parquet("ais_january2018.parquet")  # geoparquet
```

> **Dependencies**: `duckdb`, `pandas`, `numpy`, `geopandas`, `shapely`, `xarray`.

### Visualization with GridPlotter and AISMapPlotter

Two plotters are available in `ecosound.visualization`:

| Plotter | Output | Best for |
|---|---|---|
| `GridPlotter.plot_static_map` | PNG/PDF (Cartopy) | Publication-ready gridded vessel-density maps |
| `GridPlotter.plot_timeseries_grid` | HTML (Folium + time slider) | Exploring vessel density over time |
| `GridPlotter.plot_summary_grid` | HTML (Folium, static) | Overview aggregated map |
| `AISMapPlotter.plot_ais_data` | HTML (Folium, interactive markers) | Exploring individual AIS positions |

#### Static map — gridded vessel counts (publication quality)

The workflow below queries weekly periods, collapses the hourly grid with `.sum()`,
then calls `plot_static_map` for each week — matching the pattern used for SST/Chla.

```python
from ecosound.environment import AISQueryHelper
from ecosound.visualization import GridPlotter
import pandas as pd
import matplotlib.pyplot as plt
import os

DB_PATH = r"C:\Users\xavier.mouy\Documents\GitHub\NERACOOS_processing_scripts\non_acoustic_data\ais_db\gulf_of_maine_ais.duckdb"
outdir  = r"C:\my_output\ais"
os.makedirs(outdir, exist_ok=True)

# --- Study area ---
min_lat, max_lat = 41.0, 44.5
min_lon, max_lon = -71.3, -64.0

# --- Hydrophone recorder locations ---
recorder_data = pd.DataFrame({
    'name':      ['SB03',    'NRS09'],
    'latitude':  [42.2554,   42.40382],
    'longitude': [-70.1786, -70.12225],
    'depth_m':   [45,        78],
})

# --- Weekly periods to process ---
start_date = pd.Timestamp("2018-01-01")
end_date   = pd.Timestamp("2018-03-31")
period_starts = pd.date_range(start=start_date, end=end_date, freq="1W")

plotter = GridPlotter()   # reuse across all weeks

for i, period_start in enumerate(period_starts):
    period_end = (period_starts[i + 1] - pd.Timedelta(seconds=1)
                  if i < len(period_starts) - 1 else end_date)
    date_str = period_start.strftime("%Y-%m-%d")
    print(f"Processing {date_str} …")

    # --- Create hourly gridded vessel counts for this week ---
    with AISQueryHelper(DB_PATH) as ais:
        vessel_grid = ais.create_gridded_vessel_counts(
            start_date=period_start.strftime("%Y-%m-%d"),
            end_date=period_end.strftime("%Y-%m-%d"),
            min_lat=min_lat, max_lat=max_lat,
            min_lon=min_lon, max_lon=max_lon,
            width_km=1, height_km=1,
            time_resolution_hours=1,
        )

    if vessel_grid is None:
        print(f"  No data for {date_str}, skipping.")
        continue

    # --- Aggregate: cumulative vessel count over the week ---
    vessel_grid_sum = vessel_grid.sum(dim="time", skipna=True, min_count=1)

    # --- Static map: cumulative vessel count ---
    fig = plotter.plot_static_map(
        vessel_grid_sum,           # 2D DataArray (no time dim)
        timestamp=date_str,        # used for title only when no time dim
        label="Weekly Cumulative Vessel Count",
        colormap="YlOrRd",
        vmin=0,   vmax=100,
        cbar_min=0, cbar_max=100,
        axes_fontsize=8,
        marker_type="o",
        marker_size=2,
        colorbar_fontsize=10,
        title_fontsize=10,
        recorder_df=recorder_data,
        show_recorder_names=True,
        bathymetry_contours=[-200],
        bathymetry_color="black",
        bathymetry_linewidth=0.3,
        bathymetry_linestyle="-",
        bathymetry_fontsize=5,
        recorder_name_fontsize=8,
        dpi=300,
        save_path=os.path.join(outdir, f"{date_str}_weekly_vessel_count.png"),
        show=False,
    )
    plt.close(fig)

    # --- Static map: peak simultaneous vessel count per cell ---
    vessel_grid_max = vessel_grid.max(dim="time", skipna=True)
    fig = plotter.plot_static_map(
        vessel_grid_max,
        timestamp=date_str,
        label="Weekly Peak Vessel Count (vessels/hour/cell)",
        colormap="YlOrRd",
        vmin=1, vmax=5,
        cbar_min=1, cbar_max=5,
        recorder_df=recorder_data,
        show_recorder_names=True,
        dpi=300,
        save_path=os.path.join(outdir, f"{date_str}_weekly_vessel_max.png"),
        show=False,
    )
    plt.close(fig)

print(f"Done. Figures saved to: {outdir}")
```

#### Interactive HTML map — gridded vessel density time slider

```python
from ecosound.environment import AISQueryHelper
from ecosound.visualization import GridPlotter
import pandas as pd

DB_PATH = r"C:\Users\xavier.mouy\Documents\GitHub\NERACOOS_processing_scripts\non_acoustic_data\ais_db\gulf_of_maine_ais.duckdb"

recorder_data = pd.DataFrame({
    'name':      ['SB03',    'NRS09'],
    'latitude':  [42.2554,   42.40382],
    'longitude': [-70.1786, -70.12225],
})

with AISQueryHelper(DB_PATH) as ais:
    vessel_grid = ais.create_gridded_vessel_counts(
        start_date="2018-01-01", end_date="2018-01-07",
        min_lat=41.0, max_lat=44.5,
        min_lon=-71.3, max_lon=-64.0,
        width_km=5, height_km=5,       # coarser grid for HTML file size
        time_resolution_hours=6,       # 6-hour bins
    )

# --- Time-slider map (one frame per 6-hour bin) ---
plotter = GridPlotter(basemap="Esri Ocean", zoom_start=7)
plotter.plot_timeseries_grid(
    vessel_grid,
    label="Unique Vessel Count",
    colormap="YlOrRd",
    opacity=0.65,
    playback_speed_ms=600,
)
plotter.add_recorder_locations(recorder_data)
plotter.save("ais_vessel_timeseries.html")

# --- Summary map (total over all time steps) ---
plotter2 = GridPlotter(basemap="Esri Ocean", zoom_start=7)
plotter2.plot_summary_grid(
    vessel_grid,
    label="Unique Vessel Count",
    aggregation="sum",
    colormap="YlOrRd",
)
plotter2.add_recorder_locations(recorder_data)
plotter2.save("ais_vessel_summary.html")
```

#### Interactive HTML map — raw AIS positions (AISMapPlotter)

`AISMapPlotter` plots individual AIS records as clickable markers on an
interactive Folium map.  Vessel categories are colour-coded and toggleable
via the layer control panel.

```python
from ecosound.environment import AISQueryHelper
from ecosound.visualization import AISMapPlotter
import pandas as pd

DB_PATH = r"C:\Users\xavier.mouy\Documents\GitHub\NERACOOS_processing_scripts\non_acoustic_data\ais_db\gulf_of_maine_ais.duckdb"

recorder_data = pd.DataFrame({
    'name':      ['SB03',    'NRS09'],
    'latitude':  [42.2554,   42.40382],
    'longitude': [-70.1786, -70.12225],
    'depth_m':   [45,        78],
})

# --- Query raw AIS positions for a small area ---
with AISQueryHelper(DB_PATH) as ais:
    gdf = ais.query_rectangle(
        start_date="2018-01-01",
        end_date="2018-01-03",
        min_lat=42.0, max_lat=42.7,
        min_lon=-70.5, max_lon=-69.6,
        limit=None,
    )
print(f"Found {len(gdf)} AIS records")

# --- Map 1: colour by vessel type (toggleable layers per type) ---
plotter = AISMapPlotter(basemap="Esri Ocean", db_path=DB_PATH)
plotter.plot_ais_data(
    gdf,
    color_by="vessel_type_name",
    color_map=plotter.COLOR_SCHEMES["vessel_type_name"],
    popup_fields=["mmsi", "vessel_name", "vessel_type_name",
                  "sog", "cog", "base_datetime"],
    marker_size=6,
)
plotter.add_recorder_locations(recorder_data)
plotter.save("ais_map_by_type.html")

# --- Map 2: colour by vessel category (broader groups) ---
plotter2 = AISMapPlotter(basemap="CartoDB Positron", db_path=DB_PATH)
plotter2.plot_ais_data(
    gdf,
    color_by="vessel_category",
    color_map=plotter2.COLOR_SCHEMES["vessel_category"],
    popup_fields=["mmsi", "vessel_name", "vessel_category",
                  "length", "sog", "base_datetime"],
    marker_size=6,
)
plotter2.add_recorder_locations(recorder_data)
plotter2.save("ais_map_by_category.html")

# --- Map 3: colour by speed over ground (continuous) ---
plotter3 = AISMapPlotter(basemap="CartoDB Positron")
plotter3.plot_ais_data(
    gdf,
    color_by="sog",
    continuous_cmap="RdYlGn",
    popup_fields=["mmsi", "vessel_name", "sog", "cog", "base_datetime"],
    marker_size=5,
)
plotter3.add_recorder_locations(recorder_data)
plotter3.save("ais_map_by_speed.html")

# --- Map 4: cargo and tanker vessels only, with heatmap overlay ---
with AISQueryHelper(DB_PATH) as ais:
    gdf_cargo = ais.query_by_vessel_category(
        start_date="2018-01-01", end_date="2018-01-07",
        min_lat=42.0, max_lat=42.7,
        min_lon=-70.5, max_lon=-69.6,
        categories=["Cargo", "Tanker"],
    )

plotter4 = AISMapPlotter(basemap="CartoDB Dark", db_path=DB_PATH)
plotter4.plot_ais_data(
    gdf_cargo,
    color_by="vessel_category",
    color_map=plotter4.COLOR_SCHEMES["vessel_category"],
    popup_fields=["mmsi", "vessel_name", "vessel_category", "sog"],
    marker_size=5,
    add_heatmap=True,
)
plotter4.add_recorder_locations(recorder_data)
plotter4.save("ais_map_cargo_tanker_heatmap.html")
```

> **AISMapPlotter dependencies**: `folium`, `branca`, `geopandas`, `duckdb`.  
> **GridPlotter dependencies**: `cartopy`, `matplotlib-scalebar` (static); `folium`, `branca` (interactive).

---

## 8. Merging Multiple Sources

All classes (except AIS) return `xr.Dataset` with `(time, lat, lon)` coordinates.
Use `xr.merge` after aligning time axes with `.reindex(..., method="nearest")`.

```python
import xarray as xr
from datetime import datetime
from ecosound.environment import ERA5, NECOFS, Tides, LunarSolar

lat, lon = 42.40, -70.12
start, end = "2015-08-01", "2015-08-07"

# --- Fetch each source ---
era5   = ERA5()
necofs = NECOFS()
tides  = Tides()
ls     = LunarSolar()

ds_wind  = era5.get_wind_timeseries(lat=lat, lon=lon, start_dt=start, end_dt=end)
ds_ocean = necofs.get_vertical_profiles(lat=lat, lon=lon, start_dt=start, end_dt=end)
ds_tide  = tides.get_water_level(lat=lat, lon=lon, start_dt=start, end_dt=end)
ds_ephem = ls.get_timeseries(lat=lat, lon=lon, start_dt=start, end_dt=end, freq="1h")

# --- Align to NECOFS hourly time axis ---
ds_wind_aligned  = ds_wind.reindex(time=ds_ocean.time, method="nearest", tolerance="1h")
ds_tide_aligned  = ds_tide.reindex(time=ds_ocean.time, method="nearest", tolerance="10min")
ds_ephem_aligned = ds_ephem.reindex(time=ds_ocean.time, method="nearest", tolerance="1h")

# --- Merge (select a single depth layer from ocean profiles first) ---
ds_surface = ds_ocean.isel(sigma_layer=0)   # surface layer
ds_combined = xr.merge([
    ds_surface,
    ds_wind_aligned,
    ds_tide_aligned,
    ds_ephem_aligned,
])
print(ds_combined)

# --- Merge AIS gridded counts with the combined dataset ---
# from ecosound.environment import AISQueryHelper
# with AISQueryHelper("ais_gom.duckdb") as ais:
#     da_vessels = ais.create_gridded_vessel_counts(
#         start_date=start, end_date=end,
#         min_lat=41.65, max_lat=46.02,
#         min_lon=-71.1, max_lon=-65.0,
#         width_km=10, height_km=10)
```

---

## Quick Reference

| Class | Source | Coverage | Key Output Variables |
|---|---|---|---|
| `ERA5` | Open-Meteo / CDS | 1940–present, global, hourly | Wind speed/dir, u/v components, precipitation, rain, snowfall |
| `NECOFS` | FVCOM GOM3 OPeNDAP | Gulf of Maine, hourly | Temperature, salinity, currents (u/v), sound speed, depth |
| `Tides` | NOAA CO-OPS | ~500 US stations, 6-min | Water level, tidal phase, time since high tide |
| `LunarSolar` | PyEphem (local) | Any date & location | Sun/moon altitude & azimuth, day/night flags, lunar illumination & phase |
| `ERDDAPDataFetcher` | Any ERDDAP server | Dataset-specific | SST, chlorophyll, custom gridded variables |
| `AISDataDownloaderDuckDB` | Marine Cadastre (NOAA) | US waters, ~2009–present | Downloads AIS → local DuckDB + Parquet |
| `AISQueryHelper` | Local DuckDB/Parquet | Depends on data downloaded | Vessel position, identity, speed, heading, category; gridded vessel counts |