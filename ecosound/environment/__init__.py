# -*- coding: utf-8 -*-
"""
ecosound.environment
====================
Environmental context data fetchers for soundscape interpretation.

Provides classes for retrieving and computing co-variate environmental data
at a fixed location and time range.  All classes return xr.Dataset objects
with consistent (time, lat, lon) coordinates, enabling straightforward merging
with HMD acoustic data via xr.merge() or Dataset.reindex().

Classes
-------
NECOFS          : FVCOM-GOM3 ocean model vertical profiles (temp, salinity,
                  currents, sound speed) via OPeNDAP.
ERA5            : ERA5 reanalysis wind and precipitation time series via
                  Open-Meteo (default, no registration) or Copernicus CDS.
Tides           : NOAA CO-OPS water level, tidal index (time since high tide,
                  tidal phase), and detected tide event times.
LunarSolar      : Lunar (illumination, phase, altitude) and solar (altitude,
                  day/night/twilight flags) ephemeris — computed locally,
                  no network required.  Requires PyEphem.
ERDDAPDataFetcher: Generic gridded data fetcher for any ERDDAP server
                  (SST, chlorophyll, etc.).
AISQueryHelper          : Vessel traffic queries from a DuckDB + Parquet AIS database.
AISDataDownloaderDuckDB : Downloads raw AIS data from Marine Cadastre and ingests it
                          into DuckDB + Parquet (one-time setup tool).

Note
----
AIS map visualisation (interactive HTML maps via Folium) is in
ecosound.visualization.AISMapPlotter, consistent with other plotters.

Quick start
-----------
    from ecosound.environment import NECOFS, ERA5, Tides, LunarSolar
    from ecosound.environment import ERDDAPDataFetcher, AISQueryHelper

Optional dependencies (installed separately as needed)
-------------------------------------------------------
    ephem       : required by LunarSolar
    cdsapi      : required by ERA5(source="cds")
    netCDF4     : required by NECOFS (OPeNDAP access)
    duckdb      : required by AISQueryHelper
    geopandas   : required by AISQueryHelper
    scipy       : required by Tides (tidal index computation)
    requests    : optional but recommended by ERA5, Tides, LunarSolar
                  (falls back to urllib if not installed)
"""

from .necofs import NECOFS
from .era5 import ERA5
from .tides import Tides
from .lunarsolar import LunarSolar
from .erddap import ERDDAPDataFetcher
from .ais import AISQueryHelper
from .ais_downloader import AISDataDownloaderDuckDB

__all__ = [
    "NECOFS",
    "ERA5",
    "Tides",
    "LunarSolar",
    "ERDDAPDataFetcher",
    "AISQueryHelper",
    "AISDataDownloaderDuckDB",
]