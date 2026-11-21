Soundscape
==========

The ``soundscape`` module provides tools for processing and analysing
long-term soundscape data.

- :class:`~ecosound.soundscape.hmd.HMD` -- load, integrate, and analyse
  Hybrid Millidecade (HMD) spectral data stored in NetCDF files, with
  optional Dask-based parallel processing for large deployments.
- :class:`~ecosound.soundscape.hmd_qaqc.HMDQualityChecker` -- automated
  QA/QC for HMD data.  Detects hardware failures, data gaps, sensor
  degradation, tidal flow noise, and calibration drift.

.. toctree::
   :maxdepth: 1

   hmd
   hmd_qaqc
