HMD QA/QC
=========

The :class:`HMDQualityChecker` runs automated quality-control tests on
Hybrid Millidecade (HMD) spectral data loaded via
:class:`~ecosound.soundscape.hmd.HMD`.  It detects hardware failures,
data gaps, sensor degradation, tidal flow noise, and calibration drift,
returning results as ecosound
:class:`~ecosound.core.measurement.Measurement` objects with QARTOD-style
flags.

.. autoclass:: ecosound.soundscape.hmd_qaqc.HMDQualityChecker
   :members: run, summary, to_measurement, to_dataframe, to_csv, plot_overview
   :show-inheritance:
