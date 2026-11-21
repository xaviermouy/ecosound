HMD QA/QC tutorial
==================

This tutorial demonstrates how to run automated quality control on
Hybrid Millidecade (HMD) passive acoustic data using
:class:`~ecosound.soundscape.hmd_qaqc.HMDQualityChecker`.

Overview
--------

The QA/QC system runs six configurable tests on HMD spectral data and
outputs the detected issues as ecosound
:class:`~ecosound.core.measurement.Measurement` objects with QARTOD-style
quality flags.

The six detection tiers are:

+------+-------------------+-------------------------------------------+--------+
| Tier | Name              | What it detects                           | Flag   |
+======+===================+===========================================+========+
| T0   | Gross range       | Broadband SPL exceeding clipping level    | 4 Fail |
+------+-------------------+-------------------------------------------+--------+
| T1   | Structural        | Data gaps, zero-effort, duty-cycle issues | 4 / 9  |
+------+-------------------+-------------------------------------------+--------+
| T2   | Statistical       | Rolling-MAD outliers, flat-line channels  | 4 Fail |
+------+-------------------+-------------------------------------------+--------+
| T3   | Spectral (PCA)    | Unusual spectral shapes                   | 3 Susp |
+------+-------------------+-------------------------------------------+--------+
| T4   | Tidal (optional)  | Flow-noise correlated with tide           | 3 Susp |
+------+-------------------+-------------------------------------------+--------+
| T5   | Level shift       | Sustained calibration drift (PELT)        | 3 Susp |
+------+-------------------+-------------------------------------------+--------+


Loading HMD data
----------------

Start by loading data with the :class:`~ecosound.soundscape.hmd.HMD` class:

.. code-block:: python

   from ecosound.soundscape.hmd import HMD

   hmd = HMD(use_dask=False)
   hmd.load_nc_files(r"C:\data\deployment_01\nc")


Basic QA/QC run
---------------

Create a checker, run all tests, and view the summary:

.. code-block:: python

   from ecosound.soundscape.hmd_qaqc import HMDQualityChecker

   qc = HMDQualityChecker(hmd, sensitivity=2)
   qc.run()
   qc.summary()

Sample output::

   QA/QC Summary
   ======================================================================
   Per-test detections: 39
   test_name
   tidal_flow_noise    36
   zero_effort          3

   Merged events: 18

   By issue type:
   issue_type
   flow_noise              15
   flow_noise + no_data     2
   no_data                  1

   By flag:
     3 (Suspect): 15
     4 (Fail): 3


Exporting results
-----------------

Export as an ecosound Measurement (recommended):

.. code-block:: python

   # Cross-test merged events (default)
   meas = qc.to_measurement()

   # Save to NetCDF or CSV
   meas.to_netcdf("qc_events.nc")
   meas.to_csv("qc_events.csv")

   # Inspect the DataFrame
   print(meas.data[['time_min_date', 'time_max_date',
                     'label_class', 'label_subclass',
                     'comment', 'test_name', 'issue_type',
                     'severity']])

The Measurement object contains:

- **Standard Annotation columns**: ``time_min_date``, ``time_max_date``,
  ``frequency_min``, ``frequency_max``, ``label_class`` (``"QCEvent"``),
  ``label_subclass`` (QARTOD flag as string), ``comment`` (test names),
  ``confidence``.
- **QC-specific measurement columns**: ``test_name``, ``issue_type``,
  ``severity``, ``severity_unit``, ``description``.

To get the per-test (unmerged) detections instead:

.. code-block:: python

   per_test = qc.to_measurement(merged=False)

Or export a plain DataFrame / CSV:

.. code-block:: python

   df = qc.to_dataframe()
   qc.to_csv("qc_events.csv")


QC visualization
-----------------

Generate a three-panel overview figure:

.. code-block:: python

   qc.plot_overview(
       db_range=(30, 115),
       save_path="qc_overview.png",
   )

The figure shows:

1. **LTSA** with merged event boxes overlaid (orange = Suspect,
   red = Fail, gray = Missing).
2. **Decidecade band time series** (selection of 6 bands).
3. **Per-test event timeline** colored by flag.


Tuning sensitivity
------------------

The ``sensitivity`` parameter controls the MAD threshold and flat-line
detection for T2 and T3:

+-------------+-------+---------------------+
| Sensitivity | MAD k | Min variance factor |
+=============+=======+=====================+
| 1           | 10.0  | 0.03 (permissive)   |
+-------------+-------+---------------------+
| 2           | 7.0   | 0.05 (balanced)     |
+-------------+-------+---------------------+
| 3           | 5.0   | 0.08 (strict)       |
+-------------+-------+---------------------+

The level-shift detector (T5) can be tuned independently:

+-------------------------------+--------------------+
| ``level_shift_sensitivity``   | Offset threshold   |
+===============================+====================+
| 1                             | 10.0 dB            |
+-------------------------------+--------------------+
| 2                             |  6.0 dB            |
+-------------------------------+--------------------+
| 3                             |  4.0 dB            |
+-------------------------------+--------------------+

Example -- sensitive flat-line detection with permissive level-shift:

.. code-block:: python

   qc = HMDQualityChecker(
       hmd,
       sensitivity=2,
       level_shift_sensitivity=1,
   )
   qc.run()


Enabling / disabling tests
--------------------------

Each test can be toggled individually:

.. code-block:: python

   qc = HMDQualityChecker(
       hmd,
       run_t0=True,
       run_t1=True,
       run_t2=True,
       run_t3=False,            # skip PCA test
       detect_tidal_noise=True,  # enable tidal (off by default)
       run_t5=True,
   )
   qc.run()


Decidecade bands from HMD
--------------------------

The HMD class now has built-in decidecade band integration:

.. code-block:: python

   # Get decidecade band definitions
   bands = HMD.decidecade_bands(fmin=20, fmax=20000)
   print(f"{len(bands)} bands")
   print(f"First: {bands[0]}")   # (center, lower, upper)

   # Integrate loaded HMD data into decidecade bands
   dd = hmd.extract_decidecade_levels()
   print(dd)  # xarray.Dataset with one DataArray per band

   # Access a specific band
   ts_125 = dd['125Hz']


QARTOD flag reference
---------------------

+------+------------------+------------------------------+
| Flag | Meaning          | Used by                      |
+======+==================+==============================+
| 1    | Pass             | (unflagged data)             |
+------+------------------+------------------------------+
| 3    | Suspect          | T3, T4, T5                   |
+------+------------------+------------------------------+
| 4    | Fail             | T0, T1, T2                   |
+------+------------------+------------------------------+
| 9    | Missing          | T1 (data gaps)               |
+------+------------------+------------------------------+


Complete QA/QC workflow
-----------------------

.. code-block:: python

   from ecosound.soundscape.hmd import HMD
   from ecosound.soundscape.hmd_qaqc import HMDQualityChecker

   # 1. Load data
   hmd = HMD(use_dask=False)
   hmd.load_nc_files(r"C:\data\deployment_01\nc")

   # 2. Run QA/QC
   qc = HMDQualityChecker(
       hmd,
       sensitivity=2,
       level_shift_sensitivity=1,
       min_event_duration_min=60,
       detect_tidal_noise=True,
   )
   qc.run()

   # 3. Review results
   qc.summary()
   qc.plot_overview(save_path="qc_overview.png")

   # 4. Export as Measurement
   meas = qc.to_measurement()
   meas.to_netcdf("qc_events.nc")

   # 5. Export per-test detections
   per_test = qc.to_measurement(merged=False)
   per_test.to_csv("qc_per_test.csv")

   # 6. Reload later
   from ecosound.core.measurement import Measurement
   meas2 = Measurement()
   meas2.from_netcdf("qc_events.nc")
   print(meas2.data)
