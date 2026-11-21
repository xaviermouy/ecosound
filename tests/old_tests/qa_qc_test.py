from ecosound.soundscape.hmd import HMD
from ecosound.soundscape.hmd_qaqc import HMDQualityChecker
from ecosound.core.measurement import Measurement

# Load HMD data
hmd = HMD(use_dask=False)
hmd.load_nc_files(r"I:\HMD_output\NC")

# Run QA/QC with default settings
qc = HMDQualityChecker(
    hmd,
    sensitivity=3,
    rolling_window_hours=6,
    min_event_duration_min = 24*60,
    merge_gap_hours=6,
    run_t0=False,  # gross range
    run_t1=False,  # structural/gaps
    run_t2=False,  # statistical
    run_t3=False,  # PCA spectral
    detect_tidal_noise=False,  # tidal (off by default)
    run_t5=False, # level shift
    run_t6=True,  # level shift
)
qc.run()
qc.summary()
#qc.plot_overview()
#
# Export as ecosound Measurement
merged_events = qc.to_measurement()
merged_events.to_netcdf(r"I:\qc_events4.nc")
#
# # Export per-test detections separately
# per_test = qc.to_measurement(merged=False)
#
# # Separate level-shift sensitivity
# qc = HMDQualityChecker(hmd, sensitivity=2, level_shift_sensitivity=1)
# qc.run()
#
# # Disable specific tests
# qc = HMDQualityChecker(hmd, run_t3=False, detect_tidal_noise=True)
# qc.run()
