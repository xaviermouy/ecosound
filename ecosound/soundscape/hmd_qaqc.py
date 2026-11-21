# -*- coding: utf-8 -*-
"""
Automated QA/QC for Hybrid Millidecade (HMD) passive acoustic data.

Detects data quality issues in HMD spectral data and returns results as
ecosound :class:`~ecosound.core.measurement.Measurement` objects with
QARTOD-style flags.

Detection tiers
---------------
T0 - Gross range     Broadband SPL exceeding system clipping level.
T1 - Structural      Data gaps, missing data, and zero-effort periods.
T2 - Statistical     Rolling-window MAD outliers and flat-line detection.
T3 - Spectral        PCA-based spectral shape anomaly detection.
T4 - Tidal           Tidal flow-noise correlation (optional).
T5 - Level shift     PELT changepoint detection for calibration drift.
T6 - Rolling var     Per-band rolling variance flat-line detection.

QARTOD flags
------------
1 = Pass, 2 = Not evaluated, 3 = Suspect, 4 = Fail, 9 = Missing data.

Usage
-----
>>> from ecosound.soundscape.hmd import HMD
>>> from ecosound.soundscape.hmd_qaqc import HMDQualityChecker
>>>
>>> hmd = HMD(use_dask=False)
>>> hmd.load_nc_files("path/to/deployment/nc")
>>>
>>> qc = HMDQualityChecker(hmd, sensitivity=2)
>>> qc.run()
>>> qc.summary()
>>>
>>> # Get results as ecosound Measurement objects
>>> merged = qc.to_measurement()          # cross-test merged events
>>> per_test = qc.to_measurement(merged=False)  # per-test detections
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures
from scipy.ndimage import binary_dilation, label as ndimage_label
from sklearn.decomposition import PCA

from ecosound.core.measurement import Measurement
from ecosound.soundscape.hmd import HMD

warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = ["HMDQualityChecker"]


# ---------------------------------------------------------------------------
# Internal data class for an individual QC event
# ---------------------------------------------------------------------------

@dataclass
class _QCEvent:
    """Single QC detection (internal bookkeeping)."""
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    freq_min_hz: float
    freq_max_hz: float
    test_name: str
    issue_type: str
    flag: int                # QARTOD: 3=suspect, 4=fail, 9=missing
    severity: float = 0.0
    severity_unit: str = ""
    description: str = ""
    confidence: float = 1.0

    def duration_min(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60


# ---------------------------------------------------------------------------
# Helper: boolean flag matrix -> list of bounding-box events
# ---------------------------------------------------------------------------

def _flag_matrix_to_events(
    flag_matrix: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    test_name: str,
    issue_type: str,
    flag: int,
    min_duration_min: float,
    severity_matrix: np.ndarray | None = None,
    severity_unit: str = "",
    description: str = "",
) -> list[_QCEvent]:
    """Convert a boolean flag matrix (time x freq) into bounding-box events.

    Uses scipy.ndimage.label to find connected components in the flag matrix,
    then extracts the bounding box (time range, frequency range) of each
    component.  Components shorter than min_duration_min are discarded.
    If a severity_matrix is provided, the maximum severity within each
    component is recorded.
    """
    if not flag_matrix.any():
        return []
    labelled, n_components = ndimage_label(flag_matrix)
    events = []
    for comp_id in range(1, n_components + 1):
        rows, cols = np.where(labelled == comp_id)
        t_start = pd.Timestamp(times[rows.min()])
        t_end = pd.Timestamp(times[rows.max()])
        f_lo = float(freqs[cols.min()])
        f_hi = float(freqs[cols.max()])
        dur = (t_end - t_start).total_seconds() / 60
        if dur < min_duration_min:
            continue
        sev = 0.0
        if severity_matrix is not None:
            sev = float(np.nanmax(severity_matrix[rows, cols]))
        events.append(_QCEvent(
            start_time=t_start, end_time=t_end,
            freq_min_hz=f_lo, freq_max_hz=f_hi,
            test_name=test_name, issue_type=issue_type, flag=flag,
            severity=sev, severity_unit=severity_unit,
            description=description,
        ))
    return events


# ---------------------------------------------------------------------------
# Helper: merge nearby / overlapping events across all tests
# ---------------------------------------------------------------------------

def _merge_nearby_events(
    events: list[_QCEvent],
    time_gap_hours: float = 4.0,
) -> list[_QCEvent]:
    """Merge events that overlap or are close in time.

    Two events merge when they overlap or are within ``time_gap_hours``
    of each other, regardless of their frequency ranges.  Merging is
    greedy and iterative -- repeats until no more merges are possible.
    The merged event keeps the worst flag, highest severity, earliest
    start, latest end, widest frequency span, and combined test names /
    issue types.  Works across all tests (not restricted to same
    test_name).
    """
    if len(events) <= 1:
        return list(events)

    time_gap = pd.Timedelta(hours=time_gap_hours)

    def _time_close(a, b):
        return (a.start_time <= b.end_time + time_gap
                and b.start_time <= a.end_time + time_gap)

    def _do_merge(a, b):
        names = sorted(set(a.test_name.split(" + "))
                       | set(b.test_name.split(" + ")))
        issues = sorted(set(a.issue_type.split(" + "))
                        | set(b.issue_type.split(" + ")))
        return _QCEvent(
            start_time=min(a.start_time, b.start_time),
            end_time=max(a.end_time, b.end_time),
            freq_min_hz=min(a.freq_min_hz, b.freq_min_hz),
            freq_max_hz=max(a.freq_max_hz, b.freq_max_hz),
            test_name=" + ".join(names),
            issue_type=" + ".join(issues),
            flag=max(a.flag, b.flag),
            severity=max(a.severity, b.severity),
            severity_unit=a.severity_unit or b.severity_unit,
            description=(a.description if a.severity >= b.severity
                         else b.description),
            confidence=min(a.confidence, b.confidence),
        )

    merged = list(events)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            current = merged[i]
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                if _time_close(current, merged[j]):
                    current = _do_merge(current, merged[j])
                    used.add(j)
                    changed = True
            new_merged.append(current)
            used.add(i)
        merged = new_merged
    return merged


# ===================================================================
# Main checker class
# ===================================================================

class HMDQualityChecker:
    """Automated QA/QC for HMD passive acoustic data.

    Takes a loaded :class:`~ecosound.soundscape.hmd.HMD` object and runs a
    configurable set of quality-control tests.  Results are stored internally
    and can be exported as ecosound
    :class:`~ecosound.core.measurement.Measurement` objects, pandas
    DataFrames, or CSV files.

    Parameters
    ----------
    hmd : HMD
        A loaded HMD object (``hmd.load_nc_files()`` must have been called).
    sensitivity : int
        Global sensitivity preset: 1 = permissive, 2 = balanced, 3 = strict.
        Controls T2 (statistical) and T3 (PCA) thresholds.
    level_shift_sensitivity : int, optional
        Override sensitivity for T5 only.
        1 = 10 dB, 2 = 6 dB, 3 = 4 dB.  Defaults to ``sensitivity``.
    min_event_duration_min : float
        Ignore events shorter than this (minutes).  Default 60.
    rolling_window_hours : float
        Window for T2 rolling statistics (hours).  Default 6.
    gap_tolerance_sec : float
        Tolerance for T1 gap detection (seconds).  Default 5.
    merge_gap_hours : float
        Merge events separated by less than this many hours.  Default 4.
    freq_range : tuple, optional
        ``(fmin, fmax)`` in Hz to restrict the analysis.
    run_t0 : bool
        Enable T0 gross range check.  Default True.
    run_t1 : bool
        Enable T1 structural checks.  Default True.
    run_t2 : bool
        Enable T2 statistical checks.  Default True.
    run_t3 : bool
        Enable T3 PCA spectral anomaly detection.  Default True.
    detect_tidal_noise : bool
        Enable T4 tidal flow noise detection.  Default False.
    run_t5 : bool
        Enable T5 level-shift detection.  Default True.
    run_t6 : bool
        Enable T6 rolling-variance anomaly detection.  Default False.

    Examples
    --------
    >>> from ecosound.soundscape.hmd import HMD
    >>> from ecosound.soundscape.hmd_qaqc import HMDQualityChecker
    >>>
    >>> # Load HMD data
    >>> hmd = HMD(use_dask=False)
    >>> hmd.load_nc_files("path/to/nc_files")
    >>>
    >>> # Run QA/QC with default settings
    >>> qc = HMDQualityChecker(hmd, sensitivity=2)
    >>> qc.run()
    >>> qc.summary()
    >>>
    >>> # Export as ecosound Measurement
    >>> merged_events = qc.to_measurement()
    >>> merged_events.to_netcdf("qc_events.nc")
    >>>
    >>> # Export per-test detections separately
    >>> per_test = qc.to_measurement(merged=False)
    >>>
    >>> # Separate level-shift sensitivity
    >>> qc = HMDQualityChecker(hmd, sensitivity=2,
    ...                        level_shift_sensitivity=1)
    >>> qc.run()
    >>>
    >>> # Disable specific tests
    >>> qc = HMDQualityChecker(hmd, run_t3=False, detect_tidal_noise=True)
    >>> qc.run()
    """

    # Sensitivity presets: (mad_k, pct_fail_lo, pct_fail_hi, min_var_factor)
    _PRESETS = {
        1: (10.0, 0.01, 99.99, 0.03),  # permissive
        2: (7.0,  0.05, 99.95, 0.05),  # balanced
        3: (5.0,  0.1,  99.9,  0.08),  # strict
    }

    def __init__(
        self,
        hmd: HMD,
        sensitivity: int = 2,
        level_shift_sensitivity: int | None = None,
        min_event_duration_min: float = 60,
        rolling_window_hours: float = 6,
        gap_tolerance_sec: float = 5,
        merge_gap_hours: float = 4.0,
        freq_range: tuple | None = None,
        run_t0: bool = True,
        run_t1: bool = True,
        run_t2: bool = True,
        run_t3: bool = True,
        detect_tidal_noise: bool = False,
        run_t5: bool = True,
        run_t6: bool = False,
    ):
        if hmd.ds is None:
            raise ValueError("HMD object has no data. Call hmd.load_nc_files() first.")
        if sensitivity not in self._PRESETS:
            raise ValueError(f"sensitivity must be 1, 2, or 3 (got {sensitivity})")

        self.hmd = hmd
        self.sensitivity = sensitivity
        self.level_shift_sensitivity = (
            level_shift_sensitivity if level_shift_sensitivity is not None
            else sensitivity
        )
        self.min_event_duration_min = min_event_duration_min
        self.rolling_window_hours = rolling_window_hours
        self.gap_tolerance_sec = gap_tolerance_sec
        self.merge_gap_hours = merge_gap_hours
        self.freq_range = freq_range
        self.run_t0 = run_t0
        self.run_t1 = run_t1
        self.run_t2 = run_t2
        self.run_t3 = run_t3
        self.detect_tidal_noise = detect_tidal_noise
        self.run_t5 = run_t5
        self.run_t6 = run_t6

        preset = self._PRESETS[sensitivity]
        self.mad_k = preset[0]
        self.pct_fail_lo, self.pct_fail_hi = preset[1], preset[2]
        self.min_var_factor = preset[3]

        self.events: list[_QCEvent] = []
        self.merged_events: list[_QCEvent] = []

        # These are populated by _prepare_data()
        self.ds = None
        self.psd = None
        self.times = None
        self.freqs = None
        self.band_ts = None
        self.band_names = None
        self.band_centers = None
        self.band_info = None
        self.duty_cycle = None

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self):
        """Extract numpy arrays from the HMD xarray dataset and compute
        decidecade band levels.

        Reads the PSD matrix and time/frequency axes from the HMD xarray
        Dataset (computing lazy Dask arrays if needed).  Then integrates
        the millidecade PSD bins into decidecade (1/3-octave) bands using
        energy summation in the linear domain:
            band_dB = 10 * log10( sum( 10^(psd_bin / 10) ) )

        Populates self.psd, self.times, self.freqs, self.band_ts,
        self.band_names, self.band_centers, and self.band_info.
        """
        print("=" * 60)
        print("Preparing HMD data for QA/QC ...")
        print("=" * 60)

        ds = self.hmd.ds
        if self.freq_range is not None:
            ds = ds.sel(frequency=slice(*self.freq_range))
        self.ds = ds

        # PSD matrix and axes (compute if lazy/dask)
        psd_data = ds["psd"]
        if hasattr(psd_data.data, 'dask'):
            psd_data = psd_data.compute()
        self.psd = psd_data.values
        self.times = ds.time.values
        self.freqs = ds.frequency.values.astype(float)

        # Decidecade band levels
        fmin = float(self.freqs.min())
        fmax = float(self.freqs.max())
        self.band_info = HMD.decidecade_bands(fmin, fmax)

        self.band_names = [f"{fc:.0f}Hz" for fc, _, _ in self.band_info]
        print(f"Integrating {len(self.band_info)} decidecade bands ...")

        band_levels = []
        for fc, flo, fhi in self.band_info:
            mask = (self.freqs >= flo) & (self.freqs <= fhi)
            if mask.sum() == 0:
                band_levels.append(np.full(len(self.times), np.nan))
                continue
            psd_band = self.psd[:, mask]
            linear = np.power(10.0, psd_band / 10.0)
            integrated = np.nansum(linear, axis=1)
            integrated[integrated <= 0] = np.nan
            band_levels.append(10.0 * np.log10(integrated))

        self.band_ts = np.column_stack(band_levels)
        self.band_centers = np.array([fc for fc, _, _ in self.band_info])

        print(f"Data: {len(self.times)} time steps x "
              f"{len(self.freqs)} HMD bins -> "
              f"{len(self.band_info)} decidecade bands")
        print(f"Time range: {pd.Timestamp(self.times[0])} -> "
              f"{pd.Timestamp(self.times[-1])}")
        print(f"Freq range: {fmin:.0f} - {fmax:.0f} Hz\n")

    # ------------------------------------------------------------------
    # T0 -- Gross range check
    # ------------------------------------------------------------------

    def _run_t0_gross_range(self):
        """Flag time steps where broadband SPL exceeds system clipping level.

        Reads the system sensitivity (clipping level) from the NetCDF
        metadata.  Computes broadband SPL by summing all frequency bins
        in the linear domain, then flags any time step that exceeds the
        clipping threshold.  Flagged time steps span all frequencies
        (full-band event) with QARTOD flag 4 (Fail).
        """
        print("-" * 60)
        print("T0: Gross range check")
        print("-" * 60)

        # Read system clipping level from NC metadata
        if "sensitivity" in self.ds:
            psd_hi = float(self.ds["sensitivity"].values.flat[0])
            print(f"  System clipping level (from NC metadata): {psd_hi:.1f} dB")
        else:
            psd_hi = 180.0
            print(f"  No sensitivity in NC metadata -- using default: {psd_hi:.0f} dB")

        if "effort" in self.ds:
            effort = self.ds["effort"].values
            valid_data = effort != np.timedelta64(0, "ns")
        else:
            valid_data = ~np.all(np.isnan(self.psd), axis=1)

        # Broadband SPL per time step
        linear = np.power(10.0, self.psd / 10.0)
        broadband = np.nansum(linear, axis=1)
        broadband[broadband <= 0] = np.nan
        broadband_db = 10.0 * np.log10(broadband)

        flag_bb = (broadband_db > psd_hi) & valid_data
        if not flag_bb.any():
            print(f"  All broadband SPL values below {psd_hi:.1f} dB -- PASS")
            return

        print(f"  {flag_bb.sum()} time steps exceed {psd_hi:.1f} dB broadband SPL")
        flag_gross = np.repeat(flag_bb[:, np.newaxis], self.psd.shape[1], axis=1)
        self.events.extend(_flag_matrix_to_events(
            flag_gross, self.times, self.freqs,
            test_name="gross_range", issue_type="sensor_failure", flag=4,
            min_duration_min=self.min_event_duration_min,
            severity_matrix=np.repeat(broadband_db[:, np.newaxis],
                                      self.psd.shape[1], axis=1),
            severity_unit="dB broadband SPL",
            description=f"Broadband SPL exceeds system clipping level ({psd_hi:.1f} dB)",
        ))

    # ------------------------------------------------------------------
    # T1 -- Structural: gaps, missing data, zero effort
    # ------------------------------------------------------------------

    def _detect_duty_cycle_from_effort(self, zero_effort):
        """Auto-detect duty cycle from on/off effort pattern.

        Examines transitions in the zero-effort boolean array to measure
        the duration of on-periods (recording) and off-periods (silent).
        If both on and off durations are consistent (>70% within ±2 min
        of their medians), returns the duty-cycle parameters.  Returns
        None if no regular pattern is found.
        """
        changes = np.diff(zero_effort.astype(int))
        off_starts = np.where(changes == 1)[0] + 1
        on_starts = np.where(changes == -1)[0] + 1
        if len(off_starts) < 3 or len(on_starts) < 3:
            return None

        on_durs, off_durs = [], []
        for start in on_starts:
            nxt = off_starts[off_starts > start]
            if len(nxt) > 0:
                on_durs.append(nxt[0] - start)
        for start in off_starts:
            nxt = on_starts[on_starts > start]
            if len(nxt) > 0:
                off_durs.append(nxt[0] - start)
        if len(on_durs) < 3 or len(off_durs) < 3:
            return None

        on_med = int(np.median(on_durs))
        off_med = int(np.median(off_durs))
        on_ok = np.mean(np.abs(np.array(on_durs) - on_med) <= 2) > 0.7
        off_ok = np.mean(np.abs(np.array(off_durs) - off_med) <= 2) > 0.7
        if on_ok and off_ok:
            return {"on_min": on_med, "off_min": off_med,
                    "cycle_min": on_med + off_med,
                    "n_cycles": min(len(on_durs), len(off_durs)),
                    "on_tolerance": 2, "off_tolerance": 2}
        return None

    def _detect_duty_cycle_from_gaps(self, gap_durations):
        """Auto-detect duty cycle from time-axis gap regularity.

        Rounds gap durations to the nearest 10 seconds and looks for a
        dominant gap length that accounts for >50% of all gaps with at
        least 3 occurrences.  Returns (gap_sec, tolerance) if found, or
        None if the gaps are irregular.
        """
        if len(gap_durations) < 3:
            return None
        rounded = np.round(gap_durations / 10) * 10
        unique, counts = np.unique(rounded, return_counts=True)
        idx = np.argmax(counts)
        dur, cnt = unique[idx], counts[idx]
        if cnt >= 3 and cnt / len(gap_durations) > 0.5:
            return float(dur), float(dur * 0.1)
        return None

    def _run_t1_structural(self):
        """Detect data gaps and zero-effort periods.

        Identifies structural data issues:
        1. Attempts to auto-detect a duty cycle from effort or gap patterns.
        2. Flags time-axis gaps that do not match the detected duty cycle
           as missing-data events (QARTOD flag 9).
        3. Flags zero-effort time steps as no-data events (QARTOD flag 4),
           distinguishing expected off-periods from unexpected dropouts
           when a duty cycle is present.
        """
        print("-" * 60)
        print("T1: Structural checks (gaps, missing data)")
        print("-" * 60)

        events = []
        dt = np.diff(self.times).astype("timedelta64[s]").astype(float)
        expected = 60.0

        # Zero-effort mask
        if "effort" in self.ds:
            effort = self.ds["effort"].values
            zero_effort = effort == np.timedelta64(0, "ns")
        else:
            zero_effort = np.all(np.isnan(self.psd), axis=1)

        # Duty cycle detection
        self.duty_cycle = None
        dc = self._detect_duty_cycle_from_effort(zero_effort)
        if dc is not None:
            self.duty_cycle = dc
            print(f"  Duty cycle detected (from effort pattern): "
                  f"{dc['on_min']} min on / {dc['off_min']} min off "
                  f"({dc['n_cycles']} cycles)")

        # Gap detection
        gap_mask = dt > (expected + self.gap_tolerance_sec)
        gap_indices = np.where(gap_mask)[0]
        gap_durations = dt[gap_indices]

        if self.duty_cycle is None:
            dc_gap = self._detect_duty_cycle_from_gaps(gap_durations)
            if dc_gap is not None:
                dc_dur, dc_tol = dc_gap
                self.duty_cycle = {"gap_sec": dc_dur, "gap_tolerance": dc_tol}
                n_dc = np.sum((gap_durations >= dc_dur - dc_tol)
                              & (gap_durations <= dc_dur + dc_tol))
                print(f"  Duty cycle detected (from gaps): {dc_dur:.0f}s gap "
                      f"({n_dc}/{len(gap_durations)} gaps match)")
            else:
                print("  No duty cycle detected (continuous recording)")

        # Flag non-duty-cycle gaps
        dc_dur = (self.duty_cycle or {}).get("gap_sec", -1)
        dc_tol = (self.duty_cycle or {}).get("gap_tolerance", -1)
        for idx in gap_indices:
            gap_sec = dt[idx]
            if dc_dur > 0 and (dc_dur - dc_tol) <= gap_sec <= (dc_dur + dc_tol):
                continue
            events.append(_QCEvent(
                start_time=pd.Timestamp(self.times[idx]),
                end_time=pd.Timestamp(self.times[idx + 1]),
                freq_min_hz=float(self.freqs.min()),
                freq_max_hz=float(self.freqs.max()),
                test_name="gap_detection", issue_type="data_gap", flag=9,
                severity=gap_sec, severity_unit="seconds",
                description=f"Data gap of {gap_sec:.0f}s (not duty cycle)",
            ))

        # Zero-effort rows
        if self.duty_cycle is not None and "on_min" in self.duty_cycle:
            on_min = self.duty_cycle["on_min"]
            off_min = self.duty_cycle["off_min"]
            cycle = on_min + off_min
            tol = self.duty_cycle["on_tolerance"]
            changes = np.diff(zero_effort.astype(int))
            on_starts = np.where(changes == -1)[0] + 1
            if len(on_starts) == 0 and not zero_effort[0]:
                on_starts = np.array([0])
            expected_off = np.zeros(len(zero_effort), dtype=bool)
            for os in on_starts:
                s = max(0, int(os + on_min - tol))
                e = min(len(zero_effort), int(os + cycle + tol))
                expected_off[s:e] = True
            unexpected_zero = zero_effort & ~expected_off
            print(f"  Zero-effort rows: {zero_effort.sum()} total, "
                  f"{zero_effort.sum() - unexpected_zero.sum()} expected (duty cycle), "
                  f"{unexpected_zero.sum()} unexpected")
            if unexpected_zero.any():
                flag_mat = np.repeat(unexpected_zero[:, None],
                                     len(self.freqs), axis=1)
                events.extend(_flag_matrix_to_events(
                    flag_mat, self.times, self.freqs,
                    test_name="zero_effort", issue_type="no_data", flag=4,
                    min_duration_min=0,
                    description="Unexpected zero-effort during on-period",
                ))
        else:
            if zero_effort.any():
                flag_mat = np.repeat(zero_effort[:, None],
                                     len(self.freqs), axis=1)
                events.extend(_flag_matrix_to_events(
                    flag_mat, self.times, self.freqs,
                    test_name="zero_effort", issue_type="no_data", flag=4,
                    min_duration_min=0,
                    description="Zero-effort / no data recorded",
                ))

        n_gaps = sum(1 for e in events if e.test_name == "gap_detection")
        n_ze = sum(1 for e in events if e.test_name == "zero_effort")
        print(f"  Found {len(events)} events ({n_gaps} gaps, {n_ze} zero-effort)")
        self.events.extend(events)

    # ------------------------------------------------------------------
    # T2 -- Statistical: MAD outliers, flat-line
    # ------------------------------------------------------------------

    @staticmethod
    def _process_band(ts, win, min_var_factor):
        """Compute rolling MAD anomaly scores and flat-line flag for one band.

        Calculates a rolling-window Modified Z-score using the median
        absolute deviation (MAD).  Also flags flat-line segments where
        the rolling standard deviation drops below a fraction of the
        overall standard deviation (controlled by min_var_factor).
        Called in parallel across bands from _run_t2_statistical().
        """
        n = len(ts)
        scores = np.full(n, np.nan)
        flat = np.zeros(n, dtype=bool)
        valid = ~np.isnan(ts)
        if valid.sum() < win:
            return scores, flat
        s = pd.Series(ts)
        rolling_med = s.rolling(win, center=True, min_periods=win // 2).median()
        rolling_mad = (s - rolling_med).abs().rolling(
            win, center=True, min_periods=win // 2).median().clip(lower=0.1)
        scores = ((s - rolling_med) / rolling_mad).abs().values
        rolling_std = s.rolling(win, center=True, min_periods=win // 2).std()
        overall_std = np.nanstd(ts[valid])
        if overall_std > 0:
            flat = rolling_std.values < (overall_std * min_var_factor)
        return scores, flat

    def _run_t2_statistical(self):
        """Per-decidecade-band statistical anomaly and flat-line detection.

        Runs _process_band() in parallel across all decidecade bands to
        compute per-band MAD scores and flat-line flags.  Only flags a
        time step as anomalous if >40% of bands exceed the MAD threshold
        simultaneously (broadband agreement filter).  Also adds percentile
        extremes for each band.  Produces QARTOD flag 4 (Fail) events.
        """
        print("-" * 60)
        print("T2: Statistical checks (rolling MAD, variance, percentile)")
        print("-" * 60)

        n_times, n_bands = self.band_ts.shape
        win = int(self.rolling_window_hours * 60)
        mad_scores = np.full((n_times, n_bands), np.nan)
        flatline_flag = np.zeros((n_times, n_bands), dtype=bool)

        n_workers = min(n_bands, 8)
        print(f"  Processing {n_bands} bands with {n_workers} threads ...")
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(self._process_band, self.band_ts[:, b].copy(),
                            win, self.min_var_factor): b
                for b in range(n_bands)
            }
            for future in futures:
                b = futures[future]
                scores, flat = future.result()
                mad_scores[:, b] = scores
                flatline_flag[:, b] = flat

        # Multi-band agreement: only flag if >40% of bands are anomalous
        per_band_flag = mad_scores > self.mad_k
        band_frac = np.nanmean(per_band_flag, axis=1)
        broadband_mask = band_frac > 0.4
        mad_flag_fail = per_band_flag & broadband_mask[:, None]

        # Percentile extremes
        for b in range(n_bands):
            ts = self.band_ts[:, b]
            valid = ~np.isnan(ts)
            if valid.sum() < 100:
                continue
            plo = np.nanpercentile(ts[valid], self.pct_fail_lo)
            phi = np.nanpercentile(ts[valid], self.pct_fail_hi)
            pct_extreme = (ts < plo) | (ts > phi)
            mad_flag_fail[:, b] |= (pct_extreme & broadband_mask)

        events_fail = _flag_matrix_to_events(
            mad_flag_fail, self.times, self.band_centers,
            test_name="broadband_anomaly", issue_type="statistical_outlier",
            flag=4, min_duration_min=self.min_event_duration_min,
            severity_matrix=mad_scores, severity_unit="MAD_score",
            description="Rolling-MAD broadband anomaly",
        )
        events_flat = _flag_matrix_to_events(
            flatline_flag, self.times, self.band_centers,
            test_name="flat_line", issue_type="low_variance", flag=4,
            min_duration_min=self.min_event_duration_min,
            description="Low variance / flat-line detection",
        )
        print(f"  Broadband anomalies: {len(events_fail)} events")
        print(f"  Flat-line:           {len(events_flat)} events")
        print(f"  Total T2:            {len(events_fail) + len(events_flat)} events")
        self.events.extend(events_fail)
        self.events.extend(events_flat)

    # ------------------------------------------------------------------
    # T3 -- PCA spectral anomaly
    # ------------------------------------------------------------------

    def _run_t3_spectral(self):
        """PCA-based spectral shape anomaly detection.

        Standardises the decidecade band matrix and fits a PCA model
        retaining 95% of variance.  Time steps with high reconstruction
        error (i.e. spectral shapes not well-explained by the principal
        components) are flagged as spectral anomalies.  The threshold is
        based on the MAD of reconstruction errors.  Produces QARTOD
        flag 3 (Suspect) events.
        """
        print("-" * 60)
        print("T3: PCA spectral anomaly detection")
        print("-" * 60)

        n_times, n_bands = self.band_ts.shape
        valid_rows = ~np.any(np.isnan(self.band_ts), axis=1)
        if valid_rows.sum() < 100:
            print("  Not enough valid rows for PCA")
            return

        x_valid = self.band_ts[valid_rows, :]
        mu, sigma = np.mean(x_valid, axis=0), np.std(x_valid, axis=0)
        sigma[sigma < 1e-6] = 1e-6
        x_std = (x_valid - mu) / sigma

        pca = PCA(n_components=0.95, svd_solver="full")
        pca.fit(x_std)
        recon_error = np.sqrt(np.mean(
            (x_std - pca.inverse_transform(pca.transform(x_std))) ** 2, axis=1))
        med_err = np.median(recon_error)
        mad_err = max(np.median(np.abs(recon_error - med_err)), 1e-6)
        pca_scores = (recon_error - med_err) / mad_err

        pca_flag_full = np.zeros(n_times, dtype=bool)
        pca_flag_full[valid_rows] = pca_scores > self.mad_k
        pca_flag_mat = np.repeat(pca_flag_full[:, None], n_bands, axis=1)

        score_full = np.zeros(n_times)
        score_full[valid_rows] = pca_scores
        sev_mat = np.repeat(score_full[:, None], n_bands, axis=1)

        events = _flag_matrix_to_events(
            pca_flag_mat, self.times, self.band_centers,
            test_name="pca_spectral", issue_type="spectral_anomaly", flag=3,
            min_duration_min=self.min_event_duration_min,
            severity_matrix=sev_mat, severity_unit="MAD_recon_error",
            description="Unusual spectral shape (PCA reconstruction error)",
        )
        print(f"  PCA anomalies: {len(events)} events "
              f"({pca.n_components_} components, "
              f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
        self.events.extend(events)

    # ------------------------------------------------------------------
    # T4 -- Tidal flow noise
    # ------------------------------------------------------------------

    def _get_coordinates(self):
        """Extract deployment lat/lon from the NetCDF global attributes.

        Checks for ``geospatial_bounds`` (WKT POINT format) first, then
        falls back to ``geospatial_lat_min`` / ``geospatial_lon_min``.
        Returns ``(lat, lon)`` or ``None`` if coordinates are unavailable.
        """
        bounds = self.ds.attrs.get("geospatial_bounds", "")
        if "POINT" in bounds:
            parts = bounds.replace("POINT", "").strip(" ()")
            try:
                lat, lon = [float(x) for x in parts.split()]
                return lat, lon
            except ValueError:
                pass
        lat = self.ds.attrs.get("geospatial_lat_min")
        lon = self.ds.attrs.get("geospatial_lon_min")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        return None

    def _fetch_noaa_tide(self, lat, lon, start, end):
        """Fetch NOAA tide predictions for the nearest station.

        Queries the NOAA CO-OPS API to find the closest tide-prediction
        station (within 100 km), then downloads 6-minute interval
        predictions for the deployment time range.  Tries up to 10
        nearest stations in case some lack prediction data.  Returns a
        DataFrame with 'time' and 'height_m' columns, or None on failure.
        """
        import json
        import math
        import urllib.request

        print("  Fetching NOAA tide station list ...")
        try:
            url = ("https://api.tidesandcurrents.noaa.gov/mdapi/prod/"
                   "webapi/stations.json?type=tidepredictions")
            req = urllib.request.Request(url, headers={"User-Agent": "hmd-qaqc"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                station_data = json.loads(resp.read())
        except Exception as e:
            print(f"  Could not fetch station list: {e}")
            return None

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
            a = (math.sin(dlat / 2) ** 2
                 + math.cos(math.radians(lat1))
                 * math.cos(math.radians(lat2))
                 * math.sin(dlon / 2) ** 2)
            return R * 2 * math.asin(math.sqrt(a))

        candidates = sorted(
            [(haversine(lat, lon, s["lat"], s["lng"]), s["id"], s["name"])
             for s in station_data["stations"]])

        for dist_km, sid, sname in candidates[:10]:
            if dist_km > 100:
                print("  No tide station within 100 km")
                return None
            try:
                pred_url = (
                    f"https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
                    f"?begin_date={start}&end_date={end}"
                    f"&station={sid}&product=predictions"
                    f"&datum=MLLW&units=metric&time_zone=gmt"
                    f"&format=json&interval=6")
                req = urllib.request.Request(pred_url,
                                            headers={"User-Agent": "hmd-qaqc"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    result = json.loads(resp.read())
                if "predictions" in result and len(result["predictions"]) > 0:
                    preds = result["predictions"]
                    df = pd.DataFrame({
                        "time": pd.to_datetime([p["t"] for p in preds]),
                        "height_m": [float(p["v"]) for p in preds],
                    })
                    print(f"  Using station {sid}: {sname} "
                          f"({dist_km:.1f} km away, {len(df)} predictions)")
                    return df
            except Exception:
                continue
        print("  No working tide station found nearby")
        return None

    def _tidal_flag_from_correlation(self, low_band, tide_current, tide_times):
        """Flag time steps where low-frequency noise correlates with tidal current.

        Interpolates the tidal current time series onto the HMD time axis,
        then computes a rolling-window Pearson correlation (720-min window).
        Time steps are flagged when the absolute correlation exceeds 0.3
        and the noise level is above the 60th percentile.
        """
        valid_low = ~np.isnan(low_band)
        if valid_low.sum() < 100:
            return np.zeros(len(self.times), dtype=bool)
        hmd_t = pd.to_datetime(self.times).astype(np.int64).astype(float)
        tide_t = pd.to_datetime(tide_times).astype(np.int64).astype(float)
        current_interp = np.interp(hmd_t, tide_t, tide_current)

        win, min_valid = 720, 60
        rolling_corr = pd.Series(low_band).rolling(
            win, center=True, min_periods=min_valid
        ).corr(pd.Series(current_interp)).values

        level_threshold = np.nanpercentile(low_band[valid_low], 60)
        flag = (np.abs(rolling_corr) > 0.3) & (low_band > level_threshold)
        n_valid = (~np.isnan(rolling_corr)).sum()
        if n_valid > 0:
            frac = (np.abs(rolling_corr[~np.isnan(rolling_corr)]) > 0.3).mean()
            print(f"    Rolling correlation: {frac*100:.1f}% of windows > 0.3")
        return flag

    def _tidal_flag_from_periodicity(self, low_band):
        """Detect tidal periodicity via autocorrelation (fallback method).

        When tide predictions are unavailable, checks the broadband time
        series for peaks in its autocorrelation at semidiurnal (~6.2h) and
        diurnal (~12.4h) tidal periods.  If tidal periodicity is found,
        flags time steps above the 70th percentile.  Requires at least
        24 hours (1440 min) of valid data.
        """
        valid = ~np.isnan(low_band)
        if valid.sum() < 1440:
            return np.zeros(len(low_band), dtype=bool)
        ts = low_band.copy()
        ts[~valid] = np.nanmedian(ts)
        ts = ts - np.mean(ts)
        max_lag = min(900, len(ts) // 2)
        autocorr = np.correlate(ts, ts, mode="full")
        autocorr = autocorr[len(ts) - 1:] / autocorr[len(ts) - 1]

        has_peak = False
        for lo, hi in [(340, 400), (700, 790)]:
            if hi > max_lag:
                continue
            peak = np.max(autocorr[lo:hi])
            if peak > 0.15:
                period = lo + np.argmax(autocorr[lo:hi])
                print(f"    Tidal periodicity: {period} min "
                      f"({period/60:.1f}h), autocorr={peak:.2f}")
                has_peak = True
        if not has_peak:
            print("    No tidal periodicity detected")
            return np.zeros(len(low_band), dtype=bool)
        threshold = np.nanpercentile(low_band[valid], 70)
        return low_band > threshold

    def _run_t4_tidal(self):
        """Detect tidal flow noise contamination.

        Attempts to identify flow-noise periods correlated with tidal
        currents.  First tries NOAA tide-prediction correlation (if
        coordinates are available), then falls back to autocorrelation-
        based periodicity detection.  Flags affected bands individually
        (elevated >3 dB above quiet-period median) and bridges duty-cycle
        gaps with binary dilation.  Produces QARTOD flag 3 (Suspect).
        """
        print("-" * 60)
        print("T4: Tidal flow noise detection")
        print("-" * 60)

        n_times, n_bands = self.band_ts.shape
        all_linear = np.power(10.0, self.band_ts / 10.0)
        broadband = 10.0 * np.log10(np.nansum(all_linear, axis=1))
        broadband[np.isinf(broadband)] = np.nan

        tidal_flag = None
        method = "periodicity detection"

        coords = self._get_coordinates()
        if coords is not None:
            lat, lon = coords
            print(f"  Coordinates from NC file: {lat:.4f}, {lon:.4f}")
            t_start = pd.Timestamp(self.times[0]).strftime("%Y%m%d")
            t_end = pd.Timestamp(self.times[-1]).strftime("%Y%m%d")
            tide_df = self._fetch_noaa_tide(lat, lon, t_start, t_end)
            if tide_df is not None:
                dt_h = (np.diff(tide_df["time"].values)
                        .astype("timedelta64[s]").astype(float) / 3600)
                current = np.abs(np.diff(tide_df["height_m"].values) / dt_h)
                tidal_flag = self._tidal_flag_from_correlation(
                    broadband, current, tide_df["time"].values[:-1])
                method = "NOAA tide correlation"
                if not tidal_flag.any():
                    print("  Correlation found nothing, trying periodicity ...")
                    tidal_flag = self._tidal_flag_from_periodicity(broadband)
                    method = "periodicity detection"
        else:
            print("  No coordinates in NC file")

        if tidal_flag is None:
            print("  Falling back to periodicity detection ...")
            tidal_flag = self._tidal_flag_from_periodicity(broadband)

        if not tidal_flag.any():
            print("  No tidal flow noise detected")
            return

        # Per-band flagging
        valid_data = ~np.isnan(self.band_ts[:, 0])
        quiet_mask = ~tidal_flag & valid_data
        noisy_mask = tidal_flag & valid_data
        tidal_flag_mat = np.zeros((n_times, n_bands), dtype=bool)
        affected_bands = []
        for b in range(n_bands):
            ts = self.band_ts[:, b]
            if quiet_mask.sum() < 50 or noisy_mask.sum() < 50:
                continue
            quiet_med = np.nanmedian(ts[quiet_mask])
            band_elevated = ts > (quiet_med + 3.0)
            tidal_flag_mat[:, b] = tidal_flag & band_elevated
            if tidal_flag_mat[:, b].any():
                affected_bands.append(self.band_names[b])

        # Bridge duty-cycle gaps
        bridge = ((self.duty_cycle or {}).get("off_min", 0) + 5) or 5
        struct = np.ones(bridge, dtype=bool)
        for b in range(n_bands):
            if tidal_flag_mat[:, b].any():
                tidal_flag_mat[:, b] = binary_dilation(
                    tidal_flag_mat[:, b], structure=struct)

        self.events.extend(_flag_matrix_to_events(
            tidal_flag_mat, self.times, self.band_centers,
            test_name="tidal_flow_noise", issue_type="flow_noise", flag=3,
            min_duration_min=self.min_event_duration_min,
            description=f"Tidal flow noise ({method})",
        ))

        n_flagged = tidal_flag.sum()
        pct = n_flagged / max(1, valid_data.sum()) * 100
        if affected_bands:
            print(f"  Affected bands: {len(affected_bands)}/{n_bands} "
                  f"({affected_bands[0]} to {affected_bands[-1]})")
        print(f"  Flagged {n_flagged} minutes ({pct:.1f}%) as tidal flow noise")

    # ------------------------------------------------------------------
    # T5 -- Level shift (PELT changepoint)
    # ------------------------------------------------------------------

    def _run_t5_level_shift(self):
        """Detect sustained level shifts using PELT changepoint detection.

        Downsamples each decidecade band to hourly resolution and runs
        the PELT (Pruned Exact Linear Time) algorithm to find changepoints.
        Segments whose median level exceeds the minimum segment by more
        than the offset threshold (controlled by level_shift_sensitivity)
        are flagged as calibration drift.  Bands are processed in parallel.
        Produces QARTOD flag 3 (Suspect).
        """
        print("\n" + "-" * 60)
        print("T5: Level-shift (DC offset) detection")
        print("-" * 60)

        offset_thresholds = {1: 10.0, 2: 6.0, 3: 4.0}
        offset_thresh = offset_thresholds[self.level_shift_sensitivity]

        n_times, n_bands = self.band_ts.shape
        if "effort" in self.ds:
            effort = self.ds["effort"].values
            valid_data = effort != np.timedelta64(0, "ns")
        else:
            valid_data = ~np.all(np.isnan(self.psd), axis=1)

        band_vals = self.band_ts.copy()
        band_vals[~valid_data, :] = np.nan

        # Downsample to hourly for PELT speed
        ds_factor = 60
        min_size_ds = max(1, (3 * 1440) // ds_factor)  # 3 days
        n_ds = n_times // ds_factor
        if n_ds < 2 * min_size_ds:
            print("  Record too short for level-shift detection -- skipping")
            return

        penalty = np.log(n_ds)
        flag_up = np.zeros((n_times, n_bands), dtype=bool)
        offset_matrix = np.zeros((n_times, n_bands))
        band_centers = np.array([b[0] for b in self.band_info])

        def _pelt_band(b):
            col = band_vals[:n_ds * ds_factor, b]
            if (~np.isnan(col)).sum() < 2 * min_size_ds * ds_factor:
                return b, None
            reshaped = col[:n_ds * ds_factor].reshape(n_ds, ds_factor)
            with np.errstate(all="ignore"):
                signal_ds = np.nanmedian(reshaped, axis=1)
            nan_ds = np.isnan(signal_ds)
            if nan_ds.all():
                return b, None
            signal_ds[nan_ds] = np.nanmedian(signal_ds)
            algo = ruptures.Pelt(model="l2", min_size=min_size_ds, jump=5)
            algo.fit(signal_ds)
            cps = [cp * ds_factor for cp in algo.predict(pen=penalty)]
            boundaries = [0] + cps
            if boundaries[-1] > n_times:
                boundaries[-1] = n_times
            elif boundaries[-1] < n_times:
                boundaries.append(n_times)
            return b, boundaries

        n_shifts_total = 0
        with ThreadPoolExecutor(max_workers=min(n_bands, 8)) as pool:
            results = list(pool.map(_pelt_band, range(n_bands)))

        for b, boundaries in results:
            if boundaries is None:
                continue
            seg_means = []
            for i in range(len(boundaries) - 1):
                seg_vals = band_vals[boundaries[i]:boundaries[i+1], b]
                seg_means.append(np.nanmedian(seg_vals))
            baseline = np.nanmin(seg_means)
            for i in range(len(boundaries) - 1):
                offset = seg_means[i] - baseline
                if offset > offset_thresh:
                    flag_up[boundaries[i]:boundaries[i+1], b] = True
                    offset_matrix[boundaries[i]:boundaries[i+1], b] = offset
                    n_shifts_total += 1

        flag_up[~valid_data, :] = False
        severity = offset_matrix.copy()
        severity[~flag_up] = 0.0

        self.events.extend(_flag_matrix_to_events(
            flag_up, self.times, band_centers,
            test_name="level_shift", issue_type="calibration_drift", flag=3,
            min_duration_min=self.min_event_duration_min,
            severity_matrix=severity, severity_unit="dB above baseline",
            description=f"Level shift (>{offset_thresh:.0f} dB, PELT changepoint)",
        ))

        # Reporting
        bands_affected = np.any(flag_up, axis=0)
        n_affected = bands_affected.sum()
        if n_affected > 0:
            affected_freqs = band_centers[bands_affected]
            print(f"  PELT segments: {n_shifts_total} elevated across "
                  f"{n_affected} bands")
            print(f"  Affected bands: {n_affected}/{n_bands} "
                  f"({affected_freqs[0]:.0f}Hz to {affected_freqs[-1]:.0f}Hz)")
            max_offsets = np.nanmax(offset_matrix, axis=0)
            for bi in np.argsort(max_offsets)[::-1][:5]:
                if max_offsets[bi] > offset_thresh:
                    print(f"    {band_centers[bi]:.0f} Hz: "
                          f"+{max_offsets[bi]:.1f} dB")
        else:
            print("  No sustained level shifts detected")
        n_flagged = flag_up.sum()
        pct = n_flagged / (max(1, valid_data.sum()) * n_bands) * 100
        print(f"  Flagged {n_flagged} time-band cells ({pct:.1f}%)")

    # ------------------------------------------------------------------
    # T6 -- Rolling variance (flat-line)
    # ------------------------------------------------------------------

    def _run_t6_rolling_variance(self):
        """Flag periods of abnormally low variance per decidecade band.

        For each band, computes the rolling standard deviation over a
        centered window of ``rolling_window_hours``.  A time step is
        flagged when its rolling std falls below a fraction of the
        band's overall standard deviation::

            flagged = rolling_std < overall_std * min_var_factor

        ``min_var_factor`` is set by the ``sensitivity`` preset:

        =====  ==============  ====================================
        Level  min_var_factor  Meaning
        =====  ==============  ====================================
        1      0.03            Only near-zero variance flagged
        2      0.05            Balanced (default)
        3      0.08            Flags mildly flat periods
        =====  ==============  ====================================

        Flagged regions are converted to bounding-box events with
        QARTOD flag 4 (Fail).  Events shorter than
        ``min_event_duration_min`` are discarded.
        """
        print("-" * 60)
        print("T6: Rolling variance (flat-line detection)")
        print("-" * 60)

        n_times, n_bands = self.band_ts.shape
        win = int(self.rolling_window_hours * 60)
        flag_matrix = np.zeros((n_times, n_bands), dtype=bool)

        for b in range(n_bands):
            ts = self.band_ts[:, b]
            valid = ~np.isnan(ts)
            if valid.sum() < win:
                continue
            overall_std = np.nanstd(ts[valid])
            if overall_std == 0:
                continue
            rolling_std = pd.Series(ts).rolling(
                win, center=True, min_periods=win // 2).std()
            flag_matrix[:, b] = rolling_std.values < (
                overall_std * self.min_var_factor)

        events = _flag_matrix_to_events(
            flag_matrix, self.times, self.band_centers,
            test_name="rolling_variance", issue_type="low_variance",
            flag=4, min_duration_min=self.min_event_duration_min,
            description="Low rolling variance (flat-line)",
        )
        print(f"  {len(events)} flat-line events detected")
        self.events.extend(events)

    # ------------------------------------------------------------------
    # Event classification
    # ------------------------------------------------------------------

    def _classify_events(self):
        """Refine issue_type labels based on event frequency span.

        Events spanning more than half the full bandwidth are reclassified
        as 'sensor_failure'; narrower events become 'partial_failure' or
        'noise_contamination' depending on the originating test.  Skips
        gap and zero-effort events whose types are already correct.
        """
        full_bw = float(self.freqs.max()) - float(self.freqs.min())
        for ev in self.events:
            if ev.test_name in ("gap_detection", "zero_effort"):
                continue
            bw = ev.freq_max_hz - ev.freq_min_hz
            is_broadband = bw > full_bw * 0.5
            if ev.test_name == "flat_line":
                ev.issue_type = ("sensor_failure" if is_broadband
                                 else "partial_failure")
            elif ev.test_name == "broadband_anomaly":
                ev.issue_type = ("sensor_failure" if is_broadband
                                 else "noise_contamination")

    # ==================================================================
    # Public API
    # ==================================================================

    def run(self):
        """Run all enabled QA/QC tests.

        Populates ``self.events`` (per-test, unmerged) and
        ``self.merged_events`` (cross-test merged).

        Examples
        --------
        >>> qc = HMDQualityChecker(hmd)
        >>> qc.run()
        >>> print(f"{len(qc.merged_events)} merged events found")
        """
        self._prepare_data()
        if self.run_t0:
            self._run_t0_gross_range()
        if self.run_t1:
            self._run_t1_structural()
        if self.run_t2:
            self._run_t2_statistical()
        if self.run_t3:
            self._run_t3_spectral()
        if self.detect_tidal_noise:
            self._run_t4_tidal()
        if self.run_t5:
            self._run_t5_level_shift()
        if self.run_t6:
            self._run_t6_rolling_variance()
        self._classify_events()

        n_before = len(self.events)
        self.merged_events = _merge_nearby_events(
            self.events, time_gap_hours=self.merge_gap_hours)
        n_after = len(self.merged_events)
        print("\n" + "-" * 60)
        print(f"Event merging: {n_before} -> {n_after} merged events")
        print("-" * 60)
        print(f"\n{'='*60}")
        print(f"QA/QC complete: {n_before} detections, {n_after} merged events")
        print("=" * 60)

    def to_dataframe(self, merged=True):
        """Return events as a pandas DataFrame.

        Parameters
        ----------
        merged : bool
            If True (default), return cross-test merged events.
            If False, return per-test detections.

        Returns
        -------
        pandas.DataFrame
        """
        source = self.merged_events if merged else self.events
        rows = []
        for ev in source:
            rows.append({
                "start_time": ev.start_time,
                "end_time": ev.end_time,
                "duration_min": round(ev.duration_min(), 1),
                "freq_min_hz": ev.freq_min_hz,
                "freq_max_hz": ev.freq_max_hz,
                "test_name": ev.test_name,
                "issue_type": ev.issue_type,
                "flag": ev.flag,
                "severity": round(ev.severity, 2),
                "severity_unit": ev.severity_unit,
                "description": ev.description,
                "confidence": round(ev.confidence, 2),
            })
        return pd.DataFrame(rows)

    def to_measurement(self, merged=True):
        """Convert QC events to an ecosound Measurement object.

        Each event becomes one row with standard Annotation fields (time,
        frequency, label) plus QC-specific measurement columns (test_name,
        issue_type, severity, severity_unit, description).

        Parameters
        ----------
        merged : bool
            If True (default), export cross-test merged events.
            If False, export per-test detections.

        Returns
        -------
        Measurement
            An ecosound Measurement object.  The ``label_class`` is
            ``"QCEvent"``, ``label_subclass`` is the QARTOD flag number
            as a string (e.g. ``"3"``), and ``comment`` lists the test
            names responsible.

        Examples
        --------
        >>> qc.run()
        >>> meas = qc.to_measurement()
        >>> print(meas)
        >>> meas.to_netcdf("qc_events.nc")
        >>> meas.to_csv("qc_events.csv")
        >>>
        >>> # Access the data as a DataFrame
        >>> print(meas.data[['time_min_date', 'time_max_date',
        ...                  'label_class', 'comment', 'severity']])
        """
        source = self.merged_events if merged else self.events
        meas = Measurement(
            measurer_name="HMDQualityChecker",
            measurer_version="1.0",
            measurements_name=["test_name", "issue_type", "severity",
                               "severity_unit", "description"],
            measurements_parameters={
                "sensitivity": self.sensitivity,
                "level_shift_sensitivity": self.level_shift_sensitivity,
                "min_event_duration_min": self.min_event_duration_min,
                "rolling_window_hours": self.rolling_window_hours,
            },
        )

        if not source:
            return meas

        rows = []
        for ev in source:
            start_sec = ev.start_time.timestamp()
            end_sec = ev.end_time.timestamp()
            rows.append({
                "time_min_date": ev.start_time,
                "time_max_date": ev.end_time,
                "time_min_offset": start_sec,
                "time_max_offset": end_sec,
                "duration": ev.duration_min() * 60,  # seconds
                "frequency_min": ev.freq_min_hz,
                "frequency_max": ev.freq_max_hz,
                "label_class": "QCEvent",
                "label_subclass": str(ev.flag),
                "confidence": ev.confidence,
                "comment": ev.test_name,
                # Measurement-specific columns
                "test_name": ev.test_name,
                "issue_type": ev.issue_type,
                "severity": ev.severity,
                "severity_unit": ev.severity_unit,
                "description": ev.description,
            })
        df = pd.DataFrame(rows)

        # Build a new DataFrame with all Measurement columns, filled from df
        new_data = meas.data.reindex(range(len(df)))
        for col in df.columns:
            new_data[col] = df[col].values
        meas.data = new_data

        return meas

    def to_csv(self, path, merged=True):
        """Export events to CSV.

        Parameters
        ----------
        path : str
            Output file path.
        merged : bool
            If True, export merged events; if False, per-test detections.
        """
        df = self.to_dataframe(merged=merged)
        df.to_csv(path, index=False)
        label = "merged events" if merged else "per-test detections"
        print(f"Saved {len(df)} {label} to {path}")

    def summary(self):
        """Print a summary of detections and merged events.

        Examples
        --------
        >>> qc.run()
        >>> qc.summary()
        """
        if not self.events:
            print("No events found.")
            return

        df_pt = self.to_dataframe(merged=False)
        df_m = self.to_dataframe(merged=True)

        print(f"\n{'='*70}")
        print("QA/QC Summary")
        print(f"{'='*70}")

        print(f"\nPer-test detections: {len(df_pt)}")
        print(df_pt.groupby("test_name").size().to_string())

        print(f"\nMerged events: {len(df_m)}")
        print(f"\nBy issue type:")
        print(df_m.groupby("issue_type").size().to_string())
        print(f"\nBy flag:")
        for val, label in [(3, "Suspect"), (4, "Fail"), (9, "Missing")]:
            n = (df_m["flag"] == val).sum()
            if n > 0:
                print(f"  {val} ({label}): {n}")

        print(f"\nTime range: {df_m['start_time'].min()} -> "
              f"{df_m['end_time'].max()}")
        total_min = df_m["duration_min"].sum()
        print(f"Total flagged time: {total_min:.0f} min "
              f"({total_min / len(self.times) * 100:.1f}% of record)")

    def plot_overview(self, db_range=(30, 115), save_path=None):
        """Plot LTSA with QC event boxes overlaid.

        Three-panel figure: (1) LTSA with merged event boxes, (2) decidecade
        band time series, (3) per-test event timeline.

        Parameters
        ----------
        db_range : tuple
            Color range for the LTSA (dB re 1 uPa^2/Hz).
        save_path : str, optional
            Save figure to this path.

        Examples
        --------
        >>> qc.run()
        >>> qc.plot_overview(save_path="qc_overview.png")
        """
        if self.psd is None:
            raise RuntimeError("Run .run() first.")

        fig, axes = plt.subplots(3, 1, figsize=(18, 14),
                                 height_ratios=[3, 2, 1], sharex=True)

        # Panel 1: LTSA
        ax = axes[0]
        t_edges = mdates.date2num(pd.to_datetime(self.times))
        im = ax.pcolormesh(t_edges, self.freqs, self.psd.T,
                           vmin=db_range[0], vmax=db_range[1],
                           cmap="viridis", shading="auto", rasterized=True)
        ax.set_yscale("log")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title("LTSA with QC Event Boxes")
        cb = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
        cb.set_label("PSD (dB re 1 uPa^2/Hz)")

        colors = {3: ("orange", 0.35), 4: ("red", 0.45), 9: ("gray", 0.5)}
        for ev in self.merged_events:
            c, alpha = colors.get(ev.flag, ("blue", 0.3))
            t0 = mdates.date2num(ev.start_time)
            t1 = mdates.date2num(ev.end_time)
            ax.add_patch(mpatches.Rectangle(
                (t0, ev.freq_min_hz), t1 - t0,
                ev.freq_max_hz - ev.freq_min_hz,
                linewidth=1, edgecolor=c, facecolor=c, alpha=alpha))
        ax.legend(handles=[
            mpatches.Patch(facecolor="orange", alpha=0.5, label="Suspect (3)"),
            mpatches.Patch(facecolor="red", alpha=0.5, label="Fail (4)"),
            mpatches.Patch(facecolor="gray", alpha=0.5, label="Missing (9)"),
        ], loc="upper right", fontsize=8)

        # Panel 2: Band time series
        ax = axes[1]
        idx_sel = np.linspace(0, len(self.band_names) - 1, 6, dtype=int)
        cmap_lines = plt.cm.viridis(np.linspace(0.1, 0.9, len(idx_sel)))
        for i, bidx in enumerate(idx_sel):
            ax.plot(pd.to_datetime(self.times), self.band_ts[:, bidx],
                    label=self.band_names[bidx], color=cmap_lines[i],
                    linewidth=0.5, alpha=0.8)
        ax.set_ylabel("Band Level (dB)")
        ax.legend(loc="upper right", fontsize=7, ncol=3)
        ax.set_title("Decidecade Band Levels (selection)")

        # Panel 3: Event timeline
        ax = axes[2]
        if self.events:
            df = self.to_dataframe(merged=False)
            test_names = df["test_name"].unique()
            y_map = {name: i for i, name in enumerate(test_names)}
            for _, row in df.iterrows():
                c, _ = colors.get(row["flag"], ("blue", 0.3))
                y = y_map[row["test_name"]]
                t0 = mdates.date2num(row["start_time"])
                t1 = mdates.date2num(row["end_time"])
                ax.barh(y, t1 - t0, left=t0, height=0.6, color=c, alpha=0.7)
            ax.set_yticks(list(y_map.values()))
            ax.set_yticklabels(list(y_map.keys()), fontsize=8)
        ax.set_title("Event Timeline by Test")
        ax.set_ylabel("Test")

        axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axes[2].xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        fig.autofmt_xdate()
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Figure saved to {save_path}")
        plt.show()
