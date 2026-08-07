"""
ecosound.localization.tdoa
==========================
TDOA estimation: GCC and GCC-PHAT single-pair estimators, plus a batch helper
for all hydrophone pairs.

Two estimators are provided:

  gcc      — plain normalised cross-correlation in the frequency domain.
  gcc_phat — GCC-PHAT: cross-spectrum whitening before IFFT.  Produces a
             sharper, more impulse-like peak, more robust to noise and
             reverberation.

Both return four values:
  tdoa_sec  — estimated TDOA in seconds
  corr_val  — peak value of the (normalised) correlation function
  cc        — the full correlation function (not cropped to search window)
  lags      — corresponding lag values in samples

TDOA sign convention: positive means the signal arrives *later* at s2 than
at s1, i.e. tdoa = t_arrival(s2) - t_arrival(s1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Public API — single-pair estimators
# ---------------------------------------------------------------------------

def gcc(s1, s2, fs, tdoa_max_sec=None, subsample_interpolation=False):
    """Estimate TDOA using plain (normalised) cross-correlation.

    Parameters
    ----------
    s1 : numpy.ndarray
        1-D reference waveform.
    s2 : numpy.ndarray
        1-D second waveform at the same sampling rate.
    fs : float
        Sampling frequency in Hz.
    tdoa_max_sec : float, optional
        Restrict the peak search to ±tdoa_max_sec seconds.  Should be set to
        max_inter-hydrophone_distance / sound_speed.  If None the full lag
        range is searched.
    subsample_interpolation : bool
        If True, refine the integer-sample peak using parabolic interpolation
        for sub-sample TDOA precision (no upsampling required).
        If False (default), the TDOA is quantised to the nearest sample.

    Returns
    -------
    tdoa_sec : float
    corr_val : float
    cc : numpy.ndarray
    lags : numpy.ndarray
    """
    cc, lags = _cross_correlate(s1, s2, phat=False)
    peak_idx = _peak_in_window(lags, cc, fs, tdoa_max_sec)
    if subsample_interpolation:
        delta = _parabolic_interpolate(cc, peak_idx)
        tdoa_sec = (lags[peak_idx] + delta) / fs
    else:
        tdoa_sec = lags[peak_idx] / fs
    return float(tdoa_sec), float(cc[peak_idx]), cc, lags


def gcc_phat(s1, s2, fs, tdoa_max_sec=None, subsample_interpolation=False):
    """Estimate TDOA using GCC-PHAT (cross-spectrum whitening).

    Parameters
    ----------
    s1, s2, fs, tdoa_max_sec
        Same as :func:`gcc`.
    subsample_interpolation : bool
        Same as :func:`gcc`.

    Returns
    -------
    tdoa_sec, corr_val, cc, lags
        Same as :func:`gcc`.
    """
    cc, lags = _cross_correlate(s1, s2, phat=True)
    peak_idx = _peak_in_window(lags, cc, fs, tdoa_max_sec)
    if subsample_interpolation:
        delta = _parabolic_interpolate(cc, peak_idx)
        tdoa_sec = (lags[peak_idx] + delta) / fs
    else:
        tdoa_sec = lags[peak_idx] / fs
    return float(tdoa_sec), float(cc[peak_idx]), cc, lags


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

def compute_tdoas(waveform_stack, pairs, fs, tdoa_max_sec, method="gcc",
                  subsample_interpolation=False):
    """Estimate TDOAs for all hydrophone pairs.

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
        One waveform per hydrophone.
    pairs : list of [int, int]
        Pairs of hydrophone indices (as returned by
        :meth:`~ecosound.localization.SensorArray.define_pairs`).
    fs : float
        Sampling frequency in Hz.
    tdoa_max_sec : float
        Maximum TDOA search window in seconds.
    method : {'gcc', 'gcc_phat'}
        TDOA estimator to use.
    subsample_interpolation : bool
        If True, refine the integer-sample CC peak using parabolic
        interpolation.  Passed through to the underlying :func:`gcc` /
        :func:`gcc_phat` call.

    Returns
    -------
    tdoa_sec : numpy.ndarray, shape (n_pairs, 1)
        Estimated TDOAs in seconds.
    corr : numpy.ndarray, shape (n_pairs, 1)
        Peak correlation value per pair.
    cc_list : list of numpy.ndarray
        Full cross-correlation function per pair.
    lags_list : list of numpy.ndarray
        Lag values (samples) per pair.
    """
    fn = gcc_phat if method == "gcc_phat" else gcc
    tdoa_list, corr_list, cc_list, lags_list = [], [], [], []
    for pair in pairs:
        tdoa, corr, cc, lags = fn(
            waveform_stack[pair[0]], waveform_stack[pair[1]], fs, tdoa_max_sec,
            subsample_interpolation=subsample_interpolation,
        )
        tdoa_list.append(tdoa)
        corr_list.append(corr)
        cc_list.append(cc)
        lags_list.append(lags)
    tdoa_sec = np.array(tdoa_list).reshape(-1, 1)
    corr = np.array(corr_list).reshape(-1, 1)
    return tdoa_sec, corr, cc_list, lags_list


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parabolic_interpolate(cc, peak_idx):
    """Sub-sample peak refinement via parabolic interpolation.

    Fits a parabola through the three samples surrounding the integer peak
    and returns the fractional sample offset of the parabola's maximum.

    Parameters
    ----------
    cc : numpy.ndarray
        Cross-correlation array.
    peak_idx : int
        Integer index of the CC peak.

    Returns
    -------
    float
        Fractional sample offset (add to ``peak_idx`` for the true peak).
        Returns 0.0 when the peak is at the array boundary or the parabola
        is degenerate.
    """
    if peak_idx <= 0 or peak_idx >= len(cc) - 1:
        return 0.0
    y0, y1, y2 = cc[peak_idx - 1], cc[peak_idx], cc[peak_idx + 1]
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y0 - y2) / denom


def _peak_in_window(lags, cc, fs, tdoa_max_sec):
    """Return index of the cc peak, restricted to ±tdoa_max_sec if given."""
    if tdoa_max_sec is None:
        return int(np.argmax(cc))
    tdoa_max_samp = int(round(tdoa_max_sec * fs))
    mask = np.abs(lags) <= tdoa_max_samp
    return int(np.where(mask, cc, -np.inf).argmax())


def _cross_correlate(s1, s2, phat):
    """FFT-based cross-correlation, plain or PHAT-whitened.

    Correctness note on the fftshift/trim order
    --------------------------------------------
    np.fft.irfft returns a circular cross-correlation where negative lags
    are wrapped to the high-index end of the array.  Because nfft > n (next
    power of 2), truncating to [:n] before fftshift silently drops all
    negative lags — every TDOA would appear positive.

    The correct procedure: fftshift the full nfft-length array first (moving
    negative lags to the left half), then slice the central n valid samples.

    Normalisation
    -------------
    GCC (phat=False): divide by sqrt(energy(s1) * energy(s2)) so the peak
      for identical signals equals 1 (Cauchy-Schwarz bound).
    GCC-PHAT (phat=True): no energy normalisation.  PHAT whitening already
      sets the peak to ≈ 1 for identical signals.
    """
    n = len(s1) + len(s2) - 1
    nfft = int(2 ** np.ceil(np.log2(n)))

    S1 = np.fft.rfft(s1, n=nfft)
    S2 = np.fft.rfft(s2, n=nfft)
    # conj(S1)*S2 → peak at τ = +D when s2 is delayed by D relative to s1
    cross = np.conj(S1) * S2

    if phat:
        denom = np.abs(cross)
        denom = np.where(denom < 1e-12, 1e-12, denom)
        cross = cross / denom

    cc_circ = np.fft.irfft(cross, n=nfft)

    # fftshift full array first, then trim to valid central samples
    cc_circ = np.fft.fftshift(cc_circ)
    centre = nfft // 2
    cc = cc_circ[centre - (len(s1) - 1): centre + len(s2)]
    lags = np.arange(-(len(s1) - 1), len(s2))

    if not phat:
        norm = np.sqrt(np.dot(s1, s1) * np.dot(s2, s2))
        if norm > 0:
            cc = cc / norm

    return cc, lags
