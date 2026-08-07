"""
ecosound.localization.preprocessing
====================================
Waveform loading, tightening, and upsampling for the localization pipeline.
"""

import numpy as np
import ecosound.core.tools
from ecosound.core.audiotools import Sound, upsample as _upsample


def stack_waveforms(audio_files, detec, tdoa_max_sec):
    """Load and bandpass-filter waveforms from all hydrophones for one detection.

    Each channel is padded by ±tdoa_max_sec around the detection window so
    that the TDOA search is not clipped.

    Parameters
    ----------
    audio_files : dict
        ``{'path': [str, ...], 'channel': [int, ...]}`` as returned by
        :meth:`~ecosound.localization.DeployedArray.find_audio_files`.
    detec : pandas.Series
        One detection row.  Must contain 'time_min_offset', 'time_max_offset',
        'frequency_min', 'frequency_max'.
    tdoa_max_sec : float
        Padding in seconds added on each side of the detection window.

    Returns
    -------
    list of numpy.ndarray
        One bandpass-filtered waveform array per hydrophone.
    """
    waveform_stack = []
    for audio_file, channel in zip(audio_files["path"], audio_files["channel"]):
        chan_wav = Sound(audio_file)
        chan_wav.read(
            channel=channel,
            chunk=[
                detec["time_min_offset"] - tdoa_max_sec,
                detec["time_max_offset"] + tdoa_max_sec,
            ],
            unit="sec",
            detrend=True,
        )
        chan_wav.filter(
            "bandpass",
            [detec["frequency_min"], detec["frequency_max"]],
            verbose=False,
        )
        waveform_stack.append(chan_wav.waveform)
    return waveform_stack


def tighten_waveforms(waveform_stack, fs, ref_channel, method="envelope",
                      energy_window_perc=90,
                      peak_half_width_sec=0.010,
                      envelope_threshold_perc=10,
                      envelope_pad_sec=0.002):
    """Crop all waveforms to a tight window around the signal on the reference channel.

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
    fs : float
        Sampling frequency in Hz.
    ref_channel : int
        Index of the reference hydrophone in waveform_stack.
    method : {'cumulative_energy', 'peak_window', 'envelope'}
        Window-selection method.

        * ``'cumulative_energy'`` — retain a fraction of the cumulative energy.
        * ``'peak_window'`` — fixed window centred on the energy peak.
        * ``'envelope'`` — threshold on the signal envelope.
    energy_window_perc : float
        (M1) Percentage of cumulative energy to retain.
    peak_half_width_sec : float
        (M2) Half-width of the peak window in seconds.
    envelope_threshold_perc : float
        (M3) Envelope threshold as % of peak.
    envelope_pad_sec : float
        (M3) Padding added on each side of the envelope window (seconds).

    Returns
    -------
    waveform_stack : list of numpy.ndarray
        Cropped waveforms (all channels share the same window).
    t_ms : numpy.ndarray
        Time axis in milliseconds for the cropped waveforms.
    """
    ref_wf = waveform_stack[ref_channel]
    if method == "cumulative_energy":
        chunk = ecosound.core.tools.tighten_signal_limits(ref_wf, energy_window_perc)
    elif method == "peak_window":
        chunk = ecosound.core.tools.tighten_signal_peak_window(
            ref_wf, fs, peak_half_width_sec
        )
    else:  # 'envelope'
        chunk = ecosound.core.tools.tighten_signal_envelope(
            ref_wf, fs, envelope_threshold_perc, envelope_pad_sec
        )
    waveform_stack = [wf[chunk[0]: chunk[1]] for wf in waveform_stack]
    t_ms = np.arange(len(waveform_stack[0])) / fs * 1000
    return waveform_stack, t_ms


def upsample_stack(waveform_stack, fs, upsample_res_sec):
    """Upsample all waveforms to a finer time resolution.

    Parameters
    ----------
    waveform_stack : list of numpy.ndarray
    fs : float
        Current sampling frequency in Hz.
    upsample_res_sec : float or None
        Target time resolution in seconds.  If None or ≥ 1/fs the stack and
        fs are returned unchanged.

    Returns
    -------
    waveform_stack : list of numpy.ndarray
    fs : float
        Updated sampling frequency.
    """
    if upsample_res_sec is not None and upsample_res_sec < (1 / fs):
        upsampled = [_upsample(wf, 1 / fs, upsample_res_sec) for wf in waveform_stack]
        waveform_stack = [wf_up for wf_up, _ in upsampled]
        fs = upsampled[0][1]
    return waveform_stack, fs
