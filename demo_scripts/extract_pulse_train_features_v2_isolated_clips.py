from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.tools import list_files
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.detection.kurtosis_detector import KurtosisDetector, findPeaks
from ecosound.measurements import SpectrogramFeatures
from scipy.signal import savgol_filter
import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy.signal as spsig
from xarray.util.generate_ops import inplace
import copy
'''
This script measures characteristics of pulse trains from isolated audio clips

'''
matplotlib.use('TkAgg')
def rolling_energy(x, window, alignment='center', pad_mode='reflect'):
    """
    Rolling energy (sum of squares) over a window along the last axis.


    Args:
        x (array-like): 1D waveform (or ND, time on the last axis).
        window (int): number of samples in the window (>=1).
        alignment (str):
            - 'center': centered window (default)
            - 'past'  : causal; includes current and previous (window-1) samples
            - 'future': anti-causal; includes current and next (window-1) samples
        pad_mode (str): np.pad mode for edge handling ('reflect', 'edge', 'constant', ...)


    Returns:
        np.ndarray: same shape as x with rolling energy along the last axis.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if alignment not in ('center', 'past', 'future'):
        raise ValueError("alignment must be 'center', 'past', or 'future'")


    x = np.asarray(x, dtype=np.float64)
    kernel = np.ones(int(window), dtype=np.float64)


    # Handle multi-d arrays by applying along the last axis
    if x.ndim > 1:
        return np.apply_along_axis(
            lambda v: _rolling_energy_1d(v, window, alignment, pad_mode, kernel),
            axis=-1, arr=x
        )
    else:
        return _rolling_energy_1d(x, window, alignment, pad_mode, kernel)




def _rolling_energy_1d(x, window, alignment, pad_mode, kernel):
    x2 = x * x


    if alignment == 'center':
        left = window // 2
        right = window - 1 - left
        pad = (left, right)
    elif alignment == 'past':          # causal
        pad = (window - 1, 0)
    else:                               # 'future' anti-causal (lookahead)
        pad = (0, window - 1)


    x2_pad = np.pad(x2, pad, mode=pad_mode)
    # Convolution with 'valid' keeps output length == len(x)
    return np.convolve(x2_pad, kernel, mode='valid')


def max_peaks_by_threshold(array, threshold, strictly=True):
    """
    Find the maximum peak (value and index) for each contiguous region
    where `array` exceeds `threshold`.


    Args:
        array (1D array-like): input signal.
        threshold (float): threshold to exceed.
        strictly (bool): if True use '>' (exceed), if False use '>='.


    Returns:
        peak_idx (np.ndarray[int]): indices of the max peak in each region.
        peak_val (np.ndarray): values at those indices.
    """
    x = np.asarray(array)
    if x.ndim != 1:
        raise ValueError("array must be 1D")


    mask = (x > threshold) if strictly else (x >= threshold)
    if not mask.any():
        return np.array([], dtype=int), np.array([], dtype=x.dtype)


    # Find region boundaries where mask turns on/off
    d = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(d == 1) + 1
    ends   = np.flatnonzero(d == -1) + 1
    if mask[0]:   # started above threshold
        starts = np.r_[0, starts]
    if mask[-1]:  # ended above threshold
        ends = np.r_[ends, len(x)]


    peak_idx, peak_val = [], []
    for s, e in zip(starts, ends):
        i = s + np.argmax(x[s:e])  # first index of max if plateau
        peak_idx.append(i)
        peak_val.append(x[i])


    return np.array(peak_idx, dtype=int), np.array(peak_val)


def find_optimal_frequency_band(waveform, fs, freq_ranges, energy_window_samp,
                                 peak_threshold=0.3, filter_order=8, min_peaks=2):
    """
    Find the frequency band best suited for detecting a pulse train.

    Uses a composite score based on:
    - Number of peaks detected (more peaks = better for pulse trains)
    - Consistency of peak heights (low CV = similar amplitude pulses)
    - SNR (median peak height vs background)

    This approach favors bands with many similar-amplitude peaks rather than
    bands dominated by a single strong impulse.

    Args:
        waveform (np.ndarray): Audio waveform data.
        fs (float): Sampling frequency in Hz.
        freq_ranges (list): List of (freq_min, freq_max) tuples to test.
        energy_window_samp (int): Window size in samples for rolling energy.
        peak_threshold (float): Threshold for peak detection (0-1 normalized).
        filter_order (int): Order of the bandpass filter.
        min_peaks (int): Minimum number of peaks required for a valid band.

    Returns:
        best_band (tuple): (freq_min, freq_max) with highest pulse train score.
        best_energy (np.ndarray): The normalized/smoothed energy curve from that band.
        all_metrics (dict): Dict with all metrics for each band.
    """
    best_score = -np.inf
    best_band = None
    best_energy = None
    all_metrics = {}

    for freq_min, freq_max in freq_ranges:
        # Design bandpass filter
        nyquist = fs / 2
        low = freq_min / nyquist
        high = freq_max / nyquist
        # Ensure filter frequencies are valid
        if high >= 1.0:
            high = 0.99
        if low <= 0:
            low = 0.01
        if low >= high:
            continue

        try:
            sos = spsig.butter(filter_order, [low, high], btype='band', output='sos')
            filtered_waveform = spsig.sosfilt(sos, waveform)
        except Exception:
            continue

        # Calculate energy envelope
        E = rolling_energy(filtered_waveform, window=energy_window_samp, alignment='center')
        E = E - np.min(E)
        if np.percentile(E, 99) > 0:
            E = E / np.percentile(E, 99)
        E_smooth = savgol_filter(E, window_length=energy_window_samp, polyorder=2)

        # Detect peaks
        peak_idx, peak_vals = max_peaks_by_threshold(E_smooth, peak_threshold)
        n_peaks = len(peak_idx)

        if n_peaks >= min_peaks:
            # Coefficient of variation of peak heights (lower = more consistent)
            peak_cv = np.std(peak_vals) / (np.mean(peak_vals) + 1e-10)
            # Consistency score: 1 when CV=0, decreases as CV increases
            consistency = 1.0 / (1.0 + peak_cv)

            # SNR: median peak height relative to background (non-peak regions)
            background_level = np.percentile(E_smooth, 25)  # Use 25th percentile as background
            median_peak_snr = np.median(peak_vals) / (background_level + 1e-10)

            # Composite score: favors many consistent peaks with good SNR
            # Using log(n_peaks) to prevent very high peak counts from dominating
            pulse_train_score = np.log1p(n_peaks) * consistency * np.log1p(median_peak_snr)
        else:
            peak_cv = np.nan
            consistency = 0
            median_peak_snr = 0
            pulse_train_score = 0

        all_metrics[(freq_min, freq_max)] = {
            'pulse_train_score': pulse_train_score,
            'n_peaks': n_peaks,
            'peak_consistency': consistency,
            'peak_cv': peak_cv,
            'median_peak_snr': median_peak_snr,
            'energy': E_smooth
        }

        if pulse_train_score > best_score:
            best_score = pulse_train_score
            best_band = (freq_min, freq_max)
            best_energy = E_smooth

    return best_band, best_energy, all_metrics


## ####################################################################
audio_dir = r'C:\Users\xavier.mouy\Documents\Projects\2025_CNN_Minke_Raick\IPI\partage_XavierMouy_IPI-20260201T211635Z-3-001\exemples-beaux-sons' # folder where the corresponding audio data are
out_dir = r'C:\Users\xavier.mouy\Documents\Projects\2025_CNN_Minke_Raick\IPI\partage_XavierMouy_IPI-20260201T211635Z-3-001\exemples-beaux-sons\measurements' # folder where results are written

# Spectrogram parameters (for display/plot only - measurement actually done on waveform)
spectro_unit='sec'
spectro_nfft=0.128
spectro_frame=0.128
spectro_inc=0.04
window_type = 'hann'
disp_plots = False
#time_buffer_sec = 0.2 # nb seconds to add before and after the annotation to provide more context when manually reviewing results

resampling_fs_hz = 2000
#bkg_spectral_subtraction = True  # not used anymore

# For the energy calculation
freq_min_hz = 50  # Overall minimum frequency for spectrogram display
freq_max_hz = 800  # Overall maximum frequency for spectrogram display
energy_window_sec = 0.1
energy_threshold = 0.3

# Frequency bands to test for optimal band selection
# The band with the highest "pulse train score" will be selected
# Score = log(n_peaks) * consistency * log(SNR)
# This favors bands with many similar-amplitude peaks (pulse trains) over
# bands dominated by a single strong impulse (noise)
freq_bands_to_test = [
    (50, 150), (100, 200), (150, 300), (200, 400),
    (250, 450), (300, 500), (350, 550), (400, 600),
    (450, 650), (500, 700), (550, 750), (600, 800),
    (50, 300), (100, 400), (200, 500), (300, 600), (400, 800),  # wider bands
    (50, 400), (100, 500), (200, 600), (300, 800),  # even wider
    (50, 800),  # full range as fallback
]

# Filters
#min_duration_sec = 0.5 # minimum duration of the annotations to process (can be used to remove annotations that are too short)

## ####################################################################

audio_files = list_files(audio_dir,'.wav')
first_meas = True
# loop through detection and perform measurements
for file_path in audio_files:
    try:

        #os.path.basename(path)  # "report.txt"
        filename = os.path.splitext(os.path.basename(file_path))[0]
        print(filename)
        # define audio file
        sound = Sound(file_path)
        # load audio data
        sound.read(channel=0)
        # decimate
        sound.decimate(new_sampling_frequency=resampling_fs_hz)
        fs = sound.waveform_sampling_frequency
        energy_window_samp = round(energy_window_sec * fs)

        # Find optimal frequency band based on pulse train score
        # (favors bands with many consistent-amplitude peaks)
        optimal_band, optimal_energy, band_metrics = find_optimal_frequency_band(
            waveform=sound.waveform,
            fs=fs,
            freq_ranges=freq_bands_to_test,
            energy_window_samp=energy_window_samp,
            peak_threshold=energy_threshold,
            filter_order=8,
            min_peaks=2
        )
        selected_freq_min, selected_freq_max = optimal_band
        metrics = band_metrics[optimal_band]
        print(f"  -> Selected optimal band: {selected_freq_min}-{selected_freq_max} Hz "
              f"(score: {metrics['pulse_train_score']:.2f}, "
              f"n_peaks: {metrics['n_peaks']}, "
              f"consistency: {metrics['peak_consistency']:.2f})")

        # Apply bandpass filter with the optimal band for the filtered sound used in energy calculation
        sound_for_energy = copy.deepcopy(sound)
        sound_for_energy.filter(filter_type='bandpass', cutoff_frequencies=[selected_freq_min, selected_freq_max], order=8)

        # Also apply a broad filter for spectrogram display (full range)
        sound.filter(filter_type='bandpass', cutoff_frequencies=[freq_min_hz, freq_max_hz], order=8)

        # Calculates spectrogram (using broad frequency range for display)
        spectro = Spectrogram(spectro_frame, window_type, spectro_nfft, spectro_inc, fs, unit=spectro_unit)
        spectro.compute(sound, dB=True)
        spectro.crop(frequency_min=freq_min_hz, frequency_max=freq_max_hz, inplace=True)

        # # denoise
        # if bkg_spectral_subtraction:
        #     bkg_spec = np.mean(spectro.spectrogram, axis=1)
        #     spectro._spectrogram = spectro.spectrogram - bkg_spec[:, None]

        # Use the pre-calculated optimal energy curve
        E = optimal_energy

        # Define energy peaks
        peaks_samp, peaks_vals = max_peaks_by_threshold(E, energy_threshold)
        peaks_sec = peaks_samp/fs


        # ---- plot ----
        fs = sound.waveform_sampling_frequency
        t_x = np.arange(len(sound.waveform)) / fs
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        axes[0].pcolormesh(
                spectro.axis_times,
                spectro.axis_frequencies,
                spectro.spectrogram,
                cmap='viridis',
                vmin=np.percentile(spectro.spectrogram, 50),
                vmax=np.percentile(spectro.spectrogram, 99.9),
                shading="nearest",
            )
        # Draw horizontal lines showing the selected optimal frequency band
        axes[0].axhline(y=selected_freq_min, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[0].axhline(y=selected_freq_max, color='r', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title(f'Spectrogram - Optimal band: {selected_freq_min}-{selected_freq_max} Hz '
                          f'(score: {metrics["pulse_train_score"]:.2f}, '
                          f'n_peaks: {metrics["n_peaks"]}, '
                          f'consistency: {metrics["peak_consistency"]:.2f})')
        axes[1].plot(t_x, sound.waveform,'k', alpha=0.5, label='Waveform')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        ax2 = axes[1].twinx()
        #ax2.plot(t_x, E, alpha=0.9, label='Normalized energy')
        ax2.plot(t_x, E, 'g', alpha=0.9, label=f'Energy ({selected_freq_min}-{selected_freq_max} Hz)')
        ax2.set_ylabel('Normalized energy')
        # plot peaks
        ax2.plot(peaks_sec,peaks_vals,'.r')
        #plot threshold
        ax2.plot([t_x[0], t_x[-1]], [energy_threshold,energy_threshold], '--r')
        # Combine legends from both axes
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        axes[1].set_xlim(t_x[0], t_x[-1])
        fig.tight_layout()
        #plt.savefig('overlayed_energy.png')
        if disp_plots:
            plt.show()


        # Measurements
        if len(peaks_sec) > 1:


            IPI = peaks_sec[1:] - peaks_sec[0:-1] # Inter-pulse interval
            IPI_median_sec = np.median(IPI) # median
            IPI_mean_sec = np.mean(IPI) # mean
            IPI_std_sec = np.std(IPI) # standard deviation
            IPI_cv = IPI_std_sec/IPI_mean_sec #  coefficient of variation (CV) - measure of dispersiom
            PRR_hz = 1/IPI_mean_sec # pulse repetition rate
            dur_sec = peaks_sec[-1]-peaks_sec[0] # duration
            n_pulses = len(peaks_sec) # number of pulses


            # calculate average spectrum of pulses
            first_it = True
            for peak_sec in peaks_sec:


                start_sec=peak_sec - (energy_window_sec)
                end_sec = peak_sec + (energy_window_sec)
                mask = (spectro.axis_times >= start_sec) & (spectro.axis_times <= end_sec)
                idx_1d = np.where(mask)[0]
                #av_spec =  np.sum(spectro.spectrogram[:,idx_1d],1)
                if first_it==True:
                    spect_list = spectro.spectrogram[:,idx_1d]
                else:
                    spect_list = np.concatenate((spect_list, spectro.spectrogram[:,idx_1d]), axis=1)
                first_it = False
            av_spec = np.mean(spect_list,axis=1)
            #axes[2].plot(spectro.axis_frequencies, spect_list, 'k', alpha=0.2, label='Spectrum')
            axes[2].plot(spectro.axis_frequencies, av_spec, 'k', alpha=1, label='Average spectrum')
            axes[2].set_ylim(min(av_spec), max(av_spec))
            axes[2].set_xlim(min(spectro.axis_frequencies), max(spectro.axis_frequencies))
            axes[2].set_ylabel('Normalized energy')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_title('Mean spectrum')
            axes[2].grid()
            #axes[2].legend(loc='upper right')
            fig.tight_layout()
            plt.savefig(os.path.join(out_dir, filename+'.png'))
            #plt.show()
            plt.close('all')


            # Spectral measurements
            # peak
            peak_value, peak_position_unit, peak_position_relative = SpectrogramFeatures.peak(av_spec, spectro.axis_frequencies)
            # Position of percentiles
            percentiles_value = [5, 25, 50, 75, 95]
            percentiles_position = SpectrogramFeatures.percentiles_position(av_spec, percentiles_value, axis=spectro.axis_frequencies)
            # Inter quartile range
            inter_quart_range = percentiles_position['75'] - percentiles_position['25']
            # duration/width
            length = SpectrogramFeatures.length(av_spec, spectro.axis_frequencies[1] - spectro.axis_frequencies[0])
            # duration/width containing 90% of magnitude
            length_90 = percentiles_position['95'] - percentiles_position['5']
            # concentration
            concentration_unit = SpectrogramFeatures.concentration(av_spec, spectro.axis_frequencies)
            # centroid
            centroid = SpectrogramFeatures.centroid(av_spec, spectro.axis_frequencies)


            # save as dataframe
            features = pd.DataFrame({
                'ID': [filename],
                'selected_band_min_hz': [selected_freq_min],
                'selected_band_max_hz': [selected_freq_max],
                'selected_band_score': [metrics['pulse_train_score']],
                #'selected_band_n_peaks': [metrics['n_peaks']],
                'selected_band_consistency': [metrics['peak_consistency']],
                'selected_band_peak_cv': [metrics['peak_cv']],
                'selected_band_snr': [metrics['median_peak_snr']],
                'IPI': [IPI],
                'IPI_median_sec': [IPI_median_sec],
                'IPI_mean_sec': [IPI_mean_sec],
                'IPI_std_sec': [IPI_std_sec],
                'IPI_cv': [IPI_cv],
                'PRR_hz': [PRR_hz],
                'dur_sec': [dur_sec],
                'n_pulses': [n_pulses],
                'freq_peak_hz': [peak_position_unit],
                'freq_pct5_hz': [percentiles_position['5']],
                'freq_pct25_hz': [percentiles_position['25']],
                'freq_pct50_hz': [percentiles_position['50']],
                'freq_pct75_hz': [percentiles_position['75']],
                'freq_pct95_hz': [percentiles_position['95']],
                'freq_IQR_hz': [inter_quart_range],
                'freq_concentration_hz': [concentration_unit],
                'freq_centroid_hz': [centroid],
                'spectrum_freq_hz': [spectro.axis_frequencies],
                'spectrum_normalized_amplitude': [av_spec]
            })


            #Stack measurements
            if first_meas == True:
                first_meas = False
                measurements_df = features
            else:
                measurements_df = pd.concat([measurements_df,features], ignore_index=True)
            print('Finished!')


    except Exception as e:
        print('Issue happened with this detection')
        print(f"A general error occurred: {e}")
measurements_df.to_csv(os.path.join(out_dir,'measurements.csv'),index=False)
