from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
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

## ####################################################################
annot_dir = r'C:\Users\xavier.mouy\Documents\GitHub\Haddock-detector\data\test_features'
audio_dir = r'C:\Users\xavier.mouy\Documents\GitHub\Haddock-detector\data\test_features'
out_dir = r'C:\Users\xavier.mouy\Documents\GitHub\Haddock-detector\data\test_features\output_measurements'

# Spectrogram parameters
spectro_unit='sec'
spectro_nfft=0.08
spectro_frame=0.05
spectro_inc=0.008

window_type = 'hann'
disp_plots = True

resampling_fs_hz = 4000
bkg_spectral_subtraction = True

freq_min_hz=30
freq_max_hz=1000
energy_window_sec = 0.06
energy_threshold = 0.4

## ####################################################################

# load detections
detec = Annotation()
detec.from_raven(annot_dir)

# loop through detection and perform measurements
first_meas = True 
for idx in range(0,len(detec)):
    detec_test = detec.data.iloc[[idx]]
    try:
        file_path = os.path.join(detec_test['audio_file_dir'].values[0],detec_test['audio_file_name'].values[0])+detec_test['audio_file_extension'].values[0]
        file_channel = detec_test['audio_channel'].values[0]-1
        start_time = detec_test['time_min_offset'].values[0]
        end_time = detec_test['time_max_offset'].values[0]
        # Unique ID for that sound selection
        ID = detec_test['audio_file_name'] + '_' + str(start_time)
        ID = ID.values[0]
        print(ID)
        # load audio data
        sound = Sound(file_path)
        sound.read(channel=file_channel, chunk=[start_time, end_time],unit='sec')
        # decimate
        sound.decimate(new_sampling_frequency=resampling_fs_hz)
        fs = sound.waveform_sampling_frequency
        # bandpass filter
        sound.filter(filter_type='bandpass',cutoff_frequencies=[freq_min_hz,freq_max_hz],order=8,)
        # Calculates  spectrogram
        spectro = Spectrogram(spectro_frame, window_type, spectro_nfft, spectro_inc, fs, unit=spectro_unit)
        spectro.compute(sound, dB=True)

        spectro.crop(frequency_min=freq_min_hz, frequency_max=freq_max_hz,inplace=True)

        # denoise
        if bkg_spectral_subtraction:
            bkg_spec = np.mean(spectro.spectrogram,axis=1)
            spectro._spectrogram = spectro.spectrogram - bkg_spec[:,None]

        # calculate energy
        energy_window_samp = round(energy_window_sec * fs)
        E = rolling_energy(sound.waveform, window=energy_window_samp, alignment='center')
        E = E-min(E)
        E = E/max(E)
        E = savgol_filter(E, window_length=energy_window_samp, polyorder=2)

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
        axes[0].set_xlabel('Time (s)')
        axes[1].plot(t_x, sound.waveform,'k', alpha=0.5, label='Waveform')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude')
        ax2 = axes[1].twinx()
        #ax2.plot(t_x, E, alpha=0.9, label='Normalized energy')
        ax2.plot(t_x, E, 'g', alpha=0.9, label='Smoothed normalized energy')
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
        #plt.show()

        # Measurements
        if len(peaks_sec) > 1:

            IPI = peaks_sec[1:] - peaks_sec[0:-1] # Inter-pulse interval
            IPI_median_sec = np.median(IPI) # median
            IPI_mean_sec = np.mean(IPI) # mean
            IPI_std_sec = np.std(IPI) # standard deviation
            IPI_cv = IPI_std_sec/IPI_mean_sec #  coefficient of variation (CV) - measure of dispersiom
            PPR_hz = 1/IPI_mean_sec # pulse repetition rate
            dur_sec = peaks_sec[-1]-peaks_sec[0] # duration
            n_pulses = len(peaks_sec) # number of pulses

            # calculate average spectrum of pulses
            first_it = True
            for peak_sec in peaks_sec:

                start_sec=peak_sec - (energy_window_sec/2)
                end_sec = peak_sec + (energy_window_sec/2)
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
            plt.savefig(os.path.join(out_dir, ID+'.png'))
            #plt.show()

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
                'ID': [ID],
                'IPI': [IPI],
                'IPI_median_sec': [IPI_median_sec],
                'IPI_mean_sec':[IPI_mean_sec],
                'IPI_std_sec':[IPI_std_sec],
                'IPI_cv':[IPI_cv],
                'PPR_hz':[PPR_hz],
                'dur_sec':[dur_sec],
                'n_pulses':[n_pulses],
                'freq_peak_hz': [peak_position_unit],
                'freq_pct5_hz': [percentiles_position['5']],
                'freq_pct25_hz': [percentiles_position['25']],
                'freq_pct50_hz': [percentiles_position['50']],
                'freq_pct75_hz': [percentiles_position['75']],
                'freq_pct95_hz': [percentiles_position['95']],
                'freq_IQR_hz': [inter_quart_range],
                'freq_concentration_hz': [concentration_unit],
                'freq_centroid_hz': [centroid],
            })

            #Stack measurements
            if first_meas == True:
                first_meas = False
                measurements_df = features
            else:
                measurements_df = pd.concat([measurements_df,features], ignore_index=True)
            print('Finished!')

    except:
        print('Issue happened with this detection')

measurements_df.to_csv(os.path.join(out_dir,'measurements.csv'),index=False)