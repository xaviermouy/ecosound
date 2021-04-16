# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 10:07:53 2021

@author: xavier.mouy
"""
import os
import sys
sys.path.append(r'C:\Users\xavier.mouy\Documents\GitHub\ecosound') # Adds higher directory to python modules path.

import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib.cm
#import scipy.spatial
import numpy as np
import scipy.signal

from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.core.tools import derivative_1d
import localizationlib

## ############################################################################
#TDOA
ref_channel=3
#Linearized inversion
sound_speed_mps = 1484
InversionParams_m0=[0,0,0]
InversionParams_damping=0.1
InversionParams_Tdelta_m=0.0001 # threshold for stoping iterations (change in norm of models)
#InversionParams.Tdelta_X2=0.0001 # threshold for stoping iterations (change in data misfit)
InversionParams_maxIterations=2000 # 400 threshold for stoping iterations (change in data misfit)
# hydrophone coordinates (meters)
x=[-0.858, -0.858,  0.858, 0.028, -0.858, 0.858]
y=[-0.860, -0.860, -0.860, 0.000,  0.860, 0.860]
z=[-0.671,  0.479, -0.671,-0.002, -0.671, 0.671]
names=['Hydrophone 0', 'Hydrophone 1', 'Hydrophone 2', 'Hydrophone 3', 'Hydrophone 4', 'Hydrophone 5']
hydrophones_coords= pd.DataFrame({'name':names, 'x':x,'y':y, 'z':z})

# audio files
audio_dir = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\wav_files\localization'
audio_files = []
audio_files.append(os.path.join(audio_dir, 'AMAR173.1.20190920T161248Z.wav'))
audio_files.append(os.path.join(audio_dir, 'AMAR173.2.20190920T161248Z.wav'))
audio_files.append(os.path.join(audio_dir, 'AMAR173.3.20190920T161248Z.wav'))
audio_files.append(os.path.join(audio_dir, 'AMAR173.4.20190920T161248Z.wav'))
audio_files.append(os.path.join(audio_dir, 'AMAR173.5.20190920T161248Z.wav'))
audio_files.append(os.path.join(audio_dir, 'AMAR173.6.20190920T161248Z.wav'))

# detection parameters:
# Spectrogram parameters
frame = 0.0625
nfft = 0.0853
step = 0.01
fmin = 0
fmax = 1000
window_type = 'hann'
# start and stop time of wavfile to analyze
t1 = 1570
t2 = 1590

## ############################################################################
## ############################################################################

# plot hydrophones
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
colors = matplotlib.cm.tab10(hydrophones_coords.index.values)
# Sources
for index, hp in hydrophones_coords.iterrows():
    point = ax.scatter(hp['x'],hp['y'],hp['z'],
                    s=20,
                    color=colors[index],
                    label=hp['name'],
                    )
# Axes labels
ax.set_xlabel('X (m)', labelpad=10)
ax.set_ylabel('Y (m)', labelpad=10)
ax.set_zlabel('Z (m)', labelpad=10)
# legend
ax.legend(bbox_to_anchor=(1.07, 0.7, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()

## ###########################################################################
##               Automatic detection on reference channel
## ###########################################################################

# load audio data
sound = Sound(audio_files[ref_channel])
sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)

# Crop unused frequencies
spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)

# Denoise
spectro.denoise('median_equalizer', window_duration=3, use_dask=True, dask_chunks=(2048,1000), inplace=True)

# Detector
detector = DetectorFactory('BlobDetector', use_dask=True, dask_chunks=(2048,2000), kernel_duration=0.05, kernel_bandwidth=300, threshold=20, duration_min=0.05, bandwidth_min=40)
detections = detector.run(spectro, debug=False)

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
# graph.add_data(sound)
# graph.add_annotation(detections, panel=0, color='grey',label='Detections')
# graph.add_data(spectro)
# graph.add_annotation(detections, panel=1,color='black',label='Detections')
# graph.colormap = 'binary'
# #graph.colormap = 'jet'
# graph.show()


## ###########################################################################
##                             Plot all channels
## ###########################################################################

graph_spectros = GrapherFactory('SoundPlotter', title='Spectrograms', frequency_max=1000)
graph_waveforms = GrapherFactory('SoundPlotter', title='Waveforms', frequency_max=1000)

for idx, audio_file in enumerate(audio_files): # for each channel
    # load waveform
    sound = Sound(audio_file)
    sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)
    # Calculates  spectrogram
    spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
    spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)
    # Crop unused frequencies
    spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
    # Plot
    graph_spectros.add_data(spectro)
    graph_waveforms.add_data(sound)

graph_spectros.colormap = 'binary'
graph_spectros.add_annotation(detections, panel=ref_channel, color='green',label='Detections')
graph_spectros.show()

graph_waveforms.add_annotation(detections, panel=ref_channel, color='green',label='Detections')
graph_waveforms.show()



## ###########################################################################
##                                   TDOA
## ###########################################################################

def euclidean_dist(df1, df2, cols=['x','y','z']):
    """
    Calculates euclidean distance between two Pandas dataframes

    Parameters
    ----------
    df1 : TYPE
        DESCRIPTION.
    df2 : TYPE
        DESCRIPTION.
    cols : TYPE, optional
        DESCRIPTION. The default is ['x','y','z'].

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return np.linalg.norm(df1[cols].values - df2[cols].values,axis=0)


def calc_hydrophones_distances(hydrophones_coords):
    """
    Calculates Euclidiean distance between each hydrophone of the array

    Parameters
    ----------
    hydrophones_coords : TYPE
        DESCRIPTION.

    Returns
    -------
    hydrophones_dist_matrix : TYPE
        DESCRIPTION.

    """
    hydrophones_dist_matrix = np.empty((len(hydrophones_coords),len(hydrophones_coords)))
    for index1, row1 in hydrophones_coords.iterrows():
        for index2, row2 in hydrophones_coords.iterrows():
            dist = euclidean_dist(row1, row2)
            hydrophones_dist_matrix[index1, index2] = dist
    return hydrophones_dist_matrix


# define search window based on hydrophone separation and sound speed
hydrophones_dist_matrix = calc_hydrophones_distances(hydrophones_coords)
TDOA_limit_sec = np.max(hydrophones_dist_matrix)/sound_speed_mps
TDOA_limit_samp = int(np.round(TDOA_limit_sec*sound.waveform_sampling_frequency))

# define hydrophone pairs
hydrophone_pairs = localizationlib.defineReceiverPairs(len(hydrophones_coords), ref_receiver=ref_channel)

# pick single detection (will use loop after)
detec = detections.data.iloc[0]
#detec = detections.data.iloc[36]

# Adjust start/stop times of detection on reference channel
detec_wav = sound.select_snippet([detec['time_min_offset'],detec['time_max_offset']], unit='sec')
detec_wav.tighten_waveform_window(95)
t1_detec = detec_wav.waveform_start_sample - TDOA_limit_samp
t2_detec = detec_wav.waveform_stop_sample + TDOA_limit_samp

# load data from all channels for that detection
graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
waveform_stack=[]
envelope_stack=[]
leading_edge_stack=[]

for idx, audio_file in enumerate(audio_files): # for each channel
    # load waveform
    chan_wav = Sound(audio_file)
    chan_wav.read(channel=0, chunk=[t1_detec, t2_detec], unit='samp', detrend=True)
    # bandpass filter
    chan_wav.filter('bandpass',[detec['frequency_min'],detec['frequency_max']])
    # resample
    chan_wav.upsample(0.000001)
    # normalize amplitude
    chan_wav.normalize()
    # # envelope
    # import scipy.signal as spsig
    # import copy
    # #analytic_signal = spsig.hilbert(chan_wav.waveform-np.mean(chan_wav.waveform))
    # #amplitude_envelope = np.abs(analytic_signal)
    # chan_wav2 = copy.copy(chan_wav)
    # chan_wav2._filter_applied = False
    # #chan_wav2._waveform = np.abs(chan_wav2.waveform)
    # #temp = chan_wav2._waveform[chan_wav2._waveform<0]=0
    # #temp = chan_wav2._waveform.clip(min=0)
    # chan_wav2._waveform = chan_wav2._waveform.clip(min=0)
    # chan_wav2.filter('lowpass',[30])
    # amplitude_envelope = chan_wav2.waveform

    # #amplitude_envelope = amplitude_envelope - np.min(amplitude_envelope)
    # #amplitude_envelope = amplitude_envelope / np.max(amplitude_envelope)
    # # leading edge
    # leading_edge = derivative_1d(amplitude_envelope, order=1)
    # #leading_edge = leading_edge - np.min(leading_edge)
    # #leading_edge = leading_edge / np.max(leading_edge)


    # stack
    waveform_stack.append(chan_wav)
    #envelope_stack.append(amplitude_envelope)
    #leading_edge_stack.append(leading_edge)

    # plt.figure()
    # plt.plot(chan_wav.waveform, color='black')
    # plt.plot(amplitude_envelope, color='red')
    # plt.plot(leading_edge, color='blue')
    # # plot
    # ax[0].plot(chan_wav.waveform,color=colors[idx],label='Hydrophone ' + str(idx))
    # ax[1].plot(amplitude_envelope,color=colors[idx],label='Hydrophone ' + str(idx))
    # ax[2].plot(leading_edge,color=colors[idx],label='Hydrophone ' + str(idx))


# cross correlation
s1 = waveform_stack[0].waveform
s2 = waveform_stack[3].waveform
corr = scipy.signal.correlate(s1,s2, mode='same', method='auto')
corr = corr/(np.linalg.norm(s1)*np.linalg.norm(s2))
# find max peak
min_xcorr_value = 0.8
peaks = scipy.signal.find_peaks(corr, height=min_xcorr_value)
max_peak_value = np.max(peaks[1]['peak_heights'])
max_peak_idx = peaks[0][peaks[1]['peak_heights'].tolist().index(max_peak_value)]


fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(s1, color='black')
ax[0].plot(s2, color='red')
ax[0].set_xlabel('Time (sample)')
ax[0].set_ylabel('Amplitude')
ax[0].grid()
ax[0].set_title('TDOA: ' + str(max_peak_idx) + ' samples')
ax[1].plot(corr)
ax[1].plot(max_peak_idx, max_peak_value,marker = '.', color='r')
ax[1].set_xlabel('Lag (sample)')
ax[1].set_ylabel('Correlation')
ax[1].grid()





## TO DO
# 1 - Calc envelop
# 2 - Calc leading edge
# 3 - Calc cross correlation and TDOA
# 4 - calc TDOA on sliding window
# 5 - parabolic interpolation