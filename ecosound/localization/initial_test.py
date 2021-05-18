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

from ecosound.core.audiotools import Sound, upsample
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.core.tools import derivative_1d, envelope
from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa

## ############################################################################
#TDOA
ref_channel=3
#Linearized inversion
sound_speed_mps = 1484
inversion_params = {
    'm0': [-1,-1,-1],
    'damping_factor': 0.1,
    'stop_delta_m': 0.0001, # threshold for stoping iterations (change in norm of models)
    'max_iteration': 2000,
    }
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


# define search window based on hydrophone separation and sound speed
hydrophones_dist_matrix = calc_hydrophones_distances(hydrophones_coords)
TDOA_max_sec = np.max(hydrophones_dist_matrix)/sound_speed_mps

# define hydrophone pairs
hydrophone_pairs = defineReceiverPairs(len(hydrophones_coords), ref_receiver=ref_channel)

# pick single detection (will use loop after)
detec = detections.data.iloc[0]
#detec = detections.data.iloc[33]

# Adjust start/stop times of detection on reference channel
detec_wav = sound.select_snippet([detec['time_min_offset'],detec['time_max_offset']], unit='sec')
detec_wav.tighten_waveform_window(95)
TDOA_max_samp = int(np.round(TDOA_max_sec*sound.waveform_sampling_frequency))
t1_detec = detec_wav.waveform_start_sample - TDOA_max_samp
t2_detec = detec_wav.waveform_stop_sample + TDOA_max_samp

# load data from all channels for that detection
waveform_stack = []
for idx, audio_file in enumerate(audio_files):  # for each channel
    # load waveform
    chan_wav = Sound(audio_file)
    chan_wav.read(channel=0,
                  chunk=[t1_detec, t2_detec],
                  unit='samp',
                  detrend=True)
    sampling_frequency = chan_wav.waveform_sampling_frequency
    # bandpass filter
    chan_wav.filter('bandpass', [detec['frequency_min'], detec['frequency_max']])
    # stack
    waveform_stack.append(chan_wav.waveform)

# calculate TDOAs
tdoa_sec, corr_val = calc_tdoa(waveform_stack,
                               hydrophone_pairs,
                               sampling_frequency,
                               TDOA_max_sec=TDOA_max_sec,
                               upsample_res_sec=0.0000001,
                               normalize=False,
                               doplot=False,
                               )
## TO DO
# If correlation coef too small =>
# 1 - calc TDOA on sliding window
# 2 - calc TDOA on narrow frequency bands
# 3 - Use less Hp to localize

# Lineralized inversion



def solve_iterative_ML(d, hydrophones_coords, hydrophone_pairs, m, V, damping_factor):
    # Define the Jacobian matrix: Eq. (5) in Mouy et al. (2018)
    A = defineJacobian(hydrophones_coords, m, V, hydrophone_pairs)
    # Predicted TDOA at m (forward problem): Eq. (1) in Mouy et al. (2018)
    d0 = predict_tdoa(m, V, hydrophones_coords, hydrophone_pairs)
    # Reformulation of the problem
    delta_d = d-d0 # Delta d: original data - predicted data
    # Resolving by creeping approach(ML inverse for delta m): Eq. (6) in Mouy et al. (2018)
    delta_m = np.dot(np.dot(np.linalg.inv(np.dot(A.transpose(),A)),A.transpose()),delta_d) #general inverse
    # New model: Eq. (7) in Mouy et al. (2018)
    m_new = m.iloc[0].values + (damping_factor*delta_m.transpose())
    m['x'] = m_new[0,0]
    m['y'] = m_new[0,1]
    m['z'] = m_new[0,2]
    # ## jumping approach
    # #m=inv(A'*CdInv*A)*A'*CdInv*dprime; % retrieved model using ML
    # Data misfit
    part1 = np.dot(A,delta_m)-delta_d
    data_misfit = np.dot(part1.transpose(), part1)
    return m, data_misfit[0, 0]


def linearized_inversion(d, hydrophones_coords,hydrophone_pairs,inversion_params, sound_speed_mps, doplot=False):

    # convert parameters to numpy arrays or pandas dataframes
    m = pd.DataFrame({'x': [inversion_params['m0'][0]], 'y': [inversion_params['m0'][1]], 'z': [inversion_params['m0'][2]]})
    damping_factor = np.array(inversion_params['damping_factor'])
    Tdelta_m = np.array(inversion_params['stop_delta_m'])
    V = np.array(sound_speed_mps)
    max_iteration = np.array(inversion_params['max_iteration'])

    # Keeps track of values for each iteration
    iterations_logs = pd.DataFrame({
        'x': [inversion_params['m0'][0]],
        'y': [inversion_params['m0'][1]],
        'z': [inversion_params['m0'][2]],
        'norm': np.nan,
        'data_misfit': np.nan,
        })

    # Start iterations
    stop = False
    idx=0
    while stop == False:
        idx=idx+1
        # Linear inversion
        m_it, data_misfit_it = solve_iterative_ML(d, hydrophones_coords, hydrophone_pairs, m, V, damping_factor)
        # Save model and data misfit for each iteration
        iterations_logs = iterations_logs.append({
            'x': m_it['x'].values[0],
            'y': m_it['y'].values[0],
            'z': m_it['z'].values[0],
            'norm': np.sqrt(np.square(m_it).sum(axis=1)).values[0],
            'data_misfit': data_misfit_it,
            }, ignore_index=True)
        # Update m
        m = m_it
        # stopping criteria
        if idx>1:
             # change in norm of model diffreence
            if (iterations_logs['norm'][idx] - iterations_logs['norm'][idx-1] <= Tdelta_m):
                stop = True
                converged=True
            elif idx > max_iteration: # max iterations exceeded
                stop = True
                converged=False
        # stops if never converges
        if idx > max_iteration:
            stop = 1
            print('Inversion hasn''t converged.')

    if not converged:
        m['x']=np.nan
        m['y']=np.nan
        m['z']=np.nan

    return m, iterations_logs

[m, iterations_logs]=linearized_inversion(tdoa_sec,hydrophones_coords,hydrophone_pairs, inversion_params, sound_speed_mps,doplot=False)

## TODO: repeat with dfferent m0 and pick teh one with the lowest datamisfit
## TODO: verify that the localization correspond to teh fish/matlab

print('s')
