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
import ecosound.core.tools
from ecosound.core.tools import derivative_1d, envelope, read_yaml
from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa, linearized_inversion, solve_iterative_ML
import platform

def run_detector(sound, config, chunk=None, deployment_file=None):


    # load audio data
    if chunk:
        sound.read(channel=config['AUDIO']['channel'], chunk=[t1, t2], unit='sec', detrend=True)
    else:
        sound.read(channel=config['AUDIO']['channel'], detrend=True)

    # Calculates  spectrogram
    spectro = Spectrogram(config['SPECTROGRAM']['frame_sec'],
                                  config['SPECTROGRAM']['window_type'],
                                  config['SPECTROGRAM']['nfft_sec'],
                                  config['SPECTROGRAM']['step_sec'],
                                  sound.waveform_sampling_frequency,
                                  unit='sec',)
    spectro.compute(sound,
                    config['SPECTROGRAM']['dB'],
                    config['SPECTROGRAM']['use_dask'],
                    config['SPECTROGRAM']['dask_chunks'],)

    spectro.crop(frequency_min=config['SPECTROGRAM']['fmin_hz'],
                         frequency_max=config['SPECTROGRAM']['fmax_hz'],
                         inplace=True,
                         )
    # Denoise
    print('Denoise')
    spectro.denoise(config['DENOISER']['denoiser_name'],
                    window_duration=config['DENOISER']['window_duration_sec'],
                    use_dask=config['DENOISER']['use_dask'],
                    dask_chunks=tuple(config['DENOISER']['dask_chunks']),
                    inplace=True)
    # Detector
    print('Detector')
    file_timestamp = ecosound.core.tools.filename_to_datetime(sound.file_name)[0]
    detector = DetectorFactory(config['DETECTOR']['detector_name'],
                               kernel_duration=config['DETECTOR']['kernel_duration_sec'],
                               kernel_bandwidth=config['DETECTOR']['kernel_bandwidth_hz'],
                               threshold=config['DETECTOR']['threshold'],
                               duration_min=config['DETECTOR']['duration_min_sec'],
                               bandwidth_min=config['DETECTOR']['bandwidth_min_hz']
                               )
    detections = detector.run(spectro,
                              start_time=file_timestamp,
                              use_dask=config['DETECTOR']['use_dask'],
                              dask_chunks=tuple(config['DETECTOR']['dask_chunks']),
                              debug=False,
                              )

    # add deployment metadata
    detections.insert_metadata(deployment_file, channel = sound.channel_selected)

    # Add file informations
    file_name = os.path.splitext(os.path.basename(sound.file_full_path))[0]
    file_dir = os.path.dirname(sound.file_full_path)
    file_ext = os.path.splitext(sound.file_full_path)[1]
    detections.insert_values(operator_name=platform.uname().node,
                               audio_file_name=file_name,
                               audio_file_dir=file_dir,
                               audio_file_extension=file_ext,
                               audio_file_start_date=ecosound.core.tools.filename_to_datetime(sound.file_full_path)[0]
                               )

    print('s')

    #measurements.insert_metadata(deployment_file)

## ############################################################################

# Config files
hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\hydrophones_config_07-HI.csv'
detection_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\detection_config.yaml'
localization_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\localization_config.yaml'
deployment_info_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\deployment_info.csv'

# load configuration parameters
hydrophones_coords= pd.read_csv(hydrophones_config_file) # load hydrophone coordinates (meters)
detection_config = read_yaml(detection_config_file)
localization_config = read_yaml(localization_config_file)

# #TDOA
# ref_channel=3
# #Linearized inversion
# sound_speed_mps = 1484
# inversion_params = {
#     'start_model':[0,0,0],
#     'start_model_repeats': 5,
#     'damping_factor': 0.2,
#     'stop_delta_m': 0.01, # threshold for stoping iterations (change in norm of models)
#     'stop_max_iteration': 1000,
#     }




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


# load audio data
sound = Sound(audio_files[3])

# run detector on reference channel
run_detector(sound, detection_config, chunk = [t1, t2], deployment_file=deployment_info_file)

## ############################################################################
## ############################################################################

# # plot hydrophones
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# colors = matplotlib.cm.tab10(hydrophones_coords.index.values)
# # Sources
# for index, hp in hydrophones_coords.iterrows():
#     point = ax.scatter(hp['x'],hp['y'],hp['z'],
#                     s=20,
#                     color=colors[index],
#                     label=hp['name'],
#                     )
# # Axes labels
# ax.set_xlabel('X (m)', labelpad=10)
# ax.set_ylabel('Y (m)', labelpad=10)
# ax.set_zlabel('Z (m)', labelpad=10)
# # legend
# ax.legend(bbox_to_anchor=(1.07, 0.7, 0.3, 0.2), loc='upper left')
# plt.tight_layout()
# plt.show()

## ###########################################################################
##               Automatic detection on reference channel
## ###########################################################################




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
## ###########################################################################
##                                Localisation
## ###########################################################################

# Lineralized inversion
[m, iterations_logs] = linearized_inversion(tdoa_sec,
                                            hydrophones_coords,
                                            hydrophone_pairs,
                                            inversion_params,
                                            sound_speed_mps,
                                            doplot=False)

## TODO: verify that the localization correspond to teh fish/matlab

print('s')
