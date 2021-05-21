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
import datetime

from ecosound.core.audiotools import Sound, upsample
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
import ecosound.core.tools
from ecosound.core.tools import derivative_1d, envelope, read_yaml
from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa, linearized_inversion, solve_iterative_ML
import platform

def find_audio_files(filename, hydrophones_config):
    """ Find corresponding files and channels for all the hydrophones of the array """
    filename = os.path.basename(infile)
    # Define file tail
    for file_root in hydrophones_config['file_name_root']:
        idx = filename.find(file_root)
        if idx >= 0:
            file_tail = filename[len(file_root):]
            break
    # Loop through channels and define all files paths and audio channels
    audio_files = {'path':[], 'channel':[]}
    for row_idx, row_data in hydrophones_config.iterrows():
        file_path = os.path.join(row_data.data_path, row_data.file_name_root + file_tail)
        chan = row_data.audio_file_channel
        audio_files['path'].append(file_path)
        audio_files['channel'].append(chan)
    return audio_files

def run_detector(infile, channel, config, chunk=None, deployment_file=None):

    sound = Sound(infile)
    # load audio data
    if chunk:
        sound.read(channel=channel, chunk=[t1, t2], unit='sec', detrend=True)
        time_offset_sec = t1
    else:
        sound.read(channel=channel, detrend=True)
        time_offset_sec = 0

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
    file_timestamp = ecosound.core.tools.filename_to_datetime(sound.file_full_path)[0]
    detector = DetectorFactory(config['DETECTOR']['detector_name'],
                               kernel_duration=config['DETECTOR']['kernel_duration_sec'],
                               kernel_bandwidth=config['DETECTOR']['kernel_bandwidth_hz'],
                               threshold=config['DETECTOR']['threshold'],
                               duration_min=config['DETECTOR']['duration_min_sec'],
                               bandwidth_min=config['DETECTOR']['bandwidth_min_hz']
                               )
    start_time = file_timestamp + datetime.timedelta(seconds=time_offset_sec)
    detections = detector.run(spectro,
                              start_time=start_time,
                              use_dask=config['DETECTOR']['use_dask'],
                              dask_chunks=tuple(config['DETECTOR']['dask_chunks']),
                              debug=False,
                              )
    # add time offset in only a section of recording was analysed.
    detections.data['time_min_offset'] = detections.data['time_min_offset'] + time_offset_sec
    detections.data['time_max_offset'] = detections.data['time_max_offset'] + time_offset_sec

    # add deployment metadata
    detections.insert_metadata(deployment_file, channel=channel)

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

    return detections


def plot_data(audio_files,frame, window_type, nfft, step, fmin, fmax, chunk = None, detections=None, detections_channel=0):
    graph_spectros = GrapherFactory('SoundPlotter', title='Spectrograms', frequency_max=fmax)
    graph_waveforms = GrapherFactory('SoundPlotter', title='Waveforms')
    for audio_file, channel in zip(audio_files['path'], audio_files['channel'] ): # for each channel
        # load waveform
        sound = Sound(audio_file)
        sound.read(channel=channel, chunk=chunk, unit='sec', detrend=True)
        # Calculates  spectrogram
        spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
        spectro.compute(sound, dB=True, use_dask=False)
        # Crop unused frequencies
        spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
        # Plot
        graph_spectros.add_data(spectro, time_offset_sec=chunk[0])
        graph_waveforms.add_data(sound, time_offset_sec=chunk[0])

    graph_spectros.colormap = 'binary'
    if detections:
        graph_spectros.add_annotation(detections, panel=detections_channel, color='green',label='Detections')
        graph_waveforms.add_annotation(detections, panel=detections_channel, color='green',label='Detections')

    if chunk:
        graph_spectros.time_min = chunk[0]
        graph_spectros.time_max = chunk[1]
        graph_waveforms.time_min = chunk[0]
        graph_waveforms.time_max = chunk[1]

    graph_spectros.show()
    graph_waveforms.show()

## ############################################################################

# Config files
deployment_info_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\deployment_info.csv'
hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\hydrophones_config_07-HI.csv'
detection_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\detection_config.yaml'
localization_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\localization_config.yaml'

# load configuration parameters
hydrophones_config= pd.read_csv(hydrophones_config_file) # load hydrophone coordinates (meters)
detection_config = read_yaml(detection_config_file)
localization_config = read_yaml(localization_config_file)

# will need to loop later ?
infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\wav_files\localization\AMAR173.4.20190920T161248Z.wav'
t1 = 1570
t2 = 1590

# Look up data files for all channels
audio_files = find_audio_files(infile, hydrophones_config)

# run detector on selected channel
detections = run_detector(audio_files['path'][detection_config['AUDIO']['channel']],
                          audio_files['channel'][detection_config['AUDIO']['channel']],
                          detection_config,
                          chunk = [t1, t2],
                          deployment_file=deployment_info_file)

# plot spectrogram/waveforms of all channels and detections
plot_data(audio_files,
          detection_config['SPECTROGRAM']['frame_sec'],
          detection_config['SPECTROGRAM']['window_type'],
          detection_config['SPECTROGRAM']['nfft_sec'],
          detection_config['SPECTROGRAM']['step_sec'],
          detection_config['SPECTROGRAM']['fmin_hz'],
          detection_config['SPECTROGRAM']['fmax_hz'],
          chunk = [t1, t2],
          detections=detections,
          detections_channel=detection_config['AUDIO']['channel'])


## ###########################################################################
##                                   TDOA
## ###########################################################################

# define search window based on hydrophone separation and sound speed
hydrophones_dist_matrix = calc_hydrophones_distances(hydrophones_config)
TDOA_max_sec = np.max(hydrophones_dist_matrix)/sound_speed_mps

# define hydrophone pairs
hydrophone_pairs = defineReceiverPairs(len(hydrophones_config), ref_receiver=ref_channel)

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

# Lineralized inversion
[m, iterations_logs] = linearized_inversion(tdoa_sec,
                                            hydrophones_config,
                                            hydrophone_pairs,
                                            inversion_params,
                                            sound_speed_mps,
                                            doplot=False)

## TODO: verify that the localization correspond to teh fish/matlab

print('s')


# # plot hydrophones
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# colors = matplotlib.cm.tab10(hydrophones_config.index.values)
# # Sources
# for index, hp in hydrophones_config.iterrows():
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
