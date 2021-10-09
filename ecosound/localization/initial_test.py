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
import math

from ecosound.core.audiotools import Sound, upsample
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.measurement import Measurement
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
import ecosound.core.tools
from ecosound.core.tools import derivative_1d, envelope, read_yaml
from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa, linearized_inversion, solve_iterative_ML, defineCubeVolumeGrid, defineSphereVolumeGrid,defineSphereSurfaceGrid
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
    spectro.denoise(config['DENOISER']['denoiser_name'],
                    window_duration=config['DENOISER']['window_duration_sec'],
                    use_dask=config['DENOISER']['use_dask'],
                    dask_chunks=tuple(config['DENOISER']['dask_chunks']),
                    inplace=True)
    # Detector
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


def stack_waveforms(audio_files, detec, TDOA_max_sec):
    waveform_stack = []
    for audio_file, channel in zip(audio_files['path'], audio_files['channel'] ): # for each channel
        # load waveform
        chan_wav = Sound(audio_file)
        chan_wav.read(channel=channel,
                      chunk=[detec['time_min_offset']-TDOA_max_sec, detec['time_max_offset']+TDOA_max_sec],
                      unit='sec',
                      detrend=True)
        # bandpass filter
        chan_wav.filter('bandpass', [detec['frequency_min'], detec['frequency_max']])
        # stack
        waveform_stack.append(chan_wav.waveform)
    return waveform_stack


def calc_data_error(tdoa_sec, m, sound_speed_mps,hydrophones_config, hydrophone_pairs):
    """ Calculates tdoa measurement errors. Eq. (9) in Mouy et al. 2018"""
    tdoa_m = predict_tdoa(m, sound_speed_mps, hydrophones_config, hydrophone_pairs)
    Q = len(m)
    M = m.size # number of dimensions of the model (here: X, Y, and Z)
    N = len(tdoa_sec) # number of measurements
    if N > M:
        error_std = np.sqrt((1/(Q*(N-M))) * (sum((tdoa_sec-tdoa_m)**2)))
    else:
        error_std = np.sqrt((sum((tdoa_sec-tdoa_m)**2)))
    return error_std


def calc_loc_errors(tdoa_errors_std, m, sound_speed_mps, hydrophones_config, hydrophone_pairs):
    """ Calculates localization errors. Eq. (8) in Mouy et al. 2018."""
    A = defineJacobian(hydrophones_config, m, sound_speed_mps, hydrophone_pairs)
    Cm = (tdoa_errors_std**2) * np.linalg.inv(np.dot(A.transpose(),A)) # Model covariance matrix for IID
    err_std = np.sqrt(np.diag(Cm))
    return pd.DataFrame({'x_std': [err_std[0]], 'y_std': [err_std[1]], 'z_std': [err_std[2]]})

def cartesian2spherical(x,y,z):
    # Converting cartesian to polar coordinate
    const = 180/np.pi
    XsqPlusYsq = x**2 + y**2
    r = math.sqrt(XsqPlusYsq + z**2)               # r
    elev = math.atan2(math.sqrt(XsqPlusYsq),z)     # theta
    az = math.atan2(y,x)*180/np.pi
    return r, az, elev

# # Calculating radius
# radius = math.sqrt( x * x + y * y )
# # Calculating angle (theta) in radian
# theta = math.atan(y/x)
# # Converting theta from radian to degree
# theta = 180 * theta/math.pi

## ############################################################################

#outdir=r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization'
## ------------------------- LARGE ARRAY --------------------------------------

# # Config files XAV array hornby - quilback
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array_HF2.yaml'
# #detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.4.20190920T161248Z.wav'
# t1 = 1560#1570
# t2 = 1590
# #detec_idx_forced= 0 

# # Config files XAV array hornby - lingcod - THIS IS THE GOOD ONE!!!
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array_lingcod.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.4.20190919T204248Z.wav'
# t1 = 697
# t2 = 700
# #detec_idx_forced= 0 

# # Config files XAV array hornby - lingcod - Sep 17
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array_lingcod.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.4.20190917T011248Z.wav'
# #t1 = 318
# #t2 = 342
# #t1 = 322
# #t2 = 325

# t1 = 335
# t2 = 337
#detec_idx_forced= 0 



# Config files XAV array hornby - lingcod
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.4.20190916T204248Z.wav'
# # t1 = 589#1440
# # t2 = 604#1459

# #also try
# t1 = 631
# t2 = 635
# #detec_idx_forced= 6 #26

# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.4.20190919T194248Z.wav'
# t1 =1119 #944# 1052
# t2 =1121 #950# 1056



# # Config files XAV array hornby - lingcod
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.4.20190919T001248Z.wav'
# #t1 = 352
# #t2 = 362
# t1 = 903
# t2 = 910
# detec_idx_forced= 10


# # Config files XAV array Ogden Point - ROV on side of array, in front of fishcam2
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\hydrophones_config_04-OGD.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array_HF.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\AMAR173.4.20190617T161307Z.wav'
# t1 = 1528
# t2 = 1534
# #detec_idx_forced = 1

# # Config files XAV array Ogden Point - Lingcod
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\hydrophones_config_04-OGD.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'

# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\AMAR173.4.20190616T154307Z.wav'
# t1 = 274#330
# t2 = 276#334

# # maybe good one.
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\AMAR173.4.20190616T131307Z.wav'
# t1 = 1489
# t2 = 1492

# # good one (the one used in thesis)
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\AMAR173.4.20190617T151307Z.wav'
# t1 = 222
# t2 = 227

# #ok one
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\AMAR173.4.20190616T171307Z.wav'
# t1 = 999
# t2 = 1005


# # Config files XAV array Ogden Point - lingcod
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\hydrophones_config_04-OGD.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_large_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-06-15_OgdenPoint_AMAR_04-OGD\AMAR173.4.20190616T154307Z.wav'
# t1 = 329
# t2 = 335
# #detec_idx_forced = 1

## ------------------------- MOBILE ARRAY -------------------------------------

# # Config files mobile array - Projector MCauley Point -> fish - 0 degree
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\hydrophones_config_MCP-20200910.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\5147.200910210736.wav'
# # projector fish signal
# t1 = 106
# t2 = 119
# #detec_idx_forced=2

# # Config files mobile array - Projector MCauley Point -> fish - 90 degree
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\hydrophones_config_MCP-20200910.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\5147.200910210736.wav'
# # projector fish signal
# t1 = 769 
# t2 = 782 
# #detec_idx_forced=1


# # Config files mobile array - Projector MCauley Point -> fish - 180 degree
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\hydrophones_config_MCP-20200910.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\5147.200910210736.wav'
# # projector fish signal
# t1 = 910#928.8
# t2 = 923#940
# #detec_idx_forced = 0


# # Config files mobile array - Projector MCauley Point -> fish - -90 degree
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\hydrophones_config_MCP-20200910.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\5147.200910210736.wav'
# # projector fish signal
# t1 = 1060#928.8
# t2 = 1073#940
# #detec_idx_forced = 0

# # Config files mobile array - Projector MCauley Point -> fish - 0 degree - source 1 m above
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\hydrophones_config_MCP-20200910.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\5147.200910213736.wav'
# # projector fish signal
# t1 = 893.8
# t2 = 906.5
# detec_idx_forced = 0

# Config files mobile array - Horny Island - Black eye goby 
deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\deployment_info.csv'
hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\hydrophones_config_HI-201909.csv'
detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\671404070.190916182406.wav'
t1 = 138.6
t2 = 147.9

# t1 = 205
# t2 = 214
#detec_idx_forced = 10

# # Config files mobile array - Horny Island - Quillback part 1 
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\hydrophones_config_HI-201909.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\671404070.190918170055.wav'
# t1 = 155
# t2 = 159
# detec_idx_forced = 1

# # Config files mobile array - Horny Island - Quillback part 2 
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\hydrophones_config_HI-201909.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\671404070.190918170055.wav'
# t1 = 258
# t2 = 262
# detec_idx_forced = 1

# # Config files mobile array - Horny Island - Copper Rockfish
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\hydrophones_config_HI-201909.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mobile_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mobile_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\671404070.190918222812.wav'
# outdir = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mobile_array_copper'
# # t1 = 216
# # t2 = 223

# t1 = 844
# t2 = 895

#detec_idx_forced = 1

## -------------------------- MINI ARRAY --------------------------------------

# # Config files mini array - Mill Bay - ROV facing fishcam 
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\hydrophones_config_05-MILL.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_large_array_HF.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mini_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\671404070.190801165502.wav'
# # # ROV in front of camera
# # t1 = 853.5
# # t2 = 877

# # ROV in front of camera
# t1 = 880
# t2 = 900
# # infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\671404070.190801181002.wav'
# # # ROV on left side of camera
# # t1 = 10
# # t2 = 30

#detec_idx_forced = 8

# # Config files mini array - Mill Bay - Shiner Perch facing fishcam 
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\hydrophones_config_05-MILL.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mini_array_HF.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mini_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\671404070.190730164002.wav'
# t1 = 50
# t2 = 59
# detec_idx_forced = 0


# # Config files mini array - Mill Bay - Rockfish - localiuzed behind
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\hydrophones_config_05-MILL.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mini_array.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mini_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\671404070.190801181002.wav'
# t1 = 325
# t2 = 340
# detec_idx_forced = 9

# # Config files mini array - Mill Bay - Copper RockFish on top of fishcam 
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\hydrophones_config_05-MILL.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\detection_config_mini_array_copper.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mini_array.yaml'
# infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\671404070.190801232502.wav'
# t1 = 696
# t2 = 713
# #detec_idx_forced = 2


# load configuration parameters
hydrophones_config= pd.read_csv(hydrophones_config_file, skipinitialspace=True, dtype={'name': str, 'file_name_root': str}) # load hydrophone coordinates (meters)
detection_config = read_yaml(detection_config_file)
localization_config = read_yaml(localization_config_file)
if localization_config['METHOD']['linearized_inversion']:
    localization_method_name = 'Linearized inversion'
    if localization_config['METHOD']['grid_search']:
        raise ValueError('Only 1 localization method allowed.')
else:
    localization_method_name = 'Grid search'
    if localization_config['METHOD']['grid_search'] == False:
        raise ValueError('At least 1 localization method needs to be defined.')

# Look up data files for all channels
audio_files = find_audio_files(infile, hydrophones_config)

# run detector on selected channel
print('DETECTION')
detections = run_detector(audio_files['path'][detection_config['AUDIO']['channel']],
                          audio_files['channel'][detection_config['AUDIO']['channel']],
                          detection_config,
                          chunk = [t1, t2],
                          deployment_file=deployment_info_file)
#detections.insert_values(frequency_min=60)

print(str(len(detections)) + ' detections')

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


# localization
sound_speed_mps = localization_config['ENVIRONMENT']['sound_speed_mps']
ref_channel = localization_config['TDOA']['ref_channel']

# define search window based on hydrophone separation and sound speed
hydrophones_dist_matrix = calc_hydrophones_distances(hydrophones_config)
TDOA_max_sec = np.max(hydrophones_dist_matrix)/sound_speed_mps

# define hydrophone pairs
hydrophone_pairs = defineReceiverPairs(len(hydrophones_config), ref_receiver=ref_channel)

# pre-compute grid search if needed
if localization_config['METHOD']['grid_search']:
    sources = defineSphereVolumeGrid(
        localization_config['GRIDSEARCH']['spacing_m'],
        localization_config['GRIDSEARCH']['radius_m'],
        origin=localization_config['GRIDSEARCH']['origin'])
    if localization_config['GRIDSEARCH']['min_z']:
        sources = sources.loc[sources['z']>=localization_config['GRIDSEARCH']['min_z']]
        sources = sources.reset_index(drop=True)
    # sources = defineCubeVolumeGrid(0.2, 2, origin=[0, 0, 0])
    # sources = defineSphereSurfaceGrid(
    #     10000,
    #     localization_config['GRIDSEARCH']['radius_m'],
    #     origin=localization_config['GRIDSEARCH']['origin'])
    #sources = defineCubeVolumeGrid(0.2, 2, origin=[0, 0, 0])
    try:
        npzfile = np.load(localization_config['GRIDSEARCH']['stored_tdoas'])
        sources_tdoa = npzfile['sources_tdoa']
        sources_array = npzfile['sources']
        sources['x'] = sources_array[:,0]
        sources['y'] = sources_array[:,1]
        sources['z'] = sources_array[:,2]
        print('Succesully loaded precomputed grid TDOAs from file.')
    except:
        print("Couln't read precomputed TDOAs from file, computing grid TDOAs...")    
        sources_tdoa = np.zeros(shape=(len(hydrophone_pairs),len(sources)))
        for source_idx, source in sources.iterrows():
            sources_tdoa[:,source_idx] = predict_tdoa(source, sound_speed_mps, hydrophones_config, hydrophone_pairs).T
        np.savez(os.path.join(outdir,'tdoa_grid'),sources_tdoa=sources_tdoa,sources=sources)    
    # # Azimuth:
    # theta = np.arctan2(sources['y'].to_numpy(),sources['x'].to_numpy())*(180/np.pi)
    # theta = (((theta+90) % 360)-180) *(-1)        
    # # Elevation:
    # phi = np.arctan2(sources['y'].to_numpy()**2+sources['x'].to_numpy()**2,sources['z'].to_numpy())*(180/np.pi)
    # phi = phi - 90
    # sources['theta'] = theta
    # sources['phi'] = phi

# Define Measurement object for the localization results
# if localization_config['METHOD']['linearized_inversion']:
#     localizations = Measurement()
#     localizations.metadata['measurer_name'] = localization_method_name
#     localizations.metadata['measurer_version'] = '0.1'
#     localizations.metadata['measurements_name'] = [['x', 'y', 'z', 'x_std', 'y_std', 'z_std', 'tdoa_errors_std']]

# if localization_config['METHOD']['grid_search']:
#     localizations = Measurement()
#     localizations.metadata['measurer_name'] = localization_method_name
#     localizations.metadata['measurer_version'] = '0.1'
#     localizations.metadata['measurements_name'] = [['theta', 'phi', 'theta_std', 'phi_std', 'tdoa_errors_std']]

localizations = Measurement()
localizations.metadata['measurer_name'] = localization_method_name
localizations.metadata['measurer_version'] = '0.1'
localizations.metadata['measurements_name'] = [['x', 'y', 'z', 'x_std', 'y_std', 'z_std', 'tdoa_errors_std','tdoa_sec_1','tdoa_sec_2','tdoa_sec_3','tdoa_sec_4','tdoa_sec_5']]


# need to define what output is for grid search


# pick single detection (will use loop after)
print('LOCALIZATION')
for detec_idx, detec in detections.data.iterrows():

    if 'detec_idx_forced' in locals():
        print('Warning: forced to only process detection #', str(detec_idx_forced))
        detec = detections.data.iloc[detec_idx_forced]
    
    print( str(detec_idx+1) + '/' + str(len(detections)))

    # load data from all channels for that detection
    waveform_stack = stack_waveforms(audio_files, detec, TDOA_max_sec)

    # readjust signal boundaries to only focus on section with most energy 
    percentage_max_energy = 90
    chunk = ecosound.core.tools.tighten_signal_limits_peak(waveform_stack[detection_config['AUDIO']['channel']], percentage_max_energy)
    waveform_stack = [x[chunk[0]:chunk[1]] for x in waveform_stack]

    # calculate TDOAs
    tdoa_sec, corr_val = calc_tdoa(waveform_stack,
                                   hydrophone_pairs,
                                   detec['audio_sampling_frequency'],
                                   TDOA_max_sec=TDOA_max_sec,
                                   upsample_res_sec=localization_config['TDOA']['upsample_res_sec'],
                                   normalize=localization_config['TDOA']['normalize'],
                                   doplot=False,
                                   )

    if localization_config['METHOD']['grid_search']:    
        delta_tdoa = sources_tdoa - tdoa_sec
        delta_tdoa_norm = np.linalg.norm(delta_tdoa, axis=0)
        min_idx = np.argmin(delta_tdoa_norm)
        sources['delta_tdoa'] = delta_tdoa_norm
        #m = sources.loc[sources['delta_tdoa'] == sources['delta_tdoa'].min()]
        m = pd.DataFrame({'x': sources.loc[min_idx]['x'],
                          'y': sources.loc[min_idx]['y'],
                          'z': sources.loc[min_idx]['z']}, index=[0]
                          )
        # # 3D scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # colors = matplotlib.cm.tab10(hydrophones_config.index.values)      
        # alphas = 0.5
        # for index, hp in hydrophones_config.iterrows():
        #     point = ax.scatter(hp['x'],hp['y'],hp['z'],
        #                     s=40,
        #                     color=colors[index],
        #                     label=hp['name'],
        #                     )
        # ax.scatter(sources['x'],
        #             sources['y'],
        #             sources['z'],
        #             c=sources['delta_tdoa'],
        #             s=2,
        #             alpha=alphas,)
        # # Axes labels
        # ax.set_xlabel('X (m)', labelpad=10)
        # ax.set_ylabel('Y (m)', labelpad=10)
        # ax.set_zlabel('Z (m)', labelpad=10)
        # # legend
        # ax.legend(bbox_to_anchor=(1.07, 0.7, 0.3, 0.2), loc='upper left')
        # plt.tight_layout()
        # plt.show()
    
        # # # 2D scatter plot
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111)
        # ax2.scatter(sources.theta, sources.phi, c=sources.delta_tdoa)
        # ax2.set_xlabel('Azimuth angle theta (degree)', labelpad=10)
        # ax2.set_ylabel('Elevation angle phi (degree)', labelpad=10)
        # # sources.plot.hexbin(x="theta",
        # #                     y="phi",
        # #                     C="delta_tdoa",
        # #                     #reduce_C_function=np.mean,
        # #                     reduce_C_function=np.min,
        # #                     gridsize=80,
        # #                     cmap="viridis")
        
        # # # from scipy.stats import binned_statistic_2d
        # # # import numpy as np
        
        # # # x = sources.theta.values
        # # # y = sources.phi.values
        # # # z = sources.delta_tdoa.values
        
        # # # x_bins = np.linspace(-180, 180, 360)
        # # # y_bins = np.linspace(-90, 90, 180)
        
        # # # ret = binned_statistic_2d(x, y, z, statistic=np.mean, bins=[x_bins, y_bins])
        # # # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
        # # #ax0.scatter(x, y, c=z)
        # # # #ax1.imshow(ret.statistic.T, origin='bottom', extent=(0, 10, 10, 20))
        # # # ax1.imshow(ret.statistic.T)    
        
        # # # plt.figure()
        # # # xbins = np.arange(-180,185,5)
        # # # ybins = np.arange(-90,95,5)
        # # # hist = sources.delta_tdoa.groupby([pd.cut(sources.theta, bins=xbins), pd.cut(sources.phi, bins=ybins)]).mean().unstack(fill_value=0)
        # # # im = plt.imshow(hist.values)
        # # # plt.xticks(range(len(hist.index)), hist.index)
        # # # plt.yticks(range(len(hist.columns)), hist.columns)
        # # # plt.colorbar(im)
        # # # plt.show()


        # # Bring all detection and localization informations together
        # detec.loc['theta'] = m['theta'].values[0]
        # detec.loc['phi'] = m['phi'].values[0]        
        # detec.loc['theta_std'] = 0
        # detec.loc['phi_std'] = 0        
        # detec.loc['tdoa_errors_std'] = 0
    
    # Lineralized inversion
    if localization_config['METHOD']['linearized_inversion']:
        [m, iterations_logs] = linearized_inversion(tdoa_sec,
                                                    hydrophones_config,
                                                    hydrophone_pairs,
                                                    localization_config['INVERSION'],
                                                    sound_speed_mps,
                                                    doplot=False)
    
    # Estimate uncertainty
    tdoa_errors_std = calc_data_error(tdoa_sec, m, sound_speed_mps,hydrophones_config, hydrophone_pairs)
    loc_errors_std = calc_loc_errors(tdoa_errors_std, m, sound_speed_mps, hydrophones_config, hydrophone_pairs)

    # Bring all detection and localization informations together
    detec.loc['x'] = m['x'].values[0]
    detec.loc['y'] = m['y'].values[0]
    detec.loc['z'] = m['z'].values[0]
    detec.loc['x_std'] = loc_errors_std['x_std'].values[0]
    detec.loc['y_std'] = loc_errors_std['y_std'].values[0]
    detec.loc['z_std'] = loc_errors_std['z_std'].values[0]
    detec.loc['tdoa_errors_std'] = tdoa_errors_std[0]
    if len(tdoa_sec) >= 3:
        detec.loc['tdoa_sec_1'] = tdoa_sec[0][0]
        detec.loc['tdoa_sec_2'] = tdoa_sec[1][0]
        detec.loc['tdoa_sec_3'] = tdoa_sec[2][0]
    if len(tdoa_sec) > 3:
        detec.loc['tdoa_sec_4'] = tdoa_sec[3][0]
        detec.loc['tdoa_sec_5'] = tdoa_sec[4][0]

    # stack to results into localization object
    localizations.data = localizations.data.append(detec, ignore_index=True)

# Plot hydrophones
fig1 = plt.figure()
ax = fig1.add_subplot(111, projection='3d')
colors = matplotlib.cm.tab10(hydrophones_config.index.values)
# Sources
for index, hp in hydrophones_config.iterrows():
    point = ax.scatter(hp['x'],hp['y'],hp['z'],
                    s=20,
                    color=colors[index],
                    label=hp['name'],
                    )

localization = ax.scatter(localizations.data['x'],
                          localizations.data['y'],
                          localizations.data['z'],
                    s=30,
                    marker='*',
                    color='black',
                    label='Localizations',
                    )

# Axes labels
ax.set_xlabel('X (m)', labelpad=10)
ax.set_ylabel('Y (m)', labelpad=10)
ax.set_zlabel('Z (m)', labelpad=10)
# legend
ax.legend(bbox_to_anchor=(1.07, 0.7, 0.3, 0.2), loc='upper left')
plt.tight_layout()
plt.show()
