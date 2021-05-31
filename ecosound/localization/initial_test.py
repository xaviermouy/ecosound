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
from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa, linearized_inversion, solve_iterative_ML, defineCubeVolumeGrid, defineSphereVolumeGrid
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
    M = m.size
    N = len(tdoa_sec)
    error_std = np.sqrt((1/(Q*(N-M))) * (sum((tdoa_sec-tdoa_m)**2)))
    return error_std


def calc_loc_errors(tdoa_errors_std, m, sound_speed_mps, hydrophones_config, hydrophone_pairs):
    """ Calculates localization errors. Eq. (8) in Mouy et al. 2018."""
    A = defineJacobian(hydrophones_config, m, sound_speed_mps, hydrophone_pairs)
    Cm = (tdoa_errors_std**2) * np.linalg.inv(np.dot(A.transpose(),A)) # Model covariance matrix for IID
    err_std = np.sqrt(np.diag(Cm))
    return pd.DataFrame({'x_std': [err_std[0]], 'y_std': [err_std[1]], 'z_std': [err_std[2]]})

def cartesian2spherical (x,y,z):
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

# # Config files XAV array hornby - quilback
# deployment_info_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\deployment_info.csv'
# hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\hydrophones_config_07-HI.csv'
# detection_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\detection_config.yaml'
# localization_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\localization_config.yaml'
# # will need to loop later ?
# infile = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\data\wav_files\localization\AMAR173.4.20190920T161248Z.wav'
# t1 = 1570
# t2 = 1590

# Config files XAV array hornby - quilback
deployment_info_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\deployment_info.csv'
hydrophones_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\hydrophones_config_MC-ROV.csv'
detection_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\detection_config_MC-ROV.yaml'
localization_config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ecosound\ecosound\localization\config\localization_config_MC-ROV.yaml'
# will need to loop later ?
infile = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Fish-array_ROV\2020-09-09_Localization_experiment_projector\SoundTrap_20200910\5147\5147.200910210736.wav'
#fish
t1 = 106
t2 = 109
#sweeps
t1 = 141
t2 = 160


# load configuration parameters
hydrophones_config= pd.read_csv(hydrophones_config_file, skipinitialspace=True, dtype={'name': str, 'file_name_root': str}) # load hydrophone coordinates (meters)
detection_config = read_yaml(detection_config_file)
localization_config = read_yaml(localization_config_file)

# Look up data files for all channels
audio_files = find_audio_files(infile, hydrophones_config)

# run detector on selected channel
print('DETECTION')
detections = run_detector(audio_files['path'][detection_config['AUDIO']['channel']],
                          audio_files['channel'][detection_config['AUDIO']['channel']],
                          detection_config,
                          chunk = [t1, t2],
                          deployment_file=deployment_info_file)
print(str(len(detections)) + ' detections')

# # plot spectrogram/waveforms of all channels and detections
# plot_data(audio_files,
#           detection_config['SPECTROGRAM']['frame_sec'],
#           detection_config['SPECTROGRAM']['window_type'],
#           detection_config['SPECTROGRAM']['nfft_sec'],
#           detection_config['SPECTROGRAM']['step_sec'],
#           detection_config['SPECTROGRAM']['fmin_hz'],
#           detection_config['SPECTROGRAM']['fmax_hz'],
#           chunk = [t1, t2],
#           detections=detections,
#           detections_channel=detection_config['AUDIO']['channel'])


# localization
sound_speed_mps = localization_config['ENVIRONMENT']['sound_speed_mps']
ref_channel = localization_config['TDOA']['ref_channel']

# define search window based on hydrophone separation and sound speed
hydrophones_dist_matrix = calc_hydrophones_distances(hydrophones_config)
TDOA_max_sec = np.max(hydrophones_dist_matrix)/sound_speed_mps

# define hydrophone pairs
hydrophone_pairs = defineReceiverPairs(len(hydrophones_config), ref_receiver=ref_channel)

# pre-compute grid search if needed
sources = defineSphereVolumeGrid(0.2, 2, origin=[0, 0, 0])
#sources = defineCubeVolumeGrid(0.2, 2, origin=[0, 0, 0])
sources_tdoa = np.zeros(shape=(3,len(sources)))
for source_idx, source in sources.iterrows():
    sources_tdoa[:,source_idx] = predict_tdoa(source, sound_speed_mps, hydrophones_config, hydrophone_pairs).T
theta = np.arctan2(sources['y'].to_numpy(),sources['x'].to_numpy())*(180/np.pi) # azimuth
phi = np.arctan2(sources['y'].to_numpy()**2+sources['x'].to_numpy()**2,sources['z'].to_numpy())*(180/np.pi)
sources['theta'] = theta
sources['phi'] = phi

# Define Measurement object for the localization results
localizations = Measurement()
localizations.metadata['measurer_name'] = '3D Linearized inversion'
localizations.metadata['measurer_version'] = '0.1'
localizations.metadata['measurements_name'] = [['x', 'y', 'z', 'x_std', 'y_std', 'z_std', 'tdoa_errors_std']]

# pick single detection (will use loop after)
print('LOCALIZATION')
for detec_idx, detec in detections.data.iterrows():

    detec = detections.data.iloc[2]
    print( str(detec_idx+1) + '/' + str(len(detections)))

    # load data from all channels for that detection
    waveform_stack = stack_waveforms(audio_files, detec, TDOA_max_sec)

    # calculate TDOAs
    tdoa_sec, corr_val = calc_tdoa(waveform_stack,
                                   hydrophone_pairs,
                                   detec['audio_sampling_frequency'],
                                   TDOA_max_sec=TDOA_max_sec,
                                   upsample_res_sec=localization_config['TDOA']['upsample_res_sec'],
                                   normalize=localization_config['TDOA']['normalize'],
                                   doplot=False,
                                   )
    ## TO DO
    # If correlation coef too small =>
    # 1 - calc TDOA on sliding window
    # 2 - calc TDOA on narrow frequency bands
    # 3 - Use less Hp to localize

    delta_tdoa = sources_tdoa - tdoa_sec
    delta_tdoa_norm = np.linalg.norm(delta_tdoa, axis=0)
    sources['delta_tdoa'] = delta_tdoa_norm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = matplotlib.cm.tab10(hydrophones_config.index.values)
    #alphas = delta_tdoa_norm - min(delta_tdoa_norm)
    #alphas = alphas/max(alphas)
    #alphas = alphas - 1
    #alphas = abs(alphas)
    #alphas = np.array(alphas)
    alphas = 0.5
    for index, hp in hydrophones_config.iterrows():
        point = ax.scatter(hp['x'],hp['y'],hp['z'],
                        s=40,
                        color=colors[index],
                        label=hp['name'],
                        )
    ax.scatter(sources['x'],
                sources['y'],
                sources['z'],
                c=sources['delta_tdoa'],
                s=20,
                alpha=alphas,)
    # Axes labels
    ax.set_xlabel('X (m)', labelpad=10)
    ax.set_ylabel('Y (m)', labelpad=10)
    ax.set_zlabel('Z (m)', labelpad=10)
    # legend
    ax.legend(bbox_to_anchor=(1.07, 0.7, 0.3, 0.2), loc='upper left')
    plt.tight_layout()
    plt.show()


    # Lineralized inversion
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
