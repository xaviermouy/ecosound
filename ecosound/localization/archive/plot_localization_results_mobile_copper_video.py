# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 14:55:40 2021

@author: xavier.mouy
"""
import os
import sys
sys.path.append(r'C:\Users\xavier.mouy\Documents\GitHub\ecosound') # Adds higher directory to python modules path.

import pandas as pd
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import matplotlib.cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import scipy.spatial
import numpy as np
import datetime
import math
import cv2

from ecosound.core.audiotools import Sound, upsample
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.measurement import Measurement
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
import ecosound.core.tools
from ecosound.core.tools import derivative_1d, envelope, read_yaml
from localizationlib import euclidean_dist, calc_hydrophones_distances, calc_tdoa, defineReceiverPairs, defineJacobian, predict_tdoa, linearized_inversion, solve_iterative_ML, defineCubeVolumeGrid, defineSphereVolumeGrid
import platform
import cv2

def plot_spectrogram(audio_file,loc,geometry=(1,1,1)):
    
    fmin=0
    fmax=1000
    frame= 0.0625
    window_type= 'hann'
    nfft=0.0853
    step=0.01
    channel=0
    
    graph_spectros = GrapherFactory('SoundPlotter', title='Spectrograms', frequency_max=fmax)
    sound = Sound(audio_file)
    sound.read(channel=channel, unit='sec', detrend=True)
    t1_sec = 0
    t2_sec = sound.file_duration_sec
    # Calculates  spectrogram
    spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='sec')
    spectro.compute(sound, dB=True, use_dask=False)
    # Crop unused frequencies
    spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)
    # Plot
    graph_spectros.add_data(spectro)

    #graph_spectros.add_annotation(loc, panel=0, color='burlywood',label='Detections')
    graph_spectros.add_annotation(loc, panel=0, color='peachpuff')

    graph_spectros.colormap = 'binary' #'jet'
    fig, ax = graph_spectros.show()

    if ax.get_geometry() != geometry :
        ax.change_geometry(*geometry)
    return t1_sec,t2_sec,fig, ax

def plot_top_view(hydrophones_config,loc_data,params,cmap,norm, ax):
    
    #fig1 = plt.figure()
    #ax = fig1.add_subplot(111)
    colors = matplotlib.cm.tab10(hydrophones_config.index.values)
    
    # plot hydrophones
    for index, hp in hydrophones_config.iterrows():
        point = ax.scatter(hp['x'],hp['y'],
                           s=20,
                           #color=colors[index],
                           color='gainsboro',
                           edgecolors='dimgray',
                           #markeredgecolor='r',
                           label=hp['name'],
                           alpha=1,
                           zorder=3
                           )

        # plot frame
        frame_color = 'whitesmoke'
        frame_alpha = 1
        frame_width = 3
        
        rectangle = plt.Rectangle((-0.1,-0.2), 0.2, 0.4,
                                  linewidth=1,
                                  ec='dimgray',
                                  alpha=frame_alpha,
                                  facecolor='lightgrey')
        
        ax.add_patch(rectangle)
        
        ax.plot([-0.46, 0.48],[0,0],
              linewidth=1 ,
              alpha=frame_alpha,
              linestyle= 'solid',
              color='dimgray',
              )
        ax.plot([0, 0],[0,0.49],
              linewidth=1 ,
              alpha=frame_alpha,
              linestyle= 'solid',
              color='dimgray',
              )

    # plot localizations
    ax.scatter(loc_data['x'], loc_data['y'],c=loc_data['time_min_offset'],
                        s=params['loc_size'].values[0],
                        marker=params['loc_marker'].values[0],
                        #color=params['loc_color'].values[0],
                        alpha=params['loc_alpha'].values[0],
                        cmap=cmap,
                        norm=norm,
                        zorder=5,
                        )
    # plot uncertainties
    for idx, loc_point in loc_data.iterrows():   
        ax.plot([loc_point['x_err_low'],loc_point['x_err_high']],
                [loc_point['y'],loc_point['y']],
                #c=loc_point['time_min_offset'],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                #cmap=cmap,
                #norm=norm,
                )
    
        ax.plot([loc_point['x'],loc_point['x']],
                [loc_point['y_err_low'],loc_point['y_err_high']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                )
        
    # Axes labels
    ax.set_xlabel('X (m)', labelpad=4)
    ax.set_ylabel('Y (m)', labelpad=4)
    ax.set_xlim(params['x_min'].values[0], params['x_max'].values[0])
    ax.set_ylim(params['y_min'].values[0], params['y_max'].values[0])
    #ax.grid()
    ax.set_aspect('equal', adjustable='box')
    return ax


def plot_side_view(hydrophones_config,loc_data,params,cmap,norm, ax):
    
    #fig1 = plt.figure()
    #ax = fig1.add_subplot(111)
    colors = matplotlib.cm.tab10(hydrophones_config.index.values)
    
    # plot hydrophones
    for index, hp in hydrophones_config.iterrows():
        point = ax.scatter(hp['x'],hp['z'],
                           s=20,
                           #color=colors[index],
                           color='gainsboro',
                           edgecolors='dimgray',
                           #markeredgecolor='r',
                           label=hp['name'],
                           alpha=1,
                           zorder=3
                           )

    # plot frame
    frame_color = 'whitesmoke'
    frame_alpha = 1
    frame_width = 3
    
    rectangle = plt.Rectangle((-0.1,-0.1), 0.2, 0.1,
                          linewidth=1,
                          ec='dimgray',
                          alpha=frame_alpha,
                          facecolor='lightgrey')
        
    ax.add_patch(rectangle)
        
    ax.plot([-0.46, 0.48],[0,0],
      linewidth=1 ,
      alpha=frame_alpha,
      linestyle= 'solid',
      color='dimgray',
      )
    
    ax.plot([0, 0],[0,0.54],
      linewidth=1 ,
      alpha=frame_alpha,
      linestyle= 'solid',
      color='dimgray',
      )
    
    # plot localizations
    ax.scatter(loc_data['x'], loc_data['z'],c=loc_data['time_min_offset'],
                        s=params['loc_size'].values[0],
                        marker=params['loc_marker'].values[0],
                        #color=params['loc_color'].values[0],
                        alpha=params['loc_alpha'].values[0],
                        cmap=cmap,
                        norm=norm,
                        zorder=5
                        )
    # plot uncertainties
    for idx, loc_point in loc_data.iterrows():   
        ax.plot([loc_point['x_err_low'],loc_point['x_err_high']],
                [loc_point['z'],loc_point['z']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                )
    
        ax.plot([loc_point['x'],loc_point['x']],
                [loc_point['z_err_low'],loc_point['z_err_high']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                )
        
    # Axes labels
    ax.set_xlabel('X (m)', labelpad=4)
    ax.set_ylabel('Z (m)', labelpad=4)
    ax.set_xlim(params['x_min'].values[0], params['x_max'].values[0])
    ax.set_ylim(params['z_min'].values[0], params['z_max'].values[0])
    #ax.grid()
    ax.set_aspect('equal', adjustable='box')
    return ax

def plot_video_frame(video_file,frame_time_sec, ax):
    frame = []
    cap = cv2.VideoCapture(video_file)
    if (cap.isOpened()== False):        
        print("Error opening video stream or file")
    else:
     	# Read until video is completed
        idx=0
        while(cap.isOpened()):            
            ret, frame = cap.read()            
            if ret == True:
                idx+=1
                fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"                
                current_time = idx*(1/fps)
                if current_time >= frame_time_sec:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                    frame = frame[90:900,340:1220]
                    break                
            # Break the loop
            else:
                break
        cap.release()
        #cv2.destroyAllWindows()
    return ax.imshow(frame)

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

## ###########################################################################

def plot_full_figure(time_sec=None):
    
    indir = r'C:\Users\xavier.mouy\Documents\Publications\Mouy.etal_2022_XAV-Arrays\manuscript\data\mobile_copper-goby'
    loc_file = r'localization_results.nc'
    audio_file = r'671404070.190918233146.wav'
    hp_config_file = r'hydrophones_config_HI-201909.csv'
    localization_config_file =r'localization_config_mobile_array.yaml'

    params=pd.DataFrame({
        'loc_color': ['black'],
        'loc_marker': ['o'],
        'loc_alpha': [1],
        'loc_size': [5],
        'uncertainty_color': ['black'],
        'uncertainty_style': ['-'],
        'uncertainty_alpha': [1], #0.7
        'uncertainty_width': [0.2], #0.2
        'x_min':[-0.8],
        'x_max':[0.8],
        'y_min':[-0.5],
        'y_max':[1],
        'z_min':[-0.5],
        'z_max':[1],    
        })
    
    # add path tol file names
    loc_file = os.path.join(indir,loc_file)
    audio_file = os.path.join(indir,audio_file)
    hp_config_file = os.path.join(indir,hp_config_file)
    localization_config_file = os.path.join(indir,localization_config_file)
    
    ## ###########################################################################
    localization_config = read_yaml(localization_config_file)
    hydrophones_config = pd.read_csv(hp_config_file)
    sound_speed_mps = localization_config['ENVIRONMENT']['sound_speed_mps']
    ref_channel = localization_config['TDOA']['ref_channel']
    hydrophone_pairs = defineReceiverPairs(len(hydrophones_config), ref_receiver=ref_channel)
    
    ## load localization results
    loc = Measurement()
    loc.from_netcdf(loc_file)
    loc_data = loc.data
   
    if time_sec!= None:
        loc_data = loc_data.loc[(loc_data['time_max_offset']<=time_sec)] 
    else:
        print('Static')
    
    # update loc object
    loc.data = loc_data

    
    # Plot spectrogram
    t1_sec,t2_sec,fig_final, ax_spectro = plot_spectrogram(audio_file,loc, geometry=(5,1,1))    
    ax_spectro.set_title("")
    ax_spectro.get_xaxis().set_visible(False)
    n_colors = t2_sec-t1_sec
    cmap = mpl.cm.get_cmap('viridis', int(n_colors*4))
    norm = mpl.colors.Normalize(vmin=0, vmax=n_colors)
    divider = make_axes_locatable(ax_spectro)
    cax = divider.append_axes('bottom', 0.1, pad=0.03 )
    ax_cmap = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                    norm=norm,
                                    orientation='horizontal')
    ax_cmap.set_label('Time (s)')
    
    if time_sec:
        SFreq_min, SFreq_max = ax_spectro.get_ylim()
        ax_spectro.plot([time_sec,time_sec],[SFreq_min,SFreq_max],'r')
    
    # plot detection points on top of spectrogram
    #gs0 = fig_final.add_gridspec(60,1)
    ax_detec = fig_final.add_subplot(20,1,1)
    det_y = np.asarray(np.ones((1,len(loc_data['time_min_offset']))))[0]
    det_x = np.asarray(loc_data['time_min_offset'])
    ax_detec.scatter(det_x,det_y, c=loc_data['time_min_offset'],cmap=cmap,norm=norm,s=12)
    ax_detec.set_xlim(ax_spectro.get_xlim())
    ax_detec.get_xaxis().set_visible(False)
    ax_detec.get_yaxis().set_visible(False)
    ax_detec.axis('off')

    plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)

    gs = fig_final.add_gridspec(3,2)

    # plot localization top
    ax_toploc = fig_final.add_subplot(gs[1:,1])
    plot_top_view(hydrophones_config,loc_data,params,cmap,norm, ax_toploc)
    ax_toploc.set_anchor('E')
    
    # plot localization side
    #ax_sideloc = fig_final.add_subplot(3,3,7,sharex = ax_toploc)
    ax_sideloc = fig_final.add_subplot(gs[1:,0])
    plot_side_view(hydrophones_config,loc_data,params,cmap,norm,ax_sideloc)
    ax_sideloc.set_anchor('W')
    
    # set the spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    
    fig_final.set_size_inches(10.25, 6.72)

    box = ax_spectro.get_position()
    box.y0 = box.y0 - 0.03
    box.y1 = box.y1 - 0.03
    ax_spectro.set_position(box)
    return fig_final


def main():
    
    # static
    fig = plot_full_figure()
    size = fig.get_size_inches()
    
    # # movie
    # outdir = r'C:\Users\xavier.mouy\Documents\Publications\Mouy.etal_2022_XAV-Arrays\manuscript\data\mobile_copper-goby'
    # fps=20
    # duration_sec = 10
    
    # # create individual frames
    # times = np.arange(0,duration_sec,1/fps)
    # for idx, t in enumerate(times):
    #     plot_full_figure(time_sec=t)
    #     plt.savefig(os.path.join(outdir,str(idx)+'.jpg'))
    #     plt.close()
    
    # # stack all frames
    # img_array = []
    # for filename in range(0,idx):
    #     img = cv2.imread(os.path.join(outdir,str(filename)+'.jpg'))
    #     height, width, layers = img.shape
    #     size = (width,height)
    #     img_array.append(img)
    # # write videos
    # out = cv2.VideoWriter(os.path.join(outdir,'animation.mp4'),cv2.VideoWriter_fourcc(*'DIVX'), fps, size) 
    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()
    
if __name__ == "__main__":
    main()

