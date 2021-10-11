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

def plot_spectrogram(audio_file,loc,t1_sec, t2_sec, geometry=(1,1,1)):
    
    fmin=0
    fmax=1000
    frame= 0.0625
    window_type= 'hann'
    nfft=0.0853
    step=0.01
    channel=0
    chunk=[t1_sec,t2_sec]
    
    graph_spectros = GrapherFactory('SoundPlotter', title='Spectrograms', frequency_max=fmax)
    sound = Sound(audio_file)
    sound.read(channel=channel, chunk=chunk, unit='sec', detrend=True)
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
    return fig, ax

def plot_top_view(hydrophones_config,loc_data,params, ax, color='black',frame_on=True):
    
    #fig1 = plt.figure()
    #ax = fig1.add_subplot(111)
    #colors = matplotlib.cm.tab10(hydrophones_config.index.values)
    
    if frame_on:
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
        
        rectangle = plt.Rectangle((-0.1,-0.3), 0.2, 0.4,
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
        
    #c=z, cmap=cmap, norm=norm
    # plot localizations
    ax.scatter(loc_data['x'], loc_data['y'],#c=loc_data['time_min_offset'],
                        s=params['loc_size'].values[0],
                        marker=params['loc_marker'].values[0],
                        color=color,
                        alpha=params['loc_alpha'].values[0],
                        #cmap=cmap,
                        #norm=norm,
                        zorder=5,
                        )
    # plot uncertainties
    for idx, loc_point in loc_data.iterrows():   
        ax.plot([loc_point['x_min_CI99'],loc_point['x_max_CI99']],
                [loc_point['y'],loc_point['y']],
                #c=loc_point['time_min_offset'],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                color=color,
                #color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                #cmap=cmap,
                #norm=norm,
                )
    
        ax.plot([loc_point['x'],loc_point['x']],
                [loc_point['y_min_CI99'],loc_point['y_max_CI99']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                color=color,
                #color=cmap(norm(loc_point['time_min_offset'])),
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


def plot_side_view(hydrophones_config,loc_data,params,ax, color='black',frame_on=True):
    
    #fig1 = plt.figure()
    #ax = fig1.add_subplot(111)
    #colors = matplotlib.cm.tab10(hydrophones_config.index.values)
    
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
    ax.scatter(loc_data['x'], loc_data['z'],#c=loc_data['time_min_offset'],
                        s=params['loc_size'].values[0],
                        marker=params['loc_marker'].values[0],
                        color=color,
                        alpha=params['loc_alpha'].values[0],
                        #cmap=cmap,
                        #norm=norm,
                        zorder=5
                        )
    # plot uncertainties
    for idx, loc_point in loc_data.iterrows():   
        ax.plot([loc_point['x_min_CI99'],loc_point['x_max_CI99']],
                [loc_point['z'],loc_point['z']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                color=color,
                #color=params['uncertainty_color'].values[0],
                #color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                )
    
        ax.plot([loc_point['x'],loc_point['x']],
                [loc_point['z_min_CI99'],loc_point['z_max_CI99']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                #color=cmap(norm(loc_point['time_min_offset'])),
                color=color,
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
## ###########################################################################


hp_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2019-09-14_HornbyIsland_Trident\hydrophones_config_HI-201909.csv'
indir=r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mobile_array_ROV'
audio_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mobile_array\2020-09-10_Localization_experiment_projector\5147.200910210736.wav'
t1_sec = 108#106
t2_sec = 113.5#119


# loc_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\large-array_quillback\AMAR173.4.20190920T161248Z.nc'
# audio_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.1.20190920T161248Z.wav'
# video_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\3420_FishCam01_20190920T163627.613206Z_1600x1200_awb-auto_exp-night_fr-10_q-20_sh-0_b-50_c-0_i-400_sat-0.mp4'
# hp_config_file =  r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
# t1_sec = 1570
# t2_sec = 1587#1590

filter_x=[-5, 5]
filter_y=[-5, 5]
filter_z=[-2, 5]
filter_x_std=5
filter_y_std=5
filter_z_std=5

params=pd.DataFrame({
    'loc_color': ['black'],
    'loc_marker': ['o'],
    'loc_alpha': [1],
    'loc_size': [5],
    'uncertainty_color': ['black'],
    'uncertainty_style': ['-'],
    'uncertainty_alpha': [1], #0.7
    'uncertainty_width': [0.2], #0.2
    'x_min':[-3],
    'x_max':[3],
    'y_min':[-3],
    'y_max':[3],
    'z_min':[-3],
    'z_max':[3],    
    })
    
## ###########################################################################

## load localization results 0 degrees
file1 = '0_deg.nc'
loc1 = Measurement()
loc1.from_netcdf(os.path.join(indir,file1))
# loc1_data = loc1.data
# # Filter
# loc1_data = loc1_data.dropna(subset=['x', 'y','z']) # remove NaN
# loc1_data = loc1_data.loc[(loc1_data['x']>=min(filter_x)) & 
#                         (loc1_data['x']<=max(filter_x)) &
#                         (loc1_data['y']>=min(filter_y)) & 
#                         (loc1_data['y']<=max(filter_y)) &
#                         (loc1_data['z']>=min(filter_z)) & 
#                         (loc1_data['z']<=max(filter_z)) &
#                         (loc1_data['x_std']<= filter_x_std) & 
#                         (loc1_data['y_std']<= filter_y_std) &
#                         (loc1_data['z_std']<= filter_z_std)
#                         ]

# # Adjust detection times
# loc1_data['time_min_offset'] = loc1_data['time_min_offset'] - t1_sec
# loc1_data['time_max_offset'] = loc1_data['time_max_offset'] - t1_sec

# # update loc object
# loc1.data = loc1_data

# ## load localization results 90 degrees
# file2 = '90_degrees.nc'
# loc2 = Measurement()
# loc2.from_netcdf(os.path.join(indir,file2))
# loc2_data = loc2.data
# # Filter
# loc2_data = loc2_data.dropna(subset=['x', 'y','z']) # remove NaN
# loc2_data = loc2_data.loc[(loc2_data['x']>=min(filter_x)) & 
#                         (loc2_data['x']<=max(filter_x)) &
#                         (loc2_data['y']>=min(filter_y)) & 
#                         (loc2_data['y']<=max(filter_y)) &
#                         (loc2_data['z']>=min(filter_z)) & 
#                         (loc2_data['z']<=max(filter_z)) &
#                         (loc2_data['x_std']<= filter_x_std) & 
#                         (loc2_data['y_std']<= filter_y_std) &
#                         (loc2_data['z_std']<= filter_z_std)
#                         ]
# # update loc object
# loc2.data = loc2_data

# ## load localization results 90 degrees
# file3 = '180_degrees.nc'
# loc3 = Measurement()
# loc3.from_netcdf(os.path.join(indir,file3))
# loc3_data = loc3.data
# # Filter
# loc3_data = loc3_data.dropna(subset=['x', 'y','z']) # remove NaN
# loc3_data = loc3_data.loc[(loc3_data['x']>=min(filter_x)) & 
#                         (loc3_data['x']<=max(filter_x)) &
#                         (loc3_data['y']>=min(filter_y)) & 
#                         (loc3_data['y']<=max(filter_y)) &
#                         (loc3_data['z']>=min(filter_z)) & 
#                         (loc3_data['z']<=max(filter_z)) &
#                         (loc3_data['x_std']<= filter_x_std) & 
#                         (loc3_data['y_std']<= filter_y_std) &
#                         (loc3_data['z_std']<= filter_z_std)
#                         ]
# # update loc object
# loc3.data = loc3_data

# ## load localization results -90 degrees
# file4 = 'minus90_degrees.nc'
# loc4 = Measurement()
# loc4.from_netcdf(os.path.join(indir,file4))
# loc4_data = loc4.data
# # Filter
# loc4_data = loc4_data.dropna(subset=['x', 'y','z']) # remove NaN
# loc4_data = loc4_data.loc[(loc4_data['x']>=min(filter_x)) & 
#                         (loc4_data['x']<=max(filter_x)) &
#                         (loc4_data['y']>=min(filter_y)) & 
#                         (loc4_data['y']<=max(filter_y)) &
#                         (loc4_data['z']>=min(filter_z)) & 
#                         (loc4_data['z']<=max(filter_z)) &
#                         (loc4_data['x_std']<= filter_x_std) & 
#                         (loc4_data['y_std']<= filter_y_std) &
#                         (loc4_data['z_std']<= filter_z_std)
#                         ]
# # update loc object
# loc4.data = loc4_data

df = pd.read_csv(r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mobile_array_ROV\localizations_matlab_with_CI.csv')
loc1_data = df.loc[0:7] # 0 degrees
loc3_data = df.loc[8:14]
loc2_data = df.loc[15:19]
loc4_data = df.loc[20:27]

## load hydrophone locations
hydrophones_config = pd.read_csv(hp_config_file)


# Plot spectrogram
fig_final, ax_spectro = plot_spectrogram(audio_file,loc1,t1_sec, t2_sec, geometry=(5,1,1))    
ax_spectro.set_title("")


plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)


gs = fig_final.add_gridspec(3,2)

# plot localization top
ax_toploc = fig_final.add_subplot(gs[1:,1])
plot_top_view(hydrophones_config,loc1_data,params, ax_toploc,color='firebrick',frame_on=True)
plot_top_view(hydrophones_config,loc2_data,params, ax_toploc,color='seagreen',frame_on=False)
plot_top_view(hydrophones_config,loc3_data,params, ax_toploc,color='darkorange',frame_on=False)
plot_top_view(hydrophones_config,loc4_data,params, ax_toploc,color='darkblue',frame_on=False)
ax_toploc.set_anchor('E')

# # plot localization side
ax_sideloc = fig_final.add_subplot(gs[1:,0])
plot_side_view(hydrophones_config,loc1_data,params,ax_sideloc,color='firebrick',frame_on=True)
plot_side_view(hydrophones_config,loc2_data,params, ax_sideloc,color='seagreen',frame_on=False)
plot_side_view(hydrophones_config,loc3_data,params, ax_sideloc,color='darkorange',frame_on=False)
plot_side_view(hydrophones_config,loc4_data,params, ax_sideloc,color='darkblue',frame_on=False)


ax_sideloc.set_anchor('W')

# set the spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)



fig_final.set_size_inches(8.6,6.72)


box = ax_spectro.get_position()
box.y0 = box.y0 - 0.03
box.y1 = box.y1 - 0.03
ax_spectro.set_position(box)









#ax fig.subplots_adjust(bottom=0.5)

# # plot hydrophones
# fig1 = plt.figure()
# ax = fig1.add_subplot(111, projection='3d')
# colors = matplotlib.cm.tab10(hydrophones_config.index.values)

# for index, hp in hydrophones_config.iterrows():
#     point = ax.scatter(hp['x'],hp['y'],hp['z'],
#                     s=20,
#                     #color=colors[index],
#                     color='black',
#                     label=hp['name'],
#                     alpha=0.5,
#                     )


# # plot frame
# ax.plot([-0.95, -0.95],[-0.9, 0.9],[-0.88,-0.88],
#         linewidth=5,
#         alpha=0.2,
#         linestyle= 'solid',
#         color='red',
#         )

# ax.plot([+0.95, +0.95],[-0.9, 0.9],[-0.88,-0.88],
#         linewidth=5,
#         alpha=0.2,
#         linestyle= 'solid',
#         color='red',
#         )

# # plot localizations
# ax.scatter(loc_data['x'], loc_data['y'],loc_data['z'],
#                     s=loc_size,
#                     marker=loc_marker,
#                     color=loc_color,
#                     alpha=loc_alpha,
#                     )
# # plot uncertainties
# for idx, loc_point in loc_data.iterrows():   
#     ax.plot([loc_point['x']-loc_point['x_std'],loc_point['x']+loc_point['x_std']],
#             [loc_point['y'],loc_point['y']],
#             [loc_point['z'],loc_point['z']],             
#             linewidth=uncertainty_width,
#             linestyle=uncertainty_style,
#             color=uncertainty_color,
#             alpha=uncertainty_alpha,
#             )

#     ax.plot([loc_point['x'],loc_point['x']],
#             [loc_point['y']-loc_point['y_std'],loc_point['y']+loc_point['y_std']],
#             [loc_point['z'],loc_point['z']],
#             linewidth=uncertainty_width,
#             linestyle=uncertainty_style,
#             color=uncertainty_color,
#             alpha=uncertainty_alpha,
#             )
    
#     ax.plot([loc_point['x'],loc_point['x']],
#             [loc_point['y'],loc_point['y']],
#             [loc_point['z']-loc_point['z_std'], loc_point['z']+loc_point['z_std']],
#             linewidth=uncertainty_width,
#             linestyle=uncertainty_style,
#             color=uncertainty_color,
#             alpha=uncertainty_alpha,
#             )
# # Axes labels
# ax.set_xlabel('X (m)', labelpad=10)
# ax.set_ylabel('Y (m)', labelpad=10)
# ax.set_zlabel('Z (m)', labelpad=10)

# #ax.view_init(0, 0)
# #ax.view_init(azim=-1, elev=-1)

# # legend
# #ax.legend(bbox_to_anchor=(1.07, 0.7, 0.3, 0.2), loc='upper left')
# plt.tight_layout()
# plt.show()
