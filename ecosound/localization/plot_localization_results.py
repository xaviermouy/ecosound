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
    
    
    graph_spectros.colormap = 'binary' #'jet'
    fig, ax = graph_spectros.show()

    if ax.get_geometry() != geometry :
        ax.change_geometry(*geometry)        
    return fig, ax

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
    
    rectangle = plt.Rectangle((-0.94,-0.94), 1.88, 1.9,linewidth=frame_width,ec=frame_color,alpha=frame_alpha, facecolor='none')
    ax.add_patch(rectangle)
    
    #c=z, cmap=cmap, norm=norm
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
        ax.plot([loc_point['x']-loc_point['x_std'],loc_point['x']+loc_point['x_std']],
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
                [loc_point['y']-loc_point['y_std'],loc_point['y']+loc_point['y_std']],
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
    
    rectangle1 = plt.Rectangle((-0.94,-0.76), 1.87, 1.5,linewidth=frame_width,ec=frame_color,alpha=frame_alpha, facecolor='none')
    rectangle2 = plt.Rectangle((-0.94, 0.74), 1.87, 1,linewidth=frame_width,ec=frame_color,alpha=frame_alpha, facecolor='none')
    rectangle3 = plt.Rectangle((-0.1, 1.55), 0.2, 0.4,linewidth=frame_width,ec=frame_color,alpha=frame_alpha, facecolor=frame_color)
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    ax.add_patch(rectangle3)
    ax.plot([-0.94, -0.94],[-0.76, -1.06],
         linewidth=frame_width ,
         alpha=frame_alpha,
         linestyle= 'solid',
         color=frame_color,
         )
    ax.plot([0.93, 0.93],[-0.76, -1.06],
         linewidth=frame_width ,
         alpha=frame_alpha,
         linestyle= 'solid',
         color=frame_color,
         )
    ax.plot([0.028, 0.028],[0, -0.76],
         linewidth=frame_width ,
         alpha=frame_alpha,
         linestyle= 'solid',
         color=frame_color,
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
        ax.plot([loc_point['x']-loc_point['x_std'],loc_point['x']+loc_point['x_std']],
                [loc_point['z'],loc_point['z']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                )
    
        ax.plot([loc_point['x'],loc_point['x']],
                [loc_point['z']-loc_point['z_std'],loc_point['z']+loc_point['z_std']],
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
## ###########################################################################

loc_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\large-array_quillback\AMAR173.4.20190920T161248Z.nc'
audio_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\AMAR173.1.20190920T161248Z.wav'
video_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\3420_FishCam01_20190920T163627.613206Z_1600x1200_awb-auto_exp-night_fr-10_q-20_sh-0_b-50_c-0_i-400_sat-0.mp4'
hp_config_file =  r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\hydrophones_config_07-HI.csv'
t1_sec = 1570
t2_sec = 1587#1590

filter_x=[-1.5, 1.5]
filter_y=[-1.5, 1.5]
filter_z=[-1.5, 1.5]
filter_x_std=0.5
filter_y_std=0.5
filter_z_std=0.5

params=pd.DataFrame({
    'loc_color': ['black'],
    'loc_marker': ['o'],
    'loc_alpha': [1],
    'loc_size': [5],
    'uncertainty_color': ['black'],
    'uncertainty_style': ['-'],
    'uncertainty_alpha': [1], #0.7
    'uncertainty_width': [0.2], #0.2
    'x_min':[-1.26],
    'x_max':[1.26],
    'y_min':[-1.26],
    'y_max':[1.26],
    'z_min':[-1.5],
    'z_max':[2.1],    
    })
    
## ###########################################################################

## load localization results
loc = Measurement()
loc.from_netcdf(loc_file)
loc_data = loc.data

## load hydrophone locations
hydrophones_config = pd.read_csv(hp_config_file)

# Filter
loc_data = loc_data.dropna(subset=['x', 'y','z']) # remove NaN
loc_data = loc_data.loc[(loc_data['x']>=min(filter_x)) & 
                        (loc_data['x']<=max(filter_x)) &
                        (loc_data['y']>=min(filter_y)) & 
                        (loc_data['y']<=max(filter_y)) &
                        (loc_data['z']>=min(filter_z)) & 
                        (loc_data['z']<=max(filter_z)) &
                        (loc_data['x_std']<= filter_x_std) & 
                        (loc_data['y_std']<= filter_y_std) &
                        (loc_data['z_std']<= filter_z_std)
                        ]
# Adjust detection times
loc_data['time_min_offset'] = loc_data['time_min_offset'] - t1_sec
loc_data['time_max_offset'] = loc_data['time_max_offset'] - t1_sec

# update loc object
loc.data = loc_data

# plots
# fig, ax = plt.subplots(figsize=(6, 1))
# fig.subplots_adjust(bottom=0.5)
# n_colors = t2_sec-t1_sec
# cmap = mpl.cm.get_cmap('CMRmap', n_colors*2)
# norm = mpl.colors.Normalize(vmin=0, vmax=n_colors)
# ax_cmap = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
#                                 norm=norm,
#                                 orientation='horizontal')
# ax_cmap.set_label('Time (s)')

# Plot spectrogram
fig_final, ax_spectro = plot_spectrogram(audio_file,loc,t1_sec, t2_sec, geometry=(5,1,1))    
ax_spectro.set_title("")
ax_spectro.get_xaxis().set_visible(False)
n_colors = t2_sec-t1_sec
cmap = mpl.cm.get_cmap('viridis', n_colors*4)
norm = mpl.colors.Normalize(vmin=0, vmax=n_colors)
divider = make_axes_locatable(ax_spectro)
cax = divider.append_axes('bottom', 0.1, pad=0.03 )
ax_cmap = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
ax_cmap.set_label('Time (s)')



# plot detection points on top of spectrogram
#gs0 = fig_final.add_gridspec(60,1)
ax_detec = fig_final.add_subplot(20,1,1)
det_y = np.asarray(np.ones((1,len(loc_data['time_min_offset']))))[0]
det_x = np.asarray(loc_data['time_min_offset'])
ax_detec.scatter(det_x,det_y, c=loc_data['time_min_offset'],cmap=cmap,s=12)
ax_detec.set_xlim(ax_spectro.get_xlim())
ax_detec.get_xaxis().set_visible(False)
ax_detec.get_yaxis().set_visible(False)
ax_detec.axis('off')

# #pos =[left, bottom, width, height]
# box = ax_detec.get_position()
# box.y0 = box.y0 + 0.6
# box.y1 = box.y1 + 0.6
# ax_detec.set_position(box)

#size = fig_final.get_size_inches()


plt.subplots_adjust(left=0.08, bottom=0.1, right=0.95, top=0.95, wspace=0, hspace=0)


# divider2 = make_axes_locatable(ax_spectro)
# cax2 = divider2.append_axes('top', size=0.2, pad=10.0)
# det_y = np.asarray(np.ones((1,len(loc_data['time_min_offset']))))[0]
# det_x = np.asarray(loc_data['time_min_offset'])
# cax2.plot(det_x,det_y,'.r')
# cax2.set_xlim(ax_spectro.get_xlim())


# ax_cmap = mpl.colorbar.ColorbarBase(cax, cmap=cmap,
#                                     norm=norm,
#                                     orientation='horizontal')



gs = fig_final.add_gridspec(3,2)

# plot localization top
ax_toploc = fig_final.add_subplot(gs[1:,0])
plot_top_view(hydrophones_config,loc_data,params,cmap,norm, ax_toploc)
ax_toploc.set_anchor('W')

# plot localization side
#ax_sideloc = fig_final.add_subplot(3,3,7,sharex = ax_toploc)
ax_sideloc = fig_final.add_subplot(gs[1:,1])
plot_side_view(hydrophones_config,loc_data,params,cmap,norm,ax_sideloc)
ax_sideloc.set_anchor('E')

# set the spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)


# plot video frame 1
fig_video1, ax_video1 = plt.subplots(1,1)
frame1_sec = 152.8 # second detection -> 16:38:59.8
#ax_video1 = fig_final.add_subplot(3,3,5)
plot_video_frame(video_file,frame1_sec, ax_video1)
ax_video1.get_xaxis().set_visible(False)
ax_video1.get_yaxis().set_visible(False)

# plot video frame 2
fig_video2, ax_video2 = plt.subplots(1,1)
frame2_sec = 160 # 4th detection -> 16:39:07
#ax_video2 = fig_final.add_subplot(3,3,6)
plot_video_frame(video_file,frame2_sec, ax_video2)
ax_video2.get_xaxis().set_visible(False)
ax_video2.get_yaxis().set_visible(False)


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
