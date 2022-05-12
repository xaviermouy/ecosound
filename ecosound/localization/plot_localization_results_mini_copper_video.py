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
        # plot frame
        frame_color = 'whitesmoke'
        frame_alpha = 1
        frame_width = 3        
        # plot frame
        # plot fishcam 02

        ax.plot([-0.3,0.4],[-0.1,-0.1],
              linewidth=frame_width ,
              alpha=frame_alpha,
              linestyle= 'solid',
              color=frame_color,
              )

        rectangle3 = plt.Rectangle((-0.3, -0.7), 0.7, 0.9,linewidth=frame_width,ec=frame_color,alpha=frame_alpha, facecolor='white')        
        ax.add_patch(rectangle3)
        
        rectangle4 = plt.Rectangle((-0.04, -0.4), 0.22, 0.5,linewidth=frame_width,ec=frame_color,alpha=frame_alpha, facecolor=frame_color)
        ax.add_patch(rectangle4)
        
        ax.plot([-0.5,0.63],[0, 0],
              linewidth=1 ,
              alpha=frame_alpha,
              linestyle= 'solid',
              color='dimgray',
              )
        ax.plot([0.13, 0.13],[0,0.57],
              linewidth=1 ,
              alpha=frame_alpha,
              linestyle= 'solid',
              color='dimgray',
              )      


        
        
        # ax.plot([0.93, 1.5],[-0.76, -0.2],
        #      linewidth=frame_width ,
        #      alpha=frame_alpha,
        #      linestyle= 'solid',
        #      color=frame_color,
        #      )
        # ax.plot([1.5, 2],[-0.1, -1.06],
        #      linewidth=frame_width ,
        #      alpha=frame_alpha,
        #      linestyle= 'solid',
        #      color=frame_color,
        #      )
        

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
        ax.plot([loc_point['x_min_CI99'],loc_point['x_max_CI99']],
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
                [loc_point['y_min_CI99'],loc_point['y_max_CI99']],
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


    rectangle0 = plt.Rectangle((-0.3,-0.7), 0.7, 0.58,
                          linewidth=frame_width,                          
                          ec=frame_color,
                          alpha=frame_alpha,
                          facecolor='white')
    ax.add_patch(rectangle0)

    circ = plt.Circle((0.05,-0.11), 0.11,
                      linewidth=frame_width,                          
                      ec=frame_color,
                      alpha=frame_alpha,
                      facecolor=frame_color)
    ax.add_patch(circ)
    
    # rectangle = plt.Rectangle((-0.04,-0.22), 0.22, 0.22,
    #                       linewidth=frame_width,
    #                       ec=frame_color,
    #                       alpha=frame_alpha,
    #                       facecolor=frame_color)
    

        
    # ax.add_patch(rectangle)
        
    ax.plot([-0.5, 0.63],[0,0],
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

    ax.plot([0.13, 0.13],[0,0.14],
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
        ax.plot([loc_point['x_min_CI99'],loc_point['x_max_CI99']],
                [loc_point['z'],loc_point['z']],
                linewidth=params['uncertainty_width'].values[0],
                linestyle=params['uncertainty_style'].values[0],
                #color=params['uncertainty_color'].values[0],
                color=cmap(norm(loc_point['time_min_offset'])),
                alpha=params['uncertainty_alpha'].values[0],
                )
    
        ax.plot([loc_point['x'],loc_point['x']],
                [loc_point['z_min_CI99'],loc_point['z_max_CI99']],
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
        
    loc_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mini-array_copper\localizations_2cm_3m_v5.nc'
    loc_file_matlab = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mini-array_copper\localizations_matlab_with_CI.csv'
    audio_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\671404070.190801232502.wav'
    video_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\large_array\2019-09-15_HornbyIsland_AMAR_07-HI\3420_FishCam01_20190920T163627.613206Z_1600x1200_awb-auto_exp-night_fr-10_q-20_sh-0_b-50_c-0_i-400_sat-0.mp4'
    hp_config_file = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\data\mini_array\hydrophones_config_05-MILL_corrected.csv'
    localization_config_file =r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\config_files\localization_config_mini_array.yaml'    
    t1_sec = 696+15 #696
    #t1_sec = 696+14 #696
    t2_sec = 713

    
    filter_x=[-10, 10]
    filter_y=[-10, 10]
    filter_z=[-2, 10]
    filter_x_std=10
    filter_y_std=10
    filter_z_std=10
    
    params=pd.DataFrame({
        'loc_color': ['black'],
        'loc_marker': ['o'],
        'loc_alpha': [1],
        'loc_size': [5],
        'uncertainty_color': ['black'],
        'uncertainty_style': ['-'],
        'uncertainty_alpha': [1], #0.7
        'uncertainty_width': [0.2], #0.2
        'x_min':[-0.7],
        'x_max':[0.7],
        'y_min':[-0.8],
        'y_max':[0.8],
        'z_min':[-0.8],
        'z_max':[0.8],    
        })
        
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
    
    # used matlab CI
    loc_data = pd.read_csv(loc_file_matlab)
    
    # ## recalculate data errors
    # diff=[]
    # idx = 0
    # for idx in range(len(loc_data)):
    #     m = loc_data.loc[[idx],['x','y','z']]        
    #     tdoa_m = predict_tdoa(m, sound_speed_mps, hydrophones_config, hydrophone_pairs)
    #     tdoa_measured = loc_data.loc[[idx],['tdoa_sec_1','tdoa_sec_2','tdoa_sec_3']].to_numpy()    
    #     #diff_temp = (tdoa_m-tdoa_measured.T)**2
    #     if idx==0:
    #         diff = (tdoa_m-tdoa_measured.T)**2
    #     else:
    #         diff = np.vstack((diff,(tdoa_m-tdoa_measured.T)**2))
    
    # Q = len(loc_data)
    # #M = m.size # number of dimensions of the model (here: X, Y, and Z)
    # #N = len(tdoa_sec) # number of measurements    
    # #error_std = np.sqrt((1/(Q*(N-M))) * (sum((tdoa_sec-tdoa_m)**2)))    
    # tdoa_errors_std = np.sqrt( (1/Q)*(sum(diff)))
    
    # #tdoa_errors_std = calc_data_error(tdoa_sec, m, sound_speed_mps,hydrophones_config, hydrophone_pairs)
    # for idx in range(len(loc_data)):
    #     loc_errors_std = calc_loc_errors(tdoa_errors_std, loc_data.loc[[idx],['x','y','z']] , sound_speed_mps, hydrophones_config, hydrophone_pairs)
    #     print('m')
    
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
                            (loc_data['z_std']<= filter_z_std) &
                            (loc_data['time_min_offset']>= t1_sec) &
                            (loc_data['time_max_offset']<= t2_sec)
                            #(loc_data['frequency_max']>= 60)
                            ]
    # Adjust detection times
    loc_data['time_min_offset'] = loc_data['time_min_offset'] - t1_sec
    loc_data['time_max_offset'] = loc_data['time_max_offset'] - t1_sec
    
    if time_sec!= None:
        loc_data = loc_data.loc[(loc_data['time_max_offset']<=time_sec)] 
    else:
        print('Static')
    
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
    
    
    # # plot video frame 1
    # fig_video1, ax_video1 = plt.subplots(1,1)
    # frame1_sec = 152.8 # second detection -> 16:38:59.8
    # #ax_video1 = fig_final.add_subplot(3,3,5)
    # plot_video_frame(video_file,frame1_sec, ax_video1)
    # ax_video1.get_xaxis().set_visible(False)
    # ax_video1.get_yaxis().set_visible(False)
    
    # # plot video frame 2
    # fig_video2, ax_video2 = plt.subplots(1,1)
    # frame2_sec = 160 # 4th detection -> 16:39:07
    # #ax_video2 = fig_final.add_subplot(3,3,6)
    # plot_video_frame(video_file,frame2_sec, ax_video2)
    # ax_video2.get_xaxis().set_visible(False)
    # ax_video2.get_yaxis().set_visible(False)
    
    
    fig_final.set_size_inches(9.47, 6.72)

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
    # outdir = r'C:\Users\xavier.mouy\Documents\PhD\Thesis\phd-thesis\Figures\XAV_arrays\MobileArray_Copper2\animation'
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

