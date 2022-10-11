# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:00:26 2021

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
import xarray


indir = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\10-XAVarray_2020\results\mobile_array_copper'
#outfile = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Fish-arrays_stationary\HI07_localizations\restricted_localizations.csv'
filter_x=[-1000, 1000]
filter_y=[-1000, 1000]
filter_z=[-1000, 1000]
filter_x_std=50
filter_y_std=50
filter_z_std=50

# indir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Fish-arrays_stationary\HI07_localizations\07-HI'
# outfile = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Fish-arrays_stationary\HI07_localizations\restricted_localizations.csv'
# filter_x=[-1.5, 1.5]
# filter_y=[-1.5, 1.5]
# filter_z=[-2, 2]
# filter_x_std=0.3
# filter_y_std=0.3
# filter_z_std=0.3

# load data
print('')
print('Loading dataset')
idx = 0
for infile in os.listdir(indir):
    if infile.endswith(".nc"):
        print(infile)
        
        locs = Measurement()
        locs.from_netcdf(os.path.join(indir,infile))
        loc_data = locs.data
        
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
        
        if idx == 0:
            final_df = loc_data
        else:
            final_df = final_df.append(loc_data,ignore_index=True)
        print('s')
        idx +=1
#final_df.to_csv(outfile)