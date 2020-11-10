# -*- coding: utf-8 -*-
"""
This routine take takes false fish detections from the detector and turns them
as noise annotations for retraining the classifier

Created on Thu Nov  5 09:32:38 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
import pandas as pd
import os

xls_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_manual_analysis\Noise_files_RCA_Out_Oct17_Dec3_2018_67391492.xlsx'
in_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_Out_Oct17_Dec3_2018_67391492'
out_dir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Noise_dataset\Noise_files_RCA_Out_Oct17_Dec3_2018_67391492'
fish_label = 'FS'
min_threshold = 0.7
noise_label = 'NN'

# load names of file and start/stop times where false alarms have been manually
# identified
df = pd.read_excel(xls_file,header=None)

for idx in range(0,len(df)):
    # file name to load
    wav_file_name = df[0][idx]
    tmin_sec = df[1][idx]
    tmax_sec = df[2][idx]
    print(wav_file_name, tmin_sec, tmax_sec)
    detec_file_path = os.path.join(in_dir, wav_file_name + '.nc')
    # load detection/measurement file
    meas = Measurement()
    meas.from_netcdf(detec_file_path)
    data_df = meas.data
    # Only keep fish detections above the given confidence threshold and times
    data_df_filt = data_df[(data_df.label_class==fish_label)
                           & (data_df.confidence>=min_threshold)
                           & (data_df.time_min_offset>=tmin_sec)
                           & (data_df.time_max_offset<=tmax_sec)
                           ]
    data_df_filt.reset_index(inplace=True, drop=True)
    meas.data = data_df_filt
    # Change fish labels to noise labels
    meas.insert_values(label_class=noise_label)
    # Save to new nc file
    meas.to_netcdf(os.path.join(out_dir,wav_file_name + str(idx)))

print('done')