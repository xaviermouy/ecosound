# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:35:40 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
import os
import pickle
import numpy as np

#indir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_in_April_July2019_1342218252'
#outdir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run2\RCA_in_April_July2019_1342218252'
indir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run\RCA_Out_April_July2019_1409343536'
outdir = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\DFO_RCA_run2\RCA_Out_April_July2019_1409343536'


classif_model_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Classification\CV_20201105\RF300_model_20201105.sav'
file_ext = '.nc'


# laod classif model
classif_model = pickle.load(open(classif_model_file, 'rb'))
features = classif_model['features']
model = classif_model['model']
Norm_mean = classif_model['normalization_mean']
Norm_std = classif_model['normalization_std']
classes_encoder = classif_model['classes']
            
# loops thrugh each file
files_list = os.listdir(indir) # list of files
for file in files_list:
    if os.path.isfile(os.path.join(indir, file)) & file.endswith(file_ext):
        if os.path.isfile(os.path.join(outdir,file)) is False:
            # load file
            print(file)
            meas = Measurement()
            meas.from_netcdf(os.path.join(indir,file))
            # reclassify
            data = meas.data
            n1=len(data)
            # drop observations/rows with NaNs
            data = data.replace([np.inf, -np.inf], np.nan)
            data.dropna(subset=features, axis=0, how='any', thresh=None, inplace=True)
            n2=len(data)
            print('Deleted observations (due to NaNs): ' + str(n1-n2))
            # Classification - predictions
            X = data[features]
            X = (X-Norm_mean)/Norm_std
            pred_class = model.predict(X)
            pred_prob = model.predict_proba(X)
            pred_prob = pred_prob[range(0,len(pred_class)),pred_class]
            # Relabel
            for index, row in classes_encoder.iterrows():
                pred_class = [row['label'] if i==row['ID'] else i for i in pred_class]
            # update measurements
            data['label_class'] = pred_class
            data['confidence'] = pred_prob
            # sort detections by ascending start date/time
            data.sort_values('time_min_offset',axis=0,ascending=True,inplace=True)
            # save result as NetCDF file
            print('Saving')
            meas.data = data
            meas.to_netcdf(os.path.join(outdir,file))
