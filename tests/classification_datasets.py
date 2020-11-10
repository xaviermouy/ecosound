# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:24:56 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation
import pandas as pd

"""
Gathers measuremenst for all annotations and noise. Merges into a single dataset,
and re-label classes to create a 2-class dataset 'FS' vs 'NN'.

"""

# Define input and output files
annot_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_annotations_only.nc'
noise_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Noise_dataset'
outfile=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_FS-NN_modified_20201105145300.nc'

# Load measurements
meas_annot = Measurement()
meas_annot.from_netcdf(annot_file)
meas_noise = Measurement()
meas_noise.from_netcdf(noise_file)

## Label noise measurement as 'NN'
meas_noise.insert_values(label_class='NN')
print(meas_noise.summary())

## relabel annotations that are not 'FS' as 'NN'
print(meas_annot.summary())
meas_annot.data['label_class'].replace(to_replace=['', 'ANT','HS','KW','UN'], value='NN', inplace=True)
print(meas_annot.summary())

## merge the 2 datasets
meas_NN_FS = meas_noise + meas_annot
print(meas_NN_FS.summary())

## Save dataset to nc file
meas_NN_FS.to_netcdf(outfile)

