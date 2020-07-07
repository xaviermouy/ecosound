# -*- coding: utf-8 -*-
"""
Created on Sat May 30 10:19:38 2020

@author: xavier.mouy
"""


import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
import time
import pandas as pd


## Input paraneters ##########################################################

annotation_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\Master_annotations_dataset.nc"
detection_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset_with_metadata2"
outfile=r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_annotations_only.nc'

# load annotations
annot = Annotation()
annot.from_netcdf(annotation_file)

# load detections
detec = Measurement()
detec.from_netcdf(detection_file)
print(detec)

freq_ovp = True # default True
dur_factor_max = None # default None
dur_factor_min = 0.1 # default None
ovlp_ratio_min = 0.3 # defaulkt None
remove_duplicates = True # dfault - False
inherit_metadata = True # default False
filter_deploymentID = False # default True

detec.filter_overlap_with(annot,
                          freq_ovp=freq_ovp,
                          dur_factor_max=dur_factor_max,
                          dur_factor_min=dur_factor_min,
                          ovlp_ratio_min=ovlp_ratio_min,
                          remove_duplicates=remove_duplicates,
                          inherit_metadata=inherit_metadata,
                          filter_deploymentID=filter_deploymentID,
                          inplace=True
                          )
print(detec)
detec.to_netcdf(outfile)