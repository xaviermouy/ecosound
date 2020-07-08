# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 10:39:23 2020

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

annotation_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\Master_annotations_dataset.nc"
detec_file = r'C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\dataset_annotations_only.nc'

# # load annotations
# print('-----------------')
# print('  Annotations    ')
# annot = Annotation()
# annot.from_netcdf(annotation_file)
# print(annot.summary())
# annot_perfile = annot.summary(rows='audio_file_name',columns='label_class')
# annot_perfile.rename(columns={"FS": "FS-annot"}, inplace=True)
# annot_perfile = annot_perfile['FS-annot'].to_frame()
# #annot_perfile.to_csv('annot.csv')

print(' ')
print('-----------------')
print('  Detections     ')
# load detections
detec = Measurement()
detec.from_netcdf(detec_file)
print(detec.summary())
detec_perfile = detec.summary(rows='audio_file_name',columns='label_class')
detec_perfile.rename(columns={"FS": "FS-detec"}, inplace=True)
detec_perfile = detec_perfile['FS-detec'].to_frame()


dd= pd.concat([annot_perfile,detec_perfile], axis=1)
dd['diff'] = dd['FS-annot'] - dd['FS-detec']
dd.plot()


# outdir=r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\ecosound\tests\detec_export'
# detec.to_pamlab(outdir, single_file=False)


# outdir=r'C:\Users\xavier.mouy\Documents\Workspace\GitHub\ecosound\tests\annot_export'
# annot.to_pamlab(outdir, single_file=False)

