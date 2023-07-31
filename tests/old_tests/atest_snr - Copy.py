# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:55:51 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement
from ecosound.measurements.measurer_builder import MeasurerFactory
from ecosound.core.audiotools import Sound
import os
import numpy as np

annot_file = r"D:\NOAA\2022_BC_fish_detector\manual_annotations\Master_annotations_dataset_20221028_without_06-MILL-FS.nc"
annot_file2 = r"D:\NOAA\2022_BC_fish_detector\manual_annotations\Master_annotations_dataset_20221028_without_06-MILL-FS_withSNR.nc"
noise_win_sec = "auto"  # 0.25

# load annotations
dataset = Annotation()
dataset.from_netcdf(annot_file)

# filter
# dataset.filter('label_class=="FS"', inplace=True)
# dataset.data = dataset.data.iloc[:100]
# dataset.update_audio_dir(r"D:\NOAA\2022_BC_fish_detector\manual_annotations")
dataset.data.drop([0,1,2])


# Meausrement
snr_measurer = MeasurerFactory("SNR", noise_win_sec=noise_win_sec)
measurements_snr = snr_measurer.compute(dataset, verbose=True)
measurements_snr.to_netcdf(annot_file2)
print("done")
