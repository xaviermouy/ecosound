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

annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations.nc"
out_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\Annotations_dataset_UK-SAMS-WestScotland-202011-N1 annotations_withSNR.nc"
spectro_dir = r'D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20201102\spectrograms'
#annot_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20210522\Annotations_dataset_UK-SAMS-WestScotland-202105-N1 annotations.nc"
#out_file = r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\continuous_datasets\UK-SAMS-N1-20210522\Annotations_dataset_UK-SAMS-WestScotland-202105-N1 annotations_withSNR.nc"

noise_win_sec = 5 #"auto"  # 0.25

# load annotations
dataset = Annotation()
dataset.from_netcdf(annot_file)

# dataset.filter('label_class=="FS"', inplace=True)
# dataset.data = dataset.data.iloc[:100]
# dataset.update_audio_dir(r"D:\NOAA\2022_BC_fish_detector\manual_annotations")
# dataset.data.drop([0,1,2],inplace=True)

# Meausrement
snr_measurer = MeasurerFactory("SNR", noise_win_sec=noise_win_sec)
measurements_snr = snr_measurer.compute(dataset, verbose=True, debug=False)

measurements_snr.export_spectrograms(spectro_dir,
                                     sanpling_rate_hz=2000,
                                     file_prefix_field='snr',
                                     )

measurements_snr.to_netcdf(out_file)

print("done")
