# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:48 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.measurement import Measurement
from ecosound.core.audiotools import Sound
from ecosound.visualization.grapher_builder import GrapherFactory


dataset_file_path = r"C:\Users\xavier.mouy\Documents\GitHub\fish_detector_bc\Master_annotations_dataset_20221028_without_06-MILL-FS_withSNR.nc"
out_dir = r"D:\Detector\spectrograms\Master_annotations_dataset_20221025_SNR"


# Load dataset
dataset = Measurement()
dataset.from_netcdf(dataset_file_path)
# dataset.filter("deployment_ID=='SI-RCAOut-20181015'", inplace=True)
# dataset.filter("deployment_ID=='SI-RCAOut-20181015'", inplace=True)
dataset.filter("label_class=='FS'", inplace=True)
# dataset.data = dataset.data[0:10]


dataset.export_spectrograms(
    out_dir,
    time_buffer_sec=0.5,
    spectro_unit="sec",
    spetro_nfft=0.064,
    spetro_frame=0.064,
    spetro_inc=0.00125,
    freq_min_hz=None,
    freq_max_hz=None,
    sanpling_rate_hz=4000,
    filter_order=8,
    filter_type="iir",
    fig_size=(15, 10),
    deployment_subfolders=True,
    file_prefix_field="snr",
    channel=0,
    colormap="viridis",
)
