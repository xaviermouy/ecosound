# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:48 2022

@author: xavier.mouy
"""

# from ecosound.core.annotation import Annotation
# from ecosound.core.spectrogram import Spectrogram
from ecosound.core.measurement import Measurement

# from ecosound.core.audiotools import Sound
# from ecosound.visualization.grapher_builder import GrapherFactory


dataset_file_path = r"D:\NOAA\2022_BC_fish_detector\manual_annotations\Master_annotations_dataset_20221028_without_06-MILL-FS.nc"
out_dir = r"D:\NOAA\2022_BC_fish_detector\spectrograms\SNR_test2"


# Load dataset
dataset = Measurement()
dataset.from_netcdf(dataset_file_path)
# dataset.filter("deployment_ID=='SI-RCAOut-20181015'", inplace=True)
# dataset.filter("deployment_ID=='SI-RCAOut-20181015'", inplace=True)
dataset.filter("label_class=='FS'", inplace=True)
# dataset.data = dataset.data[0:10]

# dataset.update_audio_dir(
#     r"D:\NOAA\2022_BC_fish_detector\manual_annotations", verbose=True
# )

dataset.export_spectrograms(
    out_dir,
    time_buffer_sec=0.2,
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
    deployment_subfolders=False,
    date_subfolders=False,
    # file_name_field="time_min_offset",
    # file_name_field="uuid",
    file_name_field="uuid",
    file_prefix_field="snr",
    channel=0,
    # colormap="viridis",
    colormap="Greys",
    save_wav=True,
)
