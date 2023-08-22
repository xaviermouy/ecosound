# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:48 2022

@author: xavier.mouy
"""

# from ecosound.core.measurement import Measurement
from ecosound.core.annotation import Annotation

import datetime
import os

in_dir = (
    r"D:\NOAA\2022_Minke_whale_detector\manual_annotations\discrete_datasets"
)
file = "Annotations_dataset_NN-MW-HB-HK_20221213T142558.nc"
aggregate_time_offset = 0

# Load dataset
print("Loading detections...")
dataset = Annotation()
# dataset.from_sqlite(os.path.join(in_dir, file))
dataset.from_netcdf(os.path.join(in_dir, file))


# Filter
print("Filtering detections...")
dataset.filter("label_class=='MW'", inplace=True)
# dataset.filter("label_class!='HK'", inplace=True)
# dataset.filter("label_class=='NN'", inplace=True)
# dataset.filter(
#     "deployment_ID=='USA-NEFSC-MA-RI-202001-NS01'| deployment_ID=='USA-NEFSC-MA-RI-202202-NS02'",
#     inplace=True,
# )


# Create spectrograms and wav files
print("Extracting detection spectrograms...")
out_dir = os.path.join(in_dir, "extracted_detections")
if os.path.isdir(out_dir) == False:
    os.mkdir(out_dir)

dataset.export_spectrograms(
    out_dir,
    time_buffer_sec=5,
    spectro_unit="sec",
    spetro_nfft=0.256,
    spetro_frame=0.256,
    spetro_inc=0.03,
    freq_min_hz=0,
    freq_max_hz=1000,
    sanpling_rate_hz=2000,
    filter_order=8,
    filter_type="iir",
    fig_size=(15, 10),
    deployment_subfolders=True,
    date_subfolders=False,
    file_name_field="uuid",
    # file_name_field="audio_file_name",
    file_prefix_field="confidence",
    channel=0,
    colormap="Greys",  # "viridis",
    save_wav=True,
)

print("Done!")
