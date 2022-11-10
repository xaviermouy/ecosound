# -*- coding: utf-8 -*-
"""

@author: xavier.mouy
"""

from ecosound.core.tools import list_files
import re

from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
import soundfile as sf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import scipy
import os
import csv


annot_dataset_file = (
    r"D:\Detector\datasets\Master_annotations_dataset_20221025.nc"
)

new_data_dir = r"D:\Detector\datasets2"
new_dataset_file = (
    r"D:\Detector\datasets\Master_annotations_dataset_2022-10-25_test.nc"
)

# Load dataset
dataset = Annotation()
dataset.from_netcdf(annot_dataset_file)


# update audio dir path
dataset.update_audio_dir(new_data_dir, verbose=False)

# # list name of all audio files in dataset
# dataset_files_list = set(
#     dataset.data["audio_file_dir"]
#     + os.path.sep
#     + dataset.data["audio_file_name"]
#     + dataset.data["audio_file_extension"]
# )

# # list extension of all audio files in dataset
# dataset_ext_list = set(
#     [os.path.splitext(file)[1] for file in dataset_files_list]
# )

# # list all audio files in new folder (only for the target file extensions)
# new_dir_files_list = []
# for ext in dataset_ext_list:
#     new_dir_files_list = new_dir_files_list + list_files(
#         new_data_dir, ext, recursive=True
#     )

# # go through each file in dataset and try to find in in new data folder
# missing_files_list = []
# for file in dataset_files_list:
#     res = [
#         idx
#         for idx, new_dir_file in enumerate(new_dir_files_list)
#         if re.search(os.path.split(file)[1], new_dir_file)
#     ]
#     if len(res) == 0:
#         missing_files_list.append(file)
#     else:
#         new_path = os.path.split(new_dir_files_list[res[0]])[0]
#         dataset.data.loc[
#             dataset.data["audio_file_name"]
#             == os.path.splitext(os.path.split(file)[1])[0],
#             "audio_file_dir",
#         ] = new_path

# if len(missing_files_list) > 0:
#     print(str(len(missing_files_list)), " files could not be found.")
#     print(missing_files_list)

# # save update dataset
# dataset.to_netcdf(new_dataset_file)
