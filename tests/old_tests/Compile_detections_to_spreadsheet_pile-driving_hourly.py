# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 09:38:49 2022

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.tools import list_files, filename_to_datetime
import ecosound
import matplotlib.pyplot as plt
import os
import sqlite3
import pandas as pd
## ############################################################################
## Input parameters ###########################################################

detec_dir = r'\\nefscfile\PassiveAcoustics\Stellwagen_Old\DETECTORS\Pile_driving\Pile_driving_NEFSC_MA-RI_202304_NS02_ST'
audio_dir=r'\\nefscfile\PassiveAcoustics\Stellwagen_Old\ACOUSTIC_DATA\BOTTOM_MOUNTED\NEFSC_MA-RI\NEFSC_MA-RI_202304_NS02\NEFSC_MA-RI_202304_NS02_ST\6075_64kHz_UTC'
deployment_name = 'NEFSC_MA-RI_202304_NS02_ST'
audio_file_ext = '.wav'
aggregate_time_offset = 0
## ############################################################################
## ############################################################################

out_dir=os.path.join(detec_dir,'detections_summary')

#creates output dir
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)

# import detection results
print('Loading detections')
annot = Annotation()
annot.from_raven(detec_dir,recursive=True)

### update path
annot.data.loc[:,'audio_sampling_frequency']=0
annot.data.loc[:,'audio_bit_depth']=0
annot.data.loc[:,'audio_file_dir']=audio_dir

### save updated annot as netcdf file
annot.to_netcdf(os.path.join(detec_dir,'detections.nc'))

# # list files
# print('Creating detection summary spreadsheet')
audio_files = list_files(audio_dir,suffix=audio_file_ext,recursive=True)
dates = filename_to_datetime(audio_files)
# files_list=[]
# detec_count=[]
# for audio_file in audio_files:
#     name = os.path.basename(os.path.splitext(audio_file)[0])
#     files_list.append(name)
#     temp = annot.filter('audio_file_name == "' + name + '"')
#     detec_count.append(len(temp))
# table = pd.DataFrame({'file':files_list, 'detections': detec_count})
# table.to_csv(os.path.join(out_dir,'merged_detections_per_files.csv'),index=False)


# Create hourly agggregate.
print("Creating detections hourly aggregates...")
hourly_counts = annot.calc_time_aggregate_1D(
    integration_time="H",
    resampler="count",
    start_date= str(min(dates)),
    end_date= str(max(dates)),
)

hourly_counts.rename(columns={"value": "Detections"}, inplace=True)
hourly_counts.to_csv(
    os.path.join(out_dir, deployment_name + "_hourly_counts.csv"),
    index_label="Date (UTC" + str(aggregate_time_offset) + ")",
)


# create spectrograms
print('Creating spectrograms')
annot.export_spectrograms(
        out_dir,
        time_buffer_sec=4,
        spectro_unit="sec",
        spetro_nfft=0.1,
        spetro_frame=0.1,
        spetro_inc=0.03,
        freq_min_hz=10,
        freq_max_hz=1000,
        sanpling_rate_hz=4000,
        filter_order=8,
        filter_type="iir",
        fig_size=(20, 10),
        deployment_subfolders=False,
        date_subfolders=True,
        date_subfolders_format = "%Y-%m-%d_%H",
        #file_subfolder=True,
        #file_name_field="time_max_date",
        file_name_field="time_min_offset",
        file_prefix_field="audio_file_name",
        channel=None,
        colormap="viridis",#"Greys",  # "viridis",
        save_wav=False,
    )



