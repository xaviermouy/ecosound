import os
import pandas as pd
from ecosound.core.annotation import Annotation
from ecosound.core.audiotools import Sound
from ecosound.core.tools import list_files

data_dir = r'\\stellwagen.nefsc.noaa.gov\stellwagen\ACOUSTIC_DATA\BOTTOM_MOUNTED\NEFSC_GOM\NEFSC_GOM_202205_USTR01\6550_48kHz_UTC'
detec_dir = r'\\stellwagen.nefsc.noaa.gov\stellwagen\STAFF\Lindsey\humpback_detector_outputs\NEFSC_GOM_202205_USTR01_2023-02-24_16-08-13\thresh_0.9\NEFSC_GOM_202205_USTR01\6550_48kHz_UTC'
outfilename =r'\\stellwagen.nefsc.noaa.gov\stellwagen\STAFF\Lindsey\humpback_detector_outputs\NEFSC_GOM_202205_USTR01_2023-02-24_16-08-13\thresh_0.9\NEFSC_GOM_202205_USTR01\6550_48kHz_UTC\merged_detections\merged_tables.txt'
file_ext = '.wav'



# define dates and time offsets
data_files_dates = list_files(data_dir,suffix=file_ext)
data_files_dates.sort()
filenames=[]
files_dur =[]
for idx, file in enumerate(data_files_dates):
    s = Sound(file);
    filenames.append(os.path.split(file)[1][0:-len(file_ext)])
    files_dur.append(s.file_duration_sec)
files_dur = [0] + files_dur[0:-1]
filename_nchar = len(filenames[0])

file_offset_sec=[]
for idx, dur in enumerate(files_dur):
    file_offset_sec.append(sum(files_dur[0:idx+1]))

# load all text files
raven_files = list_files(detec_dir,suffix='.txt')
for idx, file in enumerate(raven_files):
    tmp = pd.read_csv(file, delimiter="\t")
    file_idx = filenames.index(os.path.split(file)[1][0:filename_nchar])
    tmp['Begin Time (s)'] = tmp['Begin Time (s)'] + file_offset_sec[file_idx]
    tmp['End Time (s)'] = tmp['End Time (s)'] + file_offset_sec[file_idx]
    if idx == 0:
        data = tmp
    else:
        data = pd.concat([data, tmp], ignore_index=True, sort=False)

data['Selection'] = list(range(1,len(data)+1))
data.drop(columns=['Selection.1'],inplace=True)

data.to_csv(outfilename,
            sep="\t",
            encoding="utf-8",
            header=True,
            #columns=cols,
            index=False,
            )
