# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:56:39 2020

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

audio_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\audio_data\AMAR173.4.20190916T011248Z.wav"
annotation_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\manual_annotations\AMAR173.4.20190916T011248Z.Table.1.selections.txt"
detection_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset\AMAR173.4.20190916T011248Z.wav.nc"

# Spectrogram parameters
frame = 3000
nfft = 4096
step = 500
#ovlp = 2500
fmin = 0
fmax = 1000
window_type = 'hann'

# start and stop time of wavfile to analyze
t1 = 0#24
t2 = 200#40
## ###########################################################################
tic = time.perf_counter()

# load audio data
sound = Sound(audio_file)
sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)
spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)

# load annotations
annot = Annotation()
annot.from_raven(annotation_file)

# load detections
detec = Measurement()
detec.from_netcdf(detection_file)

freq_ovp = True # default True
dur_factor_max = None # default None
dur_factor_min = 0.1 # default None
ovlp_ratio_min = 0.3 # defaulkt None
remove_duplicates = True # dfault - False
inherit_metadata = True # default False

# here <-= filter per filename and deployment ID

stack = []
det = detec.data
for index, an in annot.data.iterrows(): #for each annotation
    ovlp_dur = []
    ## check overlap in time first
    df = det[((det.time_min_offset <= an.time_min_offset) & (det.time_max_offset >= an.time_max_offset)) |  # 1- annot inside detec
             ((det.time_min_offset >= an.time_min_offset) & (det.time_max_offset <= an.time_max_offset)) |  # 2- detec inside annot
             ((det.time_min_offset < an.time_min_offset) & (det.time_max_offset < an.time_max_offset) & (det.time_max_offset > an.time_min_offset)) | # 3- only the end of the detec overlaps with annot
             ((det.time_min_offset > an.time_min_offset) & (det.time_min_offset < an.time_max_offset) & (det.time_max_offset > an.time_max_offset)) # 4- only the begining of the detec overlaps with annot
              ]
    ovlp_dur = an.duration
    # then looks at frequency overlap. Can be turned off if freq bounds are not reliable
    if (len(df) > 0) & freq_ovp:
        df = df[((df.frequency_min <= an.frequency_min) & (df.frequency_max >= an.frequency_max)) | # 1- annot inside detec
                ((df.frequency_min >= an.frequency_min) & (df.frequency_max <= an.frequency_max)) | # 2- detec inside annot
                ((df.frequency_min < an.frequency_min) & (df.frequency_max < an.frequency_max) & (df.frequency_max > an.frequency_min)) | # 3- only the top of the detec overlaps with annot
                ((df.frequency_min > an.frequency_min) & (df.frequency_min < an.frequency_max) & (df.frequency_max > an.frequency_max)) # 4- only the bottom of the detec overlaps with annot
                ]
    # discard if durations are too different 
    if (len(df) > 0) & (dur_factor_max is not None):
        df = df[df.duration < an.duration*dur_factor_max]
    if (len(df) > 0) & (dur_factor_min is not None):
        df = df[df.duration > an.duration*dur_factor_min]

    # discard if they don't overlap enough
    if (len(df) > 0) & (ovlp_ratio_min is not None):
        df_ovlp = (df['time_max_offset'].apply(lambda x: min(x,an.time_max_offset)) - df['time_min_offset'].apply(lambda x: max(x,an.time_min_offset))) / an.duration
        df = df[df_ovlp>=ovlp_ratio_min]

    if (len(df) > 1) & remove_duplicates:
        df = df.iloc[[df_ovlp.values.argmax()]] # pick teh one with max time overlap
        
    if len(df) > 0:
        if inherit_metadata:
            df['mooring_platform_name'] = an['mooring_platform_name']
            df['recorder_type'] = an['recorder_type']
            df['recorder_SN'] = an['recorder_SN']
            df['hydrophone_model'] = an['hydrophone_model']
            df['hydrophone_SN'] = an['hydrophone_SN']
            df['hydrophone_depth'] = an['hydrophone_depth']
            df['location_name'] = an['location_name']
            df['location_lat'] = an['location_lat']
            df['location_lon'] = an['location_lon']
            df['location_water_depth'] = an['location_water_depth']
            df['deployment_ID'] = an['deployment_ID']
            df['label_class'] = an['label_class']
            df['label_subclass'] = an['label_subclass']
        stack.append(df)
ovlp = pd.concat(stack, ignore_index=True)


detec.data = ovlp
# # detec.overlap_with(annot)
# toc = time.perf_counter()
# print(f"Executed in {toc - tic:0.4f} seconds")

# Plot
graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000, time_min=20, time_max=200)
graph.add_data(sound)
graph.add_annotation(detec, panel=0, color='green', label='Detections')
graph.add_data(spectro)
graph.add_annotation(annot, panel=1, color='red', label='Annotations')
graph.add_annotation(detec, panel=1, color='green', label='Detections')
graph.colormap = 'binary'
#graph.colormap = 'jet'
graph.show()

