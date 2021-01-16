# -*- coding: utf-8 -*-
"""
Created on Fri May 22 14:04:01 2020

@author: xavier.mouy
"""


# import sys
# sys.path.append("..") # Adds higher directory to python modules path.
# from ecosound.core.audiotools import Sound
# from ecosound.core.spectrogram import Spectrogram
# from ecosound.core.annotation import Annotation
# from ecosound.core.measurement import Measurement
# from ecosound.visualization.grapher_builder import GrapherFactory
# from ecosound.measurements.measurer_builder import MeasurerFactory
# import time


# from ecosound.core.measurement import Measurement
# # load detections
# detection_file = r"C:\Users\xavier.mouy\Desktop\Tutorial\results\67674121.181017060806.wav.nc"
# detec = Measurement()
# detec.from_netcdf(detection_file)
# detec.summary()

# from ecosound.core.measurement import Measurement
# from ecosound.core.audiotools import Sound
# from ecosound.core.spectrogram import Spectrogram
# from ecosound.visualization.grapher_builder import GrapherFactory

# # load detections
# detection_file = r"C:\Users\xavier.mouy\Desktop\Tutorial\results\67674121.181017060806.wav.nc"
# detec = Measurement()
# detec.from_netcdf(detection_file)

# # load audio file (first 15 sec only)
# audio_file = r"C:\Users\xavier.mouy\Desktop\Tutorial\data\67674121.181017060806.wav"
# sound = Sound(audio_file)
# sound.read(channel=0, chunk=[0, 15], unit='sec', detrend=True)

# # Calculate  spectrogram
# frame = 3000
# nfft = 4096
# step = 500
# window_type = 'hann'
# spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
# spectro.compute(sound, dB=True)
# #spectro.crop(frequency_min=0, frequency_max=1000, inplace=True)

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
# graph.add_data(sound) # add waveform data
# graph.add_data(spectro) # add spectrogram
# graph.add_annotation(detec, panel=0, color='green', label='Detections') # overlay detections on waveform plot
# graph.add_annotation(detec, panel=1, color='green', label='Detections') # overlay detections on spectrogram plot
# graph.colormap = 'binary'
# #graph.colormap = 'jet'
# graph.show()





from ecosound.core.measurement import Measurement
import seaborn as sns
# load detections
detection_file = r"C:\Users\xavier.mouy\Desktop\Tutorial\results\67674121.181017060806.wav.nc"
detec = Measurement()
detec.from_netcdf(detection_file)

detec.data['freq_peak'].groupby(detec.data['label_class'])
fig = sns.violinplot(y=detec.data['label_class'], x=detec.data['freq_peak'])
fig.set(xlabel='Detections peak frequency (Hz)', ylabel='Detections class')
#sns.show()

# ## Input paraneters ##########################################################


# audio_file = r"C:\Users\xavier.mouy\Desktop\Tutorial\data\67674121.181017060806.wav"

# #audio_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\audio_data\AMAR173.4.20190916T011248Z.wav"
# #annotation_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\datasets\UVIC_hornby-island_2019\manual_annotations\AMAR173.4.20190916T011248Z.Table.1.selections.txt"
# #detection_file = r"C:\Users\xavier.mouy\Documents\PhD\Projects\Dectector\results\Full_dataset\AMAR173.4.20190916T011248Z.wav.nc"


# # Spectrogram parameters
# frame = 3000
# nfft = 4096
# step = 500
# #ovlp = 2500
# fmin = 0
# fmax = 1000
# window_type = 'hann'

# # start and stop time of wavfile to analyze
# t1 = 00#24
# t2 = 60#40
# ## ###########################################################################
# tic = time.perf_counter()

# # load audio data
# sound = Sound(audio_file)
# sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

# # Calculates  spectrogram
# spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
# spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)
# spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)

# # load annotations
# annot = Annotation()
# annot.from_raven(annotation_file)

# # load detections
# detec = Measurement()
# detec.from_netcdf(detection_file)

# toc = time.perf_counter()
# print(f"Executed in {toc - tic:0.4f} seconds")

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000, time_min=30)
# graph.add_data(sound)
# graph.add_annotation(detec, panel=0, color='green', label='Detections')
# graph.add_data(spectro)
# graph.add_annotation(annot, panel=1, color='red', label='Annotations')
# graph.add_annotation(detec, panel=1, color='green', label='Detections')
# graph.colormap = 'binary'
# #graph.colormap = 'jet'
# graph.show()