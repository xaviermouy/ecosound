# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:41:54 2020

@author: xavier.mouy

"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
#from ecosound.core.measurement import Measurement
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
import time

## Input paraneters ##########################################################

single_channel_file = r"../ecosound/resources/67674121.181018013806.wav"

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
t2 = 1800#40
## ###########################################################################
tic = time.perf_counter()

# load audio data
sound = Sound(single_channel_file)
sound.read(channel=0, chunk=[t1, t2], unit='sec', detrend=True)

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
spectro.compute(sound, dB=True, use_dask=True, dask_chunks=40)

# Crop unused frequencies
spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)

# Denoise
spectro.denoise('median_equalizer', window_duration=3, use_dask=True, dask_chunks=(2048,1000), inplace=True)

# Detector
detector = DetectorFactory('BlobDetector', use_dask=True, dask_chunks=(2048,2000), kernel_duration=0.1, kernel_bandwidth=300, threshold=10, duration_min=0.05, bandwidth_min=40)
detections = detector.run(spectro, debug=False)

toc = time.perf_counter()
print(f"Executed in {toc - tic:0.4f} seconds")

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
# graph.add_data(sound)
# graph.add_annotation(detections, panel=0, color='red')
# graph.add_data(spectro)
# graph.add_annotation(detections, panel=1)
# #graph.colormap = 'binary'
# graph.colormap = 'jet'
# graph.show()




## To test the .crop method
#detecSpectro = spectro.crop(time_min=2,time_max=10, inplace=False)
#detecSpectro = spectro.crop(time_max=10, inplace=False)
#detecSpectro = spectro.crop(frequency_min=50, inplace=False)
#detecSpectro = spectro.crop(frequency_max=800,inplace=False)
# detecSpectro = spectro.crop(frequency_min=0,frequency_max=600,time_min=10,time_max=10.3, inplace=False)
# graph = GrapherFactory('SoundPlotter', title='Detection', frequency_max=1000)
# graph.add_data(detecSpectro)
# graph.show()



