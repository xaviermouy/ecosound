# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:41:54 2020

@author: xavier.mouy

"""
import sys
sys.path.append("..")  # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.visualization.grapher_builder import GrapherFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
import ecosound.core.tools
import time


## Input paraneters ##########################################################

single_channel_file = r"../ecosound/resources/67674121.181018013806.wav"

# Spectrogram parameters
frame = 3000
nfft = 4096
step = 500
#  ovlp = 2500
fmin = 0
fmax = 1000
window_type = 'hann'

# start and stop time of wavfile to analyze
t1 = 0 # 24
t2 = 20 # 40
## ###########################################################################
tic = time.perf_counter()


# load audio data
sound = Sound(single_channel_file)
sound.read(channel=0, chunk=[t1, t2], unit='sec')

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
spectro.compute(sound)

# Crop unused frequencies
spectro.crop(frequency_min=fmin, frequency_max=fmax, inplace=True)

# Denoise
spectro.denoise('median_equalizer', window_duration=3, inplace=True)

# Detector
file_timestamp = ecosound.core.tools.filename_to_datetime(single_channel_file)[0]
detector = DetectorFactory('BlobDetector', kernel_duration=0.1, kernel_bandwidth=300, threshold=40, duration_min=0.05, bandwidth_min=60)
detections = detector.run(spectro, start_time=file_timestamp, debug=False)

# Plot
graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
graph.add_data(sound)
graph.add_annotation(detections, panel=0, color='red')
graph.add_data(spectro)
graph.add_annotation(detections, panel=1)
#  graph.colormap = 'binary'
graph.colormap = 'jet'
graph.show()

# Maasurements
#detections.data = detections.data.iloc[4:5].reset_index()
spectro_features = MeasurerFactory('SpectrogramFeatures', resolution_time=0.001, resolution_freq=0.1, interp='linear')
measurements = spectro_features.compute(spectro, detections, debug=False, verbose=False)
#measurements.to_netcdf('test.nc')

toc = time.perf_counter()
print(f"Executed in {toc - tic:0.4f} seconds")
# #########################################
