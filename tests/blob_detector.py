# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:41:54 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detectors_builder import DetectorFactory

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
t1 = 24
t2 = 40
## ###########################################################################


# load audio data
sound = Sound(single_channel_file)
sound.read(channel=0, chunk=[t1,t2], unit='sec')
sound.plot_waveform()

# Calculates  spectrogram
spectro = Spectrogram(frame, window_type, nfft, step, sound.waveform_sampling_frequency, unit='samp')
spectro.compute(sound)

# Crop unused frequencies
spectro.crop(frequency_min=fmin, frequency_max=fmax)
spectro.show(frequency_min=fmin, frequency_max=fmax)

# Denoise
spectro.denoise('median_equalizer', window_duration=3)
spectro.show(frequency_min=fmin, frequency_max=fmax)

# Detector
detector = DetectorFactory('BlobDetector', kernel_duration=0.1, kernel_bandwidth=300, threshold=40, duration_min=0.01, bandwidth_min=40)
detections = detector.run(spectro, debug=True)



