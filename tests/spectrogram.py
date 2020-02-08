# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:35:31 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram


## Input paraneters ##########################################################

single_channel_file = r"../ecosound/resources/AMAR173.4.20190916T061248Z.wav"

# Spectrogram parameters
frame = 3000
nfft = 4096
step = 500
#ovlp = 2500
fmin = 0 
fmax = 1000
window_type = 'hann'

# start and stop time of wavfile to analyze
t1 = 1515
t2 = 1541
## ###########################################################################


# load audio data
sound = Sound(single_channel_file)
fs = sound.waveform_sampling_frequency
sound.read(channel=0, chunk=[round(t1*fs), round(t2*fs)])
sound.plot_waveform()

# Calculates  spectrogram
Spectro = Spectrogram(frame, window_type, nfft, step, fs, unit='samp')
Spectro.compute(sound)
Spectro.show()