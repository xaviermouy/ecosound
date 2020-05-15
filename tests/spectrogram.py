# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:35:31 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.visualization.grapher_builder import GrapherFactory
import time

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
tic = time.perf_counter()

# load audio data
sound = Sound(single_channel_file)
fs = sound.waveform_sampling_frequency
sound.read(channel=0, chunk=[round(t1*fs), round(t2*fs)])

# Calculates  spectrogram
Spectro = Spectrogram(frame, window_type, nfft, step, fs, unit='samp')
Spectro.compute(sound, dB=True, dask=True)

toc = time.perf_counter()
print(f"Executed in {toc - tic:0.4f} seconds")

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
# graph.add_data(sound)
# graph.add_data(Spectro)
# graph.colormap = 'jet'
# graph.show()