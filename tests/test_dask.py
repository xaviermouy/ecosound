# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:26:33 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time

from dask import delayed

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
sound.read(channel=0, chunk=[t1, t2], unit='sec')

# Calculates  spectrogram
frame_samp = 1024
overlap_samp = 512
fft_samp = 1024

## Using scipy #########################################
# tic = time.perf_counter()
# axis_frequencies, axis_times, spectrogram = signal.spectrogram(sound.waveform,
#                                                                fs=sound.waveform_sampling_frequency,
#                                                                window=signal.hann(frame_samp),
#                                                                noverlap=overlap_samp,
#                                                                nfft=fft_samp,
#                                                                scaling='spectrum')

# toc = time.perf_counter()
# print(f"Dexecuted in {toc - tic:0.4f} seconds")
# #fig, ax = plt.subplots(figsize=(16,4), sharex=True)
# #im = ax.pcolormesh(axis_times,axis_frequencies,spectrogram, cmap='jet',vmin = np.percentile(spectrogram,50), vmax= np.percentile(spectrogram,99.9))


## Using Dask #########################################
from dask import delayed
from dask.distributed import Client, progress
client = Client(n_workers=4)


# def getFFT(sig):
#     return abs(np.fft.fft(sig))

# def crop(x,fft_samp):
#     return x[0:int(fft_samp/2)]
    
# tic = time.perf_counter()
# starts = np.arange(0,len(sound.waveform),fft_samp-overlap_samp,dtype=int)
# starts = starts[starts + fft_samp < len(sound.waveform)]
# xns = []
# for start in starts:
#     # short term discrete fourier transform
#     ts_window = delayed(getFFT)(sound.waveform[start:start + fft_samp])
#     ts_window2 = delayed(crop)(ts_window,fft_samp)
#     xns.append(ts_window2)

# specX = delayed(np.array(xns).T)
# specX.visualize()
# # #spec = delayed(10*np.log10(specX))
# # result = specX.compute()

# # toc = time.perf_counter()
# # print(f"Dexecuted in {toc - tic:0.4f} seconds")

# #fig2, ax2 = plt.subplots(figsize=(16,4), sharex=True)
# #im = ax2.pcolormesh(spec, cmap='jet',vmin = np.percentile(spec,50), vmax= np.percentile(spec,99.9))
from time import sleep

def inc(x):
    sleep(1)
    return x + 1

def add(x, y):
    sleep(1)
    return x + y

x = delayed(inc)(1)
y = delayed(inc)(2)
z = delayed(add)(x, y)
z.visualize()