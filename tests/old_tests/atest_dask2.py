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
from numpy.fft import fft
from dask import delayed, compute
# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster,processes=False)


single_channel_file = r"../ecosound/resources/67674121.181018013806.wav"


# start and stop time of wavfile to analyze
t1 = 1
t2 = 120#1000
## ###########################################################################


# load audio data
sound = Sound(single_channel_file)
sound.read(channel=0, chunk=[t1, t2], unit='sec')

# Calculates  spectrogram
frame_samp = 3000
overlap_samp = 2500
fft_samp = 4096

# Using scipy #########################################
def spectro_numpy(frame_samp,overlap_samp,fft_samp):
    tic = time.perf_counter()
    axis_frequencies, axis_times, spectrogram = signal.spectrogram(sound.waveform,
                                                                    fs=sound.waveform_sampling_frequency,
                                                                    window=signal.hann(frame_samp),
                                                                    noverlap=overlap_samp,
                                                                    nfft=fft_samp,
                                                                    scaling='spectrum')
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")
#fig, ax = plt.subplots(figsize=(16,4), sharex=True)
#im = ax.pcolormesh(axis_times,axis_frequencies,spectrogram, cmap='jet',vmin = np.percentile(spectrogram,50), vmax= np.percentile(spectrogram,99.9))

def getFFT(sig,nfft):
    return abs(np.fft.fft(sig,nfft))

def crop(x,fft_samp):
    return x[0:int(fft_samp/2)]

def spectro_loop(frame_samp,overlap_samp,fft_samp):
    tic = time.perf_counter()
    starts = np.arange(0,len(sound.waveform),frame_samp-overlap_samp,dtype=int)
    starts = starts[starts + frame_samp < len(sound.waveform)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = getFFT(sound.waveform[start:start + frame_samp],fft_samp)
        ts_window2 = crop(ts_window,fft_samp)
        xns.append(ts_window2)
    specX = np.array(xns).T
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")

def slice(sound, start, frame_samp):
    return sound.waveform[start:start + frame_samp]

def spectro_loop_dask(frame_samp,overlap_samp,fft_samp):
    tic = time.perf_counter()
    #client = Client(n_workers=4)
    starts = np.arange(0,len(sound.waveform),frame_samp-overlap_samp,dtype=int)
    starts = starts[starts + frame_samp < len(sound.waveform)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        sig = delayed(slice)(sound,start,frame_samp)
        ts_window1 = delayed(fft)(sig,fft_samp)
        ts_window2 = delayed(abs)(ts_window1)
        #abs(
        #ts_window2 = delayed(crop)(ts_window,fft_samp)
        xns.append(ts_window2)
    #specX = delayed(np.array(xns).T)
    #specX = delayed(abs)(xns)
    #visualize(xns)
    compute(xns)
    #specX = specX.compute()
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")
    #client.close()
    
# ## Using Dask #########################################
# from dask import delayed
# from dask.distributed import Client, progress
# client = Client(n_workers=4)


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

# import time
# from dask.distributed import Client
# from dask import delayed
# from time import sleep
    
# ## Using Dask #########################################

# def inc(x):
#         sleep(1)
#         print('inc')
#         return x + 1

# def add(x, y):
#     sleep(1)
#     print('add')
#     return x + y
    
# def test():

#     client = Client(n_workers=10)
    
#     tic = time.perf_counter()
#     x = delayed(inc)(1)
#     y = delayed(inc)(2)
#     z = delayed(add)(x, y)
#     total = z.compute()
#     print(total)
#     toc = time.perf_counter()
#     print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
#     client.close()

if __name__ == '__main__':
    spectro_numpy(frame_samp,overlap_samp,fft_samp)
    spectro_loop(frame_samp,overlap_samp,fft_samp)
    spectro_loop_dask(frame_samp,overlap_samp,fft_samp)
