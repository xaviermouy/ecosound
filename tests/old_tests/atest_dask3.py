# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:26:33 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import time
from dask import delayed, compute
import dask.bag as db
from ecosound.core.audiotools import Sound

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster,processes=False)




# Using scipy #########################################
def spectro_numpy(sig, fs, frame_samp, overlap_samp, fft_samp):
    tic = time.perf_counter()
    axis_frequencies, axis_times, spectrogram = signal.spectrogram(sig,
                                                                    fs=fs,
                                                                    window=signal.hann(frame_samp),
                                                                    noverlap=overlap_samp,
                                                                    nfft=fft_samp,
                                                                    scaling='spectrum')
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")
    return spectrogram

def getFFT(sig,nfft):
    s = np.fft.fft(sig, nfft)
    s2 = abs(s)

    return s2

def calc_spectrogram(sig, win, starts, stops, fft_samp):
    fnyq = int(np.round(fft_samp/2))
    spectro = np.empty((fnyq,len(starts))) # the default 
    idx=0
    for start, stop in zip(starts, stops):
        s = sig[start:stop]*win
        Spectrum = np.fft.fft(s, fft_samp)
        Spectrum = abs(Spectrum) # amplitude
        Spectrum = Spectrum*2
        #Spectrum = Spectrum**2
        #ts_window = getFFT(sig[start:stop],fft_samp)
        spectro[:,idx] = Spectrum[0:fnyq]
        idx+=1
    return spectro

def spectro_loop(sig, fs, frame_samp, overlap_samp, fft_samp):
    tic = time.perf_counter()
    starts = np.arange(0,len(sig),frame_samp-overlap_samp,dtype=int)
    starts = starts[starts + frame_samp < len(sig)]
    xns = []
    for start in starts:
        # short term discrete fourier transform
        ts_window = getFFT(sig[start:start + frame_samp],fft_samp)
        xns.append(ts_window)
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")
    return xns

def spectro_loop_dask(sig, fs, frame_samp,overlap_samp,fft_samp):
    tic = time.perf_counter()
    starts = np.arange(0,len(sig), frame_samp - overlap_samp, dtype=int)
    starts = starts[starts + frame_samp < len(sig)]
    xns = []
    for start in starts:
        sig2 = sig[start:start + frame_samp]
        # short term discrete fourier transform
        ts_window = delayed(getFFT)(sig2, fft_samp)
        xns.append(ts_window)
    compute(xns)
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")
    return xns

def spectro_loop_dask2(sig, fs, frame_samp,overlap_samp,fft_samp, dask=False, dask_chunks=40):
    #dask_chunks = 40
    tic = time.perf_counter()
    step = frame_samp - overlap_samp
    starts = np.arange(0,len(sig)-frame_samp, step, dtype=int)
    stops = starts + frame_samp
    start_chunks = np.array_split(starts,dask_chunks)
    stop_chunks = np.array_split(stops,dask_chunks)
    win = np.hanning(frame_samp)
    spectrogram = []
    idx=0
    for start_chunk, stop_chunk in zip(start_chunks, stop_chunks):
        sig_chunk = sig[start_chunk[0]:stop_chunk[-1]]
        chunk_size = len(start_chunk)
        if dask:
            spectro_chunk = delayed(calc_spectrogram)(sig_chunk,win,
                                       start_chunk-start_chunk[0],
                                       stop_chunk-start_chunk[0],
                                       fft_samp)
        else:
            spectro_chunk = calc_spectrogram(sig_chunk,win,
                            start_chunk-start_chunk[0],
                            stop_chunk-start_chunk[0],
                            fft_samp)
        spectrogram.append(spectro_chunk)
        idx += chunk_size
    if dask:
        spectrogram = compute(spectrogram)
        spectrogram = np.concatenate(spectrogram[0][:], axis=1)
    else:
        spectrogram = np.concatenate(spectrogram[:], axis=1)
    axis_times = starts/fs
    freq_resolution = fs/fft_samp
    axis_frequencies = np.arange(0,fs/2,freq_resolution)
    toc = time.perf_counter()
    print(f"Dexecuted in {toc - tic:0.4f} seconds")
    return axis_frequencies, axis_times, spectrogram
if __name__ == '__main__':

    # # Create random signal
    # fs = 48000
    # sig_dur = 60#1800
    # sig = np.random.rand(sig_dur*fs)

    single_channel_file = r"../ecosound/resources/67674121.181018013806.wav"

    t1 = 24
    t2 = 120
    sound = Sound(single_channel_file)
    #sound.read(channel=0, chunk=[t1, t2], unit='sec')
    sound.read(channel=0)
    fs = sound.file_sampling_frequency
    sig = sound.waveform
    sig = sig-np.mean(sig)


    # Calculates  spectrogram
    frame_samp = 3000
    overlap_samp = 2500
    fft_samp = 4096

    #S1 = spectro_numpy(sig, fs, frame_samp, overlap_samp, fft_samp)
    #S2 = spectro_loop(sig, fs, frame_samp, overlap_samp, fft_samp)
    #S3 = spectro_loop_dask(sig, fs, frame_samp, overlap_samp, fft_samp)
    F, T, S3 = spectro_loop_dask2(sig,
                                  fs,
                                  frame_samp,
                                  overlap_samp,
                                  fft_samp,
                                  dask=True,
                                  dask_chunks=50,
                                  )
    
    
    # fig, ax = plt.subplots(figsize=(16,4), sharex=True)
    # im = ax.pcolormesh(S1, cmap='jet',vmin = np.percentile(S1,50), vmax= np.percentile(S1,99.9))

    fig, ax = plt.subplots(figsize=(16,4), sharex=True)
    im = ax.pcolormesh(T, F, S3, cmap='jet',vmin = np.percentile(S3,50), vmax= np.percentile(S3,99.9))
