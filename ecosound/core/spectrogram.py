# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:23:51 2020

@author: xavier.mouy
"""
import audiotools
import detectors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal, ndimage
import numpy as np
import cv2

class Spectrogram:
    _valid_units = ('samp','sec')
    _valid_windows = ('hann',)
    def __init__(self, frame, window_type, fft, step, sampling_frequency, unit = 'sec'):
        # Validation of the imput parameters
        assert (unit in Spectrogram._valid_units), ("Wrong unit value. Valid units: ", Spectrogram._valid_units)
        assert fft >= frame, " fft should alwyas be >= frame"
        assert step < frame, "step should always be <= frame"
        assert (window_type in Spectrogram._valid_windows),("Wrong window type. Valid values: ", Spectrogram._valid_windows)

        # Convert units in seconds/samples
        self._frame_samp,self._fft_samp,self._step_samp,self._frame_sec, \
        self._fft_sec,self._step_sec, self._overlap_perc, self._overlap_samp = \
        Spectrogram._convert_units(frame, fft, step, sampling_frequency, unit)

        # Time and frequency resolution
        self._sampling_frequency = sampling_frequency
        self._time_resolution = self.step_sec
        self._frequency_resolution = self.sampling_frequency / self.fft_samp

        # Define all other instance attributes
        self._window_type = window_type
        self._spectrogram =[]
        self._axis_frequencies =[]
        self._axis_times =[]

    @property
    def frame_samp(self):
        return self._frame_samp

    @property
    def frame_sec(self):
        return self._frame_sec

    @property
    def step_samp(self):
        return self._step_samp

    @property
    def step_sec(self):
        return self._step_sec

    @property
    def fft_samp(self):
        return self._fft_samp

    @property
    def fft_sec(self):
        return self._fft_sec

    @property
    def overlap_perc(self):
        return self._overlap_perc

    @property
    def overlap_samp(self):
        return self._overlap_samp

    @property
    def sampling_frequency(self):
        return self._sampling_frequency

    @property
    def time_resolution(self):
        return self._time_resolution

    @property
    def frequency_resolution(self):
        return self._frequency_resolution

    @property
    def window_type(self):
        return self._window_type

    @property
    def axis_frequencies(self):
        return self._axis_frequencies

    @property
    def axis_times(self):
        return self._axis_times

    @property
    def spectrogram(self):
        return self._spectrogram

    def _convert_units(frame, fft, step, sampling_frequency, unit):
        if unit == 'sec':
            frame_samp = round(frame*sampling_frequency)
            fft_samp = adjust_FFT_size(round(fft*sampling_frequency))
            step_samp = round(step*sampling_frequency)
            frame_sec = frame
            fft_sec = fft_samp*sampling_frequency
            step_sec = step
        elif unit == 'samp':
            frame_samp = frame
            fft_samp = adjust_FFT_size(fft)
            step_samp = step
            frame_sec = frame/sampling_frequency
            fft_sec = fft_samp/sampling_frequency
            step_sec = step/sampling_frequency
        overlap_samp = frame_samp-step_samp
        overlap_perc = (overlap_samp/frame_samp)*100
        return frame_samp, fft_samp, step_samp, frame_sec, fft_sec, step_sec, overlap_perc,overlap_samp

    def compute(self, sig, fs):
        assert fs == self.sampling_frequency, "The sampling frequency provided doesn't match the one from the Spectrogram object."
        # Weighting window
        if self.window_type == 'hann':
            window = signal.hann(self.frame_samp)
        # Calculates  spectrogram
        self._axis_frequencies,self._axis_times,self._spectrogram = signal.spectrogram(sig, fs=self.sampling_frequency, window=window, noverlap=self.overlap_samp,nfft=self.fft_samp, scaling='spectrum')
        self._spectrogram = 20*np.log10(self._spectrogram)
        return self._axis_frequencies, self._axis_times, self._spectrogram

    def show(self,frequency_min=0, frequency_max = [], time_min=0, time_max=[]):
        if not frequency_max:
            frequency_max = self.sampling_frequency/2
        if not time_max:
            time_max = self.axis_times[-1]
        assert len(self.spectrogram)>0, "Spectrogram not computed yet. Use the .compute() method first."
        assert frequency_min < frequency_max, "Incorrect frequency bounds, frequency_min must be < frequency_max "
        assert frequency_min < frequency_max, "Incorrect frequency bounds, frequency_min must be < frequency_max "

        fig, ax = plt.subplots(
        figsize=(16,4),
        sharex=True
        )
        im = ax.pcolormesh(self.axis_times, self.axis_frequencies, self.spectrogram, cmap = 'jet',vmin = np.percentile(self.spectrogram,50), vmax= np.percentile(self.spectrogram,99.9))
        ax.axis([time_min,time_max,frequency_min,frequency_max])
        #ax.set_clim(np.percentile(Sxx,50), np.percentile(Sxx,99.9))
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.set_title('Original spectrogram')
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return

def adjust_FFT_size(nfft):
        nfft_adjusted = next_power_of_2(nfft)
        if nfft_adjusted != nfft:
            print('Warning: FFT size automatically adjusted to ', nfft, 'samples (original size:', nfft,')')
        return nfft_adjusted

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def calcVariance2D(buffer):
    return np.var(buffer)
    #return np.median(buffer.ravel())


## Input paraneters ##########################################################
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

# bob detection
binThreshold = 50#20 10
min_area = 100 #10
minDuration = 30
minBandWidth = 10

# Example file
infile =r"data/AMAR173.4.20190920T161248Z.wav"
## ###########################################################################

# Close all existing graphs
plt.close('all')

# load audio data
sound = audiotools.Sound(infile)
fs =sound.getSamplingFrequencyHz()
sound.read(channel=0, chunk=[round(t1*fs),round(t2*fs)])
#sound.plotWaveform()

# Calculates  spectrogram
sig = sound.getWaveform()
Spectro = Spectrogram(frame, window_type, nfft, step, fs, unit='samp')
Spectro.compute(sig,fs)


# # crop spectrogram
# minRowIdx = np.where(f < fmin)
# maxRowIdx = np.where(f > fmax)
# if np.size(minRowIdx) == 0:
#     minRowIdx = 0
# else:
#     minRowIdx = minRowIdx[0][0]
# if np.size(maxRowIdx) == 0:
#     maxRowIdx = f[f.size-1]
# else:
#     maxRowIdx = maxRowIdx[0][0]
# f = f[minRowIdx:maxRowIdx]
# Sxx = Sxx[minRowIdx:maxRowIdx,:]


# # Dislays spectrogram
# displaySpectrogram(Sxx)

# # normalize
# Smed = ndimage.median_filter(Sxx, size=(1,100))
# displaySpectrogram(Smed)
# Sxx2 = Sxx-Smed
# # floor
# Sxx2[Sxx2<0]=0
# displaySpectrogram(Sxx2)

# # # blob detection
# Svar = ndimage.generic_filter(Sxx2, calcVariance2D, size=(30,10), mode='mirror') #size=(50,15)
# displaySpectrogram(Svar)
# # binarization
# Svar[Svar<binThreshold]=0
# Svar[Svar>0]=1
# displaySpectrogram(Svar)
# Svar_gray = cv2.normalize(src=Svar, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
# (im2, cnts, hierarchy) = cv2.findContours(Svar_gray.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)    

# # loop over the contours            
# isdetec=False
# boxCoord =[];
# for c in cnts:
#     # compute the bounding box for the contour
#     (x, y, w, h) = cv2.boundingRect(c)
#     # if the contour is too small, ignore it
#     if w < minDuration or  h < minBandWidth:
#         continue
#     else:
#         isdetec=True
#         # box coord
#         boxCoord.append([x,y,w,h])


