import ecosound.core.audiotools
#import detectors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy import signal, ndimage
import numpy as np
import cv2

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

