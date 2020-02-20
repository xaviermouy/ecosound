# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:41:54 2020

@author: xavier.mouy
"""

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram


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
spectro.denoise('median_equalizer', window_size=(1,100))
spectro.show(frequency_min=fmin, frequency_max=fmax)



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


def calcVariance2D(buffer):
    return np.var(buffer)
    #return np.median(buffer.ravel())