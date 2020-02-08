# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:35:31 2020

@author: xavier.mouy
"""
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound, Filter

#single_channel_file = r"../ecosound/resources/AMAR173.4.20190916T061248Z.wav"
multi_channel_file = r"../ecosound/resources/671404070.190722162836.wav"


# load part of the file and plot
print('------------------------------')
#sig = Sound() # should return error
sig = Sound(multi_channel_file)
print(len(sig))
sig.read(channel=0, chunk=[10, 1000])
print('------------------------------')
print(len(sig))
print('start sample: ', sig.waveform_start_sample)
print('stop sample: ', sig.waveform_stop_sample)
print('duration:: ', sig.waveform_duration_sample)
print(len(sig))
sig.plot_waveform(newfig=True)

# extract a sinppet from the data
sig2 = sig.select_snippet([100,1000])
sig2.plot_waveform(newfig=True)
print('------------------------------')
print('start sample: ', sig2.waveform_start_sample)
print('stop sample: ', sig2.waveform_stop_sample)
print('duration:: ', sig2.waveform_duration_sample)
print(len(sig2))

# try filter
filter_type = ['bandpass', 'lowpass', 'highpass']
cutoff_frequencies = [100, 1000]
sig2.filter(filter_type[0 ], cutoff_frequencies, order=4)
# try agfain -> should retrun error
#sig2.filter(filter_type[0], cutoff_frequencies, order=4)
sig2.plot_waveform(newfig=True)
print('------------------------------')
print('start sample: ', sig2.waveform_start_sample)
print('stop sample: ', sig2.waveform_stop_sample)
print('duration: ', sig2.waveform_duration_sample)
print(len(sig2))

# # re-adjust seletec waveform based on energy
# energy_percentage = 99.9
# sig2.tighten_waveform_window(energy_percentage)
# sig2.plot_waveform(newfig=True)
# print('------------------------------')
# print('start sample: ', sig2.waveform_start_sample)
# print('stop sample: ', sig2.waveform_stop_sample)
# print('duration:: ', sig2.waveform_duration_sample)
# print(len(sig2))


# sig3 = Sound(multi_channel_file)
# sig3.read(channel=0, chunk=[sig2.waveform_start_sample, sig2.waveform_stop_sample])
# #sig3.read(channel=0, chunk=[134, 1009])
# sig3.plot_waveform(newfig=True)
# print('------------------------------')
# print('start sample: ', sig3.waveform_start_sample)
# print('stop sample: ', sig3.waveform_stop_sample)
# print('duration:: ', sig3.waveform_duration_sample)
# print(len(sig3))
