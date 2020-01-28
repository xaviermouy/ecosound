# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:26:24 2017

@author: xavier
"""
# --------------------------------------------------------------
# TO DO:
#
# - resample waveform
# - play sound
# - compute and plot spectrogram
# - get time stamp from file name (?)
# --------------------------------------------------------------

import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import copy


class Sound:
    """
    A class to load, represent and manipulate a sound file

    Attributes
    ----------
    infile : str
        A formatted string providing the path and filename of the sound file

    Methods
    -------
    read(channel=0, chunk=[])
        Reads a sound file with the option to select a specific channel and
        read only a section of the file.
    applyFilter(Filter)
        Applies a scientific filter on the audio signal
    plotWaveform(unit='sec', newfig=False, title='')
        Displays a graph with the waveform of the audio signal
    extractWaveformSnippet(chunk)
        Extract a chunk of the waveform as a new Sound object
    tightenWavformWindow(EnergyPercentage)
        Crops the start and end times of a waveform in a Sound object to reduce
        "silences"
    getFileDur_samples()
        Returns the number of samples of the sound file
    getFileDur_sec()
        Returns the duration of the sound file in seconds
    getFileExtension()
        Returns the extension of the audio file
    getFileName()
        Returns the name of teh audio file
    getFilePath()
        Returns the path of the audio file
    getFullPath()
        Returns the path, filename, and extension of the audio file
    getFilterParameters()
        Returns the frequencies and type of filter used (if any)
    getFilterStatus()
        Indicates if a filter has alreday been applied to the sound
    getNbChannels()
        Returns the number of channels of the sound file
    getSelectedChannel()
        Returns the channel currently selected
    getSamplingFrequencyHz()
        Returns the sampling frequency of the sound
    getWaveform()
        Returns the waveform of the sound object. Need to use the read
        method first
    getWaveformDur_sec()
        Indicates the duration of the sound that was read
    getWaveformStartSample()
        Indicates the number of the first sample of the sound read relative
        to the beginning of the entire sound file
    getWaveformEndSample()
        Indicates the number of the last sample of the sound read relative
        to the beginning of the entire sound file

    """

    def __init__(self, infile):
        if os.path.isfile(infile):
            myfile = sf.SoundFile(infile)
            self.nSamples = myfile.seek(0, sf.SEEK_END)
            self.fs = myfile.samplerate
            self.dur = self.nSamples/self.fs
            self.nChannels = myfile.channels
            self.selectedChannel = []
            self.filePath = os.path.dirname(infile)
            # self.fullPath = infile
            self.fileName = os.path.basename(os.path.splitext(infile)[0])
            self.fileExtension = os.path.splitext(infile)[1]
            self.filterApplied = False
            myfile.close()
        else:
            raise ValueError("The sound file can't be found. Please verify the\
                             sound file name and path")

    def read(self, channel=0, chunk=[]):
        # check that the channel id is valid
        if (channel >= 0) & (channel <= self.nChannels - 1):
            if len(chunk) == 0:  # read the entire file
                sig, fs = sf.read(self.getFullPath(), always_2d=True)
                self.waveform = sig[:, channel]
                self.waveformStartSample = 0
                self.waveformEndSample = self.getFileDur_samples()-1
                self.waveformDur_samples = len(self.waveform)
                self.waveformDur_sec = self.waveformDur_samples/fs
            else:
                if len(chunk) == 2:  # only read a section of the file
                    # Validate input values
                    if (chunk[0] < 0) | (chunk[0] >= self.getFileDur_samples()):
                        raise ValueError('Invalid chunk start value. The sample\
                                         value chunk[0] is outside of the file\
                                         limit.')
                    elif (chunk[1] < 0) | (chunk[1] > self.getFileDur_samples()):
                        raise ValueError('Invalid chunk stop value. The sample\
                                         value chunk[1] is outside of file limit.')
                    elif chunk[1] <= chunk[0]:
                        raise ValueError('Invalid chunk values. chunk[1] must \
                                         be greater than chunk[0]')
                    # read data
                    sig, fs = sf.read(self.getFullPath(), start=chunk[0],
                                      stop=chunk[1], always_2d=True)
                    self.waveform = sig[:, channel]
                    self.waveformStartSample = chunk[0]
                    self.waveformEndSample = chunk[1]
                    self.waveformDur_samples = len(self.waveform)
                    self.waveformDur_sec = self.waveformDur_samples/fs
                else:
                    raise ValueError('Invalid chunk values. The argument chunk\
                                     must be a list of 2 elements.')
            self.selectedChannel = channel

        else:
            msg = 'Channel ' + str(channel) + ' does not exist ('
            + str(self.nChannels) + ' channels available).'
            raise ValueError(msg)

    def applyFilter(self, Filter):
        if self.filterApplied is False:
            b, a = self.calcFilterCoef(Filter)
            sigFilt = spsig.lfilter(b, a, self.waveform)
            self.waveform = sigFilt
            self.filterApplied = True
            self.filterParameters = {
                    'type': Filter.type,
                    'freqs': Filter.freqs,
                    'order': Filter.order}
        else:
            raise ValueError('This signal has been filtered already. Cannot\
                             filter twice.')

    def calcFilterCoef(self, Filter):
        nyq = 0.5 * self.getSamplingFrequencyHz()
        if Filter.type == 'bandpass':
            low = Filter.freqs[0] / nyq
            high = Filter.freqs[1] / nyq
            b, a = spsig.butter(Filter.order, [low, high], btype='band')
        elif Filter.type == 'lowpass':
            b, a = spsig.butter(Filter.order, Filter.freqs[0]/nyq, 'low')
        elif Filter.type == 'highpass':
            b, a = spsig.butter(Filter.order, Filter.freqs[0]/nyq, 'high')
        return b, a

    def plotWaveform(self, unit='sec', newfig=False, title=''):
        if len(self.waveform) == 0:
            raise ValueError('Cannot plot, waveform data enpty. Use Sound.read\
                             to load the waveform')
        if unit == 'sec':
            axis_t = np.arange(0, len(self.waveform)/self.fs, 1/self.fs)
            xlabel = 'Time (sec)'
        elif unit == 'samp':
            axis_t = np.arange(0, len(self.waveform), 1)
            xlabel = 'Time (sample)'
        if newfig:
            plt.figure()
        axis_t = axis_t[0:len(self.waveform)]
        plt.plot(axis_t, self.waveform, color='black')
        plt.xlabel(xlabel)
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.axis([axis_t[0], axis_t[-1], min(self.waveform), max(self.waveform)])
        plt.grid()
        plt.show()

    def extractWaveformSnippet(self, chunk):
        if len(chunk) != 2:
            raise ValueError('Chunk should be a list of with 2 values: \
                             chunk=[t1, t2].')
        elif chunk[0] >= chunk[1]:
            raise ValueError('Chunk[0] should be greater than chunk[1].')
        elif (chunk[0] < 0) | (chunk[0] > self.getFileDur_samples()):
            raise ValueError('Invalid chunk start value. The sample value \
                             chunk[0] is outside of file limit.')
        elif (chunk[1] < 0) | (chunk[1] > self.getFileDur_samples()):
            raise ValueError('Invalid chunk stop value. The sample value \
                             chunk[1] is outside of file limit.')
        snippet = copy.deepcopy(self)
        snippet.waveform = self.waveform[chunk[0]:chunk[1]]
        snippet.waveformEndSample = snippet.waveformEndSample + chunk[1]
        snippet.waveformStartSample = snippet.waveformStartSample + chunk[0]
        snippet.waveformDur_samples = len(snippet.waveform)
        snippet.waveformDur_sec = snippet.waveformDur_samples/snippet.getSamplingFrequencyHz() 
        return snippet

    def tightenWavformWindow(self, EnergyPercentage):
        cumEn = np.cumsum(np.square(self.waveform))
        cumEn = cumEn/max(cumEn)
        begPerc = (1-(EnergyPercentage/100))/2
        endPerc = 1 - begPerc
        chunk = [np.nonzero(cumEn > begPerc)[0][0], np.nonzero(cumEn > endPerc)[0][0]]
        print(chunk)
        # extractWaveformSnippet(self, chunk)
        snip = self.extractWaveformSnippet(chunk)
        self.__dict__.update(snip.__dict__)
        # plt.figure()
        # plt.plot(snip.waveform)
        # self = copy.snip

    def getSamplingFrequencyHz(self):
        return self.fs

    def getFileDur_samples(self):
        return self.nSamples

    def getFileDur_sec(self):
        return self.dur

    def getNbChannels(self):
        return self.nChannels

    def getSelectedChannel(self):
        return self.selectedChannel

    def getFilePath(self):
        return self.filePath

    def getFullPath(self):
        return os.path.join(self.filePath, self.fileName) + self.fileExtension

    def getFileExtension(self):
        return self.fileExtension

    def getFileName(self):
        return self.fileName

    def getWaveform(self):
        return self.waveform

    def getWaveformStartSample(self):
        return self.waveformStartSample

    def getWaveformEndSample(self):
        return self.waveformEndSample

    def getWaveformDur_samples(self):
        return self.waveformDur_samples

    def getWaveformDur_sec(self):
        return self.waveformDur_sec

    def getFilterParameters(self):
        if self.filterApplied:
            out = self.filterParameters
        else:
            out = []
        return out

    def getFilterStatus(self):
        return self.filterApplied

    # def play(self, channel):

    # def resample(self, channel):


class Filter:
    """
    Class to define a scientific filter object

    Attributes
    ----------
    type : str
        A formatted string providing the path and filename of the sound file
    freqs : list
        List with one or 2 elements defining the cut-off frequencies in Hz of
        the selected filter
    order : int
        Order of the filter

    """

    def __init__(self, type, freqs, order=4):
        """
        Parameters
        ----------
        type : {'bandpass', 'lowpass', 'highpass'}
            Type of filter
        freqs : list of float
            Cut-off frequencies of the filter sorted in increasing order (i.e.
            [lowcut, highcut]). If the filter type is 'bandpass' then freqs
            must be a list of 2 floats freqs=[lowcut, highcut], where lowcut <
            highcut. If the filter type is 'lowpass' or 'highpass' then freqs
            is a list with a single float.
        order : int, optional
            Order of the filter (default is 4)

        Raises
        ------
        ValueError
            If the filter type is not set to 'bandpass', 'lowpass', or 'highpass'
            If the freqs has not enough of too much values for the filter type
            selected or are not sorted by increasing frequencies.

        """

        # chech filter type
        if (type == 'bandpass') | (type == 'lowpass') | (type == 'highpass') == 0:
            raise ValueError('Wrong filter type. Must be "bandpass", "lowpass"\
                             , or "highpass".')
        # chech freq values
        if (type == 'bandpass'):
            if len(freqs) != 2:
                raise ValueError('The type "bandpass" requires two frepuency\
                                 values: freqs=[lowcut, highcut].')
            elif freqs[0] > freqs[1]:
                raise ValueError('The lowcut value should be smaller than the\
                                 highcut value: freqs=[lowcut, highcut].')
        elif (type == 'lowpass') | (type == 'highpass'):
            if len(freqs) != 1:
                raise ValueError('The type "lowpass" and "highpass" require \
                                 one frepuency values freqs=[cutfreq].')
        self.type = type
        self.freqs = freqs
        self.order = order


def normalizeVector(vec):
    # vec = vec+abs(min(vec))
    # normVec = vec/max(vec)
    # normVec = (normVec - 0.5)*2
    vec = vec - np.mean(vec)
    normVec = vec/max(vec)
    return normVec
