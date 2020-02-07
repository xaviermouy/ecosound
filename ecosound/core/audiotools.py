# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:26:24 2017

@author: xavier
"""
# --------------------------------------------------------------
##TODO: resample waveform
##TODO: play sound
# --------------------------------------------------------------

import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spsig
import copy
import core.tools


class Sound:
    """
    A class to load and manipulate a sound file

    Attributes
    ----------
    infile : str
        A formatted string providing the path and filename of the sound file

    Methods
    -------
    read(channel=0, chunk=[])
        Reads a sound file with the option to select a specific channel and
        read only a section of the file.
    filter(Filter)
        Applies a scientific filter on the audio signal
    plot_waveform(unit='sec', newfig=False, title='')
        Displays a graph with the waveform of the audio signal
    select_snippet(chunk)
        Extract a chunk of the waveform as a new Sound object
    tighten_waveform_window(energy_percentage)
        Crops the start and end times of a waveform in a Sound object to reduce
        "silences"
    file_duration_sample()
        Returns the number of samples of the sound file
    file_duration_sec()
        Returns the duration of the sound file in seconds
    file_extension()
        Returns the extension of the audio file
    file_name()
        Returns the name of teh audio file
    file_dir()
        Returns the path of the audio file
    file_full_path()
        Returns the path, filename, and extension of the audio file
    getFilterParameters()
        Returns the frequencies and type of filter used (if any)
    filter_applied()
        Indicates if a filter has alreday been applied to the sound
    channels()
        Returns the number of channels of the sound file
    channel_selected()
        Returns the channel currently selected
    sampling_frequency()
        Returns the sampling frequency of the sound
    waveform()
        Returns the waveform of the sound object. Need to use the read
        method first
    waveform_duration_sec()
        Indicates the duration of the sound that was read
    waveform_start_sample()
        Indicates the number of the first sample of the sound read relative
        to the beginning of the entire sound file
    waveform_stop_sample()
        Indicates the number of the last sample of the sound read relative
        to the beginning of the entire sound file

    """

    def __init__(self, infile):
        if os.path.isfile(infile):
            myfile = sf.SoundFile(infile)
            self._file_duration_sample = myfile.seek(0, sf.SEEK_END)
            self._file_sampling_frequency = myfile.samplerate
            self._file_duration_sec = self._file_duration_sample / \
                self._file_sampling_frequency
            self._channels = myfile.channels
            self._channel_selected = []
            self._file_dir = os.path.dirname(infile)
            self._file_name = os.path.basename(os.path.splitext(infile)[0])
            self._file_extension = os.path.splitext(infile)[1]
            self._filter_applied = False
            self._waveform = []
            self._waveform_start_sample = []
            self._waveform_stop_sample = []
            self._waveform_duration_sample = 0
            self._waveform_duration_sec = 0
            self._waveform_sampling_frequency = self._file_sampling_frequency
            
            myfile.close()
        else:
            raise ValueError("The sound file can't be found. Please verify"
                             + ' sound file name and path')

    def read(self, channel=0, chunk=[]):
 
        # check that the channel id is valid
        if (channel >= 0) & (channel <= self._channels - 1):
            if len(chunk) == 0:  # read the entire file
                sig, fs = sf.read(self.file_full_path(), always_2d=True)
                self._waveform = sig[:, channel]
                self._waveform_start_sample = 0
                self._waveform_stop_sample = self.file_duration_sample-1
                self._waveform_duration_sample = len(self._waveform)
                self._waveform_duration_sec = self._waveform_duration_sample/fs
            else:
                if len(chunk) == 2:  # only read a section of the file
                    # Validate input values
                    if (chunk[0] < 0) | (chunk[0] >= self.file_duration_sample):
                        raise ValueError('Invalid chunk start value. The sample'
                                         +' value chunk[0] is outside of the'
                                         +' file limits.')
                    elif (chunk[1] < 0) | (chunk[1] > self.file_duration_sample):
                        raise ValueError('Invalid chunk stop value. The sample'
                                         + ' value chunk[1] is outside of the'
                                         + ' file limits.')
                    elif chunk[1] <= chunk[0]:
                        raise ValueError('Invalid chunk values. chunk[1] must'
                                         + ' be greater than chunk[0]')
                    # read data
                    sig, fs = sf.read(self.file_full_path, start=chunk[0],
                                      stop=chunk[1], always_2d=True)
                    self._waveform = sig[:, channel]
                    self._waveform_start_sample = chunk[0]
                    self._waveform_stop_sample = chunk[1]
                    self._waveform_duration_sample = len(self._waveform)
                    self._waveform_duration_sec = self._waveform_duration_sample/fs
                else:
                    raise ValueError('Invalid chunk values. The argument chunk'
                                     + ' must be a list of 2 elements.')
            self._channel_selected = channel

        else:
            msg = ''.join(['Channel ', str(channel), ' does not exist (',
                           str(self._channels), ' channels available).'])
            raise ValueError(msg)

    def filter(self, filter_type, cutoff_frequencies, order=4):
        if self._filter_applied is False:
            my_filter = Filter(filter_type, cutoff_frequencies, order)
            self._waveform = my_filter.apply(self._waveform,
                                             self._waveform_sampling_frequency)
            self._filter_applied = True
            self._filter_params = my_filter
        else:
            raise ValueError('This signal has been filtered already. Cannot'
                             + ' filter twice.')

    def plot_waveform(self, unit='sec', newfig=False, title=''):
        if len(self._waveform) == 0:
            raise ValueError('Cannot plot, waveform data enpty. Use Sound.read'
                             + ' to load the waveform')
        if unit == 'sec':
            axis_t = np.arange(0, len(self._waveform)
                               /self._waveform_sampling_frequency, 1
                               /self._waveform_sampling_frequency)
            xlabel = 'Time (sec)'
        elif unit == 'samp':
            axis_t = np.arange(0, len(self._waveform), 1)
            xlabel = 'Time (sample)'
        if newfig:
            plt.figure()
        axis_t = axis_t[0:len(self._waveform)]
        plt.plot(axis_t, self._waveform, color='black')
        plt.xlabel(xlabel)
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.axis([axis_t[0], axis_t[-1],
                  min(self._waveform),
                  max(self._waveform)])
        plt.grid()
        plt.show()

    def select_snippet(self, chunk):
        if len(chunk) != 2:
            raise ValueError('Chunk should be a list of with 2 values: '
                             + 'chunk=[t1, t2].')
        elif chunk[0] >= chunk[1]:
            raise ValueError('Chunk[0] should be greater than chunk[1].')
        elif (chunk[0] < 0) | (chunk[0] > self.file_duration_sample):
            raise ValueError('Invalid chunk start value. The sample value '
                             + 'chunk[0] is outside of file limit.')
        elif (chunk[1] < 0) | (chunk[1] > self.file_duration_sample):
            raise ValueError('Invalid chunk stop value. The sample value '
                             + 'chunk[1] is outside of file limit.')
        snippet = copy.deepcopy(self)
        snippet._waveform = self._waveform[chunk[0]:chunk[1]]
        snippet._waveform_stop_sample = snippet._waveform_start_sample + chunk[1]
        snippet._waveform_start_sample = snippet._waveform_start_sample + chunk[0]
        snippet._waveform_duration_sample = len(snippet._waveform)
        snippet._waveform_duration_sec = snippet._waveform_duration_sec / snippet._waveform_sampling_frequency
        return snippet


    def tighten_waveform_window(self, energy_percentage):
        chunk = core.tools.tighten_signal_limits(self._waveform, energy_percentage)
        # select_snippet(self, chunk)
        snip = self.select_snippet(chunk)
        self.__dict__.update(snip.__dict__)

    def __len__(self):
        """Return number of samples."""
        return self.waveform_duration_sample

    @property
    def waveform_sampling_frequency(self):
        return self._waveform_sampling_frequency
    
    @property
    def file_sampling_frequency(self):
        return self._file_sampling_frequency

    @property
    def file_duration_sample(self):
        return self._file_duration_sample

    @property
    def file_duration_sec(self):
        return self._file_duration_sec

    @property
    def channels(self):
        return self._channels

    @property
    def channel_selected(self):
        return self._channel_selected

    @property
    def file_dir(self):
        return self._file_dir
    
    @property
    def file_full_path(self):
        return os.path.join(self._file_dir, self._file_name) + self._file_extension
    
    @property
    def file_extension(self):
        return self._file_extension
    
    @property
    def file_name(self):
        return self._file_name
    @property
    def waveform(self):
        return self._waveform
    
    @property
    def waveform_start_sample(self):
        return self._waveform_start_sample

    @property
    def waveform_stop_sample(self):
        return self._waveform_stop_sample

    @property
    def waveform_duration_sample(self):
        return self._waveform_duration_sample

    @property
    def waveform_duration_sec(self):
        return self._waveform_duration_sec

    @property
    def filter_parameters(self):
        if self._filter_applied:
            out = self._filter_params
        else:
            out = None
        return out

    @property
    def filter_applied(self):
        return self._filter_applied


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

    def __init__(self, type, cutoff_frequencies, order=4):
        """
        Parameters
        ----------
        type : {'bandpass', 'lowpass', 'highpass'}
            Type of filter
        cutoff_frequencies : list of float
            Cut-off frequencies of the filter sorted in increasing order (i.e.
            [lowcut, highcut]). If the filter type is 'bandpass' then 
            cutoff_frequencies must be a list of 2 floats 
            cutoff_frequencies=[lowcut, highcut], where lowcut < highcut. 
            If the filter type is 'lowpass' or 'highpass' then cutoff_frequencies
            is a list with a single float.
        order : int, optional
            Order of the filter (default is 4)

        Raises
        ------
        ValueError
            If the filter type is not set to 'bandpass', 'lowpass', or 'highpass'
            If the cutoff_frequencies has not enough of too much values for the
            filter type selected or are not sorted by increasing frequencies.

        """

        # chech filter type
        if (type == 'bandpass') | (type == 'lowpass') | (type == 'highpass') == 0:
            raise ValueError('Wrong filter type. Must be "bandpass", "lowpass"' 
                             +', or "highpass".')
        # chech freq values
        if (type == 'bandpass'):
            if len(cutoff_frequencies) != 2:
                raise ValueError('The type "bandpass" requires two frepuency '
                                 + 'values: cutoff_frequencies=[lowcut, '
                                 + 'highcut].')
            elif cutoff_frequencies[0] > cutoff_frequencies[1]:
                raise ValueError('The lowcut value should be smaller than the '
                                 + 'highcut value: cutoff_frequencies=[lowcut,'
                                 + ' highcut].')
        elif (type == 'lowpass') | (type == 'highpass'):
            if len(cutoff_frequencies) != 1:
                raise ValueError('The type "lowpass" and "highpass" require '
                                 + 'one frepuency values cutoff_frequencies='
                                 + '[cutfreq].')
        self.type = type
        self.cutoff_frequencies = cutoff_frequencies
        self.order = order
    
    def apply(self, waveform, sampling_frequency):
        b, a = self.coefficients(sampling_frequency)
        return spsig.lfilter(b, a, waveform)
    
    def coefficients(self, sampling_frequency):
        nyquist = 0.5 * sampling_frequency
        if self.type == 'bandpass':
            low = self.cutoff_frequencies[0] / nyquist
            high = self.cutoff_frequencies[1] / nyquist
            b, a = spsig.butter(self.order, [low, high], btype='band')
        elif self.type == 'lowpass':
            b, a = spsig.butter(self.order,
                                self.cutoff_frequencies[0]/nyquist, 'low')
        elif self.type == 'highpass':
            b, a = spsig.butter(self.order,
                                self.cutoff_frequencies[0]/nyquist, 'high')
        return b, a
