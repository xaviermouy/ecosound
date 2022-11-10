# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:26:24 2017

@author: xavier.mouy
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
import scipy
import copy
import ecosound.core.tools


class Sound:
    """
    A class to load and manipulate a sound file

    This class can load data from an entire, or part of a, sound file, filter
    the loaded data, select subsections, and plot the waveform. Currently a
    Sound object can only load data from one channel at a time.

    Attributes
    ----------
    file_full_path : str
        Path of the sound file, including file name and extension.
    file_dir : str
        Path of the sound file directory.
    file_name : str
        Name of the sound file withjout teh extension.
    file_extension : str
        Extension of the sound file (e.g. ".wav").
    file_sampling_frequency : int
        Sampling frequency of the original sound data, in Hz.
    file_duration_sample : float
        Duration of the sound data from the file, in samples.
    file_duration_sec : float
        Duration of the sound data from the file, in seconds.
    channels : int
        Number of channels available in the sound file.
    channel_selected : int
        Channel from which the waveform data was loaded from.
    waveform : numpy.ndarray
        Waveform of the loaded data for the selected channel (channel_selected)
        and time frame selected.
    waveform_sampling_frequency : float
        Sampling frequency of the loaded waveform data. It can differ from
        file_sampling_frequency if the waveform was up- or down- sampled.
    waveform_start_sample : float
        Index of the first sample of the loaded waveform data relative to the
        begining of the sound file.
    waveform_stop_sample : float
        Index of the last sample of the loaded waveform data relative to the
        begining of the sound file.
    waveform_duration_sample : float
        Duration of the loaded waveform data, in samples.
    waveform_duration_sec : float
        Duration of the loaded waveform data, in seconds.
    filter_applied : bool
        True if the waveform data was filtered.
    filter_parameters : Filter obj
        Filter object with all filter paramters and coefficients. Empty if no
        filter was applied.

    Methods
    -------
    read(channel=0, chunk=[])
        Reads a sound file with the option to select a specific channel and
        read only a section of the file.
    filter(filter_type, cutoff_frequencies, order=4)
        Applies a scientific filter on the audio signal
    plot_waveform(unit='sec', newfig=False, title='')
        Displays a graph with the waveform of the audio signal
    select_snippet(chunk)
        Extract a chunk of the waveform as a new Sound object
    tighten_waveform_window(energy_percentage)
        Crops the beginning  and end times of a waveform in a Sound object
        based on a percentage of energy.
    upsample(resolution_sec)
        upsample the waveform to a time resolution of resolution_sec.
    decimate(new_sampling_frequency, filter_order=8, filter_type="iir")
        Decimate waveform.
    normalize()
        Normalize max amplitude of waveform to 1.

    """

    def __init__(self, infile):
        """
        Initialize Sound object.

        Parameters
        ----------
        infile : str
            Path of the sound file.

        Raises
        ------
        ValueError
            If sound file can't be found.

        Returns
        -------
        Sound object.

        """
        if os.path.isfile(infile):
            myfile = sf.SoundFile(infile)
            self._file_duration_sample = myfile.seek(0, sf.SEEK_END)
            self._file_sampling_frequency = myfile.samplerate
            self._file_duration_sec = (
                self._file_duration_sample / self._file_sampling_frequency
            )
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
            self.detrended = []
            myfile.close()
        else:
            raise ValueError(
                "The sound file can't be found. Please verify"
                + " sound file name and path"
            )

    def detrend(self):
        self._waveform = self._waveform - np.mean(self._waveform)

    def write(
        self,
        outfilename,
        subtype="PCM_24",
        endian=None,
        format=None,
        closefd=True,
    ):
        sf.write(
            outfilename,
            self.waveform,
            int(self.waveform_sampling_frequency),
            subtype=subtype,
            endian=endian,
            format=format,
            closefd=closefd,
        )

    def read(self, channel=0, chunk=[], unit="samp", detrend=False):
        """
        Load data from sound file.

        Load data from a sound file with the option to select a specific
        channel and load only a section of the file. Data are loaded as a numpy
        arrayin in the object attribute "waveform".

        Parameters
        ----------
        channel : int, optional
            ID of the audio channel to load. The default is 0.
        chunk : list, optional
            List with two floats indicating the [start time, stop time], in
            samples, of the chunk of audio data to load. An empty list []
            loads data from the entire audio file. The default is [].
        unit : str, optional
            Time unit of the 'chunk' parameter. Can be set to 'sec' for seconds
            or 'samp', for samples. The default is 'samp'.
        detrend : bool, optional
            Remove DC offset of the waveform by subtracting the mean. The
            default is False.

        Raises
        ------
        ValueError
            If the chunk list has only 1 value.
            If the first value in the chunk list is greater or equal to the
               second one.
            If values in the chunk list exceed the audio file limits.
            If the channel selected does not exist.
            If samp is not set to 'samp' or 'sec'

        Returns
        -------
        None. Load audio data in the waveform attribute and update all waveform
        related attributes.

        """
        # check that the channel id is valid
        if (channel >= 0) & (channel <= self._channels - 1):
            if len(chunk) == 0:  # read the entire file
                sig, fs = sf.read(self.file_full_path, always_2d=True)
                self._waveform = sig[:, channel]
                self._waveform_start_sample = 0
                self._waveform_stop_sample = self.file_duration_sample - 1
                self._waveform_duration_sample = len(self._waveform)
                self._waveform_duration_sec = (
                    self._waveform_duration_sample / fs
                )
            else:
                if unit not in ("samp", "sec"):
                    raise ValueError(
                        'Invalid unit. Should be set to "sec" or' + '"samp".'
                    )
                # convert chunk to sampels if needed
                if unit in ("sec"):
                    chunk = np.round(
                        np.dot(chunk, self.waveform_sampling_frequency)
                    )
                if len(chunk) == 2:  # only read a section of the file
                    # Validate input values
                    if (chunk[0] < 0) | (
                        chunk[0] >= self.file_duration_sample
                    ):
                        raise ValueError(
                            "Invalid chunk start value. The sample"
                            + " value chunk[0] is outside of the"
                            + " file limits."
                        )
                    elif (chunk[1] < 0) | (
                        chunk[1] > self.file_duration_sample
                    ):
                        raise ValueError(
                            "Invalid chunk stop value. The sample"
                            + " value chunk[1] is outside of the"
                            + " file limits."
                        )
                    elif chunk[1] <= chunk[0]:
                        raise ValueError(
                            "Invalid chunk values. chunk[1] must"
                            + " be greater than chunk[0]"
                        )
                    # read data
                    sig, fs = sf.read(
                        self.file_full_path,
                        start=int(chunk[0]),
                        stop=int(chunk[1]),
                        always_2d=True,
                    )
                    self._waveform = sig[:, channel]
                    self._waveform_start_sample = chunk[0]
                    self._waveform_stop_sample = chunk[1]
                    self._waveform_duration_sample = len(self._waveform)
                    self._waveform_duration_sec = (
                        self._waveform_duration_sample / fs
                    )
                else:
                    raise ValueError(
                        "Invalid chunk values. The argument chunk"
                        + " must be a list of 2 elements."
                    )
            self._channel_selected = channel
            if detrend:  # removes DC offset
                self.detrend()
                # self._waveform = self._waveform - np.mean(self._waveform)

        else:
            msg = "".join(
                [
                    "Channel ",
                    str(channel),
                    " does not exist (",
                    str(self._channels),
                    " channels available).",
                ]
            )
            raise ValueError(msg)

    def filter(self, filter_type, cutoff_frequencies, order=4, verbose=True):
        """
        Filter the audio signal.

        Applies low-pass, high-pass, or band-pass scientific filter to the
        audio signal. The attribute waveform is updated with the filtered
        signal. The same data can only be filtered once.

        Parameters
        ----------
        filter_type : str
            Type of filter. Can be set to 'bandpass', 'lowpass' or 'highpass'.
        cutoff_frequencies : list
            Cutoff frequencies of the filter, in Hz (float). Must be a list with a
            single float if the filter_type is set to 'lowpass' or 'highpass'.
            Must be a list with two float values (i.e.,[fmin, fmax]) if the
            filter_type is set to 'bandpass'.
        order : int, optional
            Order of the filter. The default is 4.
        verbose : bool, optional
            if True, prints all notifications. The default is True.

        Raises
        ------
        ValueError
            If signal is filtered more than once.
            If the waveform attribute is empty
            If the filter type is not set to 'bandpass', 'lowpass', or 'highpass'
            If the cutoff_frequencies has not enough, or too much values for
            the filter type selected.
            If the values in cutoff_frequencies are not sorted by increasing
            frequencies.

        Returns
        -------
        None. Filtered signal in the 'waveform' attribute of the Sound object.
        """
        if self._filter_applied is False:

            # check bandpass cuttoff freq and switch to lowpass.highpass if necessary
            if (filter_type == "bandpass") and (min(cutoff_frequencies) <= 0):
                cutoff_frequencies = [max(cutoff_frequencies)]
                filter_type = "lowpass"
                if verbose:
                    print(
                        'Warning: filter type was changed from "bandpass" to "lowpass".'
                    )
            if (filter_type == "bandpass") and (
                max(cutoff_frequencies)
                >= self._waveform_sampling_frequency / 2
            ):
                cutoff_frequencies = [min(cutoff_frequencies)]
                filter_type = "highpass"
                if verbose:
                    print(
                        'Warning: filter type was changed from "bandpass" to "highpass".'
                    )
            # Instantiate filter object
            my_filter = Filter(filter_type, cutoff_frequencies, order)
            self._waveform = my_filter.apply(
                self._waveform, self._waveform_sampling_frequency
            )
            self._filter_applied = True
            self._filter_params = my_filter
        else:
            raise ValueError(
                "This signal has been filtered already. Cannot"
                + " filter twice."
            )

    def upsample(self, resolution_sec):
        """
        Upsample  waveform

        Increase the number of samples in the waveform and interpolate.

        Parameters
        ----------
        resolution_sec : float
            Sample resolution of the upsampled waveform, in second. The new
            sampling frequency will be 1/resolution_sec.

        Returns
        -------
        None. Updates the waveform and sampling frequency of the Sound object.

        """
        self._waveform, self._waveform_sampling_frequency = upsample(
            self._waveform,
            1 / self._waveform_sampling_frequency,
            resolution_sec,
        )
        self._waveform_duration_sec = (
            len(self._waveform) / self._waveform_sampling_frequency
        )
        self._waveform_duration_sample = (
            self._waveform_duration_sec * self._waveform_sampling_frequency
        )

    def decimate(
        self, new_sampling_frequency, filter_order=8, filter_type="iir"
    ):
        """
        Decimate  waveform

        Filter and reduce the number of samples in the waveform.

        Parameters
        ----------
        new_sampling_frequency : float
            Sampling frequency requested, in Hz.
        filter_order : int, optional
            Order of the low-pass filter to use. The default is 8.
        filter_type : str, optional
            Type of low-pass filter to use. The default is 'iir'.

        Returns
        -------
        None. Updates the waveform and sampling frequency of the Sound object.

        """

        # downsample to user-defined sampling rate
        downsampling_factor = int(
            np.round(self.waveform_sampling_frequency / new_sampling_frequency)
        )

        # decimate signal (the cutoff frequency of the filter is 0.8 x new_sampling_frequency)
        sig_decimated = scipy.signal.decimate(
            self.waveform,
            downsampling_factor,
            n=filter_order,
            ftype=filter_type,
            axis=0,
            zero_phase=True,
        )
        # update object
        self._waveform = sig_decimated
        self._waveform_sampling_frequency = (
            self.waveform_sampling_frequency / downsampling_factor
        )
        self._waveform_duration_sec = (
            len(sig_decimated) / self._waveform_sampling_frequency
        )
        self._waveform_duration_sample = (
            self._waveform_duration_sec * self._waveform_sampling_frequency
        )

    def normalize(self, method="amplitude"):
        if method == "amplitude":
            self._waveform = self._waveform / np.max(self._waveform)

    def plot(
        self,
        unit="sec",
        newfig=False,
        label=[],
        linestyle="-",
        marker="",
        color="black",
        title="",
    ):
        """
        Plot waveform of the audio signal.

        PLots the waveform of the audio signal. Both the plot title and time
        units can be asjusted. The plot can be displayed on a new or an
        existing figure.

        Parameters
        ----------
        unit : str, optional
            Time units to use. Can be either 'sec' for seconds, or 'samp' for
            samples. The default is 'sec'.
        newfig : bool, optional
            PLots on a new figure if set to True. The default is False.
        title : str, optional
            Title of the plot. The default is ''.
        linestyle : str, optional
            Linestyle of the plot. The default is '-'.
        marker : str, optional
            Marker of the plot. The default is '-'.

        Raises
        ------
        ValueError
            If the waveform attribute is empty.

        Returns
        -------
        None.

        """
        if len(self._waveform) == 0:
            raise ValueError(
                "Cannot plot, waveform data enpty. Use Sound.read"
                + " to load the waveform"
            )
        if unit == "sec":
            axis_t = np.arange(
                0,
                len(self._waveform) / self._waveform_sampling_frequency,
                1 / self._waveform_sampling_frequency,
            )
            xlabel = "Time (sec)"
        elif unit == "samp":
            axis_t = np.arange(0, len(self._waveform), 1)
            xlabel = "Time (sample)"
        if newfig:
            plt.figure()
        axis_t = axis_t[0 : len(self._waveform)]
        plt.plot(
            axis_t,
            self._waveform,
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
        plt.xlabel(xlabel)
        plt.ylabel("Amplitude")
        plt.title(title)
        plt.axis(
            [axis_t[0], axis_t[-1], min(self._waveform), max(self._waveform)]
        )
        plt.grid()
        plt.show()

    def select_snippet(self, chunk, unit="samp"):
        """
        Select section of the loaded waveform.

        Create a new Sound object from a section of the sound data laoded.

        Parameters
        ----------
        chunk : list
            List of two int values representing the [start time, stop time] of
            the sound data to select. Start time must be smaller than stop
            time.
        unit : str, optional
            Time unit of the 'chunk' parameter. Can be set to 'sec' for seconds
            or 'samp', for samples. The default is 'samp'.

        Raises
        ------
        ValueError
            If chunk has only one value
            If the start time is greater tahn the stop time
            If the start or stop times fall outside of the wavform limits.

        Returns
        -------
        snippet : Sound obj
            Sound object with the selected audio data.

        """
        if len(chunk) != 2:
            raise ValueError(
                "Chunk should be a list of with 2 values: " + "chunk=[t1, t2]."
            )
        elif unit not in ("samp", "sec"):
            raise ValueError('Invalid unit. Should be set to "sec" or "samp".')
        elif chunk[0] >= chunk[1]:
            raise ValueError("Chunk[0] should be greater than chunk[1].")

        if unit == "sec":
            chunk[0] = int(
                np.floor(chunk[0] * self.waveform_sampling_frequency)
            )
            chunk[1] = int(
                np.ceil(chunk[1] * self.waveform_sampling_frequency)
            )

        if (chunk[0] < 0) | (chunk[0] > self.file_duration_sample):
            raise ValueError(
                "Invalid chunk start value. The start value "
                + "chunk[0] is outside of file limit."
            )
        elif (chunk[1] < 0) | (chunk[1] > self.file_duration_sample):
            raise ValueError(
                "Invalid chunk stop value. The stop value "
                + "chunk[1] is outside of file limit."
            )

        snippet = copy.deepcopy(self)
        snippet._waveform = self._waveform[chunk[0] : chunk[1]]
        snippet._waveform_stop_sample = (
            snippet._waveform_start_sample + chunk[1]
        )
        snippet._waveform_start_sample = (
            snippet._waveform_start_sample + chunk[0]
        )
        snippet._waveform_duration_sample = len(snippet._waveform)
        snippet._waveform_duration_sec = (
            snippet._waveform_duration_sec
            / snippet._waveform_sampling_frequency
        )
        return snippet

    def tighten_waveform_window(self, energy_percentage):
        """
        Adjust waveform window.

        Crops the begining and end of the waveform to only capture the most
        intense part of the signal (i.e., with most energy). The percentage of
        energy is defined by the energy_percentage parameter. The attribute
        'waveform' and all its related attricbutes are updated automatically.

        Parameters
        ----------
        energy_percentage : float
            Percentage of the energy the updated waveform should have.

        Returns
        -------
        None. Updates the 'waveform' attribute alomg with all the waveform
        -related attributes.

        """
        chunk = ecosound.core.tools.tighten_signal_limits(
            self._waveform, energy_percentage
        )
        snip = self.select_snippet(chunk)
        self.__dict__.update(snip.__dict__)

    def __len__(self):
        """Return number of samples of the waveform."""
        return self.waveform_duration_sample

    @property
    def waveform_sampling_frequency(self):
        """Return the waveform_sampling_frequency attribute."""
        return self._waveform_sampling_frequency

    @property
    def file_sampling_frequency(self):
        """Return the file_sampling_frequency attribute."""
        return self._file_sampling_frequency

    @property
    def file_duration_sample(self):
        """Return the file_duration_sample attribute."""
        return self._file_duration_sample

    @property
    def file_duration_sec(self):
        """Return the file_duration_sec attribute."""
        return self._file_duration_sec

    @property
    def channels(self):
        """Return the channels attribute."""
        return self._channels

    @property
    def channel_selected(self):
        """Return the channel_selected attribute."""
        return self._channel_selected

    @property
    def file_dir(self):
        """Return the file_dir attribute."""
        return self._file_dir

    @property
    def file_full_path(self):
        """Return the file_full_path attribute."""
        return (
            os.path.join(self._file_dir, self._file_name)
            + self._file_extension
        )

    @property
    def file_extension(self):
        """Return the file_extension attribute."""
        return self._file_extension

    @property
    def file_name(self):
        """Return the file_name attribute."""
        return self._file_name

    @property
    def waveform(self):
        """Return the waveform attribute."""
        return self._waveform

    @property
    def waveform_start_sample(self):
        """Return the waveform_start_sample attribute."""
        return self._waveform_start_sample

    @property
    def waveform_stop_sample(self):
        """Return the waveform_stop_sample attribute."""
        return self._waveform_stop_sample

    @property
    def waveform_duration_sample(self):
        """Return the waveform_duration_sample attribute."""
        return self._waveform_duration_sample

    @property
    def waveform_duration_sec(self):
        """Return the waveform_duration_sec attribute."""
        return self._waveform_duration_sec

    @property
    def filter_parameters(self):
        """Return the filter_parameters attribute."""
        if self._filter_applied:
            out = self._filter_params
        else:
            out = None
        return out

    @property
    def filter_applied(self):
        """Return the filter_applied attribute."""
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

    Methods
    -------
    apply(waveform, sampling_frequency)
        Apply filter to time vector/waveform.
    coefficients(sampling_frequency)
        Defines coeeficient of the filter.

    """

    def __init__(self, type, cutoff_frequencies, order=4):
        """
        Initialize the filter.

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
        Returns
        -------
        None. Filter object.

        """
        # chech filter type
        if (type == "bandpass") | (type == "lowpass") | (
            type == "highpass"
        ) == 0:
            raise ValueError(
                'Wrong filter type. Must be "bandpass", "lowpass"'
                + ', or "highpass".'
            )
        # chech freq values
        if type == "bandpass":
            if len(cutoff_frequencies) != 2:
                raise ValueError(
                    'The type "bandpass" requires two frepuency '
                    + "values: cutoff_frequencies=[lowcut, "
                    + "highcut]."
                )
            elif cutoff_frequencies[0] > cutoff_frequencies[1]:
                raise ValueError(
                    "The lowcut value should be smaller than the "
                    + "highcut value: cutoff_frequencies=[lowcut,"
                    + " highcut]."
                )
        elif (type == "lowpass") | (type == "highpass"):
            if len(cutoff_frequencies) != 1:
                raise ValueError(
                    'The type "lowpass" and "highpass" require '
                    + "one frequency value cutoff_frequencies="
                    + "[cutfreq]."
                )
        self.type = type
        self.cutoff_frequencies = cutoff_frequencies
        self.order = order

    def apply(self, waveform, sampling_frequency):
        """
        Apply filter to time series.

        Parameters
        ----------
        waveform : numpy.ndarray
            Time series to filter.
        sampling_frequency : float
            Sampling frequency of the time series to filter, in Hz.

        Returns
        -------
        numpy.ndarray
            Filtered time series.

        """
        # b, a = self.coefficients(sampling_frequency)
        # return spsig.sosfiltfilt (b, a, waveform)
        sos = self.coefficients(sampling_frequency)
        return spsig.sosfiltfilt(sos, waveform)

    def coefficients(self, sampling_frequency):
        """
        Get filter coefficients.

        Parameters
        ----------
        sampling_frequency : float
            Sampling frequency of the time series to filter, in Hz.

        Returns
        -------
        b : float
            Filter coefficient b.
        a : float
            Filter coefficient a.

        """
        nyquist = 0.5 * sampling_frequency
        if self.type == "bandpass":
            low = self.cutoff_frequencies[0] / nyquist
            high = self.cutoff_frequencies[1] / nyquist
            # b, a = spsig.butter(self.order, [low, high], btype='band')
            sos = spsig.butter(
                self.order, [low, high], btype="band", output="sos"
            )
        elif self.type == "lowpass":
            # b, a = spsig.butter(self.order,
            #                     self.cutoff_frequencies[0]/nyquist, 'low')
            sos = spsig.butter(
                self.order,
                self.cutoff_frequencies[0] / nyquist,
                "low",
                output="sos",
            )
        elif self.type == "highpass":
            # b, a = spsig.butter(self.order,
            #                     self.cutoff_frequencies[0]/nyquist, 'high')
            sos = spsig.butter(
                self.order,
                self.cutoff_frequencies[0] / nyquist,
                "high",
                output="sos",
            )
        return sos


def upsample(waveform, current_res_sec, new_res_sec):
    """
    Upsample  waveform

    Increase the number of samples in the waveform and interpolate.

    Parameters
    ----------
    waveform: 1D array
        Waveform to upsample
    current_res_sec : float
        Time resolution of waveform in seconds. It is the inverse of the
        sampling frequency.
    new_res_sec : float
        New time resolution of waveform after interpolation (in seconds).

    Returns
    -------
    waveform: 1D array
        waveform upsampled to have a time resolution of "new_res_sec".

    """
    axis_t = np.arange(0, len(waveform) * current_res_sec, current_res_sec)
    new_fs = round(1 / new_res_sec)
    nb_samp = round(axis_t[-1] * new_fs)
    new_waveform, new_axis_t = spsig.resample(
        waveform,
        nb_samp,
        t=axis_t,
        window="hann",
    )
    return new_waveform, new_fs
