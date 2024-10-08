# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:23:51 2020

@author: xavier.mouy
"""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy import signal, ndimage
from dask import delayed, compute
import copy
from numba import njit
import dask
import dask.array
import dask_image.ndfilters

## TODO: change Asserts by Raise


class Spectrogram:
    """A class for spectrograms.

    The Spectrogram object computes, denoises, and displays a spectrogram from
    a Sound object.

    Attributes
    ----------
    sampling_frequency : float
        Sampling frequency of the sound data.
    time_resolution : float
        Time resolution of the spectrogram, in seconds.
    frequency_resolution : float
        Frequency resolution of the spectrogram, in Hz.
    frame_samp : int
        Frame size, in samples
    frame_sec : float
        Frame size, in seconds
    step_samp : int
        Time step between conscutive spectrogram frames, in samples.
    step_sec : float
        Time step between conscutive spectrogram frames, in seconds.
    overlap_perc : float
        Percentage of overlap between conscutive spectrogram frames.
    overlap_samp : int
        Overlap between conscutive spectrogram frames, in samples.
    fft_samp : int
        Size to the Fast Fourier Transform, in samples.
    fft_sec : float
        Size to the Fast Fourier Transform, in seconds.
    window_type : str
        Type of weighting window applied to the signal before the FFT.
    axis_frequencies : numpy.ndarray
        1-D array with frequency values, in Hz, for each spectrogram rows.
    axis_times : numpy.ndarray
        1-D array with time values, in seconds, for each spectrogram column.
    spectrogram : numpy.ndarray
        2-D array with spectrogram energy data.

    Methods
    -------
    compute(sig, fs)
        Compute spectrogram.
    crop(frequency_min, frequency_max)
        Crop frequencies from the spectrogram.
    denoise(method, **kwargs)
        Denoise the spectrogram using various methods.
        Methods implemented:
        METHODS           :    INPUT ARGUMENTS
        'median_equalizer':    window_duration in seconds.
    """

    _valid_units = ("samp", "sec")
    _valid_windows = ("hann",)

    def __init__(
        self,
        frame,
        window_type,
        fft,
        step,
        sampling_frequency,
        unit="sec",
        verbose=True,
    ):
        """
        Initialize Spectrogram object.

        Defines spectrogram parameters. 'step' and 'fft' sizes can be defined
        in seconds (unit='sec') or in samples (unit='samp'). The 'fft' size is
        automatically adjusted to the next power of two. Zero padding is
        possible by defining a 'fft' size greater than the 'frame' size.

        Parameters
        ----------
        frame : float
            Frame size in seconds or samples, depending on 'unit'.
        window_type : str
            Weighting window to teh signal before the FFT. Currently, only
            'hann' is supported.
        fft : float
            Size of the Fast Fourier Transform, in seconds or samples,
            depending on 'unit'.
        step : float
            Time step between conscutive spectrogram frames. In samples or
            seconds depending on 'unit'.
        sampling_frequency : float
            Sampling frequency of the signal, in Hz.
        unit : str, optional
            Unit used when defining the 'frame' and 'fft' parameters. For
            seconds, use 'sec'. For samples, use 'samp'. The default is 'sec'.
        verbose : bool
            if True, prints all notification messages. The default is True.

        Returns
        -------
        None. Spectrogram object.

        """
        # Validation of the imput parameters
        assert unit in Spectrogram._valid_units, (
            "Wrong unit value. Valid \
                                           units: ",
            Spectrogram._valid_units,
        )
        assert fft >= frame, " fft should alwyas be >= frame"
        assert step < frame, "step should always be <= frame"
        assert window_type in Spectrogram._valid_windows, (
            "Wrong window type\
                                . Valid values: ",
            Spectrogram._valid_windows,
        )

        # Convert units in seconds/samples
        (
            self._frame_samp,
            self._fft_samp,
            self._step_samp,
            self._frame_sec,
            self._fft_sec,
            self._step_sec,
            self._overlap_perc,
            self._overlap_samp,
        ) = Spectrogram._convert_units(
            frame, fft, step, sampling_frequency, unit, verbose=verbose
        )

        # Time and frequency resolution
        self._sampling_frequency = sampling_frequency
        self._time_resolution = self.step_sec
        self._frequency_resolution = self.sampling_frequency / self.fft_samp

        # Define all other instance attributes
        self._window_type = window_type
        self._spectrogram = []
        self._axis_frequencies = []
        self._axis_times = []

    def _convert_units(
        frame, fft, step, sampling_frequency, unit, verbose=True
    ):
        """Convert frame, fft, and step values to samples/seconds"""
        if unit == "sec":
            frame_samp = round(frame * sampling_frequency)
            fft_samp = adjust_FFT_size(
                round(fft * sampling_frequency), verbose=verbose
            )
            step_samp = round(step * sampling_frequency)
            frame_sec = frame
            fft_sec = fft_samp * sampling_frequency
            step_sec = step
        elif unit == "samp":
            frame_samp = frame
            fft_samp = adjust_FFT_size(fft, verbose=verbose)
            step_samp = step
            frame_sec = frame / sampling_frequency
            fft_sec = fft_samp / sampling_frequency
            step_sec = step / sampling_frequency

        overlap_samp = frame_samp - step_samp
        overlap_perc = (overlap_samp / frame_samp) * 100
        return (
            frame_samp,
            fft_samp,
            step_samp,
            frame_sec,
            fft_sec,
            step_sec,
            overlap_perc,
            overlap_samp,
        )

    def _compute_old(self, sig):
        """
        Compute spectrogram.

        Compute spectrogram from sound signal and return values in the object
        attribute 'spectrogram'.

        Parameters
        ----------
        sig : Sound object
            Sound object (core.audiotools.Sound) with the signal to anayse.

        Returns
        -------
        Populate the 'spectrogram' attribute of the Spectrogram object.
        axis_frequencies, numpy.ndarray
            1-D array with time axis values, in seconds.
        axis_times, numpy.ndarray
            1-D array with frequency axis values, in Hertz.
        spectrogram, numpy.ndarray
            2-D array with spectrogram values.

        """
        assert (
            sig.waveform_sampling_frequency == self.sampling_frequency
        ), "The sampling frequency provided doesn't match the one from the Spectrogram object."
        # Weighting window
        if self.window_type == "hann":
            window = signal.hann(self.frame_samp)
        # Calculates  spectrogram
        (
            self._axis_frequencies,
            self._axis_times,
            self._spectrogram,
        ) = signal.spectrogram(
            sig.waveform,
            fs=self.sampling_frequency,
            window=window,
            noverlap=self.overlap_samp,
            nfft=self.fft_samp,
            scaling="spectrum",
        )
        self._spectrogram = 20 * np.log10(self._spectrogram)
        return self._axis_frequencies, self._axis_times, self._spectrogram

    def compute(self, sig, dB=False, use_dask=False, dask_chunks=40):
        """
        Compute spectrogram.

        Compute spectrogram from sound signal and return values in the object
        attribute 'spectrogram'.

        Parameters
        ----------
        sig : Sound object
            Sound object (core.audiotools.Sound) with the signal to anayse.
        dB : bool, optional
            If set to True, returns spectrogram values in dB. The default is
            False.
        use_dask : bool, optional
            If set to True, parallelize the computation using Dask. The default
            is False.
        dask_chunks : int, optional
            Set the number of chunks the signal is broken in to for the
            parallelized computation using Dask. Only used if use_dask is set
            to True. The default is 40.

        Returns
        -------
        Populate the 'spectrogram' attribute of the Spectrogram object.
        axis_frequencies, numpy.ndarray
            1-D array with time axis values, in seconds.
        axis_times, numpy.ndarray
            1-D array with frequency axis values, in Hertz.
        spectrogram, numpy.ndarray
            2-D array with spectrogram values.
        """
        # Weighting window
        if self.window_type == "hann":
            win = np.hanning(self.frame_samp)
        # if signal is shorter than spectrogram frame -> add zeros
        if len(sig.waveform) < self.frame_samp:
            vec = np.zeros(self.frame_samp)
            vec[0 : len(sig.waveform)] = sig.waveform
            sig._waveform = vec
            sig._waveform_duration_sample = len(vec)
            sig._waveform_duration_sec = (
                sig._waveform_duration_sample / sig.waveform_sampling_frequency
            )
            starts = np.array([0])
            stops = np.array([len(vec)])
        else:
            starts = np.arange(
                0,
                len(sig.waveform) - self.frame_samp,
                self.step_samp,
                dtype=int,
            )
            stops = starts + self.frame_samp
        start_chunks = np.array_split(starts, dask_chunks)
        stop_chunks = np.array_split(stops, dask_chunks)

        spectrogram = []
        idx = 0
        for start_chunk, stop_chunk in zip(start_chunks, stop_chunks):
            if (len(start_chunk) > 0) & (len(stop_chunk) > 0):
                # print(start_chunk)
                # print(stop_chunk)
                # print("---------")
                sig_chunk = sig.waveform[start_chunk[0] : stop_chunk[-1]]
                chunk_size = len(start_chunk)
                if use_dask:
                    spectro_chunk = delayed(Spectrogram._calc_spectrogram)(
                        sig_chunk,
                        win,
                        start_chunk - start_chunk[0],
                        stop_chunk - start_chunk[0],
                        self.fft_samp,
                    )
                else:
                    spectro_chunk = Spectrogram._calc_spectrogram(
                        sig_chunk,
                        win,
                        start_chunk - start_chunk[0],
                        stop_chunk - start_chunk[0],
                        self.fft_samp,
                    )
                spectrogram.append(spectro_chunk)
                idx += chunk_size
        if use_dask:
            spectrogram = compute(spectrogram)
            spectrogram = np.concatenate(spectrogram[0][:], axis=1)
        else:
            spectrogram = np.concatenate(spectrogram[:], axis=1)
        if dB:
            spectrogram = 20 * np.log10(spectrogram)
        self._spectrogram = spectrogram
        self._axis_times = starts / self.sampling_frequency  # ?? needed ?
        self._axis_frequencies = np.arange(
            0, self._sampling_frequency / 2, self._frequency_resolution
        )  # ?? needed ?
        return self._axis_frequencies, self._axis_times, self._spectrogram

    @staticmethod
    # @njit
    def _calc_spectrogram(sig, win, starts, stops, fft_samp):
        fft_samp = np.int32(fft_samp)
        fnyq0 = np.round(fft_samp / 2)
        #fnyq = np.int(fnyq0)
        fnyq = int(fnyq0)
        spectro = np.empty((fnyq, len(starts)))  # the default
        idx = 0
        for start, stop in zip(starts, stops):
            s = sig[start:stop] * win
            Spectrum = np.fft.fft(s, fft_samp)
            Spectrum = abs(Spectrum)  # amplitude
            Spectrum = Spectrum * 2
            # Spectrum = Spectrum**2
            # ts_window = getFFT(sig[start:stop],fft_samp)
            spectro[:, idx] = Spectrum[0:fnyq]
            idx += 1
        return spectro

    def crop(
        self,
        frequency_min=None,
        frequency_max=None,
        time_min=None,
        time_max=None,
        inplace=False,
    ):
        """
        Crop frequencies from the spectrogram.

        Crop the spectrogram matrix by keeping only frequency rows above
        frequency_min and below frequency_max. If frequency_min is not provided
        then, only spectrogram rows above frequency_max will be removed. If
        frequency_max is not provided then, only spectrogram rows below
        frequency_min will be removed. The axis_frequencies attribute of the
        spectrogram object are automatically updated.

        Parameters
        ----------
        frequency_min : float, optional
            Minimum frequency limit, in Hz. The default is None.
        frequency_max : float, optional
            Maximum frequency limit, in Hz. The default is None.
        time_min : float, optional
            Minimum time limit, in sec. The default is None.
        time_max : float, optional
            Maximum time limit, in sec. The default is None.
        inplace : bool, optional
            If True, do operation inplace and return None. The default is False

        Returns
        -------
        None. Cropped spectrogram matrix.

        """
        # Find frequency indices
        if frequency_min is None:
            min_row_idx = 0
        else:
            min_row_idx = np.where(self._axis_frequencies < frequency_min)
            if np.size(min_row_idx) == 0:
                min_row_idx = 0
            else:
                min_row_idx = min_row_idx[0][-1] + 1
        if frequency_max is None:
            max_row_idx = self._axis_frequencies.size - 1
        else:
            max_row_idx = np.where(self._axis_frequencies > frequency_max)
            if np.size(max_row_idx) == 0:
                max_row_idx = self._axis_frequencies.size - 1
            else:
                max_row_idx = max_row_idx[0][0]
        # Find time indices
        if time_min is None:
            min_col_idx = 0
        else:
            min_col_idx = np.where(self._axis_times < time_min)
            if np.size(min_col_idx) == 0:
                min_col_idx = 0
            else:
                min_col_idx = min_col_idx[0][-1] + 1
        if time_max is None:
            max_col_idx = self._axis_times.size - 1
        else:
            max_col_idx = np.where(self._axis_times > time_max)
            if np.size(max_col_idx) == 0:
                max_col_idx = self._axis_times.size - 1
            else:
                max_col_idx = max_col_idx[0][0]
        # update spectrogram and axes
        if inplace:
            # self._axis_frequencies = self._axis_frequencies[min_row_idx:max_row_idx]
            # self._axis_times = np.arange(0,(max_col_idx - min_col_idx)*self._time_resolution,self._time_resolution)
            # self._spectrogram = self._spectrogram[min_row_idx:max_row_idx, min_col_idx:max_col_idx]
            # out_object = None
            self._axis_frequencies = self._axis_frequencies[
                min_row_idx : max_row_idx + 1
            ]
            self._axis_times = (
                self._axis_times[min_col_idx : max_col_idx + 1]
                - self._axis_times[min_col_idx]
            )
            self._spectrogram = self._spectrogram[
                min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1
            ]
            out_object = None
        else:
            out_object = copy.copy(self)
            out_object._axis_frequencies = out_object._axis_frequencies[
                min_row_idx : max_row_idx + 1
            ]
            # out_object._axis_times = np.arange(0,(max_col_idx - min_col_idx)*out_object._time_resolution,out_object._time_resolution)
            out_object._axis_times = (
                out_object._axis_times[min_col_idx : max_col_idx + 1]
                - out_object._axis_times[min_col_idx]
            )
            # out_object._spectrogram = out_object._spectrogram[min_row_idx:max_row_idx, min_col_idx:max_col_idx]
            out_object._spectrogram = out_object._spectrogram[
                min_row_idx : max_row_idx + 1, min_col_idx : max_col_idx + 1
            ]
            if out_object._spectrogram.shape[1] != len(out_object._axis_times):
                raise ValueError(
                    "Spectrogram axes don't match spectrogram matrix."
                )
        return out_object

    def denoise(self, method, **kwargs):
        """
        Denoise spectrogram.

        Denoise the spectrogram using various methods. The methods implemented
        are:
            METHODS           :    INPUT ARGUMENTS
            'median_equalizer':    window_duration in seconds.
                                   inplace

        Parameters
        ----------
        method : str
            DESCRIPTION.
        **kwargs : variable
            Parameters for the methods selected.

        Raises
        ------
        ValueError
            If method is not valid.

        Returns
        -------
        None. Denoised spectrogram matrix.

        """
        denoise_methods = ("median_equalizer",)
        if method in denoise_methods:
            eval("self._" + method + "(**kwargs)")
        else:
            raise ValueError(
                "Method not recognized. Methods available:"
                + str(denoise_methods)
            )

    def _median_equalizer(
        self,
        window_duration,
        use_dask=False,
        dask_chunks=(1000, 1000),
        inplace=False,
    ):
        """
        Median equalizer.

        Denoises the spectrogram matrix by subtracting the meidan spectrogram
        compuetd with a median filter of window "window_sixe" to the original
        spectrogram. Negative values of teh denoised spectrogram are set to
        zero.

        Parameters
        ----------
        window_duration : float
            Durations of the median filter, in seconds.
        use_dask : bool, optional
            If True, to parallelize the processing with Dask. The default is
            False
        dask_chunks : tuple-> (int, int), optional
            Size of the sepctrogram chunks for the parallelized computing in
            number of bins. dask_chunks=(number of frequency bins, number of
            time bins). Only used if use_dask is True. The default is
            (1000,1000).
        inplace : bool, optional
            If True, do operation inplace and return None. The default is False

        Returns
        -------
        Denoised spectrogram matrix.

        """
        if use_dask:
            dask_spectro = dask.array.from_array(
                self._spectrogram, chunks=dask_chunks
            )
            Smed = dask_image.ndfilters.median_filter(
                dask_spectro,
                size=(1, round(window_duration / self.time_resolution)),
                mode="mirror",
            )
            Smed = Smed.compute()
        else:
            Smed = ndimage.median_filter(
                self._spectrogram,
                (1, round(window_duration / self.time_resolution)),
            )
        if inplace:
            self._spectrogram = self._spectrogram - Smed
            self._spectrogram[self._spectrogram < 0] = 0  # floor
            out_object = None
        else:
            out_object = copy.copy(self)
            out_object._spectrogram = out_object._spectrogram - Smed
            out_object._spectrogram[out_object._spectrogram < 0] = 0  # floor
        return out_object

    @property
    def frame_samp(self):
        """Return the frame_samp attribute."""
        return self._frame_samp

    @property
    def frame_sec(self):
        """Return the frame_sec attribute."""
        return self._frame_sec

    @property
    def step_samp(self):
        """Return the step_samp attribute."""
        return self._step_samp

    @property
    def step_sec(self):
        """Return the step_sec attribute."""
        return self._step_sec

    @property
    def fft_samp(self):
        """Return the fft_samp attribute."""
        return self._fft_samp

    @property
    def fft_sec(self):
        """Return the fft_sec attribute."""
        return self._fft_sec

    @property
    def overlap_perc(self):
        """Return the overlap_perc attribute."""
        return self._overlap_perc

    @property
    def overlap_samp(self):
        """Return the overlap_samp attribute."""
        return self._overlap_samp

    @property
    def sampling_frequency(self):
        """Return the sampling_frequency attribute."""
        return self._sampling_frequency

    @property
    def time_resolution(self):
        """Return the time_resolution attribute."""
        return self._time_resolution

    @property
    def frequency_resolution(self):
        """Return the frequency_resolution attribute."""
        return self._frequency_resolution

    @property
    def window_type(self):
        """Return the window_type attribute."""
        return self._window_type

    @property
    def axis_frequencies(self):
        """Return the axis_frequencies attribute."""
        return self._axis_frequencies

    @property
    def axis_times(self):
        """Return the axis_times attribute."""
        return self._axis_times

    @property
    def spectrogram(self):
        """Return the spectrogram attribute."""
        return self._spectrogram


def adjust_FFT_size(nfft, verbose=True):
    """Adjust nfft to the next power of two if necessary."""
    nfft_adjusted = next_power_of_2(nfft)
    if nfft_adjusted != nfft:
        if verbose:
            print(
                "Warning: FFT size automatically adjusted to",
                nfft_adjusted,
                "samples (original size:",
                nfft,
                ")",
            )
    return nfft_adjusted


def next_power_of_2(x):
    """Calculate the next power of two for x."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()
