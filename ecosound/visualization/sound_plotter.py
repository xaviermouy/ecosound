# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:43:47 2020

@author: xavier.mouy
"""

from .grapher_builder import BaseClass
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation


class SoundPlotter(BaseClass):
    grapher_parameters = ('kernel_duration','kernel_bandwidth', 'threshold','duration_min','bandwidth_min')

    def __init__(self, *args, **kwargs):
        # Initialize all grapher parameters to None
        self.__dict__.update(dict(zip(self.grapher_parameters,
                                      [None]*len(self.grapher_parameters))))
        # Unpack kwargs as grapher parameters if provided on instantiation
        self.__dict__.update(**kwargs)


    def plot_spectrogram(self, frequency_min=0, frequency_max=[], time_min=0, time_max=[]):
        """
        Display spectrogram.

        Parameters
        ----------
        frequency_min : float, optional
            Minimum frequency limit of the plot, in Hz. The default is 0.
        frequency_max : float, optional
            Maximum frequency limit of the plot, in Hz. The default is [].
        time_min : float, optional
            Minimum time limit of the plot, in seconds. The default is 0.
        time_max : float, optional
            Maximum time limit of the plot, in seconds. The default is [].

        Returns
        -------
        None.

        """
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

    def plot_waveform(self, unit='sec', newfig=False, title=''):
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
    
        Raises
        ------
        ValueError
            If the waveform attribute is empty.
    
        Returns
        -------
        None.
    
        """
        if len(self._waveform) == 0:
            raise ValueError('Cannot plot, waveform data enpty. Use Sound.read'
                             + ' to load the waveform')
        if unit == 'sec':
            axis_t = np.arange(0, len(self._waveform)
                               / self._waveform_sampling_frequency, 1
                               / self._waveform_sampling_frequency)
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