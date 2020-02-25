# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:43:47 2020

@author: xavier.mouy
"""

from .grapher_builder import BaseClass
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.core.annotation import Annotation

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class SoundPlotter(BaseClass):
    grapher_parameters = ('frequency_min','frequency_max','time_min','unit','title')

    
    def __init__(self, *args, **kwargs):
        # Initialize all grapher parameters to None
        self.__dict__.update(dict(zip(self.grapher_parameters,
                                      [None]*len(self.grapher_parameters))))
        # Define default values:
        self.frequency_min = 0
        self.frequency_max=[]
        self.time_min=0 
        self.time_max=[]
        self.unit='sec'
        self.figsize=(16,4)
        
        # Unpack kwargs as grapher parameters if provided on instantiation
        self.__dict__.update(**kwargs)

        self.data =[]

    def add_data(self, *args):
        if  len(args)>3:
            raise ValueError("There can't be more than 3 input arguments")
        if len(args)<1:
            raise ValueError('There must be at least one input argument')
        # Check  type of each input arguments
        self._stack_data(args)

    def show(self):
        if len(self.data) == 0:
            raise ValueError('No data to plot. Use method .add_data to define'
                             ' the data to plot')
        # Define new figure
        nb_plots = len(self.data)
        fig, ax = plt.subplots(nb_plots, 1,
        figsize=self.figsize,
        sharex=True
        )

        for idx, data in enumerate(self.data):
            if nb_plots == 1:
                current_ax = ax
            else:
                current_ax = ax[idx]
            if data['type'] is 'waveform':
                self._plot_wavform(data['data'], current_ax)
            elif data['type'] is 'spectrogram':
                self._plot_spectrogram(data['data'], current_ax)
        
        #fig.colorbar(im, ax=ax)
        fig.tight_layout()
        
    def _stack_data(self, args):
        for idx, arg in enumerate(args):
            if isinstance(arg, Sound):                
                self.data.append({'data': arg, 'type': 'waveform'})
            elif isinstance(arg, Spectrogram):
                self.data.append({'data': arg, 'type': 'spectrogram'})
            else:
                raise ValueError('Type of input argument not recognized.'
                                 'Accepted object types: Spectrogram, Sound')

    def _plot_spectrogram(self, spectro, current_ax):
        if len(self.frequency_max) == 0:
            self.frequency_max = spectro.sampling_frequency/2
        if len(self.time_max) == 0:
            self.time_max = spectro.axis_times[-1]
        assert len(spectro.spectrogram)>0, "Spectrogram not computed yet. Use the .compute() method first."
        assert self.frequency_min < self.frequency_max, "Incorrect frequency bounds, frequency_min must be < frequency_max "
        assert self.frequency_min < self.frequency_max, "Incorrect frequency bounds, frequency_min must be < frequency_max "

        im = current_ax.pcolormesh(spectro.axis_times, spectro.axis_frequencies, spectro.spectrogram, cmap = 'jet',vmin = np.percentile(spectro.spectrogram,50), vmax= np.percentile(spectro.spectrogram,99.9))
        current_ax.axis([self.time_min,self.time_max,self.frequency_min,self.frequency_max])
        #ax.set_clim(np.percentile(Sxx,50), np.percentile(Sxx,99.9))
        current_ax.set_ylabel('Frequency [Hz]')
        current_ax.set_xlabel('Time [sec]')
        current_ax.set_title('tbd...')
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