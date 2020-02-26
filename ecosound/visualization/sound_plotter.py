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
import matplotlib
import numpy as np

class SoundPlotter(BaseClass):
    grapher_parameters = ('frequency_min',
                          'frequency_max',
                          'time_min',
                          'time_max',
                          'unit',
                          'fig_size',
                          'share_xaxis',
                          'grid',
                          'title',
                          )

    def __init__(self, *args, **kwargs):
        # Initialize all grapher parameters to None
        self.__dict__.update(dict(zip(self.grapher_parameters,
                                      [None]*len(self.grapher_parameters))))
        # Define default values:
        self.frequency_min = 0
        self.frequency_max = None
        self.time_min = 0
        self.time_max = None
        self.unit = 'sec'
        self.fig_size = (16, 4)
        self.share_xaxis = True
        self.grid = True
        self.title = None

        # Unpack kwargs as grapher parameters if provided on instantiation
        self.__dict__.update(**kwargs)
        
        # Initialize containers
        self.data = []
        self.annotations = []

    def add_data(self, *args):
        if len(args)<1:
            raise ValueError('There must be at least one input argument')
        # Check  type of each input arguments
        self._stack_data(args)
    
    def add_annotation(self, annotation, panel=None, label=False, color='red'):
        if isinstance(annotation, Annotation):
            self.annotations.append({'data': annotation,
                                     'panel': panel,
                                     'label': label,
                                     'color': color,
                                     })
        else:
            raise ValueError('Type of input argument not recognized.'
                             'Accepted object type: Annotation')

    def show(self, display=True):
        if len(self.data) == 0:
            raise ValueError('No data to plot. Use method .add_data to define'
                             ' the data to plot')
        # Display plot on screen?
        if display:
            matplotlib.use('Qt5Agg')
        else:
            matplotlib.use('Agg')

        # Define new figure and subplots
        nb_plots = len(self.data)
        fig, ax = plt.subplots(nb_plots, 1,
        figsize=self.fig_size,
        sharex=self.share_xaxis,
        constrained_layout=True,
        ) #gridspec_kw={'hspace': self.hspace}

        # Subplot titles
        titles = [None]*nb_plots
        if self.title is None: # no titles
            pass
        if type(self.title) is str:
            titles[0] = self.title
        if type(self.title) is list:
            if len(self.title) > nb_plots:
                raise ValueError("More titles than subplots")
            else:
                titles[0:len(self.title)-1] = self.title
        # Plot data
        for idx, data in enumerate(self.data):
            if nb_plots == 1:
                current_ax = ax
            else:
                current_ax = ax[idx]
            if data['type'] is 'waveform':
                self._plot_waveform(data['data'], current_ax, title=titles[idx])
            elif data['type'] is 'spectrogram':
                self._plot_spectrogram(data['data'], current_ax, title=titles[idx])
            # only dipslay x label of bottom plot if shared axes
            if self.share_xaxis and (idx != nb_plots-1):
                current_ax.set_xlabel('')

        # Plot annotations
        for idx_annot, annot in enumerate(self.annotations): # for each set of annotations
            # display annotations on all panels
            if annot['panel'] is None:
                annot['panel'] = range(0,nb_plots)
            # Make panel idx a list if not already
            if (type(annot['panel']) is float) or (type(annot['panel']) is int):
                annot['panel']=[annot['panel']]
            # Check panel indices
            if max(annot['panel']) > nb_plots-1:
                raise ValueError("Invalid panel index")
            # PLot annotations on appropriate panels
            for idx_panel in annot['panel']: # for each panel
                self._plot_annotations(annot['data'], ax[idx_panel],
                                       panel_idx=idx_panel,
                                       label=annot['label'],
                                       color=annot['color'],
                                       )

        return fig, ax

    def to_file(self, filename):
        fig, _ = self.show(display=False)
        fig.savefig(filename,transparent=False, bbox_inches='tight',)

    def _plot_annotations(self, annot, ax, label, panel_idx, color):
        panel_type = self.data[panel_idx]['type']
        
        for index, row in annot.data.iterrows():
            # plot annotations on spectrograms
            if panel_type is 'spectrogram':
                alpha = 1
                facecolor='none'
                if self.unit is 'sec':
                    x = row['time_min_offset']
                    y = row['frequency_min']
                    width = row['duration']
                    height = row['frequency_max']-row['frequency_min']
                elif self.unit is 'samp':
                    time_resolution = self.data[panel_idx]['data'].time_resolution
                    x = round(row['time_min_offset']/time_resolution)
                    y = row['frequency_min']
                    width = round(row['duration']/time_resolution)
                    height = row['frequency_max']-row['frequency_min']
            elif panel_type is 'waveform':
                alpha = 0.2
                facecolor = color
                if self.unit is 'sec':
                    x = row['time_min_offset']
                    y = min(ax.get_ylim())
                    width = row['duration']
                    height = max(ax.get_ylim()) - min(ax.get_ylim())
                elif self.unit is 'samp':
                    time_resolution = self.data[panel_idx]['data'].waveform_sampling_frequency
                    x = round(row['time_min_offset']*time_resolution)
                    y = min(ax.get_ylim())
                    width = round(row['duration']*time_resolution)
                    height =max(ax.get_ylim()) - min(ax.get_ylim())

            rect = plt.Rectangle((x,y), width, height,
                                 linewidth=1,
                                 edgecolor=color,
                                 facecolor=facecolor,
                                 alpha=alpha)
            ax.add_patch(rect)
        
    def _stack_data(self, args):
        for idx, arg in enumerate(args):
            if isinstance(arg, Sound):                
                self.data.append({'data': arg, 'type': 'waveform'})
            elif isinstance(arg, Spectrogram):
                self.data.append({'data': arg, 'type': 'spectrogram'})
            else:
                raise ValueError('Type of input argument not recognized.'
                                 'Accepted object types: Spectrogram, Sound')

    def _plot_spectrogram(self, spectro, current_ax, title=None):
        if self.frequency_max is None:
            self.frequency_max = spectro.sampling_frequency/2
        assert len(spectro.spectrogram)>0, "Spectrogram not computed yet. Use the .compute() method first."
        assert self.frequency_min < self.frequency_max, "Incorrect frequency bounds, frequency_min must be < frequency_max "
        assert self.frequency_min < self.frequency_max, "Incorrect frequency bounds, frequency_min must be < frequency_max "

        if self.unit == 'sec':
            if self.time_max is None:
                self.time_max = spectro.axis_times[-1]
            im = current_ax.pcolormesh(spectro.axis_times, spectro.axis_frequencies, spectro.spectrogram, cmap = 'jet',vmin = np.percentile(spectro.spectrogram,50), vmax= np.percentile(spectro.spectrogram,99.9))
            xlabel = 'Time (sec)'
        elif self.unit == 'samp':
            axis_t = np.arange(0, len(spectro.axis_times), 1)
            if self.time_max is None:
                self.time_max = axis_t[-1]
            im = current_ax.pcolormesh(axis_t, spectro.axis_frequencies, spectro.spectrogram, cmap = 'jet',vmin = np.percentile(spectro.spectrogram,50), vmax= np.percentile(spectro.spectrogram,99.9))
            xlabel = 'Time (bin)'
        else:
            raise ValueError("Keyword 'unit' must be set to either 'sec' or"
                             " 'samp'.")
        current_ax.axis([self.time_min,self.time_max,self.frequency_min,self.frequency_max])
        current_ax.set_ylabel('Frequency (Hz)')
        current_ax.set_xlabel(xlabel)
        current_ax.set_title(title)
        #if self.grid:
        #    current_ax.grid()
        return

    def _plot_waveform(self, sound, current_ax, title=None):
        
        if len(sound._waveform) == 0:
            raise ValueError('Cannot plot, waveform data enpty. Use Sound.read'
                             + ' to load the waveform')
        if self.unit == 'sec':
            axis_t = np.arange(0, len(sound._waveform)
                               / sound._waveform_sampling_frequency, 1
                               / sound._waveform_sampling_frequency)
            xlabel = 'Time (sec)'
        elif self.unit == 'samp':
            axis_t = np.arange(0, len(sound._waveform), 1)
            xlabel = 'Time (sample)'
        else:
            raise ValueError("Keyword 'unit' must be set to either 'sec' or"
                             " 'samp'.")

        axis_t = axis_t[0:len(sound._waveform)]
        #plt.plot(axis_t, sound._waveform, color='black')
        current_ax.plot(axis_t, sound._waveform, color='black')
        current_ax.set_xlabel(xlabel)
        current_ax.set_ylabel('Amplitude')
        current_ax.set_title(title)
        current_ax.axis([axis_t[0], axis_t[-1],
                  min(sound._waveform),
                  max(sound._waveform)])
        if self.grid:
            current_ax.grid()