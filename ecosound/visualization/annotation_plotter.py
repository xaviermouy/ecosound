# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:43:47 2020

@author: xavier.mouy
"""

from .grapher_builder import BaseClass
from ecosound.core.annotation import Annotation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mc # For the legend
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime as dt

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from matplotlib.figure import Figure
# import matplotlib
# import numpy as np


class HeatmapPlotter(BaseClass):
    """A Grapher class to visualize annotation data.

    Display heatmap of a annotations (date vs time-of-day).

    The HeatmapPlotter grapher must be instantiated using the GrapherFactory
    with the positional argument 'HeatmapPlotter':

    from ecosound.visualization.grapher_builder import GrapherFactory
    graph = GrapherFactory('HeatmapPlotter',kwargs)

    Attributes
    ----------
    name : str
        Name of the grapher
    version : str
        Version of the grapher
    fig_size : (float, float)
        Width, height of the figure, in inches.
    title : [str, str, ...] or str
        Title(s) of the figure/subplots.
    colormap : str
        Color map palette spectrogram. Uses names from Matplotlib. The default
        is 'viridis'

    Methods
    -------
    show(display=True)
        Display graph.
    to_file(filename)
        Save graph to file.
    """

    grapher_parameters = ('date_format',
                          'colormap_label'
                          'integration_time',
                          'is_binary',
                          'norm_value',
                          'fig_size',
                          'grid',
                          'title',
                          'colormap',
                          )

    def __init__(self, *args, **kwargs):
        """
        Initialize graher object.

        Initialize the grapher object to display sound data.

        Parameters
        ----------
        *args : str
            Do not use. Only used by the GrapherFactory.
        frequency_min : float, optional
            Minimum frequency of the spectrograms, in Hz. The default is 0.
        frequency_max : float, optional
            Maximum frequency of the spectrograms, in Hz. The default is half
            of the Nyquist frequency.
        time_min : float, optional
            Start time to graphs,in seconds. The default is 0.
        time_max : float, optional
            End time to graphs,in seconds. The default is the end of the sound
            data.
        unit : str, optional
            Unit of the time axis: 'sec' for seconds; 'samp' for discrete (i.e.
            samples or bins for waveform and spectrogram, respectively). The
            default is 'sec'.
        fig_size : (float, float), optional
            Width, height of the figure, in inches. The default is (16, 4).
        share_xaxis : bool, optional
            Share x axis for all subplots. The default is True.
        grid : bool, optional
            Display grid in waveform plots. The default is True.
        title : [str, str, ...], str, None,  optional
            Title(s) of the figure/subplots. If a str, a single title is set
            for the entire figure at the top. If a list of str, each element i
            of the list defines the title for the subplot of index i. The
            number of elements in the list can equal or smaller than the number
            of subplots in the graph. A number of elements in teh list greater
            than the number of subplot will return an error. If set to None, no
            title will be displayed. The default is None.

        colormap : str
            Color map for the spectrogram. Uses names from Matplotlib (e.g.
            'jet', 'binary', 'gray', etc.). The default is 'jet'

        Returns
        -------
        None.

        """
        # Initialize all grapher parameters to None
        self.__dict__.update(dict(zip(self.grapher_parameters,
                                      [None]*len(self.grapher_parameters))))
        # Define default values:        
        self.date_format = '%d-%b-%Y'
        self.colormap_label = None
        self.integration_time = '1H'
        self.is_binary = False
        self.norm_value = None
        self.fig_size = (16, 4)        
        self.title = None
        self.colormap = 'viridis'
 
        # Unpack kwargs as grapher parameters if provided on instantiation
        self.__dict__.update(**kwargs)


    @property
    def name(self):
        """Return name of the grapher."""
        grapher_name = 'HeatmapPlotter'
        return grapher_name

    @property
    def version(self):
        """Return version of the grapher."""
        version = '0.1'
        return version

    def add_data(self, *args, time_offset_sec=0):
        """
        Define annotation data to plot.

        Add Annotation objects to plot. There is no restriction on the number 
        of object to add. Each object will be displayed on a different subplot
        of the same figure. The order in which the objects are passed to this
        method defines the order in which they are displayed (i.e. subplots).
        When this method is called several times; objects are appended to the
        existing list of objects from previous calls.

        Parameters
        ----------
        *args : Annotation objects
            Annotation objects to display.

        Raises
        ------
        ValueError
            If no input arguments are provided.
            If the input arguments are not Annotation objects.

        Returns
        -------
        None. Updated grapher object.

        """
        if len(args) < 1:
            raise ValueError('There must be at least one input argument')
        # Check  type of each input arguments
        self._stack_data(args, time_offset_sec=time_offset_sec)


    def show(self, display=True):
        """
        Display graph on screen.

        Display graph made of vertical subplots for each of the Annotation 
        objects defined by the add_data method. The order in which the objects
        have been defined by add_data defines the order in which subplot are 
        displayed. 

        Parameters
        ----------
        display : bool, optional
            Display figure on screen. The default is True.

        Raises
        ------
        ValueError
            If no data have been defined (see add_data method).

        Returns
        -------
        fig : Figure instance
            Figure object.
        ax : Axis instance
            Axis object with all subplots.

        """
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
                               )  # gridspec_kw={'hspace': self.hspace}
        # Subplot titles
        titles = [None]*nb_plots
        if self.title is None:  # no titles
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
            if data['type'] == 'annotation':
                self._plot_heatmap(data['data'], current_ax, time_offset_sec=data['time_offset_sec'], title=titles[idx])            
            # only dipslay x label of bottom plot if shared axes
            if self.share_xaxis and (idx != nb_plots-1):
                current_ax.set_xlabel('')
        return fig, ax

    def to_file(self, filename):
        """
        Save the figure to an image file.

        Parameters
        ----------
        filename : str
            Full path of the image file to write.

        Returns
        -------
        None.

        """
        fig, _ = self.show(display=False)
        fig.savefig(filename, transparent=False, bbox_inches='tight',)


    def _stack_data(self, args, time_offset_sec=0):
        """Stack data to be plotted."""
        for idx, arg in enumerate(args):
            if isinstance(arg, Annotation):
                self.data.append({'data': arg, 'type': 'annotation'})           
            else:
                raise ValueError('Type of input argument not recognized.'
                                 'Accepted object types: Annotation')

    def _plot_heatmap(self, annot, current_ax, title=None):
        """Plot heatmap on the current axis"""
        
        # add time offset if defined
        spectro._axis_times = spectro.axis_times + time_offset_sec

        if self.time_max is None:
            self.time_max = spectro.axis_times[-1]
        current_ax.pcolormesh(spectro.axis_times,
                              spectro.axis_frequencies,
                              spectro.spectrogram,
                              cmap=self.colormap,
                              vmin=np.percentile(spectro.spectrogram, 50),
                              vmax=np.percentile(spectro.spectrogram, 99.9),
								  shading='nearest',
                              )
        xlabel = 'Time (sec)'  


        current_ax.axis([self.time_min,
                         self.time_max,
                         self.frequency_min,
                         self.frequency_max]
                        )
        current_ax.set_ylabel('Frequency (Hz)')
        current_ax.set_xlabel(xlabel)
        current_ax.set_title(title)
        # if self.grid:
        #    current_ax.grid()
        return
