# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:43:47 2020

@author: xavier.mouy
"""

from .grapher_builder import BaseClass
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
try:
    from ecosound.core.annotation import Annotation
except ImportError:
    pass

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib
import ipympl
import numpy as np


class SoundPlotter(BaseClass):
    """A Grapher class to visualize sound data.

    Display wavforms, spectrograms of a sound signal and can also overlay
    annotations on each sound representation.

    The SoundPlotter grapher must be instantiated using the GrapherFactory with
    the positional argument 'SoundPlotter':

    from ecosound.visualization.grapher_builder import GrapherFactory
    graph = GrapherFactory('SoundPlotter',kwargs)

    Attributes
    ----------
    name : str
        Name of the grapher
    version : str
        Version of the grapher
    frequency_min : float
        Minimum frequency of the spectrograms, in Hz.
    frequency_max : float
        Maximum frequency of the spectrograms, in Hz.
    time_min : float
        Start time to graphs,in seconds.
    time_max : float
        End time to graphs,in seconds.
    unit : str
        Unit of the time axis. 'sec' or 'samp'
    fig_size : (float, float)
        Width, height of the figure, in inches.
    share_xaxis : bool
        Share x axis for all subplots.
    grid : bool
        Display grid in waveform plots.
    title : [str, str, ...] or str
        Title(s) of the figure/subplots.
    colormap : str
        Color map for the spectrogram. Uses names from Matplotlib. The default
        is 'jet'

    Methods
    -------
    add_data(*args)
        Add Sound or Spectrogram objects to plot.
    add_annotations(Annotation, panel=None, label=False, color='red')
        Add Annotation object to overlay on graphs.
    show(display=True)
        Display graph.
    to_file(filename)
        Save graph to file.
    """

    grapher_parameters = ('frequency_min',
                          'frequency_max',
                          'time_min',
                          'time_max',
                          'unit',
                          'fig_size',
                          'share_xaxis',
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
        self.frequency_min = 0
        self.frequency_max = None
        self.time_min = 0
        self.time_max = None
        self.unit = 'sec'
        self.fig_size = (16, 4)
        self.share_xaxis = True
        self.grid = True
        self.title = None
        self.colormap = 'jet'
        # Unpack kwargs as grapher parameters if provided on instantiation
        self.__dict__.update(**kwargs)
        # Initialize containers
        self.data = []
        self.annotations = []

    @property
    def name(self):
        """Return name of the grapher."""
        grapher_name = 'SoundPlotter'
        return grapher_name

    @property
    def version(self):
        """Return version of the grapher."""
        version = '0.1'
        return version

    def add_data(self, *args, time_offset_sec=0):
        """
        Define sound or spectrogram data to plot.

        Add Sound or Spectrogram objects to plot. There is no restriction on
        the number of object to add. Each object will be displayed on a
        different subplot of the same figure. The order in which the objects
        are passed to this method defines the order in which they are displayed
        (i.e. subplots). When this method is called several times; objects are
        appended to the existing list of objects from previous calls.

        Parameters
        ----------
        *args : Spectrogram, Sound objects
            Sound and/or Spectrogram objects to display.

        Raises
        ------
        ValueError
            If no input arguments are provided.
            If the input arguments are not Sound or Spectrogram objects.

        Returns
        -------
        None. Updated grapher object.

        """
        if len(args) < 1:
            raise ValueError('There must be at least one input argument')
        # Check  type of each input arguments
        self._stack_data(args, time_offset_sec=time_offset_sec)


    def add_annotation(self, annotation, panel=None, label=False, color='red', tag=False, line_width=1):
        """
        Define annotations to display.

        When this method is called several times; ANnotation objects are
        appended to the existing list of objects from previous calls.

        Parameters
        ----------
        annotation : Annotation object.
            Annotation object to overlay on top of the waveforem or spectrogram
            plots.
        panel : int, list, None, optional
            Define on which subplot (i.e. data defined with the add_data
            method) to overlay the annotations on. If set to None, annotations
            will be displayed on all subplots. If set to int, annotations will
            only be displayed on the subplot index defined by panel. If set as
            a list of int, annotations will be displayed on the subplots
            indices defined in the panel panel. The default is None.
        label : bool, optional
            Display label for each annotation. Not implemented yet. The default
            is False.
        color : str, optional
            Color of the annotation boxes. Uses the color name as matplotlib
            (e.g. 'black', 'white','red', 'yellow', etc). The default is 'red'.
        tag : bool, optional
            If set to True, displays the classification confidence over each
            annotation box. The default is False.
		line_width : int, optional
            Width of the annotation line. The default is 1.

        Raises
        ------
        ValueError
            If the input argument is not an Annotation object.

        Returns
        -------
        None. Updated grapher object.

        """
        from ecosound.core.annotation import Annotation
        if isinstance(annotation, Annotation):
            self.annotations.append({'data': annotation,
                                     'panel': panel,
                                     'label': label,
                                     'color': color,
                                     'tag': tag,
									 'line_width': line_width,
                                     })
        else:
            raise ValueError('Type of input argument not recognized.'
                             'Accepted object type: Annotation')

    def show(self, display=True, is_in_notebook=False):
        """
        Display graph on screen.

        Display graph made of vertical subplots for each of the Sound and
        Spectrogram objects defined by the add_data method. The order in which
        the objects have been defined by add_data defines the order in which
        subplot they are displayed. Waveform and Spectrogram objects are
        identified automatically and plotted accordingly. Annotations are
        overlayed on subplots (i.e. panels) defined by the method
        add_annotations. Annotations are displayed as boxes in spectrograms,
        and as shaded areas in waveforms.

        Parameters
        ----------
        display : bool, optional
            Display figure on screen. The default is True.

        Raises
        ------
        ValueError
            If no data have been defined (see add_data method).
            If a Spectrogram has no data (i.e. spectrogram.compute() not excuted).
            If unit is not equal to 'sec' or 'samp'.

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
            if data['type'] == 'waveform':
                self._plot_waveform(data['data'], current_ax, time_offset_sec=data['time_offset_sec'], title=titles[idx])
            elif data['type'] == 'spectrogram':
                self._plot_spectrogram(data['data'], current_ax, time_offset_sec=data['time_offset_sec'], title=titles[idx])
            # only dipslay x label of bottom plot if shared axes
            if self.share_xaxis and (idx != nb_plots-1):
                current_ax.set_xlabel('')

        # Plot annotations
        for idx_annot, annot in enumerate(self.annotations):  # for each set of annotations
            # display annotations on all panels
            if annot['panel'] is None:
                annot['panel'] = range(0, nb_plots)
            # Make panel idx a list if not already
            if (type(annot['panel']) is float) or (type(annot['panel']) is int):
                annot['panel'] = [annot['panel']]
            # Check panel indices
            if max(annot['panel']) > nb_plots-1:
                raise ValueError("Invalid panel index")
            # PLot annotations on appropriate panels
            for idx_panel in annot['panel']:  # for each panel
                if len(self.data)==1:
                    current_ax = ax
                else:
                    current_ax = ax[idx_panel]
                self._plot_annotations(annot['data'], current_ax,
                                       panel_idx=idx_panel,
                                       label=annot['label'],
                                       color=annot['color'],
									   line_width = annot['line_width'],
                                       )
                if annot['label'] is not False:
                    handles, labels = current_ax.get_legend_handles_labels()
                    unique_labels=list(set(labels))
                    new_handles=[]
                    for l in unique_labels:
                        new_handles.append(handles[labels.index(l)])
                    current_ax.legend(new_handles,unique_labels,loc='upper right')

                if annot['tag'] is True:
                    bbox_props = dict(boxstyle="square", fc="w", ec="w", alpha=0.8)
                    panel_type = self.data[annot['panel'][0]]['type']
                    for index, row in annot['data'].data.iterrows():
                        if self.unit == 'sec':
                            #height = row['frequency_max']-row['frequency_min']
                            x = row['time_min_offset']
                            if panel_type == 'spectrogram':
                                y = row['frequency_max']
                            elif panel_type == 'waveform':
                                y = max(current_ax.get_ylim())
                            conf = str(round(row['confidence'],2))
                        elif self.unit == 'samp':
                            x = row['time_min_offset']
                            if panel_type == 'spectrogram':
                                y = row['frequency_max']
                            elif panel_type == 'waveform':
                                y = max(current_ax.get_ylim())
                            conf = str(round(row['confidence'],2))
                        current_ax.text(x, y, conf, size=8, bbox=bbox_props)


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

    def _plot_annotations(self, annot, ax, label, panel_idx, color, line_width):
        """Plot annotations on top of the waveform or spectrogram axes."""
        panel_type = self.data[panel_idx]['type']
        for index, row in annot.data.iterrows():
            # plot annotations on spectrograms
            if panel_type == 'spectrogram':
                alpha = 1
                facecolor = 'none'
                if self.unit == 'sec':
                    x = row['time_min_offset']
                    y = row['frequency_min']
                    width = row['duration']
                    height = row['frequency_max']-row['frequency_min']
                elif self.unit == 'samp':
                    time_resolution = self.data[panel_idx]['data'].time_resolution
                    x = round(row['time_min_offset']/time_resolution)
                    y = row['frequency_min']
                    width = round(row['duration']/time_resolution)
                    height = row['frequency_max']-row['frequency_min']
            elif panel_type == 'waveform':
                alpha = 0.2
                facecolor = color
                if self.unit == 'sec':
                    x = row['time_min_offset']
                    y = min(ax.get_ylim())
                    width = row['duration']
                    height = max(ax.get_ylim()) - min(ax.get_ylim())
                elif self.unit == 'samp':
                    time_resolution = self.data[panel_idx]['data'].waveform_sampling_frequency
                    x = round(row['time_min_offset']*time_resolution)
                    y = min(ax.get_ylim())
                    width = round(row['duration']*time_resolution)
                    height = max(ax.get_ylim()) - min(ax.get_ylim())
            rect = plt.Rectangle((x, y), width, height,
                                 linewidth=line_width,
                                 edgecolor=color,
                                 facecolor=facecolor,
                                 alpha=alpha,
                                 label=label)
            ax.add_patch(rect)

    def _stack_data(self, args, time_offset_sec=0):
        """Stack data to be plotted."""
        for idx, arg in enumerate(args):
            if isinstance(arg, Sound):
                self.data.append({'data': arg, 'type': 'waveform', 'time_offset_sec': time_offset_sec})
            elif isinstance(arg, Spectrogram):
                self.data.append({'data': arg, 'type': 'spectrogram', 'time_offset_sec': time_offset_sec})
            else:
                raise ValueError('Type of input argument not recognized.'
                                 'Accepted object types: Spectrogram, Sound')

    def _plot_spectrogram(self, spectro, current_ax, time_offset_sec=0, title=None):
        """Plot spectrogram on the current axis"""
        if self.frequency_max is None:
            self.frequency_max = spectro.sampling_frequency/2
        assert len(spectro.spectrogram) > 0, "Spectrogram not computed yet. "
        "Use the .compute() method first."
        # add time offset if defined
        spectro._axis_times = spectro.axis_times + time_offset_sec
        if self.unit == 'sec':
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
        elif self.unit == 'samp':
            axis_t = np.arange(0, len(spectro.axis_times), 1)
            axis_t = axis_t + round(time_offset_sec/spectro.time_resolution)
            if self.time_max is None:
                self.time_max = axis_t[-1]
            current_ax.pcolormesh(axis_t,
                                  spectro.axis_frequencies,
                                  spectro.spectrogram,
                                  cmap=self.colormap,
                                  vmin=np.percentile(spectro.spectrogram, 50),
                                  vmax=np.percentile(spectro.spectrogram, 99.9)
                                  )
            xlabel = 'Time (bin)'
        else:
            raise ValueError("Keyword 'unit' must be set to either 'sec' or"
                             " 'samp'.")
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

    def _plot_waveform(self, sound, current_ax, time_offset_sec=0, title=None):
        """Plot waveform of a sound object on the current axis."""
        if len(sound._waveform) == 0:
            raise ValueError('Cannot plot, waveform data enpty. Use Sound.read'
                             + ' to load the waveform')
        if self.unit == 'sec':
            axis_t = np.arange(0, sound.waveform_duration_sample
                               / sound.waveform_sampling_frequency, 1
                               / sound.waveform_sampling_frequency)
            axis_t = axis_t + time_offset_sec
            xlabel = 'Time (sec)'
        elif self.unit == 'samp':
            axis_t = np.arange(0, len(sound._waveform), 1)
            axis_t = axis_t + (time_offset_sec*sound.waveform_sampling_frequency)
            xlabel = 'Time (sample)'
        else:
            raise ValueError("Keyword 'unit' must be set to either 'sec' or"
                             " 'samp'.")
        if self.time_max is None:
                self.time_max = axis_t[-1]
        #axis_t = axis_t[0:len(sound._waveform)]
        current_ax.plot(axis_t[0:len(sound._waveform)], sound._waveform, color='black')
        current_ax.set_xlabel(xlabel)
        current_ax.set_ylabel('Amplitude')
        current_ax.set_title(title)

        current_ax.axis([self.time_min,
                         self.time_max,
                         min(sound._waveform),
                         max(sound._waveform)]
        # current_ax.axis([axis_t[0],
        #                  axis_t[-1],
        #                  min(sound._waveform),
        #                  max(sound._waveform)]
                        )
        if self.grid:
            current_ax.grid()