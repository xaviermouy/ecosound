# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:43:47 2020

@author: xavier.mouy
"""

from .grapher_builder import BaseClass

try:
    from ecosound.core.annotation import Annotation
except ImportError:
    pass


import numpy as np
import pandas as pd
import ecosound
import matplotlib.pyplot as plt
import matplotlib.colors as mc  # For the legend
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime as dt

# import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# from matplotlib.figure import Figure
import matplotlib

# import numpy as np


class AnnotHeatmap(BaseClass):
    """A Grapher class to visualize annotation data.

    Display heatmap of annotations (date vs time-of-day).

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
    colorbar_label: str
        Label to use for the clolorbar.
    integration_time: str
        Integration time for the time aggregates.
    date_format: str
        Date format to use for the x-axis tick labels.
    is_binary: bool
        Makes binary aggreagates rather than counts.
    norm_value: int
        Maximum value to use for the colomap.
    share_xaxis: bool
        Share xaxis in case of subplots (not implememnted yet)

    Methods
    -------
    add_data(Annotation obj)
        Add Annotation object to plot.
    show(display=True)
        Display graph.
    to_file(filename)
        Save graph to file.
    """

    grapher_parameters = (
        "date_format",
        "colorbar_label" "integration_time",
        "is_binary",
        "norm_value",
        "fig_size",
        "share_xaxis",
        "title",
        "colormap",
    )

    def __init__(self, *args, **kwargs):
        """
        Initialize grapher object.

        Initialize the grapher object to display sound data.

        Parameters
        ----------
        *args : str
            Do not use. Only used by the GrapherFactory.
        integration_time: str, optional
            Integration time for the aggregate. Uses the pandas offset aliases
            (i.e. '2H'-> 2 hours, '15min'=> 15 minutes, '1D'-> 1 day) see pandas
            documnentation here:https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        fig_size : (float, float), optional
            Width, height of the figure, in inches. The default is (16, 4).
        share_xaxis : bool, optional
            Share x axis for all subplots. The default is False.
        is_binary: bool, optional
            If set to True, calculates the aggregates in term on presence (1)
            or absence (0). The default is False.
        title : [str, str, ...], str, None,  optional
            Title(s) of the figure/subplots. If a str, a single title is set
            for the entire figure at the top. If a list of str, each element i
            of the list defines the title for the subplot of index i. The
            number of elements in the list can equal or smaller than the number
            of subplots in the graph. A number of elements in teh list greater
            than the number of subplot will return an error. If set to None, no
            title will be displayed. The default is None.
        colormap : str, optional
            Color map for the spectrogram. Uses names from Matplotlib (see
            https://matplotlib.org/stable/tutorials/colors/colormaps.html). The
            default is 'viridis'
        colorbar_label: str, optional
            Label for the color bar. If set to 'auto', the colorbar label is
            automatically defined as "Annotations" or "Detections" based on
            the field "is_detector" of the Annotation object, and "(count)" or
            "(presence/absence)" based on the grapher attribute 'is_binary'.
            The default is 'auto'.
        norm_value: flaat, None
            Maximum value to use for the colobar. The default is None.
        date_format: str, optional
            Date format to use for the x-axis tick labels. Uses standard python
            datetime format codes (see https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes)
            The default is '%d-%b-%Y'.

        Returns
        -------
        None.

        """
        # Initialize all grapher parameters to None
        self.__dict__.update(
            dict(
                zip(
                    self.grapher_parameters,
                    [None] * len(self.grapher_parameters),
                )
            )
        )
        # Define default values:
        self.date_format = "%d-%b-%Y"
        self.colorbar_label = "auto"
        self.share_xaxis = False
        self.integration_time = "1H"
        self.is_binary = False
        self.norm_value = None
        self.fig_size = (16, 4)
        self.title = None
        self.colormap = "viridis"
        self.data = []
        # Unpack kwargs as grapher parameters if provided on instantiation
        self.__dict__.update(**kwargs)

    @property
    def name(self):
        """Return name of the grapher."""
        grapher_name = "AnnotHeatmap"
        return grapher_name

    @property
    def version(self):
        """Return version of the grapher."""
        version = "0.1"
        return version

    def add_data(self, *args):
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
            raise ValueError("There must be at least one input argument")
        # Check  type of each input arguments
        self._stack_data(args)

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
            raise ValueError(
                "No data to plot. Use method .add_data to define"
                " the data to plot"
            )
        # Display plot on screen?
        if display:
            matplotlib.use("Qt5Agg")
        else:
            matplotlib.use("Agg")
        # Define new figure and subplots
        nb_plots = len(self.data)
        fig, ax = plt.subplots(
            nb_plots,
            1,
            figsize=self.fig_size,
            sharex=self.share_xaxis,
            constrained_layout=True,
        )  # gridspec_kw={'hspace': self.hspace}
        # Subplot titles
        titles = [None] * nb_plots
        if self.title is None:  # no titles
            pass
        if type(self.title) is str:
            titles[0] = self.title
        if type(self.title) is list:
            if len(self.title) > nb_plots:
                raise ValueError("More titles than subplots")
            else:
                titles[0 : len(self.title) - 1] = self.title
        # normalization values
        norm_values = [None] * nb_plots
        if self.norm_value is None:  # no titles
            pass
        if (type(self.norm_value) is float) or (type(self.norm_value) is int):
            norm_values = [self.norm_value] * nb_plots
        if type(self.norm_value) is list:
            if len(self.norm_value) > nb_plots:
                raise ValueError("More norm_value than subplots")
            else:
                norm_values[0 : len(self.norm_value) - 1] = self.norm_value
        # Plot data
        for idx, data in enumerate(self.data):
            if nb_plots == 1:
                current_ax = ax
            else:
                current_ax = ax[idx]
            if data["type"] == "annotation":
                self._plot_heatmap(
                    data["data"],
                    current_ax,
                    title=titles[idx],
                    norm_value=norm_values[idx],
                )
            # only dipslay x label of bottom plot if shared axes
            if self.share_xaxis and (idx != nb_plots - 1):
                current_ax.set_xlabel("")
        # fig.tight_layout()
        # fig.set_tight_layout(True)
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
        fig.savefig(
            filename,
            transparent=False,
            bbox_inches="tight",
        )

    def _stack_data(self, args):
        """Stack data to be plotted."""
        for idx, arg in enumerate(args):
            if isinstance(arg, ecosound.core.annotation.Annotation):
                self.data.append({"data": arg, "type": "annotation"})
            else:
                raise ValueError(
                    "Type of input argument not recognized."
                    "Accepted object types: Annotation"
                )

    def _plot_heatmap(self, annot, current_ax, title=None, norm_value=None):
        """Plot heatmap on the current axis"""
        # calulate 1D aggreagate
        data_grid = annot.calc_time_aggregate_2D(
            integration_time=self.integration_time, is_binary=self.is_binary
        )
        axis_date = data_grid.columns.to_list()
        # Plot matrix
        x_lims = mdates.date2num(
            [axis_date[0], axis_date[-1] + dt.timedelta(1)]
        )
        y_lims = mdates.date2num(
            [
                dt.datetime.combine(axis_date[0], dt.time(0, 0)),
                dt.datetime.combine(axis_date[0], dt.time(23, 59, 59)),
            ]
        )
        if self.is_binary:
            norm_value = 1
        im = current_ax.imshow(
            data_grid,
            extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
            vmin=0,
            vmax=norm_value,
            aspect="auto",
            origin="lower",
            cmap=self.colormap,
        )
        current_ax.xaxis_date()
        current_ax.yaxis_date()
        current_ax.xaxis.set_major_formatter(
            mdates.DateFormatter(self.date_format)
        )
        current_ax.yaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        divider = make_axes_locatable(current_ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        current_ax.set_title(title)
        current_ax.set_ylabel("Time of day")
        if self.colorbar_label == "auto":
            if bool(annot.data.from_detector.iloc[0]):
                if self.is_binary:
                    cbar.set_label("Detections \n(presence)")
                else:
                    cbar.set_label("Detections \n(count)")
            else:
                if self.is_binary:
                    cbar.set_label("Annotations \n(presence)")
                else:
                    cbar.set_label("Annotations \n(count)")
        else:
            cbar.set_label(self.colorbar_label)
        return
