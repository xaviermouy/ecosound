Visualization
=============

The ``visualization`` module provides graphers for displaying annotation data,
waveforms, spectrograms, gridded environmental data, and AIS vessel traffic on
interactive maps. Graphers follow the same plug-in pattern as detectors and
measurers: every grapher inherits from
:class:`~ecosound.visualization.grapher_builder.BaseClass` and is instantiated
through :func:`~ecosound.visualization.grapher_builder.GrapherFactory`.

.. toctree::
   :maxdepth: 1

   grapher_builder
   annotation_plotter
   sound_plotter
   grid_plotter
   ais_plotter
