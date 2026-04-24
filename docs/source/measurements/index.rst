Measurements
============

The ``measurements`` module provides a plug-in architecture for extracting
acoustic measurements from annotated signals. Every measurer inherits from
:class:`~ecosound.measurements.measurer_builder.BaseClass` and is instantiated
through :func:`~ecosound.measurements.measurer_builder.MeasurerFactory`.
Included measurers cover signal-to-noise ratio and a comprehensive set of
spectral and temporal features.

.. toctree::
   :maxdepth: 1

   measurer_builder
   snr
   spectrogram_features
