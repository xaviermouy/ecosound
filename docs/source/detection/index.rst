Detection
=========

The ``detection`` module provides a plug-in architecture for acoustic event
detectors. New detectors are built by subclassing :class:`~ecosound.detection.detector_builder.BaseClass`
and are loaded at runtime by :func:`~ecosound.detection.detector_builder.DetectorFactory`.
Two production detectors are included: a blob detector based on local spectral
variance and a kurtosis-based transient detector.

.. toctree::
   :maxdepth: 1

   detector_builder
   blob_detector
   kurtosis_detector
