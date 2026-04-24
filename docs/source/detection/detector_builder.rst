Detector Builder
================

The ``detector_builder`` module defines the plug-in infrastructure that all
detectors share. Every detector must inherit from :class:`BaseClass` and
implement the required interface; :func:`DetectorFactory` discovers and
instantiates the correct subclass at runtime from a plain string name.

BaseClass
---------

.. autoclass:: ecosound.detection.detector_builder.BaseClass
   :members:
   :undoc-members:
   :show-inheritance:

DetectorFactory
---------------

.. autofunction:: ecosound.detection.detector_builder.DetectorFactory
