Kurtosis Detector
=================

The :class:`KurtosisDetector` detects impulsive acoustic events by computing
the kurtosis of the waveform over a rolling window and identifying peaks above
a threshold. The module also provides helper classes for storing detections and
for visualising the kurtosis time-series and peak picks.

KurtosisDetector
----------------

.. autoclass:: ecosound.detection.kurtosis_detector.KurtosisDetector
   :members:
   :undoc-members:
   :show-inheritance:

Detections
----------

.. autoclass:: ecosound.detection.kurtosis_detector.Detections
   :members:
   :undoc-members:
   :show-inheritance:

Figure
------

.. autoclass:: ecosound.detection.kurtosis_detector.Figure
   :members:
   :undoc-members:
   :show-inheritance:
