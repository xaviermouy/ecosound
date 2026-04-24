Blob Detector
=============

The :class:`BlobDetector` finds transient acoustic events in a spectrogram by
computing local 2-D variance, binarising the result against a threshold, and
tracing connected regions using Moore's Neighbourhood algorithm. Detections are
filtered by minimum duration and bandwidth before being returned as an
:class:`~ecosound.core.annotation.Annotation` object.

.. autoclass:: ecosound.detection.blob_detector.BlobDetector
   :members:
   :undoc-members:
   :show-inheritance:
