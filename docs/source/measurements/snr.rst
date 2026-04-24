SNR
===

The :class:`SNR` measurer computes the signal-to-noise ratio of annotated
acoustic events. Noise is estimated from a window around each detection, with
half the window placed before and half after the signal of interest.

.. autoclass:: ecosound.measurements.snr.SNR
   :members:
   :undoc-members:
   :show-inheritance:
