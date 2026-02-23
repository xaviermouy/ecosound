Audio Tools
===========

The ``audiotools`` module provides classes and functions for loading, processing,
and writing audio data. The :class:`Sound` class wraps a waveform array and
exposes methods for reading WAV files, applying bandpass filters, computing
signal envelopes, normalising amplitude, and generating plots. The
:class:`Filter` class designs Butterworth bandpass filters. The standalone
:func:`upsample` function resamples audio to a higher sampling rate using
polyphase filtering.

Sound
-----

.. autoclass:: ecosound.core.audiotools.Sound
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __len__

Filter
------

.. autoclass:: ecosound.core.audiotools.Filter
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: ecosound.core.audiotools.upsample