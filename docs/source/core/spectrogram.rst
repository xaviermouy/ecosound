Spectrogram
===========

The :class:`Spectrogram` class computes and stores a short-time Fourier
transform (STFT) spectrogram from a :class:`~ecosound.core.audiotools.Sound`
object. It provides methods for computing the spectrogram with configurable
window length, overlap, and FFT size; applying frequency-domain denoising
(median equaliser); converting between linear and decibel scales; and
generating plots. The computed time–frequency representation is stored as a
2-D NumPy array together with the corresponding time and frequency axes.

.. autoclass:: ecosound.core.spectrogram.Spectrogram
   :members:
   :undoc-members:
   :show-inheritance: