Spectrogram Features
====================

The :class:`SpectrogramFeatures` measurer extracts a comprehensive set of
spectral and temporal features from a spectrogram for each annotated
detection. Features are based on measurements from the software Raven and
from the R package Seewave (Sueur et al.), and include energy, kurtosis,
skewness, entropy, and frequency centroid among others.

.. autoclass:: ecosound.measurements.spectrogram_features.SpectrogramFeatures
   :members:
   :undoc-members:
   :show-inheritance:
