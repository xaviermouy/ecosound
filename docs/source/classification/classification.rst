Classifier
==========

The :class:`Classifier` class loads a trained scikit-learn model from a pickle
file and applies it to ecosound :class:`~ecosound.core.measurement.Measurement`
data. It handles feature selection, z-score normalisation, and class-label
decoding automatically, returning predicted class labels and confidence scores.

.. autoclass:: ecosound.classification.classification.Classifier
   :members:
   :undoc-members:
   :show-inheritance:
