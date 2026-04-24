Cross-Validation
================

The ``CrossValidation`` module extends scikit-learn's splitter API with
group-aware, stratified cross-validation strategies. These are needed when
data contain groups (e.g. recordings from the same individual or deployment)
that must not be split across folds.

StratifiedGroupKFold
--------------------

.. autoclass:: ecosound.classification.CrossValidation.StratifiedGroupKFold
   :members:
   :undoc-members:
   :show-inheritance:

RepeatedStratifiedGroupKFold
----------------------------

.. autoclass:: ecosound.classification.CrossValidation.RepeatedStratifiedGroupKFold
   :members:
   :undoc-members:
   :show-inheritance:
