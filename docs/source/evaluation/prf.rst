PRF
===

The :class:`PRF` class computes Precision, Recall, and F-score for acoustic
detectors against ground-truth annotations. Two evaluation modes are supported:

* **count** — compares individual detection events to annotations file by file,
  sweeping over a range of confidence thresholds.
* **presence** — aggregates annotations and detections into fixed time bins and
  evaluates presence/absence agreement across thresholds.

.. autoclass:: ecosound.evaluation.prf.PRF
   :members:
   :undoc-members:
   :show-inheritance:
