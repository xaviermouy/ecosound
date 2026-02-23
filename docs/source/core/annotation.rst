Annotation
==========

The :class:`Annotation` class is the central data structure in ecosound. It
stores annotation data from manual analysis tools (Raven, PAMlab) and from
automatic detectors and classifiers in a unified pandas DataFrame. It
supports import and export in multiple formats (Raven, PAMlab, Parquet,
NetCDF, SQLite, CSV) and provides filtering, aggregation, and integrity
checking methods.

.. autoclass:: ecosound.core.annotation.Annotation
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __add__, __len__, __repr__, __str__
