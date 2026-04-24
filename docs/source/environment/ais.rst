AIS Query
=========

The :class:`AISQueryHelper` class queries Automatic Identification System (AIS)
vessel-traffic data stored in a local DuckDB database backed by Parquet files.
It supports spatial queries by bounding box or radius, vessel track retrieval,
category-based filtering, and gridded vessel-count aggregation. Results are
returned as GeoDataFrames by default.

.. autoclass:: ecosound.environment.ais.AISQueryHelper
   :members:
   :undoc-members:
   :show-inheritance:
