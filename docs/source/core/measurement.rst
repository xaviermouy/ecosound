Measurement
===========

The :class:`Measurement` class extends :class:`~ecosound.core.annotation.Annotation`
with acoustic measurement data. In addition to the standard annotation fields,
it stores one extra column per measurement (e.g. peak frequency, SNR, spectral
centroid) and carries a ``metadata`` attribute that records the measurer name,
version, measurement column names, and parameters. The class supports import
and export in NetCDF and Raven formats, and multiple :class:`Measurement`
objects produced by the same measurer can be concatenated with the ``+``
operator.

.. autoclass:: ecosound.core.measurement.Measurement
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __add__