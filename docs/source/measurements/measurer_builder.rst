Measurer Builder
================

The ``measurer_builder`` module defines the plug-in infrastructure shared by
all measurers. Every measurer must inherit from :class:`BaseClass` and
implement the required interface; :func:`MeasurerFactory` discovers and
instantiates the correct subclass at runtime from a plain string name.

Measurer BaseClass
------------------

.. autoclass:: ecosound.measurements.measurer_builder.BaseClass
   :members:
   :undoc-members:
   :show-inheritance:

MeasurerFactory
---------------

.. autofunction:: ecosound.measurements.measurer_builder.MeasurerFactory
