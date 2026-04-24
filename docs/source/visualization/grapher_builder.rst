Grapher Builder
===============

The ``grapher_builder`` module defines the plug-in infrastructure shared by
all graphers. Every grapher must inherit from :class:`BaseClass` and implement
the required interface; :func:`GrapherFactory` discovers and instantiates the
correct subclass at runtime from a plain string name.

Grapher BaseClass
-----------------

.. autoclass:: ecosound.visualization.grapher_builder.BaseClass
   :members:
   :undoc-members:
   :show-inheritance:

GrapherFactory
--------------

.. autofunction:: ecosound.visualization.grapher_builder.GrapherFactory
