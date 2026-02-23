Decorators
==========

The ``decorators`` module provides utility decorators used internally by
ecosound. :func:`listinput` ensures that a function always receives its first
argument as a list, making it easy to write functions that accept both a single
item and a list of items without duplicating logic. :func:`timeit` wraps any
function to print its wall-clock execution time, which is useful for profiling
and debugging.

.. automodule:: ecosound.core.decorators
   :members:
   :undoc-members: