Decorators Tutorial
===================

.. contents:: Contents
   :local:
   :depth: 2

The :mod:`ecosound.core.decorators` module provides two small utility
decorators used internally by ecosound and available for use in your own code.

.. code-block:: python

   from ecosound.core.decorators import listinput, timeit


listinput
---------

``@listinput`` ensures that the first argument of the decorated function is
always a list.  If a single (non-list) value is passed, it is automatically
wrapped in a list before the function is called.  This lets functions be written
to expect a list while still accepting a scalar for convenience.

**Example — defining a list-aware function:**

.. code-block:: python

   @listinput
   def process_files(files):
       """Print every file name in the list."""
       for f in files:
           print('Processing:', f)

   # Works with a list:
   process_files(['file_a.wav', 'file_b.wav'])

   # Also works with a single string (wrapped automatically):
   process_files('file_a.wav')

.. code-block:: text

   Processing: file_a.wav
   Processing: file_b.wav

   Processing: file_a.wav

**Why this is useful:**  functions like
:func:`~ecosound.core.tools.list_files` and annotation importers in ecosound
accept both a single path and a list of paths.  The ``@listinput`` decorator
implements this behaviour with a single line.


timeit
------

``@timeit`` wraps a function so that its wall-clock execution time is printed
to the console every time it is called.  Useful during development for quick
benchmarking without importing a profiler.

**Example:**

.. code-block:: python

   import time

   @timeit
   def slow_function(n):
       """Simulate work by sleeping."""
       time.sleep(n)
       return n * 2

   result = slow_function(0.5)
   print('Result:', result)

.. code-block:: text

   Executed in 0.5003 seconds
   Result: 1

The decorator preserves the original function's name and docstring
(via :func:`functools.wraps`), so it is transparent to introspection tools and
documentation generators.

**Combining both decorators:**

.. code-block:: python

   @timeit
   @listinput
   def analyse(items):
       total = 0
       for item in items:
           total += len(item)
       return total

   # Single string — automatically wrapped by @listinput, timed by @timeit
   analyse('hello')

.. code-block:: text

   Executed in 0.0000 seconds
