.. ecosound documentation master file, created by
   sphinx-quickstart on Fri Jan 15 17:59:15 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ecosound's documentation!
====================================

.. image:: https://img.shields.io/pypi/v/ecosound.svg
        :target: https://pypi.python.org/pypi/ecosound

.. image:: https://readthedocs.org/projects/ecosound/badge/?version=latest
        :target: https://ecosound.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://static.pepy.tech/badge/ecosound
    :target: https://pepy.tech/project/ecosound
    :alt: Total PyPI downloads

.. image:: https://img.shields.io/pypi/dm/ecosound
    :target: https://pypi.python.org/pypi/ecosound
    :alt: Monthly PyPI downloads

.. image:: https://img.shields.io/github/stars/xaviermouy/ecosound?style=social
    :target: https://github.com/xaviermouy/ecosound
    :alt: GitHub stars

.. image:: https://img.shields.io/github/forks/xaviermouy/ecosound?style=social
    :target: https://github.com/xaviermouy/ecosound
    :alt: GitHub forks



**Ecosound** is an open source python package to facilitate the analysis of passive acoustic data. It includes modules for manual annotation
processing and visualization, automatic detection, signal classification, and localization. It heavily relies on libraries such as xarray,
pandas, numpy and scikit-learn. Under the hood it also uses dask which supports the processing of large data sets that don’t fit into memory,
and makes processing scalable through distributed computing (on either local clusters or on the cloud). Outputs from ecosound are compatible 
with popular bioacoustics software such as `Raven <https://ravensoundsoftware.com/>`_ and 
`PAMlab <https://static1.squarespace.com/static/52aa2773e4b0f29916f46675/t/5be5b07088251b9f59268184/1541779574284/PAMlab+Brochure.pdf>`_.


Installation
------------

Ecosound can be installed from PyPI using pip:

.. code-block:: bash

   pip install ecosound


Quick Start Example
-------------------

The example below loads a Raven annotation file, filters detections by
confidence, and plots a summary heatmap:

.. code-block:: python

   from ecosound.core.annotation import Annotation

   # Load annotations from a Raven selection table
   annot = Annotation()
   annot.from_raven('my_annotations.txt', class_header='Sound type')

   # Keep only high-confidence detections
   annot.data = annot.data[annot.data['confidence'] >= 0.8]

   # Aggregate and visualise
   annot.plot_heatmap()


.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   core/index
   classification/index
   detection/index
   environment/index
   evaluation/index
   measurements/index
   soundscape/index
   visualization/index


Status
------
Ecosound is very much a work in progress and is still under heavy development. 
At this stage, it is recommended to contact the main contributor before using
ecosound for your projects.


GitHub repository
-----------------
https://github.com/xaviermouy/ecosound


Contributors
------------

`Xavier Mouy <https://xaviermouy.weebly.com/>`_ (@XavierMouy), Acoustics and Conservation Technology (ACT) Lab, Woods Hole Oceanographic Institution (WHOI).

Support
-------

This project has received funding and support from:

* `Woods Hole Oceanographic Institution (WHOI) <https://www.whoi.edu/>`_
* `NOAA Fisheries <https://www.fisheries.noaa.gov/>`_
* `Canadian Healthy Oceans Network (CHONe) <https://chone2.ca/>`_
* `University of Victoria <https://www.uvic.ca/>`_
* `Fisheries and Oceans Canada <https://www.dfo-mpo.gc.ca/>`_


License
-------
Ecosound is licensed under the open source `BSD-3-Clause License <https://choosealicense.com/licenses/bsd-3-clause/>`_.

