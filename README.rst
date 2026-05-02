
.. image:: docs/source/_static/ecosound_logo_small.png


Welcome to ecosound!
====================

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


**Ecosound** is an open source Python package to facilitate the analysis of
passive acoustic data. It includes modules for manual annotation processing
and visualization, automatic detection, signal classification, and
localization. It heavily relies on libraries such as xarray, pandas, numpy,
and scikit-learn. Under the hood it also uses dask, which supports the
processing of large data sets that don't fit into memory and makes processing
scalable through distributed computing (on either local clusters or on the
cloud). Outputs from ecosound are compatible with popular bioacoustics
software such as `Raven <https://ravensoundsoftware.com/>`_ and
`PAMlab <https://static1.squarespace.com/static/52aa2773e4b0f29916f46675/t/5be5b07088251b9f59268184/1541779574284/PAMlab+Brochure.pdf>`_.


Features
--------

* **Annotation** — load, filter, merge, and export manual annotations from Raven and PAMlab
* **Audio tools** — read audio files, apply filters, compute spectrograms
* **Detection** — plug-in detectors (blob, kurtosis) with a common factory interface
.. * **Classification** — apply trained scikit-learn classifiers to acoustic measurements
.. * **Measurements** — extract spectral and temporal features from annotated signals
* **Evaluation** — compute Precision, Recall, and F-score curves for detectors
.. * **Environment** — fetch co-located oceanographic, weather, tidal, and AIS data
.. * **Soundscape** — process Hybrid Millidecade (HMD) spectral data for long-term soundscape analysis
* **Visualization** — plot waveforms, spectrograms, annotation heatmaps, and interactive AIS maps


Installation
------------

.. code-block:: bash

   pip install ecosound


Quick Start
-----------

.. code-block:: python

   from ecosound.core.annotation import Annotation

   # Load annotations from a Raven selection table
   annot = Annotation()
   annot.from_raven('my_annotations.txt', class_header='Sound type')

   # Keep only high-confidence detections with the label "MW"
   annot.data = annot.filter('label_class == "MW" & confidence >= 0.8')

   # Aggregate and visualise
   annot.plot_heatmap()


Status
------
Ecosound is very much a work in progress and is still under heavy development.
At this stage, it is recommended to contact the main contributor before using
ecosound for your projects.


Documentation
-------------
API documentation is available at https://ecosound.readthedocs.io.


Contributors
------------

`Xavier Mouy <https://xaviermouy.weebly.com/>`_ (@XavierMouy),
Acoustics and Conservation Technology (ACT) Lab,
Woods Hole Oceanographic Institution (WHOI).


Support
-------

Over the years, this project has received support from:

* `Woods Hole Oceanographic Institution (WHOI) <https://www.whoi.edu/>`_
* `NOAA Fisheries <https://www.fisheries.noaa.gov/>`_
* `Canadian Healthy Oceans Network (CHONe) <https://chone2.ca/>`_
* `University of Victoria <https://www.uvic.ca/>`_
* `Fisheries and Oceans Canada <https://www.dfo-mpo.gc.ca/>`_


License
-------
Ecosound is licensed under the open source `BSD-3-Clause License <https://choosealicense.com/licenses/bsd-3-clause/>`_.
