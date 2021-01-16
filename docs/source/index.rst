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
		
.. image:: https://travis-ci.com/xaviermouy/ecosound.svg?branch=master
    :target: https://travis-ci.com/xaviermouy/ecosound

.. image:: https://coveralls.io/repos/github/xaviermouy/ecosound/badge.svg?branch=master
	:target: https://coveralls.io/github/xaviermouy/ecosound?branch=master



**Ecosound** is an open source python package to facilitate the analysis of passive acoustic data. It includes modules for manual annotation
processing and visualization, automatic detection, signal classification, and localization. It heavily relies on libraries such as xarray,
pandas, numpy and scikit-learn. Under the hood it also uses dask which supports the processing of large data sets that don’t fit into memory,
and makes processing scalable through distributed computing (on either local clusters or on the cloud). Outputs from ecosound are compatible 
with popular bioacoustics software such as `Raven <https://ravensoundsoftware.com/>`_ and 
`PAMlab <https://static1.squarespace.com/static/52aa2773e4b0f29916f46675/t/5be5b07088251b9f59268184/1541779574284/PAMlab+Brochure.pdf>`_.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


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

`Xavier Mouy <https://xaviermouy.weebly.com/>`_ (@XavierMouy) leads this project as part of his PhD in the `Juanes Lab <https://juaneslab.weebly.com/>`_ 
at the University of Victoria (British Columbia, Canada).

Credits
-------

* This project was initiated in the `Juanes Lab <https://juaneslab.weebly.com/>`_ at the University of Victoria (British Columbia, Canada) and received funding from the `Canadian Healthy Oceans Network <https://chone2.ca/>`_ and `Fisheries and Oceans Canada - Pacific Region <https://www.dfo-mpo.gc.ca/contact/regions/pacific-pacifique-eng.html#Nanaimo-Lab>`_. 


License
-------
Ecosound is licensed under the open source `BSD-3-Clause License <https://choosealicense.com/licenses/bsd-3-clause/>`_.

