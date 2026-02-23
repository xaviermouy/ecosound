Metadata Tutorial
=================

.. contents:: Contents
   :local:
   :depth: 2

The :mod:`ecosound.core.metadata` module provides the
:class:`~ecosound.core.metadata.DeploymentInfo` class for managing hydrophone
deployment metadata.  A CSV template is generated, filled in by the user, and
then read back into the object.  Once loaded, the metadata can be inserted into
:class:`~ecosound.core.annotation.Annotation` or
:class:`~ecosound.core.measurement.Measurement` objects.

.. code-block:: python

   from ecosound.core.metadata import DeploymentInfo
   import pandas as pd


DeploymentInfo class
--------------------

Writing a template
~~~~~~~~~~~~~~~~~~

:meth:`~ecosound.core.metadata.DeploymentInfo.write_template` creates a
one-row CSV file with all required column headers and empty values.  Open it in
any spreadsheet application, fill in the values for your deployment, and save.

.. code-block:: python

   dep = DeploymentInfo()
   dep.write_template('deployment_template.csv')

   df_tmpl = pd.read_csv('deployment_template.csv')
   print('Template columns:')
   for col in df_tmpl.columns:
       print('  -', col)

.. code-block:: text

   Template columns:
     - audio_channel_number
     - UTC_offset
     - sampling_frequency
     - bit_depth
     - mooring_platform_name
     - recorder_type
     - recorder_SN
     - hydrophone_model
     - hydrophone_SN
     - hydrophone_depth
     - location_name
     - location_lat
     - location_lon
     - location_water_depth
     - deployment_ID
     - deployment_date
     - recovery_date


Reading a filled template
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the CSV has been filled in, call
:meth:`~ecosound.core.metadata.DeploymentInfo.read` to load the metadata into
the object.  The data are accessible via the ``data`` attribute as a
:class:`pandas.DataFrame`.

.. code-block:: python

   dep2 = DeploymentInfo()
   dep2.read('deployment_filled.csv')
   print(dep2.data.T.to_string())

.. code-block:: text

                                      0
   audio_channel_number               0
   UTC_offset                        -8
   sampling_frequency             32000
   bit_depth                         24
   mooring_platform_name  Bottom lander
   recorder_type                   AMAR
   recorder_SN                      173
   hydrophone_model          HTI-96-MIN
   hydrophone_SN                  12345
   hydrophone_depth                40.0
   location_name          Hornby Island
   location_lat                   49.52
   location_lon                 -124.68
   location_water_depth            55.0
   deployment_ID            HB-2019-001
   deployment_date           2019-09-16
   recovery_date             2020-03-15


Inserting metadata into an Annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After loading, pass the :class:`DeploymentInfo` object to
:meth:`Annotation.insert_metadata` to stamp every row in the annotation table
with the deployment fields.

.. code-block:: python

   from ecosound.core.annotation import Annotation

   annot = Annotation()
   annot.from_raven('data/Raven_annotations/AMAR173.4.20190916T061248Z.Table.1.selections.txt',
                    class_header='Sound type',
                    verbose=False)

   dep = DeploymentInfo()
   dep.read('deployment_filled.csv')
   annot.insert_metadata(dep)

   # The deployment fields are now available in annot.data:
   print(annot.data[['location_name', 'recorder_type', 'hydrophone_depth']].head(3))

.. tip::

   :meth:`~ecosound.core.annotation.Annotation.insert_metadata` can be called at
   any point before exporting.  It is a convenient way to enrich annotation
   tables with location, instrument, and deployment information before saving to
   SQLite, NetCDF, or Parquet.


Complete metadata workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ecosound.core.annotation import Annotation
   from ecosound.core.metadata import DeploymentInfo

   # 1. Write a template, fill it in, then read it back
   dep = DeploymentInfo()
   dep.write_template('my_deployment.csv')
   # ... edit my_deployment.csv in Excel / LibreOffice Calc ...
   dep.read('my_deployment.csv')

   # 2. Load annotations
   annot = Annotation()
   annot.from_raven('data/Raven_annotations/AMAR173.4.20190916T061248Z.Table.1.selections.txt',
                    class_header='Sound type',
                    verbose=False)

   # 3. Stamp with deployment metadata
   annot.insert_metadata(dep)

   # 4. Export enriched table
   annot.to_sqlite('annotations_with_metadata.sqlite')
