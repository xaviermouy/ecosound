Measurement Tutorial
====================

.. contents:: Contents
   :local:
   :depth: 2

:class:`~ecosound.core.measurement.Measurement` extends
:class:`~ecosound.core.annotation.Annotation` by adding a structured way to
attach acoustic feature measurements to each detection.  In addition to the
standard annotation columns, a :class:`Measurement` object stores:

- **measurer metadata** — name and version of the algorithm that computed the
  features.
- **measurement columns** — one extra column per feature (e.g. SNR, peak
  frequency, duration).

Because :class:`Measurement` inherits from :class:`Annotation`, every import,
export, filtering, and visualisation method described in the
:doc:`annotation` tutorial also works here.

.. code-block:: python

   from ecosound.core.measurement import Measurement
   from ecosound.core.annotation import Annotation


Creating a Measurement object
------------------------------

Pass the measurer name, version, feature names, and any algorithm parameters
to the constructor.  The ``data`` DataFrame will have all standard annotation
columns plus one column for each entry in ``measurements_name``.

.. code-block:: python

   meas = Measurement(
       measurer_name='SpectrogramFeatures',
       measurer_version='1.0',
       measurements_name=['SNR_dB', 'peak_freq_Hz', 'duration_s'],
       measurements_parameters={'window_sec': 0.1},
   )
   print(meas)
   print()
   print('metadata DataFrame:')
   print(meas.metadata.to_string())
   print()
   print('Extra measurement columns in data:', ['SNR_dB', 'peak_freq_Hz', 'duration_s'])
   print('Total columns in data DataFrame   :', len(meas.data.columns))

.. code-block:: text

   0 annotation(s)

   metadata DataFrame:
            measurer_name measurer_version                   measurements_name measurements_parameters
   0  SpectrogramFeatures              1.0  [SNR_dB, peak_freq_Hz, duration_s]     {'window_sec': 0.1}

   Extra measurement columns in data: ['SNR_dB', 'peak_freq_Hz', 'duration_s']
   Total columns in data DataFrame   : 38

The ``metadata`` property is a one-row :class:`pandas.DataFrame` that records
what algorithm produced the measurements and how it was configured.


Inspecting metadata
-------------------

.. code-block:: python

   print('measurer_name    :', meas.metadata['measurer_name'].values[0])
   print('measurer_version :', meas.metadata['measurer_version'].values[0])
   print('measurements_name:', meas.metadata['measurements_name'].values[0])
   print('parameters       :', meas.metadata['measurements_parameters'].values[0])

.. code-block:: text

   measurer_name    : SpectrogramFeatures
   measurer_version : 1.0
   measurements_name: ['SNR_dB', 'peak_freq_Hz', 'duration_s']
   parameters       : {'window_sec': 0.1}


Loading measurements from SQLite
----------------------------------

Because :class:`Measurement` inherits :meth:`~ecosound.core.annotation.Annotation.from_sqlite`,
you can load a SQLite database of detections (produced by a detector or
classifier) directly into a :class:`Measurement` object.

.. code-block:: python

   annot_meas = Annotation()
   annot_meas.from_sqlite('data/sqlite_annotations/read/detections1.sqlite',
                          verbose=True)

   print()
   print('label_class values :', annot_meas.get_labels_class())
   print()
   print('confidence describe:')
   print(annot_meas.data['confidence'].describe().round(3))

.. code-block:: text

   Duplicate entries removed: 0
   Integrity test successful
   19240 annotations imported.

   label_class values : ['MW']

   confidence describe:
   count    19240.000
   mean         0.753
   std          0.137
   min          0.263
   25%          0.642
   50%          0.760
   75%          0.867
   max          1.000
   Name: confidence, dtype: float64


Filtering by confidence
-----------------------

All filtering methods from the :class:`Annotation` base class are available.
For example, to keep only high-confidence detections:

.. code-block:: python

   high_conf = annot_meas.data[annot_meas.data['confidence'] >= 0.8].copy()
   print('Total detections       :', len(annot_meas.data))
   print('High-confidence (>=0.8):', len(high_conf))

.. tip::

   The :meth:`~ecosound.core.annotation.Annotation.filter_by_values` method
   (inherited from :class:`Annotation`) provides a more concise interface for
   filtering on label, confidence, or any other column.


Complete Measurement workflow
------------------------------

A typical workflow with :class:`Measurement` looks like this:

.. code-block:: python

   from ecosound.core.measurement import Measurement
   from ecosound.core.metadata import DeploymentInfo

   # 1. Create an empty Measurement with feature schema
   meas = Measurement(
       measurer_name='MyDetector',
       measurer_version='2.3',
       measurements_name=['SNR_dB', 'peak_freq_Hz'],
       measurements_parameters={'threshold_dB': 6.0},
   )

   # 2. Run your detector and populate meas.data with rows
   #    (each row is one detection, with all standard annotation columns
   #     plus the SNR_dB and peak_freq_Hz columns filled in)

   # 3. Insert deployment metadata
   dep = DeploymentInfo()
   dep.read('deployment_filled.csv')
   meas.insert_metadata(dep)

   # 4. Save to SQLite for later use
   meas.to_sqlite('my_detections.sqlite')

   # 5. Reload
   meas2 = Measurement(
       measurer_name='MyDetector',
       measurer_version='2.3',
       measurements_name=['SNR_dB', 'peak_freq_Hz'],
       measurements_parameters={'threshold_dB': 6.0},
   )
   meas2.from_sqlite('my_detections.sqlite', verbose=True)
