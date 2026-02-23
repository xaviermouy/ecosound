Metadata
========

The ``metadata`` module provides the :class:`DeploymentInfo` class for
managing acoustic deployment metadata. A deployment record typically describes
the recorder hardware, hydrophone, mooring platform, geographic location, and
deployment dates. This information can be loaded from a CSV file and then
applied to :class:`~ecosound.core.annotation.Annotation` or
:class:`~ecosound.core.measurement.Measurement` objects to populate their
metadata fields. A blank CSV template with all required column headers can be
generated with :meth:`~DeploymentInfo.write_template`.

DeploymentInfo
--------------

.. autoclass:: ecosound.core.metadata.DeploymentInfo
   :members:
   :undoc-members:
   :show-inheritance: