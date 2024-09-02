API Reference
=============

.. toctree::
   :maxdepth: 2
   :caption: Modules
=================

This package has two modules, detailed below.  

Connectors
----------
.. automodule:: src.Connectors
   :members:
   :undoc-members:
   :show-inheritance:

1. GDC Endpoint Connectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gdc_endpt_base.GDCEndptBase
   :members:
   :undoc-members:
   :show-inheritance:


2. GDC Filters
^^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gdc_filters.GDCQueryFilters
   :members:
   :undoc-members:
   :show-inheritance:


.. autoclass:: src.Connectors.gdc_filters.GDCFacetFilters
   :members:
   :undoc-members:
   :show-inheritance:



3. GDC Parser
^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gdc_parser.GDCJson2DfParser
   :members:
   :undoc-members:
   :show-inheritance:



4. Google Cloud Connector
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gcp_bigquery_utils.BigQueryUtils
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: src.Connectors.gcp_bigquery_utils.BigQueryQueries
   :members:
   :undoc-members:
   :show-inheritance:

Engines
-------
.. automodule:: src.Engines
   :members:
   :undoc-members:
   :show-inheritance:

1. Analysis Engine
^^^^^^^^^^^^^^^^^^
.. autoclass:: src.Engines.analysis_engine.AnalysisEngine
   :members:
   :undoc-members:
   :show-inheritance:


2. BigQuery Engine
^^^^^^^^^^^^^
.. autoclass:: src.Engines.bigquery_engine.BigQueryEngine
   :members:
   :undoc-members:
   :show-inheritance:


