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

1. GDC Files Endpoint Connector
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gdc_files_endpt.GDCFilesEndpt
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods
   .. autofunction:: src.Connectors.gdc_files_endpt.GDCFilesEndpt.__init__
   .. autofunction:: src.Connectors.gdc_files_endpt.GDCFilesEndpt.rna_seq_query_to_json

2. GDC Filters
^^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gdc_filters.GDCQueryFilters
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.__init__
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.create_and_filters
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.all_projects_by_exp_filter
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.rna_seq_data_filter
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.all_diseases
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.create_single_facet_filter
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.get_files_endpt_facet_filter
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.create_single_facet_df
   .. autofunction:: src.Connectors.gdc_filters.GDCQueryFilters.get_files_facet_data


.. autoclass:: src.Connectors.gdc_filters.GDCFacetFilters
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods
   .. autofunction:: src.Connectors.gdc_filters.GDCFacetFilters.__init__
   .. autofunction:: src.Connectors.gdc_filters.GDCFacetFilters.create_single_facet_filter
   .. autofunction:: src.Connectors.gdc_filters.GDCFacetFilters.get_files_endpt_facet_filter
   .. autofunction:: src.Connectors.gdc_filters.GDCFacetFilters.create_single_facet_df
   .. autofunction:: src.Connectors.gdc_filters.GDCFacetFilters.get_files_facet_data


3. GDC Parser
^^^^^^^^^^^^^
.. autoclass:: src.Connectors.gdc_parser.GDCParser
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods
   .. autofunction:: src.Connectors.gdc_parser.GDCParser.__init__
   .. autofunction:: src.Connectors.gdc_parser.GDCParser.get_unnested_dict_for_rna_seq
   .. autofunction:: src.Connectors.gdc_parser.GDCParser.make_df_rna_seq

4. Google Cloud Connector
^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: src.Connectors.google_cloud_conn.BigQueryUtils
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.__init__
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.table_exists
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.dataset_exists
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.upload_df_to_bq
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.create_bigquery_table_with_schema
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.df_to_json
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.load_json_data
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.create_identifier
   .. autofunction:: src.Connectors.google_cloud_conn.BigQueryUtils.upload_partitioned_clustered_table

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

   .. rubric:: Methods

   .. autofunction:: src.Engines.analysis_engine.AnalysisEngine.expand_data_from_bq
   .. autofunction:: src.Engines.analysis_engine.AnalysisEngine.counts_from_bq_df
   .. autofunction:: src.Engines.analysis_engine.AnalysisEngine.metadata_for_pydeseq
   .. autofunction:: src.Engines.analysis_engine.AnalysisEngine.run_pydeseq
   .. autofunction:: src.Engines.analysis_engine.AnalysisEngine.run_gsea
   .. autofunction:: src.Engines.analysis_engine.AnalysisEngine.data_for_ml

2. GDC Engine
^^^^^^^^^^^^^
.. autoclass:: src.Engines.gdc_engine.GDCEngine
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autofunction:: src.Engines.gdc_engine.GDCEngine.set_params
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._check_data_type
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._check_exp_type
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._get_raw_data
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._make_file_id_url_map
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._get_urls_content
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._get_rna_seq_metadata
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._make_rna_seq_data_matrix
   .. autofunction:: src.Engines.gdc_engine.GDCEngine._process_data_matrix_rna_seq
   .. autofunction:: src.Engines.gdc_engine.GDCEngine.run_rna_seq_data_matrix_creation

