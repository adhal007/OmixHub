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

   Example usage:

   .. code-block:: python

      from Connectors.gdc_filters import GDCQueryFilters

      gdc_filters = GDCQueryFilters()

      # Create a custom RNA-Seq data filter
      custom_params = {
         "primary_site": ["Breast"],
         "cases.demographic.gender": ["female"]
      }
      custom_rna_seq_filter = gdc_filters.rna_seq_data_filter(field_params=custom_params)

   For more examples and detailed usage, see the :ref:`examples` section.

.. autoclass:: src.Connectors.gdc_filters.GDCFacetFilters
   :members:
   :undoc-members:
   :show-inheritance:

   Example usage:

   .. code-block:: python

      from Connectors.gdc_filters import GDCFacetFilters

      facet_filters = GDCFacetFilters()

      # Create a facet filter
      facet_filter = facet_filters.create_facet_filter("cases.project.primary_site", ["Breast", "Lung"])



3. Google Cloud Connector
^^^^^^^^^^^^^^^^^^^^^^^^^
.. class:: src.Connectors.gcp_bigquery_utils.BigQueryUtils(project_id)

   Utility class for interacting with Google BigQuery.

   .. method:: table_exists(table_ref)

      Example:
      
      .. code-block:: python

         bq_utils = BigQueryUtils("my-project-id")
         table_ref = "my-project.my_dataset.my_table"
         exists = bq_utils.table_exists(table_ref)
         print(f"Table exists: {exists}")

   .. method:: dataset_exists(dataset_id)

      Example:
      
      .. code-block:: python

         bq_utils = BigQueryUtils("my-project-id")
         dataset_id = "my-project.my_dataset"
         exists = bq_utils.dataset_exists(dataset_id)
         print(f"Dataset exists: {exists}")

   .. method:: upload_df_to_bq(table_id, df)

      Example:
      
      .. code-block:: python

         import pandas as pd

         bq_utils = BigQueryUtils("my-project-id")
         df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
         table_id = "my-project.my_dataset.my_table"
         job = bq_utils.upload_df_to_bq(table_id, df)
         job.result()  # Wait for the job to complete

   .. method:: create_bigquery_table_with_schema(table_id, schema, partition_field=None, clustering_fields=None)

      Example:
      
      .. code-block:: python

         from google.cloud import bigquery

         bq_utils = BigQueryUtils("my-project-id")
         table_id = "my-project.my_dataset.my_table"
         schema = [
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("age", "INTEGER"),
         ]
         table = bq_utils.create_bigquery_table_with_schema(table_id, schema)

   .. method:: df_to_json(df, file_path="data.json")

      Example:
      
      .. code-block:: python

         import pandas as pd

         bq_utils = BigQueryUtils("my-project-id")
         df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
         bq_utils.df_to_json(df, "output.json")

   .. method:: load_json_data(json_object, schema, table_id)

      Example:
      
      .. code-block:: python

         from google.cloud import bigquery

         bq_utils = BigQueryUtils("my-project-id")
         json_object = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]
         schema = [
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("age", "INTEGER"),
         ]
         table_id = "my-project.my_dataset.my_table"
         job = bq_utils.load_json_data(json_object, schema, table_id)
         job.result()  # Wait for the job to complete

   .. method:: run_query(query)

      Example:
      
      .. code-block:: python

         bq_utils = BigQueryUtils("my-project-id")
         query = "SELECT * FROM `my-project.my_dataset.my_table` LIMIT 10"
         df = bq_utils.run_query(query)
         print(df.head())


.. autoclass:: src.Connectors.gcp_bigquery_utils.BigQueryQueries
   :members:
   :undoc-members:
   :show-inheritance:
   Example usage:

   .. code-block:: python

      from src.Connectors.gcp_bigquery_utils import BigQueryQueries

      # Initialize BigQueryQueries
      project_id = "your-project-id"
      dataset_id = "your-dataset-id"
      table_id = "your-table-id"
      bq_queries = BigQueryQueries(project_id, dataset_id, table_id)

      # Get primary site options
      primary_sites = bq_queries.get_primary_site_options()
      print("Primary site options:", primary_sites)

      # Get primary diagnosis options for a specific primary site
      primary_site = "Breast"
      diagnoses = bq_queries.get_primary_diagnosis_options(primary_site)
      print(f"Primary diagnosis options for {primary_site}:", diagnoses)

      # Get DataFrame for PyDeSeq analysis
      primary_diagnosis = "Invasive Ductal Carcinoma"
      df_pydeseq = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
      print("DataFrame for PyDeSeq analysis:")
      print(df_pydeseq.head())

      # Get DataFrame for recurrence-free survival analysis
      df_rfs = bq_queries.get_df_for_recurrence_free_survival_exp(primary_site, primary_diagnosis)
      print("DataFrame for recurrence-free survival analysis:")
      print(df_rfs.head())

      # Get all primary diagnoses for a primary site
      all_diagnoses_df = bq_queries.get_all_primary_diagnosis_for_primary_site(primary_site)
      print(f"All primary diagnoses for {primary_site}:")
      print(all_diagnoses_df)

   This example demonstrates how to use all public methods of the `BigQueryQueries` class:

   1. Initialize the `BigQueryQueries` instance with project, dataset, and table IDs.
   2. Get a list of primary site options using `get_primary_site_options()`.
   3. Get primary diagnosis options for a specific primary site using `get_primary_diagnosis_options()`.
   4. Retrieve a DataFrame for PyDeSeq analysis with `get_df_for_pydeseq()`.
   5. Get a DataFrame for recurrence-free survival analysis using `get_df_for_recurrence_free_survival_exp()`.
   6. Fetch all primary diagnoses for a given primary site with `get_all_primary_diagnosis_for_primary_site()`.

   These methods provide a convenient interface to query and retrieve data from your customly created BigQuery Database 
   for various genomic analyses. The custom creation of BigQuery Tables with partitioning and clustering for optimized 
   query performance is shown in the example :ref:`upload_data_to_bq` method and :ref:`examples` section.

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

   Example usage:

   .. code-block:: python

      import pandas as pd
      from Engines.analysis_engine import AnalysisEngine
      from Connectors.gcp_bigquery_utils import BigQueryQueries

      # Initialize BigQuery connection
      project_id = "your_project_id"
      dataset_id = "your_dataset_id"
      table_id = "your_table_id"
      bq_queries = BigQueryQueries(project_id, dataset_id, table_id)

      # Fetch data from BigQuery
      primary_site = "Breast"
      primary_diagnosis = "Invasive Ductal Carcinoma"
      df = bq_queries.get_df_for_pydeseq(primary_site, primary_diagnosis)
      data_from_bq = df.copy()

      # Optional: Add simulated samples if available
      simulated_samples = None  # Replace with actual simulated samples if available
      if simulated_samples is not None:
         data_from_bq = pd.concat([data_from_bq, simulated_samples], ignore_index=True)

      # Initialize AnalysisEngine
      analysis_cls = AnalysisEngine(data_from_bq, analysis_type='DE')

      # Check if there are enough tumor and normal samples
      if not analysis_cls.check_tumor_normal_counts():
         raise ValueError("Tumor and Normal counts should be at least 10 each")

      # Load gene IDs (adjust the path as needed)
      gene_ids_or_gene_cols = list(pd.read_csv('path/to/gene_id_to_gene_name_mapping.csv')['gene_id'])

      # Expand data for differential expression analysis
      exp_data = analysis_cls.expand_data_from_bq(data_from_bq, gene_ids_or_gene_cols, 'DE')

      # Prepare counts and metadata for PyDESeq2
      counts_for_de = analysis_cls.counts_from_bq_df(exp_data, gene_ids_or_gene_cols)
      metadata = analysis_cls.metadata_for_pydeseq(exp_data)

      # Perform differential expression analysis
      de_results = analysis_cls.run_pydeseq(metadata=metadata, counts=counts_for_de)

      # Display results
      print(de_results.head())

      # Optional: Perform Gene Set Enrichment Analysis (GSEA)
      gene_set = "path/to/your/gene_set.gmt"  # Replace with actual gene set file path
      gsea_results, _, _ = analysis_cls.run_gsea(de_results, gene_set)

      # Display GSEA results
      print(gsea_results.head())

   This example demonstrates how to use the `AnalysisEngine` class for differential expression analysis:

   1. It starts by fetching data from BigQuery using the `BigQueryQueries` class.
   2. Optionally adds simulated samples to the dataset.
   3. Initializes the `AnalysisEngine` with the data and specifies the analysis type as 'DE' (Differential Expression).
   4. Checks if there are enough tumor and normal samples for analysis.
   5. Loads gene IDs from a CSV file.
   6. Expands the data for differential expression analysis.
   7. Prepares counts and metadata for PyDESeq2.
   8. Runs the differential expression analysis using PyDESeq2.
   9. Optionally performs Gene Set Enrichment Analysis (GSEA) on the differential expression results.

   This workflow showcases the key functionalities of the `AnalysisEngine` class for genomic data analysis, particularly focusing on differential expression and enrichment analysis.

2. BigQuery Engine
^^^^^^^^^^^^^
.. autoclass:: src.Engines.bigquery_engine.BigQueryEngine
   :members:
   :undoc-members:
   :show-inheritance:

3. GDC Engine
^^^^^^^^^^^^
.. autoclass:: src.Engines.gdc_engine.GDCEngine
   :members:
   :undoc-members:
   :show-inheritance:

   Example usage:

   .. code-block:: python

      from Engines.gdc_engine import GDCEngine
      import json

      # Initialize GDCEngine
      params = {
         'files.experimental_strategy': 'RNA-Seq', 
         'data_type': 'Gene Expression Quantification'
      }
      gdc_eng_inst = GDCEngine(**params)

      # Set parameters
      new_params = {'cases.project.primary_site': 'Lung'}
      gdc_eng_inst.set_params(**new_params)

      # Get RNA-Seq metadata
      rna_seq_metadata = gdc_eng_inst._get_rna_seq_metadata()
      print(rna_seq_metadata['metadata'].head())

      # Run RNA-Seq data matrix creation
      primary_site = 'Lung'
      ml_data_matrix = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site, downstream_analysis='ML')
      print(ml_data_matrix.head())

      # Create identifier
      sample_row = ml_data_matrix.iloc[0]
      identifier = gdc_eng_inst.create_identifier(sample_row)
      print(f"Identifier: {identifier}")

      # Make count data for BigQuery
      json_object, gene_cols = gdc_eng_inst.make_count_data_for_bq(primary_site, downstream_analysis='DE', format='json')
      print(f"Number of gene columns: {len(gene_cols)}")
      print(json.dumps(json_object[0], indent=2))

      # Make data for recurrence-free survival
      rfs_data, rfs_gene_cols = gdc_eng_inst.make_data_for_recurrence_free_survival(primary_site, downstream_analysis='ML', format='dataframe')
      print(rfs_data.head())
      print(f"Number of gene columns for RFS: {len(rfs_gene_cols)}")

   This example demonstrates the usage of all public methods in the `GDCEngine` class:

   1. Initializing the `GDCEngine` with parameters.
   2. Setting new parameters using `set_params()`.
   3. Retrieving RNA-Seq metadata with `_get_rna_seq_metadata()`.
   4. Running RNA-Seq data matrix creation with `run_rna_seq_data_matrix_creation()`.
   5. Creating a unique identifier for a row using `create_identifier()`.
   6. Making count data for BigQuery with `make_count_data_for_bq()`.
   7. Preparing data for recurrence-free survival analysis with `make_data_for_recurrence_free_survival()`.

   These methods provide a comprehensive toolkit for working with GDC data, from initial querying to preparing data for various types of analyses, including differential expression and machine learning tasks.