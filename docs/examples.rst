Examples
=============

Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to clone the repository and view the example notebooks.



Useful query filters for GDC API endpoints
------------------------------------------

The following examples demonstrate how to use various filters from the GDCQueryFilters class to query different GDC API endpoints.
These examples demonstrate how to create filters for various GDC data types and endpoints.

.. collapse:: RNA-Seq Filter

    .. code-block:: python

        from Connectors.gdc_filters import GDCQueryFilters

        gdc_filters = GDCQueryFilters()
        rna_seq_filter = gdc_filters.rna_seq_filter()
        
        # Use this filter with the 'files' endpoint
        # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": rna_seq_filter, "size": 10})

.. collapse:: WGS Filter

    .. code-block:: python

        wgs_filter = gdc_filters.wgs_filter()
        
        # Use this filter with the 'files' endpoint
        # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": wgs_filter, "size": 10})

.. collapse:: Methylation Filter

    .. code-block:: python

        methylation_filter = gdc_filters.methylation_filter()
        
        # Use this filter with the 'files' endpoint
        # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": methylation_filter, "size": 10})

.. collapse:: Top Mutated Genes Filter

    .. code-block:: python

        top_mutated_genes_filter = gdc_filters.top_mutated_genes_by_project_filter("TCGA-BRCA", top_n=5)
        
        # Use this filter with the 'analysis/top_mutated_genes_by_project' endpoint
        # Example: requests.get("https://api.gdc.cancer.gov/analysis/top_mutated_genes_by_project", 
        #                       params={"filters": json.dumps(top_mutated_genes_filter), "fields": "gene_id,symbol,score", "size": 5})

.. collapse:: Custom RNA-Seq Data Filter

    .. code-block:: python

        custom_params = {
            "cases.project.primary_site": ["Breast"],
            "cases.demographic.gender": ["female"]
        }
        custom_rna_seq_filter = gdc_filters.rna_seq_data_filter(field_params=custom_params)
        
        # Use this filter with the 'files' endpoint
        # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": custom_rna_seq_filter, "size": 10})

Data Processing and Analysis Examples
-------------------------------------

.. collapse:: Cohort Creation of Bulk RNA Seq Experiments from Genomic Data Commons (GDC)

    .. code-block:: python

        """
        This example demonstrates how to create a data matrix for Differential gene expression (DE) or machine learning analysis.
        You can select the primary site of the samples and the downstream analysis you want to perform.
        """

        import grequests
        import src.Engines.gdc_engine as gdc_engine
        from importlib import reload
        reload(gdc_engine)

        # Create Dataset for differential gene expression
        rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='DE')

        # Create Dataset for machine learning analysis
        rna_seq_ML_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='ML')

.. collapse:: Migrating GDC RNA-Seq Expression Data to your BigQuery Database

    Make sure to run this code in a jupyter notebook or script in the Root directory of OmixHub
    This example demonstrates a comprehensive workflow for uploading RNA-Seq data from multiple primary sites to BigQuery:

    1. It initializes the `BigQueryUtils` class with a specific project ID.
    2. Defines a schema for the BigQuery table, including various fields related to RNA-Seq data.
    3. Creates a new BigQuery table with the defined schema, including partitioning and clustering for optimized performance.
    4. Initializes a `GDCEngine` instance to fetch data from the GDC API.
    5. Iterates through a list of primary sites, fetching data for each site from GDC.
    6. Loads the fetched data into the BigQuery table for each primary site.

    This strategy allows for efficient uploading of data from multiple primary sites into a single, well-structured BigQuery table. The use of partitioning and clustering can significantly improve query performance on large datasets.

    Key features demonstrated:
    - Creating a table with a specific schema
    - Implementing partitioning and clustering for better query performance
    - Batch processing of multiple primary sites
    - Integration with GDCEngine for data retrieval
    - Using tqdm for progress tracking during the upload process

    This approach is particularly useful for large-scale genomic data analysis, allowing researchers to efficiently store and query RNA-Seq data across multiple primary sites in a cloud-based environment.

    .. code-block:: python

        """
        For downstream applications, it is tedious to make API calls to GDC every time you need to access the data for analysis.
        This example demonstrates how to create a BigQuery database for the data you need so that downstream applications can access the data easily.
        """

        import gevent.monkey
        gevent.monkey.patch_all(thread=False, select=False)

        from Connectors.gcp_bigquery_utils import BigQueryUtils
        from google.cloud import bigquery
        from tqdm import tqdm
        from Engines.gdc_engine import GDCEngine

        # Initialize BigQueryUtils with your project
        project_id = 'rnaseqml'
        bq_utils = BigQueryUtils(project_id=project_id)

        # Define the table ID
        table_id = 'rnaseqml.rnaseqexpression.expr_clustered'

        # Define the schema for your table
        schema = [
            bigquery.SchemaField("case_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("file_id", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("expr_unstr_count", "INTEGER", mode="REPEATED"),
            bigquery.SchemaField("tissue_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("sample_type", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("primary_site", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("tissue_or_organ_of_origin", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("age_at_diagnosis", "FLOAT", mode="NULLABLE"),
            bigquery.SchemaField("primary_diagnosis", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("race", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("gender", "STRING", mode="NULLABLE"),
            bigquery.SchemaField("group_identifier", "INTEGER", mode="NULLABLE")
        ]

        # Create table with partitioning and clustering
        bq_utils.create_bigquery_table_with_schema(
            table_id=table_id, 
            schema=schema, 
            partition_field="group_identifier", 
            clustering_fields=["primary_site", "tissue_type"]
        )

        # Initialize GDCEngine
        params = {
            'files.experimental_strategy': 'RNA-Seq', 
            'data_type': 'Gene Expression Quantification'
        }
        gdc_eng_inst = GDCEngine(**params)

        # List of primary sites to process
        primary_sites = ['Esophagus', 'Lung', 'Breast']  # Add more sites as needed

        # Specify the kind of downstream analysis you want to perform
        downstream_analysis = 'DE'

        # Process each primary site
        for site in tqdm(primary_sites):
            # Get data from GDC
            json_object = gdc_eng_inst.get_data_for_bq(site, downstream_analysis=downstream_analysis, format='json')

            # Load data into BigQuery
            job = bq_utils.load_json_data(json_object, schema, table_id)
            job.result()  # Wait for the job to complete
            print(f"Data for {site} loaded successfully.")

        print("All data loaded successfully.")

.. collapse:: Run an analysis for Differential Gene Expression (DE) and Gene Set Enrichment Analysis (GSEA)

    .. code-block:: python

        """
        This example demonstrates how to create a data matrix for Differential gene expression (DE) or machine learning analysis.
        You can select the primary site of the samples and the downstream analysis you want to perform.
        """

        import pandas as pd
        from importlib import reload
        import src.Engines.analysis_engine as analysis_engine
        import src.Connectors.gcp_bigquery_utils as gcp_bigquery_utils
        reload(analysis_engine)
        reload(gcp_bigquery_utils)
        
        # 1. Download Dataset from BigQuery for a given Primary Diagnosis By Primary Site and the Normal Tissue for the Primary site
        project_id = 'rnaseqml'
        dataset_id = 'rnaseqexpression'
        table_id = 'expr_clustered_08082024'
        bq_queries = gcp_bigquery_utils.BigQueryQueries(project_id=project_id, 
                                                    dataset_id=dataset_id,
                                                    table_id=table_id)
        pr_site = 'Head and Neck'
        pr_diag = 'Squamous cell carcinoma, NOS'
        data_from_bq = bq_queries.get_df_for_pydeseq(primary_site=pr_site, primary_diagnosis=pr_diag)

        # 2. Data Preprocessing for PyDeSeq and GSEA
        # Intialize the Analysis Engine
        analysis_eng = analysis_engine.AnalysisEngine(data_from_bq, analysis_type='DE')
        if not analysis_eng.check_tumor_normal_counts():
            raise ValueError("Tumor and Normal counts should be at least 10 each")
        gene_ids_or_gene_cols_df = pd.read_csv('/Users/abhilashdhal/Projects/personal_docs/data/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')
        gene_ids_or_gene_cols = list(gene_ids_or_gene_cols_df['gene_id'].to_numpy())

        # Expand the nested expression Data From BigQuery
        exp_df = analysis_eng.expand_data_from_bq(data_from_bq, gene_ids_or_gene_cols=gene_ids_or_gene_cols, analysis_type='DE')

        # Get Metadata and Counts for PyDeSeq
        metadata = analysis_eng.metadata_for_pydeseq(exp_df=exp_df)
        counts_for_de = analysis_eng.counts_from_bq_df(exp_df, gene_ids_or_gene_cols)

        # 3. Run PyDeSeq
        res_pydeseq = analysis_eng.run_pydeseq(metadata=metadata, counts=counts_for_de)

        # Merge Gene Names as it is required for GSEA and more informative 
        res_pydeseq_with_gene_names = pd.merge(res_pydeseq, gene_ids_or_gene_cols_df, left_on='index', right_on='gene_id')
        
        # 4. Run GSEA for the given Primary Diagnosis By Primary Site and the Normal Tissue for the Primary site using a gene set database
        # Explore the gene set options from gseapy
        from gseapy.plot import gseaplot
        import gseapy as gp
        from gseapy import dotplot
        gsea_options = gp.get_library_name()
        print(gsea_options)

        ## Select Gene Set, run GSEA and plot the results
        gene_set = 'Human_Gene_Atlas'
        result, plot = analysis_eng.run_gsea(res_pydeseq_with_gene_names, gene_set)