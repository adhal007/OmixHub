Examples
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to clone the repository and view the example notebooks.



Useful query filters for GDC API endpoints
------------------------------------------

The following examples demonstrate how to use various filters from the GDCQueryFilters class to query different GDC API endpoints.
These examples demonstrate how to create filters for various GDC data types and endpoints.

RNA-Seq Filter
~~~~~~~~~~~~~~

.. code-block:: python

    from Connectors.gdc_filters import GDCQueryFilters

    gdc_filters = GDCQueryFilters()
    rna_seq_filter = gdc_filters.rna_seq_filter()
    
    # Use this filter with the 'files' endpoint
    # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": rna_seq_filter, "size": 10})

WGS Filter
~~~~~~~~~~

.. code-block:: python

    wgs_filter = gdc_filters.wgs_filter()
    
    # Use this filter with the 'files' endpoint
    # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": wgs_filter, "size": 10})

Methylation Filter
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    methylation_filter = gdc_filters.methylation_filter()
    
    # Use this filter with the 'files' endpoint
    # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": methylation_filter, "size": 10})

Top Mutated Genes Filter
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    top_mutated_genes_filter = gdc_filters.top_mutated_genes_by_project_filter("TCGA-BRCA", top_n=5)
    
    # Use this filter with the 'analysis/top_mutated_genes_by_project' endpoint
    # Example: requests.get("https://api.gdc.cancer.gov/analysis/top_mutated_genes_by_project", 
    #                       params={"filters": json.dumps(top_mutated_genes_filter), "fields": "gene_id,symbol,score", "size": 5})

Custom RNA-Seq Data Filter
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    custom_params = {
        "cases.project.primary_site": ["Breast"],
        "cases.demographic.gender": ["female"]
    }
    custom_rna_seq_filter = gdc_filters.rna_seq_data_filter(field_params=custom_params)
    
    # Use this filter with the 'files' endpoint
    # Example: requests.post("https://api.gdc.cancer.gov/files", json={"filters": custom_rna_seq_filter, "size": 10})



Useful Examples for Data Processing and Analysis in OmixHub
-----------------------------------------------------------

Cohort Creation of Bulk RNA Seq Experiments from Genomic Data Commons (GDC)
**********************************************************************************
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
**************************************************

Migrating GDC RNA-Seq Expression Data to your BigQuery Database
********************************************************************************
Make sure to run this code in a jupyter notebook or script in the Root directory of OmixHub

.. code-block:: python

    """
    For downstream applications, it is tedious to make API calls to GDC every time you need to access the data for analysis.
    This example demonstrates how to create a BigQuery database for the data you need so that downstream applications can access the data easily.
    """

    import gevent.monkey
    gevent.monkey.patch_all(thread=False, select=False)
    import pandas as pd
    import numpy as np

    import os
    from importlib import reload
    from flatten_json import flatten
    from tqdm import tqdm 
    import src.Engines.gdc_engine as gdc_engine
    import src.Connectors.gcp_bigquery_utils as gcp_bigquery_utils
    from google.cloud import bigquery
    from google.cloud.exceptions import NotFound
    reload(gcp_bigquery_utils)
    reload(gdc_engine)


    # Initialize the GDC Engine
    params = {
        'files.experimental_strategy': 'RNA-Seq', 
        'data_type': 'Gene Expression Quantification'
    }

    gdc_eng_inst = gdc_engine.GDCEngine(**params)

    primary_sites = [
        'Esophagus'
    ]

    ## Initialize BigQueryUtils with your project
    bq_utils = gcp_bigquery_utils.BigQueryUtils(project_id='rnaseqml')
    table_id = 'rnaseqml.rnaseqexpression.expr_clustered'

    ## Give Schema of your table to be created or updated 
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

    ## Create table with partitioning and clustering
    bq_utils.create_bigquery_table_with_schema(
        table_id=table_id, schema=schema, partition_field="group_identifier", clustering_fields=["primary_site", "tissue_type"]
    )

    ## Specify the Kind of Downstream Analysis you want to perform
    downstream_analysis = 'DE'
    for site in tqdm(primary_sites):
        json_object = gdc_eng_inst.get_data_for_bq(site, downstream_analysis='DE', format='json')

        # Load data into BigQuery
        job = bq_utils.load_json_data(json_object, schema, table_id)
        job.result()  # Wait for the job to complete
        print(f"Data for {site} loaded successfully.")
******************************************************

Run an analysis for Differential Gene Expression (DE) and Gene Set Enrichment Analysis (GSEA)
********************************************************************************************************
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