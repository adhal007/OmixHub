Examples
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to clone the repository and view the example notebooks.

Example 1. Cohort Creation of Bulk RNA Seq Experiments from Genomic Data Commons (GDC)
**************************************************
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

Example 2. Migrating GDC RNA-Seq Expression Data to your BigQuery Database
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

Example 3. Run an analysis for Differential Gene Expression (DE) and Gene Set Enrichment Analysis (GSEA)
********************************************************************************************************
.. code-block:: python

    """
    This example demonstrates how to create a data matrix for Differential gene expression (DE) or machine learning analysis.
    You can select the primary site of the samples and the downstream analysis you want to perform.
    """

    import src.ClassicML.DGE.pydeseq_utils as pydeseq_utils
    import pandas as pd 
    from gseapy.plot import gseaplot
    import gseapy as gp
    import numpy as np
    import matplotlib.pyplot as plt
    from gseapy import dotplot
    ## Preprocess the data
    ## Load the count data saved from example 1. 
    rna_seq_DGE_data  = pd.read_csv('./de_gsea_data/kidney_unstr_tumor_normal.csv')
    unique_data_by_case_id =  rna_seq_DGE_data.drop_duplicates(['case_id']).reset_index(drop=True)
    kidney_cancer_count_data = unique_data_by_case_id.iloc[:, :60660].T
    counts = kidney_cancer_count_data.copy().reset_index()
    counts = counts.set_index('index')
    counts = counts.T
    counts = pd.concat([unique_data_by_case_id[['case_id']], counts],axis=1)  

    ## Run DE analysis
    pydeseq_obj = pydeseq_utils.PyDeSeqWrapper(count_matrix=counts, metadata=metadata, design_factors='Condition', groups = {'group1':'Tumor', 'group2':'Normal'})
    design_factor = 'Condition'
    result = pydeseq_obj.run_deseq(design_factor=design_factor, group1 = 'Tumor', group2 = 'Normal')

    ## Prepare the data for GSEA
    results_df = result.results_df
    results_df_filtered = results_df.dropna()
    results_df_filtered = results_df_filtered.reset_index()
    results_df_filtered['nlog10'] = -1*np.log10(results_df_filtered.padj)

    ## Create ranking for GSEA
    df = results_df_filtered.copy()
    df['Rank'] = -np.log10(df.padj)*df.log2FoldChange
    df = df.sort_values('Rank', ascending = False).reset_index(drop = True)
    ranking = df[['Gene', 'Rank']]
    pre_res = gp.prerank(rnk = ranking, gene_sets = 'RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO', seed = 6, permutation_num = 100)

    ## Plot the GSEA results
    out = []
    for term in list(pre_res.results):
        out.append([term,
                pre_res.results[term]['fdr'],
                pre_res.results[term]['es'],
                pre_res.results[term]['nes']])

    out_df = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop = True)
    terms = pre_res.res2d.Term
    axs = pre_res.plot(terms=terms[1]) 

    # Create dotplot of most enrichment terms from Gene Set 
    ax = dotplot(pre_res.res2d,
                column="FDR q-val",
                title='KEGG_2016',
                cmap=plt.cm.viridis,
                size=6, # adjust dot size
                figsize=(4,5), cutoff=0.25, show_ring=False)
