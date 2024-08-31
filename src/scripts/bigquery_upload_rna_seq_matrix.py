import gevent.monkey
gevent.monkey.patch_all(thread=False, select=False)
import grequests  # Import grequests after monkey-patching
import hashlib
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from google.cloud import bigquery
from src.Connectors.gcp_bigquery_utils import BigQueryUtils
from src.Engines import gdc_engine

if __name__ == "__main__": 

    def create_identifier(row):
        identifier_str = f"{row['primary_site']}_{row['tissue_type']}_{row['primary_diagnosis']}"
        return int(hashlib.md5(identifier_str.encode()).hexdigest(), 16) % (10 ** 8)

    params = {
        'files.experimental_strategy': 'RNA-Seq', 
        'data_type': 'Gene Expression Quantification'
    }

    gdc_eng_inst = gdc_engine.GDCEngine(**params)
    cohort_metadata = gdc_eng_inst._get_rna_seq_metadata()
    cohort_metadata = cohort_metadata['metadata']
    primary_sites = [
        'Esophagus'
    ]

    # Initialize BigQueryUtils
    bq_utils = BigQueryUtils(project_id='rnaseqml')
    table_id = 'rnaseqml.rnaseqexpression.expr_clustered'

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
        table_id=table_id, schema=schema, partition_field="group_identifier", clustering_fields=["primary_site", "tissue_type"]
    )

    for site in tqdm(primary_sites):
        df = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site=site, downstream_analysis='DE')
        df = df.set_index('file_id').reset_index()
        gene_cols = np.unique(df.columns.to_numpy()[1:60661])
        df['expr_unstr_count'] = df[np.sort(gene_cols)].agg(list, axis=1)
        df_unq = df.drop_duplicates(['case_id']).reset_index(drop=True)

        data_for_bq = df_unq[['case_id', 'file_id', 'expr_unstr_count', 'tissue_type', 'sample_type', 'primary_site']]
        data_bq_with_labels = pd.merge(
            data_for_bq, 
            cohort_metadata[['file_id', 'case_id', 'tissue_or_organ_of_origin', 'age_at_diagnosis', 'primary_diagnosis', 'race', 'gender']], 
            on=['file_id', 'case_id']
        )

        data_bq_with_labels['group_identifier'] = data_bq_with_labels.apply(create_identifier, axis=1)
        json_data = data_bq_with_labels.to_json(orient='records')
        json_object = json.loads(json_data)

        # Load data into BigQuery
        job = bq_utils.load_json_data(json_object, schema, table_id)
        job.result()  # Wait for the job to complete

        print(f"Data for {site} loaded successfully.")





