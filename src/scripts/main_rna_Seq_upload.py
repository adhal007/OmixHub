import argparse
import hashlib
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import src.Connectors.google_cloud_conn as gcp_bq
import src.Engines.gdc_engine as gdc_engine  # Replace 'your_module' with the actual module name
 

def main(project_id, dataset_id, table_id, primary_sites, experimental_strategy, data_type):
    params = {
        'files.experimental_strategy': experimental_strategy, 
        'data_type': data_type
    }

    gdc_eng_inst = gdc_engine.GDCEngine(**params)
    cohort_metadata = gdc_eng_inst._get_rna_seq_metadata()
    cohort_metadata = cohort_metadata['metadata']

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

    bq_utils = gcp_bq.BigQueryUtils(project_id)
    full_table_id = f"{project_id}.{dataset_id}.{table_id}"
    bq_utils.create_bigquery_table_with_schema(full_table_id, schema, partition_field='group_identifier', clustering_fields=["primary_site", "tissue_type"])

    for site in tqdm(primary_sites):
        df = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site=site, downstream_analysis='DE')
        df = df.set_index('file_id').reset_index()
        gene_cols = df.columns.to_numpy()[1:60661]
        df['expr_unstr_count'] = df[np.sort(gene_cols)].agg(list, axis=1)
        df_unq = df.drop_duplicates(['case_id']).reset_index(drop=True)
        data_for_bq = df_unq[['case_id', 'file_id', 'expr_unstr_count', 'tissue_type', 'sample_type', 'primary_site']]
        data_bq_with_labels = pd.merge(data_for_bq, cohort_metadata[['file_id', 'case_id', 'tissue_or_organ_of_origin', 'age_at_diagnosis', 'primary_diagnosis', 'race', 'gender']], on=['file_id', 'case_id'])
        
        bq_utils.upload_partitioned_clustered_table(full_table_id, data_bq_with_labels, schema, 'primary_site', 'tissue_type', 'primary_diagnosis')

    print("Data loaded successfully for all primary sites.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload RNA-Seq data to BigQuery with partitioning and clustering.")
    parser.add_argument('--project_id', type=str, required=True, help='Google Cloud Project ID')
    parser.add_argument('--dataset_id', type=str, required=True, help='BigQuery Dataset ID')
    parser.add_argument('--table_id', type=str, required=True, help='BigQuery Table ID')
    parser.add_argument('--primary_sites', type=str, nargs='+', required=True, help='List of primary sites to process')
    parser.add_argument('--experimental_strategy', type=str, default='RNA-Seq', help='Experimental strategy (default: RNA-Seq)')
    parser.add_argument('--data_type', type=str, default='Gene Expression Quantification', help='Data type (default: Gene Expression Quantification)')

    args = parser.parse_args()
    main(args.project_id, args.dataset_id, args.table_id, args.primary_sites, args.experimental_strategy, args.data_type)