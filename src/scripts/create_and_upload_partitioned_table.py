import grequests
import pandas as pd
import hashlib
import json
from tqdm import tqdm
from google.cloud import bigquery
import pandas as pd
import numpy as np
import src.Engines.gdc_engine as gdc_engine
import os
from importlib import reload
from flatten_json import flatten
from tqdm import tqdm 
import requests
import json
from google.cloud import bigquery
import hashlib
import numpy as np
if __name__ == "__main__":

    def create_identifier(row, existing_identifiers):
        identifier_str = f"{row['primary_site']}_{row['tissue_type']}_{row['primary_diagnosis']}"
        identifier = int(hashlib.md5(identifier_str.encode()).hexdigest(), 16) % (10 ** 8)
        
        while identifier in existing_identifiers:
            # Create a new identifier if there is a clash
            identifier_str += "_duplicate"
            identifier = int(hashlib.md5(identifier_str.encode()).hexdigest(), 16) % (10 ** 8)
        
        existing_identifiers.add(identifier)
        return identifier

    params = {
        'files.experimental_strategy': 'RNA-Seq', 
        'data_type': 'Gene Expression Quantification'
    }

    gdc_eng_inst = gdc_engine.GDCEngine(**params)
    cohort_metadata = gdc_eng_inst._get_rna_seq_metadata()
    cohort_metadata = cohort_metadata['metadata']
    primary_sites = [
        'Prostate',
        'Colorectal',
        'Liver',
        'Head and Neck',
        'Stomach',
        'Uterus'
    ]

    client = bigquery.Client(project='rnaseqml')
    table_id = 'rnaseqml.rnaseqexpression.expr_clustered_08082024'
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

    table = bigquery.Table(table_id, schema=schema)
    # Set partitioning based on group_identifier
    table.range_partitioning = bigquery.RangePartitioning(
        field="group_identifier",
        range_=bigquery.PartitionRange(start=0, end=100000000, interval=1000000),
    )

    # Set clustering fields
    table.clustering_fields = ["primary_site", "tissue_type"]

    client.create_table(table, exists_ok=True)
    print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")

    existing_identifiers = set()
    
    for site in tqdm(primary_sites):
        df = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site=site, downstream_analysis='DE')
        df = df.set_index('file_id')
        df = df.reset_index()
        gene_cols = np.unique(df.columns.to_numpy()[1:60661]) 
        df['expr_unstr_count'] = df[np.sort(gene_cols)].agg(list, axis=1)
        df_unq = df.drop_duplicates(['case_id']).reset_index(drop=True)
        data_for_bq = df_unq[['case_id', 'file_id', 'expr_unstr_count', 'tissue_type', 'sample_type', 'primary_site']]
        data_bq_with_labels = pd.merge(data_for_bq, cohort_metadata[['file_id', 'case_id', 'tissue_or_organ_of_origin', 'age_at_diagnosis', 'primary_diagnosis', 'race', 'gender']], on=['file_id', 'case_id'])
        
        data_bq_with_labels['group_identifier'] = data_bq_with_labels.apply(lambda row: create_identifier(row, existing_identifiers), axis=1)
        
        json_data = data_bq_with_labels.to_json(orient='records')
        json_object = json.loads(json_data)

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )
        job = client.load_table_from_json(json_object, table_id, job_config=job_config)
        job.result()  # Wait for the job to complete
        print(f"Loaded data for primary site: {site}")

    print("Data loaded successfully for all primary sites.")