import grequests
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
reload(gdc_engine)


if __name__ ==  "__main__": 

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
        # 'Blood',
        # 'Kidney',
        # 'Breast',
        'Thyroid',
        'Prostate',
        'Colorectal',
        'Liver',
        'Head and Neck',
        'Stomach',
        'Uterus'
        ]

    client = bigquery.Client(project='rnaseqml')
    table_id = 'rnaseqml.rnaseqexpression.expr_clustered'
    schema = [
        bigquery.SchemaField("case_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("file_id", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("expr_unstr_count","INTEGER",mode="REPEATED",
        ),

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

    for site in tqdm(primary_sites):
        df = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site=site,downstream_analysis='DE')
        df = df.set_index('file_id')
        df = df.reset_index()
        gene_cols = np.unique(df.columns.to_numpy()[1:60661])
         
        df['expr_unstr_count'] = df[np.sort(gene_cols)].agg(list, axis=1)
        df_unq = df.drop_duplicates(['case_id']).reset_index(drop=True)
        data_for_bq = df_unq[['case_id', 'file_id', 'expr_unstr_count', 'tissue_type', 'sample_type', 'primary_site']]
        data_bq_with_labels = pd.merge(data_for_bq, cohort_metadata[['file_id', 'case_id', 'tissue_or_organ_of_origin', 'age_at_diagnosis', 'primary_diagnosis', 'race', 'gender']], on=['file_id', 'case_id'])
        # data_bq_with_labels['cohort'] = site
        df['group_identifier'] = df.apply(create_identifier, axis=1)
        json_data = data_bq_with_labels.to_json(orient = 'records')
        json_object = json.loads(json_data)

        job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )
        job = client.load_table_from_json(json_object, table_id, job_config = job_config)
    # df.to_csv('./de_gsea_data/all_unstr_tumor_normal.csv', index=False)
    