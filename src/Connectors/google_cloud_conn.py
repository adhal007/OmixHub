from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import hashlib
import pandas as pd
import json


class BigQueryUtils:
    def __init__(self, project_id) -> None:
        self.project_id = project_id
        self._client = bigquery.Client(project=self.project_id)

    def table_exists(self, table_ref):
        try:
            self._client.get_table(table_ref)
            return True
        except NotFound:
            return False

    def dataset_exists(self, dataset_id):
        try:
            self._client.get_dataset(dataset_id)  # Make an API request.
            return True
        except NotFound:
            return False

    def upload_df_to_bq(self, table_id, df):
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,
            autodetect=True,
        )

        job = self._client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        return job

    def create_bigquery_table_with_schema(
        self, table_id, schema, partition_field=None, clustering_fields=None
    ):
        if not self.table_exists(table_id):
            table = bigquery.Table(table_id, schema=schema)
            if partition_field:
                table.range_partitioning = bigquery.RangePartitioning(
                    field=partition_field,
                    range_=bigquery.PartitionRange(
                        start=0, end=100000000, interval=1000000
                    ),
                )
            if clustering_fields:
                table.clustering_fields = clustering_fields
            table = self._client.create_table(table)
            print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
            return table
        else:
            print("Table Already Exists")
            return None

    def df_to_json(self, df, file_path="data.json"):
        json_data = df.to_json(file_path, orient="records", lines=True)
        json_object = json.loads(json_data)
        return json_object

    def load_json_data(self, json_object, schema, table_id):
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )
        job = self._client.load_table_from_json(
            json_object, table_id, job_config=job_config
        )
        return job

    def create_identifier(self, row, existing_identifiers):
        identifier_str = (
            f"{row['primary_site']}_{row['tissue_type']}_{row['primary_diagnosis']}"
        )
        identifier = int(hashlib.md5(identifier_str.encode()).hexdigest(), 16) % (10**8)

        while identifier in existing_identifiers:
            # Create a new identifier if there is a clash
            identifier_str += "_duplicate"
            identifier = int(hashlib.md5(identifier_str.encode()).hexdigest(), 16) % (
                10**8
            )

        existing_identifiers.add(identifier)
        return identifier

    def upload_partitioned_clustered_table(
        self,
        table_id,
        df,
        schema,
        primary_site_col,
        tissue_type_col,
        primary_diagnosis_col,
    ):
        existing_identifiers = set()
        df["group_identifier"] = df.apply(
            lambda row: self.create_identifier(row, existing_identifiers), axis=1
        )

        self.df_to_json(df)
        with open("data.json", "rb") as source_file:
            json_data = json.load(source_file)

        job = self.load_json_data(json_data, schema, table_id)
        job.result()  # Wait for the job to complete
        print("Data loaded successfully.")
        return job


############## Redundant functions ##################################################################################
# def upload_partitioned_df_to_bq(self, table_id, df, clust_flds):
#     for fld in clust_flds:
#         if fld not in df.columns:
#             raise ValueError(f"Incorrect clustering fields provided: Should be one of {','.join(df.columns)}")
#     job_config = bigquery.LoadJobConfig(
#         source_format=bigquery.SourceFormat.CSV,
#         skip_leading_rows=1,
#         clustering_fields=clust_flds,
#     )

#     job = self._client.load_table_from_dataframe(
#         df, table_id, job_config=job_config
#     )
#     return job
