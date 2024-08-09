from google.cloud import bigquery
from google.cloud import bigquery_storage
from google.cloud.exceptions import NotFound
import pandas as pd 

class BigQueryUtils:
    def __init__(self, project_id) -> None:
        self.project_id = project_id
        self._client = bigquery.Client(project=self.project_id)
        self._bqstorage_client = bigquery_storage.BigQueryReadClient()

    def table_exists(self, table_ref):
        try:
            self._client.get_table(table_ref)
            return True
        except NotFound:
            return False
    
    def dataset_exists(self, dataset_id):
        try:
            self._client.get_dataset(dataset_id)  # Make an API request.
            print(f"Dataset {dataset_id} already exists")
            return True
        except NotFound:
            print(f"Dataset {dataset_id} is not found")
            return False

    def upload_df_to_bq(self, table_id, df):
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
        )

        job = self._client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        return job

    def create_bigquery_table_with_schema(self, table_id, schema, partition_field=None, clustering_fields=None):
        if not self.table_exists(table_id):
            table = bigquery.Table(table_id, schema=schema)
            if partition_field:
                table.range_partitioning = bigquery.RangePartitioning(
                    field=partition_field,
                    range_=bigquery.PartitionRange(start=0, end=100000000, interval=1000000),
                )
            if clustering_fields:
                table.clustering_fields = clustering_fields
            table = self._client.create_table(table)
            print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
            return table
        else:
            print("Table Already Exists")
            return None

    def df_to_json(self, df, file_path='data.json'):
        df.to_json(file_path, orient='records', lines=True)

    def load_json_data(self, json_object, schema, table_id):
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )
        job = self._client.load_table_from_json(json_object, table_id, job_config=job_config)
        return job


class BigQueryQueries(BigQueryUtils):
    def __init__(self, project_id, dataset_id, table_id):
        super().__init__(project_id)
        self.dataset_id = dataset_id
        self.table_id = table_id

    def get_primary_diagnosis_options(self, primary_site):
        query = f"""
        SELECT DISTINCT primary_diagnosis
        FROM `{self.dataset_id}.{self.table_id}`
        WHERE primary_site = @primary_site AND tissue_type = 'Tumor'
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site)
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        results = query_job.result()
        return [row.primary_diagnosis for row in results]

    def get_df_for_pydeseq(self, primary_site, primary_diagnosis):
        query = f"""
        SELECT
            case_id,
            primary_site,
            sample_type,
            tissue_type,
            primary_diagnosis,
            expr_unstr_count
        FROM
            `{self.dataset_id}.{self.table_id}`
        WHERE
            (primary_site = @primary_site AND primary_diagnosis = @primary_diagnosis AND tissue_type = 'Tumor') OR 
            (primary_site = @primary_site AND tissue_type = 'Normal')
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site),
                bigquery.ScalarQueryParameter("primary_diagnosis", "STRING", primary_diagnosis)
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        result = query_job.result()  # Wait for the query job to complete.
        df = result.to_dataframe()
        # Expand 'expr_unstr_count' into separate columns using apply with pd.Series
        # expr_unstr_df = df['expr_unstr_count'].apply(pd.Series)

        # # Optionally rename the new columns to something meaningful
        # expr_unstr_df.columns = [f'expr_unstr_count_{i}' for i in expr_unstr_df.columns]

        # # Concatenate the expanded columns back to the original dataframe
        # df = pd.concat([df.drop(columns=['expr_unstr_count']), expr_unstr_df], axis=1)
        return df 
    
    def get_all_primary_diagnosis_for_primary_site(self, primary_site):

        query = f"""
        SELECT
            case_id,
            primary_site,
            sample_type,
            tissue_type,
            primary_diagnosis
        FROM
            `{self.dataset_id}.{self.table_id}`
        WHERE
            (primary_site = @primary_site AND tissue_type = 'Tumor')
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site),
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        result = query_job.result()  # Wait for the query job to complete.
        df = result.to_dataframe()
        value_counts_df = df['primary_diagnosis'].value_counts().reset_index()
        value_counts_df.columns = ['primary_diagnosis', 'number_of_cases']
        return value_counts_df