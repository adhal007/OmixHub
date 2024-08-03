from google.cloud import bigquery
from google.cloud.exceptions import NotFound
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
            print("Dataset {} already exists".format(dataset_id))
            return True
        except NotFound:
            print("Dataset {} is not found".format(dataset_id))
            return False

    def upload_df_to_bq(self, table_id, df):
        job_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
                )

        job = self._client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        return job
            
    def create_biqguery_table_with_schema(self, table_id, schema):
        if not self.table_exists(table_id):
            table = bigquery.Table(table_id, schema=schema)
            return table
        else:
            return "Table Already Exists"
    
    def load_json_data(self, json_object, schema, table_id):
        job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )

        job = self._client.load_table_from_json(json_object, table_id, job_config = job_config)
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