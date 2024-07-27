from google.cloud import bigquery


class BigQueryUtils:
    def __init__(self, project_id) -> None:
        self.project_id = project_id
        self._client = bigquery.Client(project=self.project_id)
        
    def upload_df_to_bq(self, table_id, df):
        job_config = bigquery.LoadJobConfig(
                    source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1, autodetect=True,
                )

        job = self._client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        return job
    
    # def upload_partitioned_df_to_bq(self, table_id, df):
    #     job_config = bigquery.LoadJobConfig(
    #         source_format=bigquery.SourceFormat.CSV,
    #         skip_leading_rows=1,
    #         range_partitioning=bigquery.RangePartitioning(
    #             type_=bigquery.TimePartitioningType.DAY,
    #             field="date",  # Name of the column to use for partitioning.
    #             expiration_ms=7776000000,  # 90 days.
    #         ),
    #     )