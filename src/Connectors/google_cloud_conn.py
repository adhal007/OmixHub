from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import hashlib
import pandas as pd
import json


class BigQueryUtils:
    """
    Utility class for interacting with Google BigQuery.

    Attributes:
        project_id (str): The Google Cloud project ID.
        _client (bigquery.Client): The BigQuery client.
    """

    def __init__(self, project_id) -> None:
        """
        Initialize the BigQueryUtils class.

        Args:
            project_id (str): The Google Cloud project ID.
        """
        self.project_id = project_id
        self._client = bigquery.Client(project=self.project_id)

    def table_exists(self, table_ref) -> bool:
        """
        Check if a BigQuery table exists.

        Args:
            table_ref (str): The reference to the BigQuery table.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            self._client.get_table(table_ref)
            return True
        except NotFound:
            return False

    def dataset_exists(self, dataset_id) -> bool:
        """
        Check if a BigQuery dataset exists.

        Args:
            dataset_id (str): The ID of the BigQuery dataset.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        try:
            self._client.get_dataset(dataset_id)  # Make an API request.
            return True
        except NotFound:
            return False

    def upload_df_to_bq(self, table_id, df) -> bigquery.LoadJob:
        """
        Upload a DataFrame to a BigQuery table.

        Args:
            table_id (str): The ID of the BigQuery table.
            df (pd.DataFrame): The DataFrame to upload.

        Returns:
            bigquery.LoadJob: The load job object.
        """
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
    ) -> bigquery.Table:
        """
        Create a BigQuery table with a specified schema, partitioning, and clustering.

        Args:
            table_id (str): The ID of the BigQuery table.
            schema (list): The schema of the BigQuery table.
            partition_field (str, optional): The field to partition the table by. Defaults to None.
            clustering_fields (list, optional): The fields to cluster the table by. Defaults to None.

        Returns:
            bigquery.Table: The created BigQuery table object, or None if the table already exists.
        """
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

    def df_to_json(self, df, file_path="data.json") -> dict:
        """
        Convert a DataFrame to a JSON object and save it to a file.

        Args:
            df (pd.DataFrame): The DataFrame to convert.
            file_path (str, optional): The path to save the JSON file. Defaults to "data.json".

        Returns:
            dict: The JSON object.
        """
        json_data = df.to_json(file_path, orient="records", lines=True)
        json_object = json.loads(json_data)
        return json_object

    def load_json_data(self, json_object, schema, table_id) -> bigquery.LoadJob:
        """
        Load JSON data into a BigQuery table.

        Args:
            json_object (dict): The JSON object to load.
            schema (list): The schema of the BigQuery table.
            table_id (str): The ID of the BigQuery table.

        Returns:
            bigquery.LoadJob: The load job object.
        """
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )
        job = self._client.load_table_from_json(
            json_object, table_id, job_config=job_config
        )
        return job

    def create_identifier(self, row, existing_identifiers) -> int:
        """
        Create a unique identifier for a row based on specific fields.

        Args:
            row (pd.Series): The row of data.
            existing_identifiers (set): A set of existing identifiers to avoid duplicates.

        Returns:
            int: The unique identifier.
        """
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
    ) -> bigquery.LoadJob:
        """
        Upload a DataFrame to a partitioned and clustered BigQuery table.

        Args:
            table_id (str): The ID of the BigQuery table.
            df (pd.DataFrame): The DataFrame to upload.
            schema (list): The schema of the BigQuery table.
            primary_site_col (str): The column name for the primary site.
            tissue_type_col (str): The column name for the tissue type.
            primary_diagnosis_col (str): The column name for the primary diagnosis.

        Returns:
            bigquery.LoadJob: The load job object.
        """
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