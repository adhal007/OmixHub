from google.cloud import bigquery, bigquery_storage_v1
from google.api_core.exceptions import NotFound, PermissionDenied, Forbidden
from google.oauth2 import service_account
import pandas as pd
import json
from typing import Union, Optional, Tuple
class BigQueryUtils:
    """
    Utility class for interacting with Google BigQuery.

    Attributes:
        project_id (str): The Google Cloud project ID.
        _client (bigquery.Client): The BigQuery client.
        _bqstorage_client (bigquery_storage_v1.BigQueryReadClient): The BigQuery storage client.
    """

    def __init__(self, project_id, credentials_path: Optional[str] = None) -> None:
        """
        Initialize the BigQueryUtils class.

        Args:
            project_id (str): The Google Cloud project ID.
        """
        self.project_id = project_id
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            self._client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
            self._bqstorage_client = bigquery_storage_v1.BigQueryReadClient(
                credentials=credentials
            )
        else:
            # Use application default credentials
            self._client = bigquery.Client(project=self.project_id)
            self._bqstorage_client = bigquery_storage_v1.BigQueryReadClient()

        # self._client = bigquery.Client(project=self.project_id)
        # self._bqstorage_client = bigquery_storage_v1.BigQueryReadClient()

    def project_exists(self) -> Tuple[bool, str]:
        """
        Check if the project exists and is accessible.

        Returns:
            Tuple[bool, str]: A tuple containing:
                - bool: True if the project exists and is accessible, False otherwise
                - str: A message describing the status or error
        """
        try:
            # Try to list datasets in the project (this will fail if project doesn't exist)
            next(self._client.list_datasets(max_results=1))
            return True, f"Project {self.project_id} exists and is accessible"
        except StopIteration:
            # Project exists but has no datasets
            return True, f"Project {self.project_id} exists but contains no datasets"
        except (PermissionDenied, Forbidden) as e:
            # Project might exist but user doesn't have access
            return False, f"Project {self.project_id} might exist but you don't have access: {str(e)}"
        except NotFound:
            # Project doesn't exist
            return False, f"Project {self.project_id} does not exist"
        except Exception as e:
            # Other unexpected errors
            return False, f"Error checking project {self.project_id}: {str(e)}"

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
            print(f"Dataset {dataset_id} already exists")
            return True
        except NotFound:
            print(f"Dataset {dataset_id} is not found")
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
            skip_leading_rows=0,
            autodetect=True,
        )

        job = self._client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        return job
    def create_bigquery_table_with_schema(
        self, table_id: str, schema: list, partition_field: Optional[str] = None, 
        clustering_fields: Optional[list] = None
    ) -> Optional[bigquery.Table]:
        """
        Create a BigQuery table with a specified schema, partitioning, and clustering.

        Args:
            table_id (str): The ID of the BigQuery table in format 'dataset_id.table_id' 
                        or 'project_id.dataset_id.table_id'
            schema (list): The schema of the BigQuery table.
            partition_field (str, optional): The field to partition the table by. Defaults to None.
            clustering_fields (list, optional): The fields to cluster the table by. Defaults to None.

        Returns:
            bigquery.Table: The created BigQuery table object, or None if the table already exists.
        """
        # Ensure table_id is fully qualified
        if table_id.count('.') == 1:
            # If only dataset.table provided, add project
            table_id = f"{self.project_id}.{table_id}"
        elif table_id.count('.') == 0:
            raise ValueError(
                "table_id must be in format 'dataset_id.table_id' or 'project_id.dataset_id.table_id'"
            )
        
        # Split the table_id into its components
        parts = table_id.split('.')
        if len(parts) == 3:
            project_id, dataset_id, table_name = parts
        else:
            project_id, dataset_id, table_name = self.project_id, parts[0], parts[1]

        # Ensure dataset exists
        dataset_ref = bigquery.DatasetReference(project_id, dataset_id)
        try:
            self._client.get_dataset(dataset_ref)
        except NotFound:
            # Create the dataset if it doesn't exist
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = "US"  # You might want to make this configurable
            self._client.create_dataset(dataset, exists_ok=True)
            print(f"Created dataset {project_id}.{dataset_id}")

        # Create or get table
        try:
            table = self._client.get_table(table_id)
            print(f"Table {table_id} already exists")
            return None
        except NotFound:
            # Create the table
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
    # def create_bigquery_table_with_schema(
    #     self, table_id, schema, partition_field=None, clustering_fields=None
    # ) -> bigquery.Table:
    #     """
    #     Create a BigQuery table with a specified schema, partitioning, and clustering.

    #     Args:
    #         table_id (str): The ID of the BigQuery table.
    #         schema (list): The schema of the BigQuery table.
    #         partition_field (str, optional): The field to partition the table by. Defaults to None.
    #         clustering_fields (list, optional): The fields to cluster the table by. Defaults to None.

    #     Returns:
    #         bigquery.Table: The created BigQuery table object, or None if the table already exists.
    #     """
    #     try:
    #         table = self._client.get_table(table_id)
    #         print("Table Already Exists")
    #         return None
    #     except NotFound:
    #         table = bigquery.Table(table_id, schema=schema)
    #         if partition_field:
    #             table.range_partitioning = bigquery.RangePartitioning(
    #                 field=partition_field,
    #                 range_=bigquery.PartitionRange(
    #                     start=0, end=100000000, interval=1000000
    #                 ),
    #             )
    #         if clustering_fields:
    #             table.clustering_fields = clustering_fields
    #         table = self._client.create_table(table)
    #         print(f"Created table {table.project}.{table.dataset_id}.{table.table_id}")
    #         return table

    def df_to_json(self, df, file_path="data.json") -> None:
        """
        Convert a DataFrame to a JSON file.

        Args:
            df (pd.DataFrame): The DataFrame to convert.
            file_path (str, optional): The path to save the JSON file. Defaults to "data.json".
        """
        df.to_json(file_path, orient="records", lines=True)

    def load_json_data(self, json_object, schema, table_id) -> bigquery.LoadJob:
        """
        Load JSON data into a BigQuery table. If the table doesn't exist, it will create the table.

        Args:
            json_object (dict): The JSON object to load.
            schema (list): The schema of the BigQuery table.
            table_id (str): The ID of the BigQuery table.

        Returns:
            bigquery.LoadJob: The load job object.
        """
        if not self.table_exists(table_id):
            print(f"Table {table_id} does not exist, creating table...")
            self.create_bigquery_table_with_schema(table_id, schema)
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON, schema=schema
        )
        job = self._client.load_table_from_json(
            json_object, table_id, job_config=job_config
        )
        return job

    def run_query(self, query):
        """
        Run a BigQuery SQL query and return the results as a DataFrame.

        Args:
            query (str): The SQL query to run.

        Returns:
            pd.DataFrame: The query results as a DataFrame.
        """
        query_job = self._client.query(query)
        return query_job.result().to_dataframe()


class BigQueryQueries(BigQueryUtils):
    """
    Class for executing queries on Google BigQuery.

    Attributes:
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The ID of the BigQuery table.
    """

    def __init__(self, project_id, dataset_id, table_id) -> None:
        """
        Initialize the BigQueryQueries class.

        Args:
            project_id (str): The Google Cloud project ID.
            dataset_id (str): The ID of the BigQuery dataset.
            table_id (str): The ID of the BigQuery table.
        """
        super().__init__(project_id)
        self.dataset_id = dataset_id
        self.table_id = table_id

    def get_primary_site_options(self, dry_run=False) -> Union[list, str]:
        query = f"""
        SELECT DISTINCT primary_site
        FROM `{self.dataset_id}.{self.table_id}`
        """
        if dry_run:
            return query

        query_job = self._client.query(query)
        results = query_job.result()
        return [row.primary_site for row in results]

    def get_primary_diagnosis_options(self, primary_site, dry_run=False) -> Union[list, str]:
        query = f"""
        SELECT DISTINCT primary_diagnosis
        FROM `{self.dataset_id}.{self.table_id}`
        WHERE primary_site = @primary_site AND tissue_type = 'Tumor'
        """
        if dry_run:
            return query

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site)
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        results = query_job.result()
        return [row.primary_diagnosis for row in results]

    def get_df_for_pydeseq(self, primary_site, primary_diagnosis, dry_run=False) -> Union[pd.DataFrame, str]:
        query = f"""
        SELECT
            case_id,
            primary_site,
            sample_type,
            tissue_type,
            tissue_or_organ_of_origin,
            primary_diagnosis,
            expr_unstr_count
        FROM
            `{self.dataset_id}.{self.table_id}`
        WHERE
            (primary_site = @primary_site AND primary_diagnosis = @primary_diagnosis AND tissue_type = 'Tumor') OR 
            (primary_site = @primary_site AND tissue_type = 'Normal')
        """
        if dry_run:
            return query

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site),
                bigquery.ScalarQueryParameter(
                    "primary_diagnosis", "STRING", primary_diagnosis
                ),
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        result = query_job.result()
        return result.to_dataframe()

    def get_df_for_recurrence_free_survival_exp(self, primary_site, primary_diagnosis, dry_run=False) -> Union[pd.DataFrame, str]:
        query = f"""
        SELECT
            case_id,
            primary_site,
            sample_type,
            tissue_type,
            tissue_or_organ_of_origin,
            primary_diagnosis,
            days_to_recurrence,
            expr_unstr_count
        FROM
            `{self.dataset_id}.{self.table_id}`
        WHERE
            (primary_site = @primary_site AND primary_diagnosis = @primary_diagnosis AND tissue_type = 'Tumor') OR 
            (primary_site = @primary_site AND tissue_type = 'Normal')
        """
        if dry_run:
            return query

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site),
                bigquery.ScalarQueryParameter(
                    "primary_diagnosis", "STRING", primary_diagnosis
                ),
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        result = query_job.result()
        return result.to_dataframe()
    
    def get_all_primary_diagnosis_for_primary_site(self, primary_site, dry_run=False) -> Union[pd.DataFrame, str]:
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
        if dry_run:
            return query

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("primary_site", "STRING", primary_site),
            ]
        )
        query_job = self._client.query(query, job_config=job_config)
        result = query_job.result()
        df = result.to_dataframe()
        value_counts_df = df["primary_diagnosis"].value_counts().reset_index()
        value_counts_df.columns = ["primary_diagnosis", "number_of_cases"]
        return value_counts_df
    
    def get_geneid2genename(self, dry_run=False) -> Union[pd.DataFrame, str]:
        query = f"""
        SELECT *
        FROM `{self.dataset_id}.{self.table_id}`
        """
        if dry_run:
            return query

        query_job = self._client.query(query)
        results = query_job.result()
        return results.to_dataframe()