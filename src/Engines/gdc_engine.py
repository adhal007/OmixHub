import grequests
import src.Connectors.gdc_endpt_base as gdc_endpt_base
import src.Connectors.gdc_filters as gdc_filters
import src.Connectors.gdc_fields as gdc_fields
import src.Connectors.gdc_parser as gdc_prs
import src.Connectors.gdc_field_validator as gdc_vld
from tqdm import tqdm
import pandas as pd
import requests
import re
import io
import numpy as np
import hashlib
import json

# The `GDCEngine` class is a Python class that facilitates fetching and processing RNA sequencing
# metadata and data from the Genomic Data Commons API.

class GDCEngine:
    """
    GDCEngine class to fetch and process data from the GDC API.

    Attributes:
        params (dict): Parameters for the GDCEngine.
        _query_params (dict): Query parameters for the GDC API.
        _default_params (dict): Default parameters for the GDCEngine.

    Methods:
        set_params(params: dict) -> None:
            Set the parameters for the GDCEngine.

        _check_data_type(data_type: str) -> bool:
            Check if the specified data type is supported.

        _get_raw_data() -> dict:
            Get the raw data from the API response.

        _make_file_id_url_map(data: dict) -> dict:
            Create a mapping of file IDs to download URLs.

        _get_urls_content(urls: list) -> list:
            Download the content from the specified URLs.

        get_normalized_RNA_seq_metadata() -> pd.DataFrame:
            Fetch the normalized RNA sequencing metadata.

        make_RNA_seq_data_matrix() -> pd.DataFrame:
            Create a data matrix for RNA sequencing data.

        make_count_data_for_bq() -> pd.DataFrame:
            Get data for BigQuery based on primary site and downstream analysis type.

        make_data_for_recurrence_free_survival() -> pd.DataFrame:
            Get data for recurrence free survival analysis based on primary site and downstream analysis type.

        run() -> None:
            Run the GDCEngine to fetch and process the data.
    """

    def __init__(self, **params: dict) -> None:
        """
        Initialize the GDCEngine with the given parameters.

        Args:
            **params (dict): Parameters to initialize the GDCEngine.
        """
        self._default_params = {
            "endpt": "files",
            "homepage": "https://api.gdc.cancer.gov",
            "cases.project.primary_site": None,
            "new_fields": None,
            "cases.demographic.race": None,
            "cases.demographic.gender": None,
            "files.experimental_strategy": None,
            "data_type": None,
            "op_params": None,
        }
        ## public attributes
        self.params = self._default_params | params
        self.gene_cols = None 
        self._query_params = params
        
        self._files_endpt = gdc_endpt_base.GDCEndptBase(endpt="files")
        self._cases_endpt = gdc_endpt_base.GDCEndptBase(endpt="cases")
        self._projects_endpt = gdc_endpt_base.GDCEndptBase(endpt="projects")
        
        self._filters = gdc_filters.GDCQueryFilters()
        self._facet_filters = gdc_filters.GDCFacetFilters()
        self._fields = gdc_fields.GDCQueryDefaultFields()
        self._validator = gdc_vld.GDCValidator()
        self._parser = gdc_prs.GDCJson2DfParser(
            self._files_endpt, self._cases_endpt, self._projects_endpt
        )
        self._column_names_from_bq = None
        self._op_params = self.params["op_params"]
        self._exp_types = ["RNA-Seq", "SNP", "Total RNA-Seq"]
        self._data_types = [
            "Gene Expression Quantification",
            "Exon Expression Quantification",
            "Isoform Expression Quantification",
            "Splice Junction Quantification",
        ]
        
        self._field_names_from_data_matrix = ['case_id', 'file_id', 'tissue_type', 'sample_type', 'primary_site']
        self._available_feature_norms = ["fpkm_unstranded", "tpm_unstranded"]

    def set_params(self, **params: dict) -> None:
        """
        Set the parameters for the GDCEngine.

        Args:
            params (dict): A dictionary containing the parameters to set.

        Returns:
            None
        """
        self.params = self.params | params
        return self.params

    def _check_data_type(self):
        """
        Check if the specified data type is supported.

        Raises:
            ValueError: If the specified data type is not supported.

        Returns:
            bool: True if the data type is supported, False otherwise.
        """
        if self.params["data_type"] not in self._data_types:
            raise ValueError(
                f"Data type '{self.params['data_type']}' not supported. Choose from {self._data_types}"
            )
        return True

    def _check_exp_type(self):
        """
        Check if the specified experiment type is supported.

        Raises:
            ValueError: If the specified experiment type is not supported.

        Returns:
            bool: True if the experiment type is supported, False otherwise.
        """
        if self.params["files.experimental_strategy"] not in self._exp_types:
            raise ValueError(
                f"Experiment type '{self.params['exp_type']}' not supported. Choose from {self._exp_types}"
            )
        return True

    def _get_raw_data(self, response):
        """
        Get the raw data from the API response.

        Args:
            response: The API response object.

        Returns:
            pd.DataFrame: The raw data as a pandas DataFrame.
        """
        if response is None or response.status_code != 200:
            return None

        content = response.content.decode("utf-8")
        lines = content.splitlines()
        if len(lines) <= 1:
            # Handle the case where there's only one line or empty content
            print(f"Discarding response with insufficient data: {response.url}")
            return None

        # urlData = response.content

        ## Need to add a check for 1 line files

        rawData = pd.read_csv(io.StringIO(content), sep="\t", header=1)
        return rawData

    def _make_file_id_url_map(self, file_ids: list[str]):
        """
        Create a mapping of file IDs to download URLs.

        Args:
            file_ids (list): A list of file IDs.

        Returns:
            dict: A dictionary mapping file IDs to download URLs.
        """
        urls = [f"{self.params['homepage']}/data/{file_id}" for file_id in file_ids]
        return dict(zip(file_ids, urls))

    def _get_urls_content(self, file_id_url_map: dict[str, str]):
        """
        Download the content from the specified URLs.

        Args:
            file_id_url_map (dict): A dictionary mapping file IDs to download URLs.

        Returns:
            dict: A dictionary mapping file IDs to raw data as pandas DataFrames.
        """
        rs = (
            grequests.get(u, headers={"Content-Type": "application/json"})
            for u in file_id_url_map.values()
        )
        responses = grequests.map(rs)
        rawDataMap = {
            file_id: self._get_raw_data(response)
            for file_id, response in zip(file_id_url_map.keys(), responses)
        }
        return rawDataMap

    def _get_rna_seq_metadata(self):
        """
        Fetch all the samples for RNA-Sequencing metadata

        Returns:
            dict: A dictionary containing the metadata and filters.
                - metadata (pd.DataFrame): The metadata as a pandas DataFrame.
                - filters (dict): The filters used to fetch the metadata.
        """
        json_data, filters = self._files_endpt._query_to_json(
            params=self._query_params
        )
        metadata = self._parser.make_df_rna_seq(json_data)
        return {"metadata": metadata, "filters": filters}

    def _make_rna_seq_data_matrix(
        self,
        rawDataMap: dict[str, pd.DataFrame],
        metadata: pd.DataFrame,
        feature_col="fpkm_unstranded",
    ):
        """
        Create a data matrix for RNA sequencing data.

        Args:
            rawDataMap (dict): A dictionary mapping file IDs to raw data as pandas DataFrames.
            metadata (pd.DataFrame): The metadata as a pandas DataFrame.

        Returns:
            pd.DataFrame: The RNA sequencing data matrix.
        """
        df_list = []
        for key, value in tqdm(rawDataMap.items()):
            if value is not None:
                df_tmp = value.dropna()
                cols = df_tmp[["gene_id"]].to_numpy().flatten()
                df_tmp = df_tmp[[feature_col]].T
                df_tmp.columns = cols
                df_tmp["file_id"] = key
                df_list.append(df_tmp)
        rna_seq_data_matrix = pd.concat(df_list)
        return rna_seq_data_matrix

    def _process_data_matrix_rna_seq(
        self, meta, primary_site=None, downstream_analysis="DE", num_chunks=50
    ):
        if primary_site is not None:
            sub_meta = meta[meta["primary_site"] == primary_site].reset_index(drop=True)
        else:
            sub_meta = meta.copy()
        chunks = sub_meta.shape[0] // num_chunks
        chunk_ls = []
        if downstream_analysis == "DE":
            feature_col_for_extraction = "unstranded"
        elif downstream_analysis == "ML":
            feature_col_for_extraction = "tpm_unstranded"

        for chunk_i in tqdm(range(chunks)):
            sub_meta_i = sub_meta.iloc[
                chunk_i * num_chunks : (chunk_i * num_chunks + num_chunks), :
            ].reset_index(drop=True)
            file_ids = sub_meta_i["file_id"].to_list()
            file_id_url_map = self._make_file_id_url_map(file_ids)
            rawDataMap = self._get_urls_content(file_id_url_map)
            ids_with_none = [
                key for key in rawDataMap.keys() if rawDataMap[key] is None
            ]
            rna_seq_data_matrix = self._make_rna_seq_data_matrix(
                rawDataMap, sub_meta_i, feature_col=feature_col_for_extraction
            )

            sub_meta_sub_i = sub_meta_i[~sub_meta_i["file_id"].isin(ids_with_none)]
            rna_seq_data_matrix["project_id"] = sub_meta_sub_i["project_id"].to_numpy()
            rna_seq_data_matrix["tissue_type"] = sub_meta_sub_i[
                "tissue_type"
            ].to_numpy()
            rna_seq_data_matrix["sample_type"] = sub_meta_sub_i[
                "sample_type"
            ].to_numpy()
            rna_seq_data_matrix["primary_site"] = sub_meta_sub_i[
                "primary_site"
            ].to_numpy()
            rna_seq_data_matrix["case_id"] = sub_meta_sub_i["case_id"].to_numpy()
            chunk_ls.append(rna_seq_data_matrix)
        df = pd.concat(chunk_ls)
        return df

    def run_rna_seq_data_matrix_creation(self, primary_site, downstream_analysis="DE"):
        """
        Run the GDCEngine to fetch and process the data.

        Returns:
            pd.DataFrame: The processed data matrix for machine learning.
        """
        if self._check_data_type():
            rna_seq_metadata = self._get_rna_seq_metadata()
            meta = rna_seq_metadata["metadata"]
            ml_data_matrix = self._process_data_matrix_rna_seq(
                meta=meta,
                primary_site=primary_site,
                downstream_analysis=downstream_analysis,
                
            )
            return ml_data_matrix
    
    def create_identifier(self, row):
        """
        Create a unique identifier for a row based on specific fields. 
        This will be useful for partitioning and clustering in BigQuery

        Args:
            row (pd.Series): The row of data containing 'primary_site', 'tissue_type', and 'primary_diagnosis'.

        Returns:
            int: The unique identifier.
        """
        identifier_str = f"{row['primary_site']}_{row['tissue_type']}_{row['primary_diagnosis']}"
        return int(hashlib.md5(identifier_str.encode()).hexdigest(), 16) % (10 ** 8)

    def make_count_data_for_bq(self, primary_site, downstream_analysis='DE', format='dataframe'):
        """
        Get data for BigQuery based on primary site and downstream analysis type.

        Args:
            primary_site (str): The primary site to filter by.
            downstream_analysis (str, optional): The type of downstream analysis ('DE' or other). Defaults to 'DE'.
            format (str, optional): The format of the output ('dataframe' or 'json'). Defaults to 'dataframe'.

        Returns:
            pd.DataFrame or dict: The data formatted as a DataFrame or JSON object.
        """
        field_names_from_cohort = ['file_id', 'case_id', 'tissue_or_organ_of_origin', 'age_at_diagnosis', 'primary_diagnosis', 'race', 'gender']
        cohort_metadata = self._get_rna_seq_metadata()
        cohort_metadata = cohort_metadata['metadata']
        df = self.run_rna_seq_data_matrix_creation(primary_site=primary_site, downstream_analysis=downstream_analysis)
        df = df.set_index('file_id').reset_index()
        self.gene_cols = df.columns.to_numpy()[1:60661]  # Store gene_cols as an attribute
        if downstream_analysis == 'DE':
            feature_values_column = 'expr_unstr_count'
            
        elif downstream_analysis == 'ML':
            feature_values_column = 'expr_tpm'
            
        self._field_names_from_data_matrix.append(feature_values_column)    
        df[feature_values_column] = df[np.sort(self.gene_cols)].agg(list, axis=1)
        df_unq = df.drop_duplicates(['case_id']).reset_index(drop=True)
        
        data_for_bq = df_unq[self._field_names_from_data_matrix]
        data_bq_with_labels = pd.merge(data_for_bq, cohort_metadata[field_names_from_cohort], on=['file_id', 'case_id'])
        if format == 'dataframe':
            return data_bq_with_labels, self.gene_cols
        elif format == 'json':
            json_data = data_bq_with_labels.to_json(orient='records')
            json_object = json.loads(json_data)
            return json_object, self.gene_cols
    
    def make_data_for_recurrence_free_survival(self, primary_site, downstream_analysis='ML', format='dataframe'):
        """
        Get data for recurrence free survival analysis based on primary site and downstream analysis type.

        Args:
            primary_site (str): The primary site to filter by.
            downstream_analysis (str, optional): The type of downstream analysis ('ML' or other). Defaults to 'ML'.
            format (str, optional): The format of the output ('dataframe' or 'json'). Defaults to 'dataframe'.

        Returns:
            pd.DataFrame or dict: The data formatted as a DataFrame or JSON object.
        """
        field_names_from_cohort =['file_id', 'case_id', 'tissue_or_organ_of_origin', 'age_at_diagnosis', 'primary_diagnosis', 'race', 'gender', 'days_to_recurrence']
        cohort_metadata = self._get_rna_seq_metadata()
        cohort_metadata = cohort_metadata['metadata']
        df = self.run_rna_seq_data_matrix_creation(primary_site=primary_site, downstream_analysis=downstream_analysis)
        df = df.set_index('file_id').reset_index()
        gene_cols = df.columns.to_numpy()[1:60661]
        if downstream_analysis == 'DE':
            feature_values_column = 'expr_unstr_count'
        elif downstream_analysis == 'ML':
            feature_values_column = 'expr_tpm'
        df[feature_values_column] = df[np.sort(gene_cols)].agg(list, axis=1)
        
        df_unq = df.drop_duplicates(['case_id']).reset_index(drop=True)
        data_for_bq = df_unq[self._field_names_from_data_matrix]
        data_bq_with_labels = pd.merge(data_for_bq, cohort_metadata[field_names_from_cohort], on=['file_id', 'case_id'])
        if format == 'dataframe':
            return data_bq_with_labels, gene_cols
        elif format == 'json':
            json_data = data_bq_with_labels.to_json(orient='records')
            json_object = json.loads(json_data)
            return json_object, gene_cols
