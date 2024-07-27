# The `GDCEngine` class is a Python class that facilitates fetching and processing RNA sequencing
# metadata and data from the Genomic Data Commons API.
import grequests
import src.Connectors.gdc_files_endpt as gdc_files
import src.Connectors.gdc_cases_endpt as gdc_cases
import src.Connectors.gdc_projects_endpt as gdc_projects
import src.Connectors.gdc_filters as gdc_filters
import src.Connectors.gdc_fields as gdc_fields
import src.Connectors.gdc_parser as gdc_prs
import src.Connectors.gdc_field_validator as gdc_vld
import pandas as pd
import requests
import re 
import io

class GDCEngine:
    """
    A class representing the GDCEngine.

    The GDCEngine is responsible for fetching and processing data from the GDC (Genomic Data Commons) API.
    It provides methods to set parameters, fetch metadata, and create data matrices for RNA sequencing data.

    Attributes:
        params (dict): A dictionary containing the parameters for the GDCEngine.
            - endpt (str): The endpoint for the GDC API. Default is 'files'.
            - homepage (str): The homepage URL for the GDC API. Default is 'https://api.gdc.cancer.gov'.
            - ps_list (list): A list of program names to filter the data. Default is None.
            - new_fields (list): A list of additional fields to include in the metadata. Default is None.
            - race_list (list): A list of races to filter the data. Default is None.
            - gender_list (list): A list of genders to filter the data. Default is None.
            - data_type (str): The type of data to fetch. Default is 'RNASeq'.

    Methods:
        set_params: Set the parameters for the GDCEngine.
        _check_data_type: Check if the specified data type is supported.
        _get_raw_data: Get the raw data from the API response.
        _make_file_id_url_map: Create a mapping of file IDs to download URLs.
        _get_urls_content: Download the content from the specified URLs.
        get_normalized_RNA_seq_metadata: Fetch the normalized RNA sequencing metadata.
        make_RNA_seq_data_matrix: Create a data matrix for RNA sequencing data.
        run: Run the GDCEngine to fetch and process the data.

    """

    def __init__(self, **params: dict) -> None:
        self._default_params = {
            'endpt': 'files',
            'homepage': 'https://api.gdc.cancer.gov',
            'cases.project.primary_site': None,
            'new_fields': None,
            'cases.demographic.race': None,
            'cases.demographic.gender': None,
            'files.experimental_strategy': None,
            'data_type': None
        }
        ## public attributes 
        self.params = self._default_params | params
        self._query_params = params 
        
        ## private attributes
        self._files_endpt = gdc_files.GDCFilesEndpt()
        self._cases_endpt = gdc_cases.GDCCasesEndpt()
        self._projects_endpt = gdc_projects.GDCProjectsEndpt()
        self._filters = gdc_filters.GDCQueryFilters()
        self._facet_filters = gdc_filters.GDCFacetFilters()
        self._fields = gdc_fields.GDCQueryDefaultFields(self.params['endpt'])
        self._validator = gdc_vld.GDCValidator()
        self._parser = gdc_prs.GDCJson2DfParser(self._files_endpt, self._cases_endpt, self._projects_endpt)
        self._exp_types = ['RNA-Seq', 'SNP', 'Total RNA-Seq']
        self._data_types = ['Gene Expression Quantification', 'Exon Expression Quantification', 'Isoform Expression Quantification', 'Splice Junction Quantification']
    

        
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
        if self.params['data_type'] not in self._data_types:
            raise ValueError(f"Data type '{self.params['data_type']}' not supported. Choose from {self._data_types}")
        return True
    
    def _check_exp_type(self):
        """
        Check if the specified experiment type is supported.

        Raises:
            ValueError: If the specified experiment type is not supported.

        Returns:
            bool: True if the experiment type is supported, False otherwise.
        """
        if self.params['files.experimental_strategy'] not in self._exp_types:
            raise ValueError(f"Experiment type '{self.params['exp_type']}' not supported. Choose from {self._exp_types}")
        return True
    
    def _get_raw_data(self, response):
        """
        Get the raw data from the API response.

        Args:
            response: The API response object.

        Returns:
            pd.DataFrame: The raw data as a pandas DataFrame.
        """
        urlData = response.content
        
        ## Need to add a check for 1 line files 
        
        rawData = pd.read_csv(io.StringIO(urlData.decode('utf-8')), sep="\t", header=1)
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
        rs = (grequests.get(u, headers = {"Content-Type": "application/json"}) for u in file_id_url_map.values())
        responses = grequests.map(rs)
        file_id_response_map = dict(zip(file_id_url_map.keys(), responses))
        responses = [r for r in file_id_response_map.values() if r.status_code == 200 or r is not None]
        rawData = [self._get_raw_data(r) for r in responses]
        rawDataMap = dict(zip(file_id_url_map.keys(), rawData))
        return rawDataMap

    def get_normalized_RNA_seq_metadata(self, filtered: bool = True):
        """
        Fetch the normalized RNA sequencing metadata.

        Returns:
            dict: A dictionary containing the metadata and filters.
                - metadata (pd.DataFrame): The metadata as a pandas DataFrame.
                - filters (dict): The filters used to fetch the metadata.
        """
        json_data, filters = self._files_endpt.fetch_rna_seq_star_counts_data(params=self._query_params)
        metadata = self._parser.create_df_from_rna_star_count_q_op(json_data, filtered=filtered)
        return {'metadata': metadata, 'filters': filters}
    
    def make_RNA_seq_data_matrix(self, rawDataMap: dict[str, pd.DataFrame], metadata: pd.DataFrame, feature_col='fpkm_unstranded'):
        """
        Create a data matrix for RNA sequencing data.

        Args:
            rawDataMap (dict): A dictionary mapping file IDs to raw data as pandas DataFrames.
            metadata (pd.DataFrame): The metadata as a pandas DataFrame.

        Returns:
            pd.DataFrame: The RNA sequencing data matrix.
        """
        df_list= []
        for key, value in rawDataMap.items():
            df_tmp = value.dropna()
            cols = df_tmp[['gene_name']].to_numpy().flatten()
            df_tmp = df_tmp[[feature_col]].T
            df_tmp.columns = cols
            df_tmp['file_id'] = key
            df_list.append(df_tmp)
        rna_seq_data_matrix = pd.concat(df_list)
        return rna_seq_data_matrix
    
    def run(self):
        """
        Run the GDCEngine to fetch and process the data.

        Returns:
            pd.DataFrame: The processed data matrix for machine learning.
        """
        if self._check_data_type():
            metadata = self.get_normalized_RNA_seq_metadata()
            file_ids = metadata['metadata']['file_id'].to_list()
            file_id_url_map = self._make_file_id_url_map(file_ids)
            rawDataMap = self._get_urls_content(file_id_url_map)
            rna_seq_data_matrix = self.make_RNA_seq_data_matrix(rawDataMap, metadata['metadata'])
            ml_data_matrix = rna_seq_data_matrix.merge(metadata['metadata'], on='file_id')
            return ml_data_matrix
    
    
        
        