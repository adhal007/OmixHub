import json
import requests
import re
import pandas as pd
import src.Connectors.gdc_filters as gdc_flt
import src.Connectors.gdc_fields as gdc_fld
import src.Connectors.gdc_field_validator as gdc_vld
# Example of usage
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Base Utils class and high-level API functions Only Returns JSON like Objects for different Queries
@author: Abhilash Dhal
@date:  2024_22_27
"""
class GDCEndptBase:
    """
    Base class for interacting with the GDC API.

    This class provides utility functions to query different endpoints and handle JSON responses.
    It includes methods for constructing endpoint URLs, fetching data, and downloading files by ID.

    Attributes:
        homepage (str): The base URL for the GDC API.
        endpt (str): The specific endpoint to query.
        headers (dict): The headers to include in API requests.
        gdc_flt (GDCQueryFilters): An instance of the GDCQueryFilters class.
        gdc_fld (GDCQueryDefaultFields): An instance of the GDCQueryDefaultFields class.
        gdc_vld (GDCValidator): An instance of the GDCValidator class.
    """

    def __init__(self, homepage: str = "https://api.gdc.cancer.gov", endpt: str = None):
        """
        Initialize the GDCEndptBase class.

        Args:
            homepage (str): The base URL for the GDC API. Defaults to "https://api.gdc.cancer.gov".
            endpt (str): The specific endpoint to query. Defaults to None.
        """
        self.endpt = endpt
        self._endpts = {"files": "files", "projects": "projects", "cases": "cases"}
        self.homepage = homepage
        self.headers = {"Content-Type": "application/json"}
        self.gdc_flt = gdc_flt.GDCQueryFilters()
        self.gdc_fld = gdc_fld.GDCQueryDefaultFields(self.endpt)
        self.gdc_vld = gdc_vld.GDCValidator()
        self._mapping = "_mapping"
        self._files_endpt_url = None
        self._projects_endpt_url = None
        self._cases_endpt_url = None
        self._endpt_fields = None

    @property
    def files_endpt_url(self):
        """
        Get the URL for the files endpoint.

        Returns:
            str: The URL for the files endpoint.
        """
        return self._get_endpt_url("files")

    @property
    def projects_endpt_url(self):
        """
        Get the URL for the projects endpoint.

        Returns:
            str: The URL for the projects endpoint.
        """
        return self._get_endpt_url("projects")

    @property
    def cases_endpt_url(self):
        """
        Get the URL for the cases endpoint.

        Returns:
            str: The URL for the cases endpoint.
        """
        return self._get_endpt_url("cases")

    @property
    def endpt_fields(self):
        """
        Get the fields for each endpoint.

        Returns:
            dict: A dictionary containing the fields for each endpoint.
        """
        if self._endpt_fields is None:
            self._endpt_fields = {
                endpt: self._get_endpt_fields(endpt) for endpt in self._endpts.keys()
            }
        return self._endpt_fields

    def _make_endpt_url(self, endpt: str):
        """
        Construct the URL for a given endpoint.

        Args:
            endpt (str): The endpoint to construct the URL for.

        Returns:
            str: The constructed URL.
        """
        if self.endpt is None:
            return f"{self.homepage}/{endpt}"
        else:
            return f"{self.homepage}/{self.endpt}"

    def _get_endpt_fields(self, endpt: str):
        """
        Get the fields for a given endpoint.

        Args:
            endpt (str): The endpoint to get the fields for.

        Returns:
            dict: A dictionary containing the fields for the endpoint.
        """
        url = f"{self.homepage}/{endpt}/{self._mapping}"
        response = requests.get(url)
        mappings = response.json()
        return mappings["fields"]

    def _get_endpt_url(self, endpt):
        """
        Get the URL for a given endpoint.

        Args:
            endpt (str): The endpoint to get the URL for.

        Returns:
            str: The URL for the endpoint.
        """
        if getattr(self, f"_{endpt}_endpt_url") is None:
            setattr(
                self, f"_{endpt}_endpt_url", self._make_endpt_url(self._endpts[endpt])
            )
        return getattr(self, f"_{endpt}_endpt_url")

    def get_json_data(self, url: str, params: dict):
        """
        Fetch JSON data from a given URL with specified parameters.

        Args:
            url (str): The URL to fetch data from.
            params (dict): The parameters to include in the request.

        Returns:
            dict: The JSON data fetched from the URL.
        """
        response = requests.get(url, params=params)
        json_data = json.loads(response.text)
        return json_data

    @staticmethod
    def get_response(url: str, params: str):
        """
        Get the response from a GDC API query.

        Args:
            url (str): The URL to query.
            params (dict): The parameters to pass to the query.

        Returns:
            requests.models.Response: The response object.
        """
        response = requests.get(url, params=params)
        return response

    def query(self, endpoint: str, params: dict = None, method: str = "GET", data=None):
        """
        General purpose method to query the GDC API.

        Args:
            endpoint (str): The endpoint to query.
            params (dict, optional): The parameters to include in the request. Defaults to None.
            method (str, optional): The HTTP method to use. Defaults to "GET".
            data (dict, optional): The data to include in the request body. Defaults to None.

        Returns:
            dict: The JSON response from the API.

        Raises:
            requests.exceptions.HTTPError: If the response status code is not 200.
        """
        url = f"{self.homepage}{endpoint}"
        response = requests.request(
            method, url, headers=self.headers, params=params, json=data
        )
        if response.status_code == 200:
            try:
                print("Valid Connection")
                return response.json()
            except json.JSONDecodeError:
                print(
                    f"Failed to decode JSON. Status code: {response.status_code}, Response text: {response.text}"
                )
                raise
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            response.raise_for_status()

    def download_by_file_id(self, file_id: str):
        """
        Download a file from the GDC API by file ID.

        Args:
            file_id (str): The ID of the file to download.

        Returns:
            None
        """
        data_endpt = "{}/data/{}".format(self.homepage, file_id)
        response = requests.get(
            data_endpt, headers={"Content-Type": "application/json"}
        )
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]
        print(file_name)
        with open(file_name, "wb") as output_file:
            output_file.write(response.content)


    def _query_to_json(self, params: dict, op_params=None):
        """
        Fetches RNA-Seq STAR counts data from the GDC Files API endpoint.

        This method fetches RNA-Seq STAR counts data from the GDC Files API endpoint based on the provided parameters.

        Args:
            params (dict): A dictionary containing the parameters for the API request.
                accepted keys:
                - cases.project.primary_site (list): A list of primary sites to filter the data.
                - new_fields (list): A list of additional fields to include in the metadata. Default is None.
                - data_type (str): The type of data to fetch. Default is 'RNA-Seq'.

        Returns:
            tuple: A tuple containing the JSON data response and the applied filters.

        Raises:
            ValueError: If the list of primary sites is not provided or if the provided field is not in the list of fields by GDC.

        """
        param_keys = params.keys()
        print(param_keys)     
        
        if "new_fields" not in list(param_keys):
            fields = self.gdc_fld.dft_rna_seq_star_count_data_fields
        elif params["new_fields"] is None:
            fields = self.gdc_fld.dft_rna_seq_star_count_data_fields
        else:
            endpt_fields = self.endpt_fields[self.endpt]
            new_fields = params["new_fields"]
            for x in new_fields:
                if x not in endpt_fields:
                    raise ValueError("Field provided is not in the list of fields by GDC")
            self.gdc_fld.update_fields('dft_rna_seq_star_count_data_fields', new_fields)
            fields = self.gdc_fld.dft_rna_seq_star_count_data_fields        
        fields = ",".join(fields)
        print(fields)

        
        filters  = self.gdc_flt.rna_seq_data_filter(params, op_params=op_params)
        url_params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "format": "json",
            "size": "50000"
            }

        url = self._make_endpt_url(self.endpt)
        response = requests.get(url=url, params=url_params)
        json_data = json.loads(response.text)
        return json_data, filters

### Redundant code: Will possibly be used in hindsight
# print(file_name)
# with open(file_name, "wb") as output_file:
#     output_file.write(response.content)


# @property
# def list_of_programs(self):
#     return self.get_files_facet_data(url=self.projects_endpt_url, facet_key='list_of_projects_flt', method_name='list_of_projects_flt')

# @property
# def list_of_primary_sites(self):
#     return self.get_files_facet_data(url=self.cases_endpt_url, facet_key='list_of_primary_sites_flt', method_name='list_of_primary_sites_flt')

# @property
# def list_of_exp_strategies(self):
#     return self.get_files_facet_data(url=self.files_endpt_url, facet_key='list_of_exp_flt', method_name='list_of_exp_flt')
