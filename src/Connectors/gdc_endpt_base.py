import json
import requests
import re
import pandas as pd
from abc import abstractmethod

# Example of usage
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Base Utils class and high-level API functions Only Returns JSON like Objects for different Queries
@author: Abhilash Dhal
@date:  2024_22_27
"""


class GDCEndptBase:
    def __init__(self, homepage: str = "https://api.gdc.cancer.gov", endpt: str = None):
        self.endpt = endpt
        self._endpts = {"files": "files", "projects": "projects", "cases": "cases"}
        self.homepage = homepage
        self.headers = {"Content-Type": "application/json"}

        self._mapping = "_mapping"
        self._files_endpt_url = None
        self._projects_endpt_url = None
        self._cases_endpt_url = None
        self._endpt_fields = None
        # self._list_of_projects = None
        # self._list_of_primary_sites = None
        # self._list_of_exp_strategies = None
        # self._list_of_tumor_types = None
        # self._list_of_primary_diagnoses = None

    ### These properties should be in the base class
    @property
    def files_endpt_url(self):
        return self._get_endpt_url("files")

    @property
    def projects_endpt_url(self):
        return self._get_endpt_url("projects")

    @property
    def cases_endpt_url(self):
        return self._get_endpt_url("cases")

    @property
    def endpt_fields(self):
        if self._endpt_fields is None:
            self._endpt_fields = {
                endpt: self._get_endpt_fields(endpt) for endpt in self._endpts.keys()
            }
        return self._endpt_fields

    def _make_endpt_url(self, endpt: str):
        if self.endpt is None:
            return f"{self.homepage}/{endpt}"
        else:
            return f"{self.homepage}/{self.endpt}"

    def _get_endpt_fields(self, endpt: str):
        url = f"{self.homepage}/{endpt}/{self._mapping}"
        # Send a GET request
        response = requests.get(url)
        mappings = response.json()
        return mappings["fields"]

    def _get_endpt_url(self, endpt):
        if getattr(self, f"_{endpt}_endpt_url") is None:
            setattr(
                self, f"_{endpt}_endpt_url", self._make_endpt_url(self._endpts[endpt])
            )
        return getattr(self, f"_{endpt}_endpt_url")

    ####### COMMON API calls for GDC ####################################################
    def get_json_data(self, url: str, params: dict):
        response = requests.get(url, params=params)
        json_data = json.loads(response.text)
        return json_data

    @staticmethod
    def get_response(url: str, params: str):
        """
        Function to get response from GDC API

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
        General purpose method to query GDC API.
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
        data_endpt = "{}/data/{}".format(self.homepage, file_id)
        response = requests.get(
            data_endpt, headers={"Content-Type": "application/json"}
        )
        # The file name can be found in the header within the Content-Disposition key.
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]
        print(file_name)
        with open(file_name, "wb") as output_file:
            output_file.write(response.content)


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
