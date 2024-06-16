import json
import requests

import src.Connectors.gdc_filters as gdc_flt
import src.Connectors.gdc_fields as gdc_fld
import re
import pandas as pd 
from abc import abstractmethod
# Example of usage
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Base Utils class and high-level API functions

@author: Abhilash Dhal
@date:  2024_22_27
"""
class GDCEndptBase:
    def __init__(self, homepage='https://api.gdc.cancer.gov', endpt=None):
        self.endpt = endpt
        self.homepage = homepage
        self.headers = {
            'Content-Type': 'application/json'
        }

    def make_endpt_url(self, endpt:str):
        if self.endpt is not None:
            return f"{self.homepage}/{endpt}"
        else:
            return None

####### COMMON API calls for GDC ####################################################
    @abstractmethod
    def get_json_data(self, url, params):
        response = requests.get(url, params = params)
        json_data = json.loads(response.text)
        return json_data

    @abstractmethod
    def make_params_dict(self, filters, fields, size=100, format='tsv'):
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "format": format,
            "size": "100"
            }
        return params
    

    def get_response(self, url, params):
        """
        Function to get response from GDC API 

        Args:
        url (str): The URL to query.
        params (dict): The parameters to pass to the query.

        Returns:
        requests.models.Response: The response object.
        """
        response = requests.get(url, params = params)
        return response


    def query(self, endpoint, params=None, method='GET', data=None):
        """
        General purpose method to query GDC API.
        """
        url = f"{self.homepage}{endpoint}"
        response = requests.request(method, url, headers=self.headers, params=params, data=json.dumps(data) if data else data)
        if response.status_code == 200:
            try:
                print("Valid Connection")
                return response.json()
            except json.JSONDecodeError as e:
                print(f"Failed to decode JSON. Status code: {response.status_code}, Response text: {response.text}")
                raise
        else:
            print(f"Error: {response.status_code}, Response: {response.text}")
            response.raise_for_status()
            
    def download_by_file_id(self, file_id):
        data_endpt = "{}/data/{}".format(self.homepage, file_id)

        response = requests.get(data_endpt, headers = {"Content-Type": "application/json"})

        # The file name can be found in the header within the Content-Disposition key.
        response_head_cd = response.headers["Content-Disposition"]
        file_name = re.findall("filename=(.+)", response_head_cd)[0]
        print(file_name)
        with open(file_name, "wb") as output_file:
            output_file.write(response.content)

    def create_single_facet_df(self, url, facet_key:str, params:dict):
        response = self.get_response(url, params=params)
        data = response.json()
        facet_df = pd.DataFrame(data['data']['aggregations'][facet_key]['buckets'])
        facet_df['key'] = facet_df['key'].apply(lambda x: x['key'])
        return facet_df
    
    def get_files_facet_data(self, facet_key, method_name):
        raise NotImplementedError("Method will be implemented in child class")


    # @abstractmethod
    # def search(self, endpoint, filters, fields, size=100, format='json'):
    #     """
    #     Search data in the GDC using filters and expansion, now including format handling.
    #     """
    #     data = {
    #         'filters': json.dumps(filters),
    #         'fields': ','.join(fields),
    #         'format': format,
    #         'size': size
    #     }
    #     return self.query(endpoint, method='POST', data=data)
##### Need to update this check tomorrow ###############################
    # def check_valid_endpt(self):
    #     return self.query(self.endpt)

### Methods to be added based on application by user/bioinformatician/ 
### 1. Get list of all disease_types available on gdc platform 
### 2. Fetch Gene expression files by tcga barcodes 
### 3. Fetch Metadata for files based on a list of UUIDS 
### 4. Fetch Metadata by primate site query 
### 5. Get Gene expression data by primary site query 
### 6. Get Gene expression data by project query