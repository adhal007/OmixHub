
import json
import requests
import src.Connectors.gdc_endpt_base as gdc_endpt_base
import src.Connectors.gdc_filters as gdc_flt
import src.Connectors.gdc_fields as gdc_fld
import src.Connectors.gdc_field_validator as gdc_vld
import pandas as pd
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Files Endpt Class and high-level API functions Only Returns JSON like Objects for different Queries

@author: Abhilash Dhal
@date:  2024_06_07
"""
class GDCFilesEndpt(gdc_endpt_base.GDCEndptBase):
    """
    A class representing the GDC Files Endpoint.

    This class provides methods to interact with the GDC Files API endpoint and fetch RNA-Seq STAR counts data.

    Args:
        homepage (str, optional): The base URL of the GDC API. Defaults to 'https://api.gdc.cancer.gov'.
        endpt (str, optional): The endpoint name. Defaults to 'files'.

    Attributes:
        gdc_flt (GDCQueryFilters): An instance of the GDCQueryFilters class for handling query filters.
        gdc_fld (GDCQueryDefaultFields): An instance of the GDCQueryDefaultFields class for handling default fields.
        gdc_vld (GDCValidator): An instance of the GDCValidator class for validating fields.

    """

    def __init__(self, homepage='https://api.gdc.cancer.gov', endpt='files'):
        super().__init__(homepage, endpt='files')
        self.gdc_flt = gdc_flt.GDCQueryFilters()
        self.gdc_fld = gdc_fld.GDCQueryDefaultFields(self.endpt)
        self.gdc_vld = gdc_vld.GDCValidator()

    def rna_seq_query_to_json(self, params: dict, op_params=None):
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
        # if params["cases.project.primary_site"] is None:
        #     raise ValueError("List of primary sites must be provided") 
        
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
    
