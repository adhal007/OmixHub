
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
    def __init__(self, homepage='https://api.gdc.cancer.gov', endpt='files'):
        super().__init__(homepage, endpt='files')
        # if self.check_valid_endpt():
        self.gdc_flt = gdc_flt.GDCQueryFilters()
        self.gdc_fld = gdc_fld.GDCQueryDefaultFields(self.endpt)
        self.gdc_vld = gdc_vld.GDCValidator()

######### APPLICATION ORIENTED python functions ################################################
################################################################################################
    def fetch_rna_seq_star_counts_data(self, new_fields:list[str]=None, ps_list:list[str]=None, race_list:list[str]=None, gender_list:list[str]=None):
        if ps_list is None:
            raise ValueError("List of primary sites must be provided") 
        if new_fields is None:
            fields = self.gdc_fld.dft_rna_seq_star_count_data_fields
        else:
            ## Adding logic for checking fields
            endpt_fields = self.gdc_vld.endpt_fields[self.endpt]
            for x in new_fields:
                ## create file_endpt_fields in gdc_vld
                if x not in endpt_fields:
                    raise ValueError("Field provided is not in the list of fields by GDC")
            self.gdc_fld.update_fields('dft_rna_seq_star_count_data_fields', new_fields)
            fields = self.gdc_fld.dft_rna_seq_star_count_data_fields        
        fields = ",".join(fields)


        filters = self.gdc_flt.rna_seq_star_count_filter(ps_list=ps_list, race_list=race_list, gender_list=gender_list)
        # Here a GET is used, so the filter parameters should be passed as a JSON string.
        print(filters)
        params = {
            "filters": json.dumps(filters),
            "fields": fields,
            "format": "json",
            "size": "50000"
            }
        print(params)
        
        url = self.make_endpt_url(self.endpt)
        response = requests.get(url=url, params = params)
        json_data = json.loads(response.text)
        return json_data, filters
    
