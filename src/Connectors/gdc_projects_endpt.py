
import src.Connectors.gdc_endpt_base as gdc_endpt_base
import src.Connectors.gdc_filters as gdc_flt
import src.Connectors.gdc_fields as gdc_fld
import json

"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Projects Endpoint Class and high-level API functions

@author: Abhilash Dhal
@date:  2024_22_27
"""

class GDCProjectsEndpt(gdc_endpt_base.GDCEndptBase):
    def __init__(self, homepage='https://api.gdc.cancer.gov', endpt='projects'):
        super().__init__(homepage, endpt='projects')

        self.gdc_flt = gdc_flt.GDCFilters(self.endpt)
        self.gdc_fld = gdc_fld.GDCQueryFields(self.endpt)

######### APPLICATION ORIENTED python functions for projects endpoint ################################################
################################################################################################
    def list_all_projects_by_exp(self, experimental_strategy=None, new_fields=None, size=100, format='json'):
        
        pbe_filter = self.gdc_flt.all_projects_by_exp_filter(experimental_strategy=experimental_strategy)
        if new_fields is None:
            fields = self.gdc_fld.dft_list_all_project_fields
        else:
            self.gdc_fld.update_fields('dft_list_all_project_fields', new_fields)
            fields = self.gdc_fld.dft_list_all_project_fields            
        fields = ",".join(fields)

        params = self.make_params_dict(filters=pbe_filter, fields=fields, size=size, format=format)
        json_data = self.get_json_data(self.files_endpt, params)
        # return self.search('/projects', filters=pbd_filter, fields=fields)
        return json_data
    
    def list_projects_by_disease(self, disease_type, new_fields=None, size=100, format='json'):
        """
        List projects filtered by disease type with specified fields.
        """

        pbd_filter = self.gdc_flt.projects_by_disease_filter(disease_type)

        if new_fields is None:
            fields = self.gdc_fld.dft_project_by_disease_fields
        else:
            self.gdc_fld.update_fields('dft_project_by_disease_fields', new_fields)
            fields = self.gdc_fld.dft_primary_site_race_gender_exp_fields            
        fields = ",".join(fields)
        params = self.make_params_dict(filters=pbd_filter, fields=fields, size=size, format=format)
        json_data = self.get_json_data(self.files_endpt, params)
        # return self.search('/projects', filters=pbd_filter, fields=fields)
        return json_data