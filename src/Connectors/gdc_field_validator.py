import json
import requests
import src.Connectors.gdc_filters as gdc_flt
import src.Connectors.gdc_endpt_base as gdc_endpt_base
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Endpoint Fields Validator Class and high-level API functions

@author: Abhilash Dhal
@date:  2024_06_05
"""
### Next steps: Would be Nice to Have a Base Validator Class and then subsequent child validators for each endpoint 
### gdc_field_validtor, gdc_file_validator, gdc_annotation_validator, gdc_project_validator 
class GDCValidator(gdc_flt.GDCFacetFilters):
    def __init__(self, homepage='https://api.gdc.cancer.gov', endpt=None):
        super().__init__(homepage, endpt)
        self.mapping = '_mapping'
        self.endpts = {'projects': 'projects', 'cases':'cases', 'files':'files'}
        self.imp_facet_keys = self.imp_facet_keys
        self._endpt_fields = None

        self._files_endpt_url = None
        self._projects_endpt_url = None
        self._cases_endpt_url = None        
        self._list_of_projects = None 
        self._list_of_primary_sites = None
        self._list_of_exp_strategies = None
        self._list_of_tumor_types = None
        self._list_of_primary_diagnoses = None


    def _get_endpt_url(self, endpt):
        if getattr(self, f'_{endpt}_endpt_url') is None:
            setattr(self, f'_{endpt}_endpt_url', self.make_endpt_url(self.endpts[endpt]))
        return getattr(self, f'_{endpt}_endpt_url')
    
    @property
    def files_endpt_url(self):
        return self._get_endpt_url('files')
    
    @property 
    def projects_endpt_url(self):
        return self._get_endpt_url('projects')
    
    @property
    def cases_endpt_url(self):
        return self._get_endpt_url('cases')
    
    @property
    def list_of_programs(self):
        return self.get_files_facet_data(url=self.projects_endpt_url, facet_key='list_of_projects_flt', method_name='list_of_projects_flt')

    @property
    def list_of_primary_sites(self):
        return self.get_files_facet_data(url=self.cases_endpt_url, facet_key='list_of_primary_sites_flt', method_name='list_of_primary_sites_flt')

    @property
    def list_of_exp_strategies(self):
        return self.get_files_facet_data(url=self.files_endpt_url, facet_key='list_of_exp_flt', method_name='list_of_exp_flt')
    
    @property
    def endpt_fields(self):
        if self._endpt_fields is None:
            self._endpt_fields = {endpt: self.get_endpt_fields(endpt) for endpt in self.endpts.keys()}
        return self._endpt_fields
     
    def get_endpt_fields(self, endpt:str):
        url = f'{self.homepage}/{endpt}/{self.mapping}'
        # Send a GET request
        response = requests.get(url)
        mappings = response.json()
        return mappings['fields']

    def validate_project_fields(self, input_fields):
        """
        Function to validate project fields specified by the user against the fields available in the GDC API's projects endpoint

        Args:
        input_fields (list): A list of fields to validate

        Returns:
        bool: True if all fields are valid, False otherwise
        """
        invalid_fields = [field for field in input_fields if field not in self.endpt_fields['projects']]
        if invalid_fields:
            raise ValueError(f"Invalid project fields: {', '.join(invalid_fields)}")
        return True
    
    def validate_case_fields(self, input_fields):
        """
        Function to validate case fields specified by the user against the fields available in the GDC API's cases endpoint

        Args:
        input_fields (list): A list of fields to validate

        Returns:
        bool: True if all fields are valid, False otherwise
        """
        invalid_fields = [field for field in input_fields if field not in self.endpt_fields['cases']]
        if invalid_fields:
            raise ValueError(f"Invalid case fields: {', '.join(invalid_fields)}")
        return True
    
    def validate_file_fields(self, input_fields):
        """
        Function to validate file fields specified by the user against the fields available in the GDC API's files endpoint

        Args:
        input_fields (list): A list of fields to validate

        Returns:
        bool: True if all fields are valid, False otherwise
        """
        invalid_fields = [field for field in input_fields if field not in self.endpt_fields['files']]
        if invalid_fields:
            raise ValueError(f"Invalid file fields: {', '.join(invalid_fields)}")
        return True
    
    def validate_annotation_fields(self, input_fields):
        """
        Function to validate annotation fields specified by the user against the fields available in the GDC API's annotation endpoint

        Args:
        input_fields (list): A list of fields to validate

        Returns:
        bool: True if all fields are valid, False otherwise
        """
        invalid_fields = [field for field in input_fields if field not in self.endpt_fields['annotation']]
        if invalid_fields:
            raise ValueError(f"Invalid annotation fields: {', '.join(invalid_fields)}")
        return True
    

    def get_files_facet_data(self, url, facet_key, method_name):
        facet_key_value = self.imp_facet_keys.get(facet_key, None)
        print(facet_key_value)
        if facet_key_value is None:
            raise ValueError(f"Invalid facet_key: {facet_key}")
        
        if getattr(self, facet_key, None) is None:
            params = self.get_files_endpt_facet_filter(method_name=method_name)
            print(params)
            data = self.create_single_facet_df(url=url, facet_key_value=facet_key_value, params=params)
            data.columns = ['count', f"{facet_key_value}"]
        return data