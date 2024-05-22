import json
import re
import src.Connectors.gdc_fields as gdc_fld
import src.Connectors.gdc_field_validator as gdc_vld

"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC filter class and high-level API functions

@author: Abhilash Dhal
@date:  2024_22_27
"""
class GDCFilters(gdc_fld.GDCQueryFields):
    def __init__(self, endpt) -> None:
        super().__init__(endpt)
        self.gdc_vld = gdc_vld.GDCValidator()

    def generate_filters(self, filter_list, operation='and'):
        # Initialize the main filters dictionary
        filters = {
            "op": operation,
            "content": []
        }

        # Loop through each filter specification and append it to the filters["content"]
        for filter_spec in filter_list:
            filter_op = {
                "op": "in",
                "content": {
                    "field": filter_spec["field"],
                    "value": filter_spec["value"]
                }
            }
            filters["content"].append(filter_op)
        return filters
    
    def create_filters(self, filter_specs):
        """
        Creates a list of filters based on given specifications.
        
        Args:
        filter_specs (list of tuples): Each tuple contains the field name and the corresponding values list.
        
        Returns:
        list of dicts: List containing filter dictionaries.
        """
        filters = []
        for field, values in filter_specs:
            filter_op = {
                "op": "in",
                "content": {
                    "field": field,
                    "value": values
                }
            }
            filters.append(filter_op)
        return filters 
        
    def all_projects_by_exp_filter(self, experimental_strategy):
        filters = {
            "op": "in",
            "content":
            {
                "field": "files.experimental_strategy",
                "value": experimental_strategy
            }
        }
        return filters 
    
    def primary_site_filter(self, ps_value:list)->None:
        # cases_endpt = "https://api.gdc.cancer.gov/cases"
        if ps_value is None:
            raise ValueError('Please provide a valid list of primary sites')
        

        filters = {
            "op": "in",
            "content":{
                "field": "cases.primary_site",
                "value": ps_value
                }
            }

        return filters

    def projects_by_disease_filter(self,  disease_type:list):
        filters = {
            "op": "in",
            "content": {
                "field": "projects.disease_type",
                "value": disease_type
            }
        }
        return filters
    
    def ps_race_gender_exp_filter(self, ps_list:list=None, race_list:list=None, exp_list:list=None, gender_list:list=['male', 'female']):
        
        filters = {
            "op": "and",
            "content":[
                {
                "op": "in",
                "content":{
                    "field": "cases.project.primary_site",
                    "value": ps_list
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "cases.demographic.race",
                    "value": race_list
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "cases.demographic.gender",
                    "value": gender_list
                    }
                },
                {
                "op": "in",
                "content":{
                    "field": "files.experimental_strategy",
                    "value": exp_list
                    }
                }
            ]
        }
        return filters

    def primary_site_exp_filter(self, primary_sites:list, experimental_strategies:list, data_formats:list):
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.primary_site",
                        "value": primary_sites
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.experimental_strategy",
                        "value": experimental_strategies
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "files.data_format",
                        "value": data_formats
                    }
                }
            ]
        }
        return filters
    
    def rna_seq_star_count_filter(self, ps_list=None, race_list=None, gender_list=None):
        default_filter_specs = [
            ("files.experimental_strategy", ["RNA-Seq"]),
            ("data_type", ["Gene Expression Quantification"]),
            ("analysis.workflow_type", ["STAR - Counts"])
        ]
        # Start with default filters
        combined_filter_specs = default_filter_specs.copy()

        # Append conditionally based on the content of user inputs
        if ps_list:
            combined_filter_specs.append(("cases.project.primary_site", ps_list))

        if race_list:
            combined_filter_specs.append(("cases.demographic.race", race_list))

        if gender_list:
            combined_filter_specs.append(("cases.demographic.gender", gender_list))

        # Output the filter list to verify its contents
        final_filters = self.create_filters(combined_filter_specs)
        filters_for_query = filters = {
            "op": "and",
            "content": final_filters
        } 
        return filters_for_query