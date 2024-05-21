import json
import requests
import re

class GDCFilters:
    def __init__(self) -> None:
        pass 

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