from __future__ import annotations
### External Imports
import json
from pandas import json_normalize
import pandas as pd 
from flatten_json import flatten
import numpy as np
### Internal Imports
import src.Connectors.gdc_endpt_base as gdc_endpt_base
import re


"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Parser for processing json_data objects into dataframes 
This is in facade object style to allow different endpt subsytems (Files, Cases or Projects)
to perform queries 
@author: Abhilash Dhal
@date:  2024_06_04
"""
class GDCJson2DfParser:
    """
    GDCJson2DfParser class for processing JSON data objects into DataFrames.

    This class uses a facade object style to allow different endpoint subsystems (Files, Cases, or Projects)
    to perform queries.

    Attributes:
        _files_sub (gdc_files.GDCFilesEndpt): The files endpoint subsystem.
        _cases_sub (gdc_cases.GDCCasesEndpt): The cases endpoint subsystem.
        _projs_sub (gdc_projects.GDCProjectsEndpt): The projects endpoint subsystem.
    """

    def __init__(self, 
                 gdc_files_sub: gdc_endpt_base.GDCEndptBase(endpt='files'), 
                 gdc_cases_sub: gdc_endpt_base.GDCEndptBase(endpt='cases'),
                 gdc_projects_sub: gdc_endpt_base.GDCEndptBase(endpt='projects')) -> None:
        """
        Initialize the GDCJson2DfParser class.

        Args:
            gdc_files_sub (gdc_files.GDCFilesEndpt): The files endpoint subsystem.
            gdc_cases_sub (gdc_cases.GDCCasesEndpt): The cases endpoint subsystem.
            gdc_projs_sub (gdc_projects.GDCProjectsEndpt): The projects endpoint subsystem.
        """
        self._files_sub = gdc_files_sub
        self._cases_sub = gdc_cases_sub 
        self._projs_sub = gdc_projects_sub 

    def get_unnested_dict_for_rna_seq(self, data: dict) -> dict:
        """
        Extract and handle missing data from RNA sequencing JSON data.

        Args:
            data (dict): The JSON data to process.

        Returns:
            dict: The unnested data dictionary.
        """
        unnested_data = {
            'id': data.get('id'),
            'submitter_id': data.get('submitter_id'),
            'case_id': data.get('cases', [{}])[0].get('case_id'),
            'project_id': data.get('cases', [{}])[0].get('project', {}).get('project_id'),
            'project_name': data.get('cases', [{}])[0].get('project', {}).get('project_name').lower().replace('-', '_'),
            'program_id': data.get('cases', [{}])[0].get('project', {}).get('program', {}).get('program_id'),
            'program_name': data.get('cases', [{}])[0].get('project', {}).get('program', {}).get('program_name'),
            'alcohol_history': data.get('cases', [{}])[0].get('exposures', [{}])[0].get('alcohol_history'),
            'years_smoked': data.get('cases', [{}])[0].get('exposures', [{}])[0].get('years_smoked'),
            'tissue_or_organ_of_origin': data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('tissue_or_organ_of_origin'),
            'days_to_last_follow_up': data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('days_to_last_follow_up'),
            'age_at_diagnosis': data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('age_at_diagnosis'),
            'primary_diagnosis': data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('primary_diagnosis'),
            'primary_site': data.get('cases', [{}])[0].get('project', {}).get('primary_site'),
            'tumor_grade': data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('tumor_grade'),
            'treatment_or_therapy': next(
                (t.get('treatment_or_therapy') for t in data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('treatments', [])
                if t.get('treatment_or_therapy') in ['yes', 'no']), 'unknown'),
            'last_known_disease_status': data.get('cases', [{}])[0].get('diagnoses', [{}])[0].get('last_known_disease_status'),
            'tissue_type': data.get('cases', [{}])[0].get('samples', [{}])[0].get('tissue_type'),
            'sample_type': data.get('cases', [{}])[0].get('samples', [{}])[0].get('sample_type'), 
            'race': data.get('cases', [{}])[0].get('demographic', {}).get('race'),
            'gender': data.get('cases', [{}])[0].get('demographic', {}).get('gender'),
            'ethnicity': data.get('cases', [{}])[0].get('demographic', {}).get('ethnicity'),
            'file_name': data.get('file_name'),
            'file_id': data.get('file_id'),
            'data_type': data.get('data_type'),
            'workflow_type': data.get('analysis', {}).get('workflow_type'),
            'experimental_strategy': data.get('experimental_strategy')
        }
        return unnested_data 
    
    def make_df_rna_seq(self, json_data: dict) -> pd.DataFrame:
        """
        Create a DataFrame from RNA sequencing JSON data.

        Args:
            json_data (dict): The JSON data to process.

        Returns:
            pd.DataFrame: The resulting DataFrame.
        """
        parsed_data = list()
        entries = json_data['data']['hits']
        for entry in entries:
            unnested_data = self.get_unnested_dict_for_rna_seq(entry)
            parsed_data.append(unnested_data) 
        df = pd.DataFrame(parsed_data)
        return df 