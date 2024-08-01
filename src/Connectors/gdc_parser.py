
from __future__ import annotations
### External Imports
import json
from pandas import json_normalize
import pandas as pd 
from flatten_json import flatten
import numpy as np
### Internal Imports
import src.Connectors.gdc_files_endpt as gdc_files
import src.Connectors.gdc_cases_endpt as gdc_cases 
import src.Connectors.gdc_projects_endpt as gdc_projects
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
    def __init__(self, 
                 gdc_files_sub:gdc_files.GDCFilesEndpt, 
                 gdc_cases_sub:gdc_cases.GDCCasesEndpt,
                 gdc_projs_sub:gdc_projects.GDCProjectsEndpt) -> None:
        
        self._files_sub = gdc_files_sub or gdc_files.GDCFilesEndpt()
        self._cases_sub = gdc_cases_sub or gdc_cases.GDCCasesEndpt()
        self._projs_sub = gdc_projs_sub or gdc_projects.GDCProjectsEndpt() 

### First building parsers for files endpoint 
    def get_unnested_dict_for_rna_seq(self, data):
    # Extract and handle missing data
        unnested_data = {
        'id': data.get('id'),
        'case_id': data.get('cases', [{}])[0].get('case_id'),
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
    
    def make_df_rna_seq(self, json_data):
        parsed_data = list()
        entries = json_data['data']['hits']
        for entry in entries:
            unnested_data = self.get_unnested_dict_for_rna_seq(entry)
            parsed_data.append(unnested_data) 
        df = pd.DataFrame(parsed_data)
        return df 
      
    # def create_df_from_rna_star_count_q_op(self, data, filtered=True):
    #     """
    #     Function to create a dataframe from the json_data returned from the query for RNA seq star count data.

    #     Args:
    #         data (dict): The json_data returned from the query.

    #     Returns:
    #         pandas.DataFrame: The dataframe created from the json_data.

    #     """
    #     data = data['data']['hits']
    #     # Preprocessing to ensure 'treatments' key exists in all 'diagnoses'
    #     for item in data:
    #         for case in item.get('cases', []):
    #             if 'project' not in case:
    #                 case['project'] = {"primary_site": "Not Available"}
    #             else:
    #                 if 'primary_site' not in case['project']:
    #                     case['project']['primary_site'] = "Not Available"

    #             # if 'samples' not in case:
    #             #     case['samples'] = {"tissue_type": "Not Available"}
    #             # else:
    #             #     if 'tissue_type' not in case['samples']:
    #             #         case['samples']['tissue_type'] = 'Not Available'
                        
    #             if 'demographic' not in case:
    #                 case['demographic']['race'] = "Not Available"
    #                 case['demographic']['gender'] = "Not Available"
    #                 case['demographic']['ethnicity'] = "Not Available"
    #             else:
    #                 if 'race' not in case['demographic']:
    #                     case['demographic']['race'] = "Not Available"
    #                 elif 'gender' not in case['demographic']:
    #                     case['demographic']['gender'] = 'Not Available'
    #                 elif 'ethnicity' not in case['demographic']:
    #                     case['demographic']['ethnicity'] = 'Not Available'

    #             if 'diagnoses' not in case:
    #                 case['diagnoses'] = [
    #                     {"days_to_last_follow_up": None,
    #                         "primary_diagnosis": "Not Available",
    #                         "tumor_grade": "Not Available",
    #                         "treatments": [{"treatment_or_therapy": "Not Available"},
    #                                     {"treatment_or_therapy": "Not Available"}],
    #                         "last_known_disease_status": "Not Available"}]
    #             else:
    #                 for diagnosis in case.get('diagnoses', []):
    #                     if 'treatments' not in diagnosis:
    #                         # Provide a default empty list or other default structure for 'treatments'
    #                         diagnosis['treatments'] = [{'treatment_or_therapy': 'Not Available'}]  # default

    #         # Assuming 'data' is already defined
    #         # Define the correct paths and structure for json_normalize
    #     dict_flattened = (flatten(record, '.') for record in data)
    #     df = pd.DataFrame(dict_flattened)

    #     if filtered:
    #         col_list = list(df.columns)
    #         uq_col_list = np.unique([col_list[i].split('.')[-1] for i in range(len(col_list))])
    #         # Flatten treatments if needed and manage multiple entries correctly
    #         for uq_col in uq_col_list:
    #             sub_data = df[[col for col in df.columns if col.__contains__(uq_col)]]
    #             if sum(sub_data.dtypes == 'float') > 0:
    #                 if sum(sub_data.dtypes == 'float') == len(sub_data.columns):
    #                     df[uq_col] = df[[col for col in df.columns if col.__contains__(uq_col)]].apply(
    #                         lambda x: np.max(x), axis=1)
    #             else:
    #                 id_col = bool(re.search('id', uq_col))
    #                 if not id_col:
    #                     df[uq_col] = df[[col for col in df.columns if col.__contains__(uq_col)]].apply(
    #                         lambda x: ', '.join(x.dropna().unique()), axis=1
    #                     )
    #                 else:
    #                     df[uq_col] = df[[col for col in df.columns if col.__contains__(uq_col)]].apply(
    #                         lambda x: x[0], axis=1
    #                     )
                        

    #         # Drop original treatments columns if they are no longer needed
    #         cols_to_select = [col for col in df.columns if col in uq_col_list]
    #         # df.drop(columns=[col for col in df.columns if uq_col in col and col != uq_col], inplace=True)

    #         # cols_to_select = [col for col in df.columns if col in uq_col_list]
    #         df = df[cols_to_select]
    #         df = df[df['treatment_or_therapy'].isin(['yes', 'no', 'not reported', 'unknown', 'Not Available'])].reset_index(
    #             drop=True)   
    #     else:
    #         df = json_normalize(
    #         data,
    #         record_path=['cases', 'diagnoses', 'treatments'],
    #         meta=[
    #                 'file_id',
    #                 'file_name',
    #                 'experimental_strategy',
    #                 'data_type',
    #                 ['cases', 'case_id'],
    #                 ['cases', 'project', 'primary_site'],
    #                 ['cases', 'diagnoses', 'last_known_disease_status'],
    #                 ['cases', 'diagnoses', 'primary_diagnosis'],
    #                 ['cases', 'diagnoses', 'tumor_grade'],
    #                 ['cases', 'diagnoses', 'days_to_last_follow_up'],
    #                 ['cases', 'diagnoses', 'age_at_diagnosis'],
    #                 ['cases', 'demographic', 'ethnicity'],
    #                 ['cases', 'demographic', 'gender'],
    #                 ['cases', 'demographic', 'race'],
    #                 ['cases', 'diagnoses', 'tissue_or_organ_of_origin'],
    #                 ['cases', 'diagnoses', 'days_to_death'],
    #                 ['cases', 'samples', 'tissue_type'], 
    #                 ['analysis', 'workflow_type']
    #         ],
    #         errors='ignore'  # This will fill missing keys with NaN, useful if not all records are uniform
    #         )

    #         # Flatten treatments if needed and manage multiple entries correctly
    #         df['treatment_or_therapy'] = df[[col for col in df.columns if col.startswith('treatment_or_therapy')]].apply(
    #         lambda x: ', '.join(x.dropna().unique()), axis=1
    #         )

    #         # Drop original treatments columns if they are no longer needed
    #         df.drop(columns=[col for col in df.columns if 'treatment_or_therapy' in col and col != 'treatment_or_therapy'], inplace=True)
    #         df = df.drop_duplicates(subset='file_id')
    #     return df
 