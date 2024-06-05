
from __future__ import annotations
### External Imports
import json
from pandas import json_normalize
import pandas as pd 

### Internal Imports
import src.Connectors.gdc_files_endpt as gdc_files
import src.Connectors.gdc_cases_endpt as gdc_cases 
import src.Connectors.gdc_projects_endpt as gdc_projects


"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Parser for processing json_data objects into dataframes 
This is in facade object style to allow different endpt subsytems (Files, Cases or Projects)
to perform queries 
@author: Abhilash Dhal
@date:  2024_06_04
"""
class GDCParser:
    def __init__(self, 
                 gdc_files_sub:gdc_files.GDCFilesEndpt, 
                 gdc_cases_sub:gdc_cases.GDCCasesEndpt,
                 gdc_projs_sub:gdc_projects.GDCProjectsEndpt) -> None:
        
        self._files_sub = gdc_files_sub or gdc_files.GDCFilesEndpt()
        self._cases_sub = gdc_cases_sub or gdc_cases.GDCCasesEndpt()
        self._projs_sub = gdc_projs_sub or gdc_projects.GDCProjectsEndpt() 

### First building parsers for files endpoint 
    def create_df_from_rna_star_count_q_op(self, data):
        """
        Function to create dataframe of json_data returned from query for RNA seq star count data


        """
        data = data['data']['hits']

        # Preprocessing to ensure 'treatments' key exists in all 'diagnoses'
        for item in data:
                for case in item.get('cases', []):
                    if 'diagnoses' not in case:
                            case['diagnoses'] = [
                                {"days_to_last_follow_up": None,
                                "primary_diagnosis": "Not Available",
                                "tumor_grade": "Not Available",
                                "treatments": [{"treatment_or_therapy": "Not Available"},
                                                {"treatment_or_therapy": "Not Available"}],
                                "last_known_disease_status": "Not Available"}]
                    else:
                        for diagnosis in case.get('diagnoses', []):
                                if 'treatments' not in diagnosis:
                                    # Provide a default empty list or other default structure for 'treatments'
                                    diagnosis['treatments'] = [{'treatment_or_therapy': 'Not Available'}]  # default
        # Assuming 'data' is already defined
        # Define the correct paths and structure for json_normalize
        df = json_normalize(
        data,
        record_path=['cases', 'diagnoses', 'treatments'],
        meta=[
                'file_id',
                'file_name',
                'experimental_strategy',
                'data_type',
                ['cases', 'case_id'],
                ['cases', 'diagnoses', 'last_known_disease_status'],
                ['cases', 'diagnoses', 'primary_diagnosis'],
                ['cases', 'diagnoses', 'tumor_grade'],
                ['cases', 'diagnoses', 'days_to_last_follow_up'],
                ['analysis', 'workflow_type']
        ],
        errors='ignore'  # This will fill missing keys with NaN, useful if not all records are uniform
        )

        # Flatten treatments if needed and manage multiple entries correctly
        df['treatment_or_therapy'] = df[[col for col in df.columns if col.startswith('treatment_or_therapy')]].apply(
        lambda x: ', '.join(x.dropna().unique()), axis=1
        )

        # Drop original treatments columns if they are no longer needed
        df.drop(columns=[col for col in df.columns if 'treatment_or_therapy' in col and col != 'treatment_or_therapy'], inplace=True)
        return df

    def get_longitudinal_data_q_op(self, data):
        raise NotImplementedError()

# def create_projects_by_ps_gender_race_exp_df(self, json_data):
#     df = pd.DataFrame(json_data['data']['hits'])
#     new_dict = {'file_id': [], 'file_name':[], 'disease_type':[], 'project_id':[], 'sample_type':[], 'submitter_id':[]}
#     for row in df.iterrows():
#         new_dict['file_id'].append(row[1]['file_id'])
#         new_dict['file_name'].append(row[1]['file_name'])
#         cases_row = row[1]['cases'][0]
#         new_dict['disease_type'].append(cases_row['disease_type'])
#         new_dict['project_id'].append(cases_row['project']['project_id'])
#         new_dict['sample_type'].append(cases_row['samples'][0]['sample_type'])
#         new_dict['submitter_id'].append(cases_row['submitter_id'])
#     return df 