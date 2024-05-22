
import json
import src.Connectors.gdc_field_validator as gdc_vld
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC fields class and high-level API functions

@author: Abhilash Dhal
@date:  2024_22_27
"""

class GDCQueryFields():
    def __init__(self, endpt) -> None:

        self.endpts = ['cases', 'annotation', 'projects', 'files']
        if endpt not in self.endpts:
            raise ValueError(f"Endpoint has be either of {','.join(self.endpts)}")
        self.endpt = endpt

        self.dft_list_all_project_fields = [
            "project_id", "project_name", "program.name", "summary.experimental_strategies.experimental_strategy"
        ]

        self.dft_primary_site_fields = [
        "file_id", "file_name", "cases.submitter_id", "cases.case_id",
        "data_category", "data_type", "cases.samples.tumor_descriptor",
        "cases.samples.tissue_type", "cases.samples.sample_type",
        "cases.samples.submitter_id", "cases.samples.sample_id",
        "cases.samples.portions.analytes.aliquots.aliquot_id",
        "cases.samples.portions.analytes.aliquots.submitter_id"
        ]

        self.dft_project_by_disease_fields = [
            "project_id", "project_name", "primary_site", "program.name"
        ]

        self.dft_primary_site_race_gender_exp_fields = [
            "file_id",
            "file_name",
            "cases.submitter_id",
            "cases.samples.sample_type",
            "cases.disease_type",
            "cases.project.project_id",
            "cases.summary.experimental_strategies.experimental_strategy"
            ]
        
        self.dft_primary_site_exp_fields = [
            "file_id", "file_name", "cases.submitter_id", "cases.case_id",
            "data_category", "data_type", "cases.samples.tumor_descriptor",
            "cases.samples.tissue_type", "cases.samples.sample_type",
            "cases.samples.submitter_id", "cases.samples.sample_id",
            "cases.samples.portions.analytes.aliquots.aliquot_id",
            "cases.samples.portions.analytes.aliquots.submitter_id"
        ]

        self.dft_rna_seq_star_count_data_fields = ['file_id', 'file_name', 
                                                   'experimental_strategy', 'data_type', 
                                                   'cases.case_id', 'analysis.workflow_type']
        
    def update_fields(self, field_name, new_fields):
        """
        General method to update field lists based on the field name and new fields.
        
        :param field_name: The name of the field list to update.
        :param new_fields: List of new fields to replace the existing ones.
        """
        if new_fields is not None and isinstance(new_fields, list):
            if hasattr(self, field_name):
                setattr(self, field_name, new_fields)
            else:
                raise ValueError(f"No such field list: {field_name}")
        else:
            raise ValueError("Invalid new fields: Must be a non-empty list.")
        
        
    
