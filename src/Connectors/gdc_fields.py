
import json
import src.Connectors.gdc_endpt_base as gdc_endpt_base

"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC fields class and high-level API functions

@author: Abhilash Dhal
@date:  2024_06_07
"""

class GDCQueryDefaultFields:
    def __init__(self, endpt) -> None:

        self._endpts = gdc_endpt_base.GDCEndptBase()._endpts
        if endpt not in self._endpts.keys():
            raise ValueError(f"Endpoint has be either of {','.join(list(self._endpts.keys()))}")
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
            "cases.summary.experimental_strategies.experimental_strategy",
            "cases.diagnoses.last_known_disease_status",
            "cases.diagnoses.primary_diagnosis",
            "cases.diagnoses.tumor_stage",
            "cases.diagnoses.tumor_grade",
            "cases.diagnoses.treatments.treatment_or_therapy"
            ]
        
        self.dft_primary_site_exp_fields = [
            "file_id", "file_name", "cases.submitter_id", "cases.case_id",
            "data_category", "data_type", "cases.samples.tumor_descriptor",
            "cases.samples.tissue_type", "cases.samples.sample_type",
            "cases.samples.submitter_id", "cases.samples.sample_id",
            "cases.samples.portions.analytes.aliquots.aliquot_id",
            "cases.samples.portions.analytes.aliquots.submitter_id"
        ]

        self.dft_rna_seq_star_count_data_fields = ['file_id', 
                                                'file_name', 
                                                'experimental_strategy', 
                                                'data_type','platform', 
                                                'cases.case_id',
                                                "cases.diagnoses.last_known_disease_status",
                                                "cases.diagnoses.primary_diagnosis",
                                                "cases.diagnoses.tumor_stage",
                                                "cases.diagnoses.tumor_grade",
                                                "cases.diagnoses.treatments.treatment_or_therapy",
                                                "cases.diagnoses.days_to_last_follow_up",
                                                'cases.diagnoses.age_at_diagnosis',
                                                'cases.diagnoses.days_to_death',
                                                "cases.project.primary_site",
                                                'analysis.workflow_type',
                                                'cases.demographic.ethnicity',
                                                'cases.demographic.gender',
                                                'cases.demographic.race',
                                                'cases.diagnoses.tissue_or_organ_of_origin',
                                                'cases.exposures.bmi',
                                                'cases.exposures.alcohol_history',
                                                'cases.exposures.years_smoked'
                                                ]
        
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
        
        
    
