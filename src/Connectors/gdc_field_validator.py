import json
import requests

"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Endpoint Fields Validator Class and high-level API functions

@author: Abhilash Dhal
@date:  2024_22_27
"""


### Next steps: Would be Nice to Have a Base Validator Class and then subsequent child validators for each endpoint 
### gdc_field_validtor, gdc_file_validator, gdc_annotation_validator, gdc_project_validator 
class GDCValidator:
    def __init__(self, 
                 homepage='https://api.gdc.cancer.gov', 
                 case_fields_filename='/Users/abhilashdhal/Projects/src_files/gdc_case_fields.txt',
                 file_fields_filename='/Users/abhilashdhal/Projects/src_files/gdc_file_fields.txt'):
        
        self.project_endpt_fields = [
            "dbgap_accession_number",
            "disease_type",
            "name",
            "primary_site",
            "project_id",
            "released",
            "state",
            "program.dbgap_accession_number",
            "program.name",
            "program.program_id",
            "summary.case_count",
            "summary.file_count",
            "summary.file_size",
            "summary.data_categories.case_count",
            "summary.data_categories.data_category",
            "summary.data_categories.file_count",
            "summary.experimental_strategies.case_count",
            "summary.experimental_strategies.experimental_strategy",
            "summary.experimental_strategies.file_count"
        ]
        self.case_endpt_fields = self.load_fields_from_file(case_fields_filename)
        self.file_endpt_fields = self.load_fields_from_file(file_fields_filename)
        self.annotation_endpt_fields = [
            "annotation_id", "case_id", "case_submitter_id", "category",
            "classification", "created_datetime", "entity_id", "entity_submitter_id",
            "entity_type", "legacy_created_datetime", "legacy_updated_datetime",
            "notes", "state", "status", "submitter_id", "updated_datetime",
            "project.code", "project.dbgap_accession_number", "project.disease_type",
            "project.name", "project.primary_site", "project.program.dbgap_accession_number",
            "project.program.name", "project.program.program_id", "project.project_id",
            "project.released", "project.state"
        ]
        self.endpoint_fields = {
            "project": self.project_endpt_fields,
            "cases": self.case_endpt_fields,
            "files": self.file_endpt_fields,
            "annotation": self.annotation_endpt_fields
        }

        self.list_of_primary_sites = [
                                    "accessory sinuses",
                                    "adrenal gland",
                                    "anus and anal canal",
                                    "base of tongue",
                                    "bladder",
                                    "bones, joints and articular cartilage of limbs",
                                    "bones, joints and articular cartilage of other and unspecified sites",
                                    "brain",
                                    "breast",
                                    "bronchus and lung",
                                    "cervix uteri",
                                    "colon",
                                    "connective, subcutaneous and other soft tissues",
                                    "corpus uteri",
                                    "esophagus",
                                    "eye and adnexa",
                                    "floor of mouth",
                                    "gallbladder",
                                    "gum",
                                    "heart, mediastinum, and pleura",
                                    "hematopoietic and reticuloendothelial systems",
                                    "hypopharynx",
                                    "kidney",
                                    "larynx",
                                    "lip",
                                    "liver and intrahepatic bile ducts",
                                    "lymph nodes",
                                    "meninges",
                                    "nasal cavity and middle ear",
                                    "nasopharynx",
                                    "not reported",
                                    "oropharynx",
                                    "other and ill-defined digestive organs",
                                    "other and ill-defined sites",
                                    "other and ill-defined sites in lip, oral cavity and pharynx",
                                    "other and ill-defined sites within respiratory system and intrathoracic organs",
                                    "other and unspecified female genital organs",
                                    "other and unspecified major salivary glands",
                                    "other and unspecified male genital organs",
                                    "other and unspecified parts of biliary tract",
                                    "other and unspecified parts of mouth",
                                    "other and unspecified parts of tongue",
                                    "other and unspecified urinary organs",
                                    "other endocrine glands and related structures",
                                    "ovary",
                                    "palate",
                                    "pancreas",
                                    "parotid gland",
                                    "penis",
                                    "peripheral nerves and autonomic nervous system",
                                    "prostate gland",
                                    "rectosigmoid junction",
                                    "rectum",
                                    "renal pelvis",
                                    "retroperitoneum and peritoneum",
                                    "skin",
                                    "small intestine",
                                    "spinal cord, cranial nerves, and other parts of central nervous system",
                                    "stomach",
                                    "testis",
                                    "thymus",
                                    "thyroid gland",
                                    "tonsil",
                                    "trachea",
                                    "unknown",
                                    "ureter",
                                    "uterus, nos",
                                    "vagina",
                                    "vulva"
                                    ]
        


    def load_fields_from_file(self, filename):
        """
        Load field names from a text file, where each line is a field name.
        """
        try:
            with open(filename, 'r') as file:
                return [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            raise Exception(f"The file {filename} was not found. Please check the file path.")        
    def validate_project_fields(self, input_fields):
        invalid_fields = [field for field in input_fields if field not in self.project_endpt_fields]
        if invalid_fields:
            raise ValueError(f"Invalid project fields: {', '.join(invalid_fields)}")

    def validate_case_fields(self, input_fields):
        invalid_fields = [field for field in input_fields if field not in self.case_endpt_fields]
        if invalid_fields:
            raise ValueError(f"Invalid case fields: {', '.join(invalid_fields)}")
    def validate_file_fields(self, input_fields):
        invalid_fields = [field for field in input_fields if field not in self.file_endpt_fields]
        if invalid_fields:
            raise ValueError(f"Invalid file fields: {', '.join(invalid_fields)}")
    def validate_annotation_fields(self, input_fields):
        invalid_fields = [field for field in input_fields if field not in self.annotation_endpt_fields]
        if invalid_fields:
            raise ValueError(f"Invalid annotation fields: {', '.join(invalid_fields)}")
