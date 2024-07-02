import json
import src.Connectors.gdc_endpt_base as gdc_endpt_base
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Endpoint Fields Validator Class and high-level API functions

@author: Abhilash Dhal
@date:  2024_06_05
"""
### Next steps: Would be Nice to Have a Base Validator Class and then subsequent child validators for each endpoint 
### gdc_field_validtor, gdc_file_validator, gdc_annotation_validator, gdc_project_validator 
class GDCValidator:
    def __init__(self):
        self._endpt_fields = gdc_endpt_base.GDCEndptBase().endpt_fields

    def validate_project_fields(self, input_fields):
        """
        Function to validate project fields specified by the user against the fields available in the GDC API's projects endpoint

        Args:
        input_fields (list): A list of fields to validate

        Returns:
        bool: True if all fields are valid, False otherwise
        """
        invalid_fields = [field for field in input_fields if field not in self._endpt_fields['projects']]
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
        invalid_fields = [field for field in input_fields if field not in self._endpt_fields['cases']]
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
        invalid_fields = [field for field in input_fields if field not in self._endpt_fields['files']]
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
        invalid_fields = [field for field in input_fields if field not in self._endpt_fields['annotation']]
        if invalid_fields:
            raise ValueError(f"Invalid annotation fields: {', '.join(invalid_fields)}")
        return True
