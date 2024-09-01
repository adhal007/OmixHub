import json
import src.Connectors.gdc_endpt_base as gdc_endpt_base
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Endpoint Fields Validator Class and high-level API functions

@author: Abhilash Dhal
@date:  2024_06_05
"""
class GDCValidator:
    """
    GDCFieldValidator class to validate fields specified by the user against the fields available in the GDC API's endpoints.

    Methods:
        validate_fields(input_fields: list, endpt_fields: dict, field_type: str) -> bool:
            Validate fields specified by the user against the fields available in the GDC API's specified endpoint.
    """

    def validate_fields(self, input_fields, endpt_fields, field_type):
        """
        Validate fields specified by the user against the fields available in the GDC API's specified endpoint.

        Args:
            input_fields (list): A list of fields to validate.
            endpt_fields (dict): A dictionary containing endpoint fields.
            field_type (str): The type of fields to validate (e.g., 'files', 'annotation').

        Returns:
            bool: True if all fields are valid, False otherwise.

        Raises:
            ValueError: If any of the input fields are invalid.
        """
        if field_type not in endpt_fields:
            raise ValueError(f"Invalid field type: {field_type}")

        invalid_fields = [field for field in input_fields if field not in endpt_fields[field_type]]
        if invalid_fields:
            raise ValueError(f"Invalid {field_type} fields: {', '.join(invalid_fields)}")
        return True

    def validate_file_fields(self, input_fields, endpt_fields):
        """
        Validate file fields specified by the user against the fields available in the GDC API's files endpoint.

        Args:
            input_fields (list): A list of fields to validate.
            endpt_fields (dict): A dictionary containing endpoint fields.

        Returns:
            bool: True if all fields are valid, False otherwise.
        """
        return self.validate_fields(input_fields, endpt_fields, 'files')

    def validate_annotation_fields(self, input_fields, endpt_fields):
        """
        Validate annotation fields specified by the user against the fields available in the GDC API's annotation endpoint.

        Args:
            input_fields (list): A list of fields to validate.
            endpt_fields (dict): A dictionary containing endpoint fields.

        Returns:
            bool: True if all fields are valid, False otherwise.
        """
        return self.validate_fields(input_fields, endpt_fields, 'annotation')