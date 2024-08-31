import json
import requests
import pandas as pd
import src.Connectors.gdc_endpt_base as gdc_endpt_base

"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC filter class and high-level API functions

@author: Abhilash Dhal
@date:  2024_06_07
"""


class GDCQueryFilters:
    """
    GDCQueryFilters class for creating and managing filters for GDC queries.
    """
    def __init__(self):
        """
        Initialize the GDCQueryFilters class.
        """
        pass

    def create_and_filters(self, filter_specs, op_specs):
        """
        Create a list of filters based on given specifications.

        Args:
            filter_specs (dict): Each key-value pair contains the field name and the corresponding values list.
            op_specs (dict): Each key-value pair contains the field name and the corresponding operation.

        Returns:
            dict: A dictionary containing the combined filters for the query.
        """
        filters = []
        for field, values in filter_specs.items():
            op = op_specs[field]
            filter_op = {"op": op, "content": {"field": field, "value": values}}
            filters.append(filter_op)

        filters_for_query = {"op": "and", "content": filters}
        return filters_for_query

    def all_projects_by_exp_filter(self, experimental_strategy):
        """
        Return a filter dictionary for retrieving all projects based on the given experimental strategy.

        Args:
            experimental_strategy (str): The experimental strategy to filter projects by.

        Returns:
            dict: A filter dictionary that can be used to query projects based on the experimental strategy.
        """
        filters = {
            "op": "in",
            "content": {
                "field": "files.experimental_strategy",
                "value": experimental_strategy,
            },
        }
        return filters

    def rna_seq_data_filter(self, field_params=None, op_params=None):
        """
        Filter for RNA sequencing data based on various parameters.

        Args:
            field_params (dict, optional): A dictionary containing the filter specifications. Defaults to None.
                Accepted keys:
                - cases.project.primary_site (list): A list of primary sites to filter the data.
                - cases.demographic.race (list): A list of races to filter the data.
                - cases.demographic.gender (list): A list of genders to filter the data.
                - data_type (str): The type of data to fetch. Default is 'RNA-Seq'.
                - files.experimental_strategy (list): A list of experimental strategies to filter the data.
                - analysis.workflow_type (list): A list of workflow types to filter the data.
            op_params (dict, optional): A dictionary containing the operation specifications. Defaults to None.

        Returns:
            dict: A nested JSON object to be used as a query for GDC.
        """
        default_filter_specs = {
            "files.experimental_strategy": ["RNA-Seq"],
            "data_type": ["Gene Expression Quantification"],
            "analysis.workflow_type": ["STAR - Counts"],
            "cases.demographic.race": ["*"],
            "cases.demographic.gender": ["*"],
        }

        default_op_specs = {key: "in" for key in default_filter_specs.keys()}
        if field_params is not None:
            combined_filter_specs = default_filter_specs | field_params
        else:
            combined_filter_specs = default_filter_specs

        if op_params is not None:
            if sorted(list(op_params.keys())) != sorted(
                list(combined_filter_specs.keys())
            ):
                op_specs = default_op_specs
                raise Warning(
                    "The query operations are not defined correctly. Using 'in' operation for all query params"
                )
            else:
                op_specs = default_op_specs | op_params
        else:
            op_specs = default_op_specs
        filter_for_query = self.create_and_filters(combined_filter_specs, op_specs)
        return filter_for_query

    def all_diseases(self):
        """
        Placeholder method for retrieving all diseases.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError()


