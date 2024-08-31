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
    def __init__(self):
        pass

    def create_and_filters(self, filter_specs, op_specs):
        """
        Creates a list of filters based on given specifications.

        Args:
        filter_specs (list of tuples): Each tuple contains the field name and the corresponding values list.

        Returns:
        list of dicts: List containing filter dictionaries.
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
        Returns a filter dictionary for retrieving all projects based on the given experimental strategy.

        Parameters:
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
        Function to filer for 1) Primary Site 2) Race 3) Demography

        Args:
        params (dict, optional): A dictionary containing the filter specifications. Defaults to None.
            accepted keys:
            - cases.project.primary_site (list): A list of primary sites to filter the data.
            - cases.demographic.race
            - cases.demographic.gender
            - data_type (str): The type of data to fetch. Default is 'RNA-Seq'.
            - files.experimental_strategy (list): A list of experimental strategies to filter the data.
            - analysis.workflow_type (list): A list of workflow types to filter the data.

        Returns:
        json_query: nested json object to be fed into query for GDC.
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
        raise NotImplementedError()


# class GDCFacetFilters:
#     def __init__(self):
#         # Mapping of method names to facet keys for different endpoints
#         # This contains facets from file endpt that are used in the API
#         self._imp_facet_keys = {
#             "list_of_primary_sites_flt": "project.primary_site",
#             "list_of_exp_flt": "experimental_strategy",
#             "list_of_projects_flt": "project.program.name",
#         }

#     def create_single_facet_filter(self, facet_key: str, sort_order: str = "asc"):
#         """
#         Generic function to create a single facet filter for a given facet key for any GDC endpt.

#         ARGS:
#         facet_key (str): The facet key to filter on.
#         sort_order (str): The sort order for the facet values. Default is 'asc'.

#         Returns:
#         dict: A dictionary containing the facet filter.
#         """
#         return {
#             "facets": facet_key,
#             "from": 0,
#             "size": 0,
#             "sort": f"{facet_key}:{sort_order}",
#         }

#     def get_files_endpt_facet_filter(self, method_name: str):
#         """
#         Function to get facet filter for the files endpoint based on the method name.

#         Args:
#         method_name (str): The method name to get the facet filter for.

#         Returns:
#         dict: The facet filter for the given method name.
#         """
#         facet_key_value = self._imp_facet_keys.get(method_name)
#         if facet_key_value is None:
#             raise ValueError(f"No facet key found for facet_key '{method_name}'")
#         # Inlined `create_single_facet_filter` functionality here
#         return {
#             "facets": facet_key_value,
#             "from": 0,
#             "size": 0,
#             "sort": f"{facet_key_value}:asc",
#         }

#     def create_single_facet_df(self, url: str, facet_key_value: str, params: dict):
#         response = gdc_endpt_base.GDCEndptBase.get_response(url, params=params)
#         data = response.json()
#         facet_df = pd.DataFrame(
#             data["data"]["aggregations"][facet_key_value]["buckets"]
#         )
#         return facet_df

#     def get_files_facet_data(self, url, facet_key, method_name):
#         facet_key_value = self._imp_facet_keys.get(facet_key, None)
#         print(facet_key_value)
#         if facet_key_value is None:
#             raise ValueError(f"Invalid facet_key: {facet_key}")

#         if getattr(self, facet_key, None) is None:
#             params = self.get_files_endpt_facet_filter(method_name=method_name)
#             print(params)
#             data = self.create_single_facet_df(
#                 url=url, facet_key_value=facet_key_value, params=params
#             )
#             data.columns = ["count", f"{facet_key_value}"]
#         return data

class GDCFacetFilters:
    """
    GDCFacetFilters class for creating and managing facet filters for GDC queries.
    """

    def __init__(self):
        """
        Initialize the GDCFacetFilters class.
        """
        # Mapping of method names to facet keys for different endpoints
        # This contains facets from file endpoint that are used in the API
        self._imp_facet_keys = {
            "list_of_primary_sites_flt": "project.primary_site",
            "list_of_exp_flt": "experimental_strategy",
            "list_of_projects_flt": "project.program.name",
        }

    def create_single_facet_filter(self, facet_key: str, sort_order: str = "asc"):
        """
        Create a single facet filter for a given facet key for any GDC endpoint.

        Args:
            facet_key (str): The facet key to filter on.
            sort_order (str): The sort order for the facet values. Default is 'asc'.

        Returns:
            dict: A dictionary containing the facet filter.
        """
        return {
            "facets": facet_key,
            "from": 0,
            "size": 0,
            "sort": f"{facet_key}:{sort_order}",
        }

    def get_files_endpt_facet_filter(self, method_name: str):
        """
        Get facet filter for the files endpoint based on the method name.

        Args:
            method_name (str): The method name to get the facet filter for.

        Returns:
            dict: The facet filter for the given method name.

        Raises:
            ValueError: If no facet key is found for the given method name.
        """
        facet_key_value = self._imp_facet_keys.get(method_name)
        if facet_key_value is None:
            raise ValueError(f"No facet key found for facet_key '{method_name}'")
        return {
            "facets": facet_key_value,
            "from": 0,
            "size": 0,
            "sort": f"{facet_key_value}:asc",
        }

    def create_single_facet_df(self, url: str, facet_key_value: str, params: dict):
        """
        Create a DataFrame from a single facet filter.

        Args:
            url (str): The URL to send the request to.
            facet_key_value (str): The facet key value to filter on.
            params (dict): The parameters for the request.

        Returns:
            pd.DataFrame: The resulting DataFrame.
        """
        response = gdc_endpt_base.GDCEndptBase.get_response(url, params=params)
        data = response.json()
        facet_df = pd.DataFrame(
            data["data"]["aggregations"][facet_key_value]["buckets"]
        )
        return facet_df

    def get_files_facet_data(self, url, facet_key, method_name):
        """
        Get facet data for the files endpoint.

        Args:
            url (str): The URL to send the request to.
            facet_key (str): The facet key to filter on.
            method_name (str): The method name to get the facet filter for.

        Returns:
            pd.DataFrame: The resulting DataFrame.

        Raises:
            ValueError: If the facet key is invalid.
        """
        facet_key_value = self._imp_facet_keys.get(facet_key, None)
        print(facet_key_value)
        if facet_key_value is None:
            raise ValueError(f"Invalid facet_key: {facet_key}")

        if getattr(self, facet_key, None) is None:
            params = self.get_files_endpt_facet_filter(method_name=method_name)
            print(params)
            data = self.create_single_facet_df(
                url=url, facet_key_value=facet_key_value, params=params
            )
            data.columns = ["count", f"{facet_key_value}"]
        return data
### Redundant functions that might be useful in hindsight
# def generate_filters(self, filter_list, operation='and'):
#     """
#     Generic function to generate filters based on given specifications.

#     Args:
#     filter_list (list of dicts): Each dictionary contains the field name and the corresponding values list.
#     operation (str): The operation to perform on the filters. Default is 'and'.

#     Returns:
#     dict: A filter dictionary that can be used to query data based on the given filter specifications.
#     """
#     # Initialize the main filters dictionary
#     filters = {
#         "op": operation,
#         "content": []
#     }

#     # Loop through each filter specification and append it to the filters["content"]
#     for filter_spec in filter_list:
#         filter_op = {
#             "op": "in",
#             "content": {
#                 "field": filter_spec["field"],
#                 "value": filter_spec["value"]
#             }
#         }
#         filters["content"].append(filter_op)
#     return filters

# def projects_by_primary_diagnosis_filter(self, disease_type):
#     """
#     Returns a filter dictionary for retrieving projects based on the given disease type.

#     Parameters:
#     disease_type (str): The disease type to filter projects by.

#     Returns:
#     dict: A filter dictionary that can be used to query projects based on the disease type.
#     """
#     filters = {
#         "op": "in",
#         "content":
#         {
#             "field": "cases.diagnoses.primary_diagnosis",
#             "value": disease_type
#         }
#     }
#     return filters
# def rna_seq_disease_filter(self, disease_list=None, op_specs=None):
#     """
#     Filter for RNA-Seq data based on disease list.

#     This function generates a filter specification for querying RNA-Seq data based on the provided disease list.
#     The default filter specifications include files with experimental strategy "RNA-Seq", data type "Gene Expression Quantification",
#     and analysis workflow type "STAR - Counts". If a disease list is provided, an additional filter specification is added
#     to filter for cases with primary diagnosis matching the diseases in the list.

#     Args:
#         disease_list (list, optional): A list of diseases to filter for. Defaults to None.

#     Returns:
#         dict: The filter specification for querying RNA-Seq data.
#     """
#     default_filter_specs = [
#         ("files.experimental_strategy", ["RNA-Seq"]),
#         ("data_type", ["Gene Expression Quantification"]),
#         ("analysis.workflow_type", ["STAR - Counts"])
#     ]

#     combined_filter_specs = default_filter_specs.copy()
#     if disease_list:
#         combined_filter_specs.append(("cases.diagnoses.primary_diagnosis", disease_list))

#     filter_for_query = self.create_and_filters(combined_filter_specs, op_specs=op_specs)
#     return filter_for_query

# def update_rna_seq_filter_by_dtype(self, field_params: dict=None, op_params: dict = None):
#     """
#     Function to update the filter for RNA-Seq data based on the data type.

#     This function updates the filter specification for querying RNA-Seq data based on the provided data type.

#     Args:
#         params (dict, optional): A dictionary containing the parameters for the API request. Defaults to None.
#             accepted keys:
#             - data_type (str): The type of data to fetch. Default is 'Gene Expression Quantification'.
#     Returns:
#         dict: The updated filter specification for querying RNA-Seq data.
#     """
#     filters, op_specs = self.rna_seq_template_filter(field_params, op_params)
#     # if field_params is None:
#     #     data_type = 'Gene Expression Quantification'
#     # else:
#     #     data_type = field_params['data_type']

#     # filters_upd = filters | {"data_type": [data_type]}
#     filter_for_query = self.create_and_filters(filters, op_specs)
#     return filter_for_query
