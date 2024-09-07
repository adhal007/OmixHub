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
################# CREATED BY CURSOR BOT ##############################################
    def create_sequencing_filter(self, data_type, experimental_strategy=None, workflow_type=None, data_format=None):
        """
        Create a filter for specific sequencing data types.

        Args:
            data_type (str): The type of data (e.g., "Gene Expression Quantification", "Copy Number Segment").
            experimental_strategy (str or list, optional): The experimental strategy (e.g., "RNA-Seq", "WXS").
            workflow_type (str or list, optional): The workflow type (e.g., "HTSeq - Counts", "VarScan2 Variant Aggregation and Masking").
            data_format (str or list, optional): The data format (e.g., "BAM", "VCF").

        Returns:
            dict: A filter dictionary for the specified sequencing data.
        """
        filters = [self.create_basic_filter("data_type", data_type)]
        
        if experimental_strategy:
            filters.append(self.create_basic_filter("files.experimental_strategy", experimental_strategy))
        
        if workflow_type:
            filters.append(self.create_basic_filter("files.analysis.workflow_type", workflow_type))
        
        if data_format:
            filters.append(self.create_basic_filter("files.data_format", data_format))
        
        return self.combine_filters(filters)

    def rna_seq_filter(self, workflow_type=None):
        """
        Create a filter for RNA-Seq data.

        Args:
            workflow_type (str or list, optional): The workflow type (e.g., "HTSeq - Counts", "STAR - Counts").

        Returns:
            dict: A filter dictionary for RNA-Seq data.
        """
        return self.create_sequencing_filter(
            data_type="Gene Expression Quantification",
            experimental_strategy="RNA-Seq",
            workflow_type=workflow_type or ["HTSeq - Counts", "STAR - Counts"]
        )

    def wgs_filter(self, workflow_type=None):
        """
        Create a filter for Whole Genome Sequencing (WGS) data.

        Args:
            workflow_type (str or list, optional): The workflow type.

        Returns:
            dict: A filter dictionary for WGS data.
        """
        return self.create_sequencing_filter(
            data_type="Aligned Reads",
            experimental_strategy="WGS",
            workflow_type=workflow_type
        )

    def wxs_filter(self, workflow_type=None):
        """
        Create a filter for Whole Exome Sequencing (WXS) data.

        Args:
            workflow_type (str or list, optional): The workflow type.

        Returns:
            dict: A filter dictionary for WXS data.
        """
        return self.create_sequencing_filter(
            data_type="Aligned Reads",
            experimental_strategy="WXS",
            workflow_type=workflow_type
        )

    def mirna_seq_filter(self, workflow_type=None):
        """
        Create a filter for miRNA-Seq data.

        Args:
            workflow_type (str or list, optional): The workflow type.

        Returns:
            dict: A filter dictionary for miRNA-Seq data.
        """
        return self.create_sequencing_filter(
            data_type="miRNA Expression Quantification",
            experimental_strategy="miRNA-Seq",
            workflow_type=workflow_type
        )

    def methylation_filter(self, platform=None):
        """
        Create a filter for DNA methylation data.

        Args:
            platform (str or list, optional): The platform used (e.g., "Illumina Human Methylation 450").

        Returns:
            dict: A filter dictionary for methylation data.
        """
        filters = [
            self.create_basic_filter("data_type", "Methylation Beta Value"),
            self.create_basic_filter("files.experimental_strategy", "Methylation Array")
        ]
        if platform:
            filters.append(self.create_basic_filter("files.platform", platform))
        return self.combine_filters(filters)

    def copy_number_variation_filter(self, workflow_type=None):
        """
        Create a filter for Copy Number Variation (CNV) data.

        Args:
            workflow_type (str or list, optional): The workflow type.

        Returns:
            dict: A filter dictionary for CNV data.
        """
        return self.create_sequencing_filter(
            data_type="Copy Number Segment",
            workflow_type=workflow_type
        )

    def somatic_mutation_filter(self, workflow_type=None):
        """
        Create a filter for somatic mutation data.

        Args:
            workflow_type (str or list, optional): The workflow type (e.g., "MuTect2 Variant Aggregation and Masking").

        Returns:
            dict: A filter dictionary for somatic mutation data.
        """
        return self.create_sequencing_filter(
            data_type="Simple Somatic Mutation",
            workflow_type=workflow_type or "MuTect2 Variant Aggregation and Masking"
        )
    
    def create_basic_filter(self, field, value, operation="in"):
        """
        Create a basic filter for any field.

        Args:
            field (str): The field to filter on.
            value (str or list): The value(s) to filter by.
            operation (str): The operation to use (e.g., "in", "=", "!=", ">", "<", ">=", "<=").

        Returns:
            dict: A basic filter dictionary.
        """
        return {
            "op": operation,
            "content": {
                "field": field,
                "value": [value] if isinstance(value, str) else value
            }
        }

    def create_range_filter(self, field, start, end):
        """
        Create a range filter for numeric fields.

        Args:
            field (str): The field to filter on.
            start (int or float): The start of the range.
            end (int or float): The end of the range.

        Returns:
            dict: A range filter dictionary.
        """
        return {
            "op": "and",
            "content": [
                {
                    "op": ">=",
                    "content": {
                        "field": field,
                        "value": start
                    }
                },
                {
                    "op": "<=",
                    "content": {
                        "field": field,
                        "value": end
                    }
                }
            ]
        }

    def create_date_range_filter(self, field, start_date, end_date):
        """
        Create a date range filter.

        Args:
            field (str): The date field to filter on.
            start_date (str): The start date in YYYY-MM-DD format.
            end_date (str): The end date in YYYY-MM-DD format.

        Returns:
            dict: A date range filter dictionary.
        """
        return self.create_range_filter(field, start_date, end_date)

    def create_exists_filter(self, field):
        """
        Create a filter to check if a field exists.

        Args:
            field (str): The field to check for existence.

        Returns:
            dict: An exists filter dictionary.
        """
        return {
            "op": "is not",
            "content": {
                "field": field,
                "value": "null"
            }
        }

    def create_missing_filter(self, field):
        """
        Create a filter to check if a field is missing.

        Args:
            field (str): The field to check for missing values.

        Returns:
            dict: A missing filter dictionary.
        """
        return {
            "op": "is",
            "content": {
                "field": field,
                "value": "null"
            }
        }

    def create_regex_filter(self, field, pattern):
        """
        Create a regex filter for string fields.

        Args:
            field (str): The field to apply the regex to.
            pattern (str): The regex pattern.

        Returns:
            dict: A regex filter dictionary.
        """
        return {
            "op": "regex",
            "content": {
                "field": field,
                "value": pattern
            }
        }

    def combine_filters(self, filters, operation="and"):
        """
        Combine multiple filters with a specified operation.

        Args:
            filters (list): A list of filter dictionaries.
            operation (str): The operation to use for combining filters ("and" or "or").

        Returns:
            dict: A combined filter dictionary.
        """
        return {
            "op": operation,
            "content": filters
        }
####### CREATED BY ME ##############################################     
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

    def top_mutated_genes_by_project_filter(self, project_name):
        """
        Create a filter for top mutated genes by project.

        Args:
            project_name (str): The project name to filter by (e.g., "TCGA-BRCA").
            top_n (int): The number of top mutated genes to retrieve.

        Returns:
            dict: A filter dictionary for top mutated genes by project.
        """
        return {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [project_name]
                    }
                },
                {
                    "op": "in",
                    "content": {
                        "field": "ssms.consequence.transcript.annotation.vep_impact",
                        "value": ["HIGH", "MODERATE"]
                    }
                }
            ]
        }  
    
                        
    def all_diseases(self):
        """
        Placeholder method for retrieving all diseases.

        Raises:
            NotImplementedError: This method is not yet implemented.
        """
        raise NotImplementedError()



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