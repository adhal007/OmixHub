GDC Query Filters
-----------------

The `GDCQueryFilters` class provides methods to generate filter dictionaries for querying data based on various specifications.

Methods:
- `generate_filters(filter_list, operation='and')`: Generates a filter dictionary based on given specifications.
- `create_and_filters(filter_specs)`: Creates a list of filters based on given specifications.
- `all_projects_by_exp_filter(experimental_strategy)`: Returns a filter dictionary for retrieving all projects based on the given experimental strategy.
- `projects_by_disease_filter(disease_type)`: Returns a filter dictionary for retrieving projects based on the given disease type.
- `rna_seq_star_count_filter(ps_list=None, race_list=None, gender_list=None)`: Generates a filter specification for querying RNA-Seq data based on primary site, race, and gender.
- `rna_seq_disease_filter(disease_list=None)`: Generates a filter specification for querying RNA-Seq data based on disease list.
- `all_diseases()`: Returns a filter dictionary for retrieving all diseases.

-----------------

GDC Facet Filters
-----------------

The `GDCFacetFilters` class assists in creating facet filters for different endpoints in the GDC API.

Methods:
- `create_single_facet_filter(facet_key, sort_order='asc')`: Creates a single facet filter for a given facet key.
- `get_files_endpt_facet_filter(method_name)`: Gets the facet filter for the files endpoint based on the method name.
- `create_single_facet_df(url, facet_key_value, params)`: Creates a single facet dataframe.
- `get_files_facet_data(url, facet_key, method_name)`: Gets the facet data for the files endpoint.