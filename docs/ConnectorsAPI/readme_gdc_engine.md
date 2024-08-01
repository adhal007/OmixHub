<!-- GDC Engine
-----------------

The `GDCEngine` class provides methods to get RNA-Seq Data Matrix for primary_site, gender or race based queries for ML Applications. -->

## GDC Engine

The `GDCEngine` class is responsible for fetching and processing data from the GDC (Genomic Data Commons) API. It provides methods to set parameters, fetch metadata, and create data matrices for RNA sequencing data. It provides methods to get RNA-Seq Data Matrix for primary_site, gender or race based queries for ML Applications

### Attributes

- `params` (dict): A dictionary containing the parameters for the GDCEngine.
    - `endpt` (str): The endpoint for the GDC API. Default is 'files'.
    - `homepage` (str): The homepage URL for the GDC API. Default is 'https://api.gdc.cancer.gov'.
    - `ps_list` (list): A list of program names to filter the data. Default is None.
    - `new_fields` (list): A list of additional fields to include in the metadata. Default is None.
    - `race_list` (list): A list of races to filter the data. Default is None.
    - `gender_list` (list): A list of genders to filter the data. Default is None.
    - `data_type` (str): The type of data to fetch. Default is 'RNASeq'.

### Methods

- `set_params`: Set the parameters for the GDCEngine.
- `_check_data_type`: Check if the specified data type is supported.
- `_get_raw_data`: Get the raw data from the API response.
- `_make_file_id_url_map`: Create a mapping of file IDs to download URLs.
- `_get_urls_content`: Download the content from the specified URLs.
- `_get_rna_seq_metadata`: Fetch the RNA sequencing metadata from GDC.
- `_make_RNA_seq_data_matrix`: Create a data matrix for RNA sequencing data.
- `run_rna_seq_data_matrix_creation`: Run the GDCEngine to fetch and process the data.