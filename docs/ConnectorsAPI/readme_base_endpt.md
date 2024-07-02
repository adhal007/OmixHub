## Documentation for the Connectors API for GDC: Base Endpoint Class

- **endpt_base class:** 
  - This class is the base class for all the connectors. 
  - It contains the basic methods to interact with the API.
  - **Functions include:**
    - make_url
    - get_url 
    - get_enpt_fields

    - get_response: gets the response from the API using url and parameters provided
    - get_json_data: Extension of get_response that returns the json data
    - make_params_dict: creates a dictionary of parameters for the API call using the fields and filters provided
    - download_file_by_id: downloads a file from the API using the file_id
    - query: queries the API using the fields and filters provided
  
  - **Properties include:**
    - {endpt}_url: gives you the url for each of the 3 endpoints
    - endpt_fields: gives a dictionary of fields available to be returned from a GDC API query for each of the 3 endpoints. 
  - **Purpose:**
    - Serves as the parent class for more specific endpoint classes.
    - Provides common API calls and properties for all the endpoint classes.
    - Provides a base structure for the endpoint classes to build on.
