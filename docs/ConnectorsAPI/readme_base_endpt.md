## Documentation for the Connectors API for GDC: Base Endpoint Class

- **endpt_base class:** 
  - This class is the base class for all the connectors. 
  - It contains the basic methods to interact with the API.
  
  - **Functions include:**
    - `make_url`
    - `_get_endpt_url`:  
    - `_get_enpt_fields`: gets the list of available fields for 3 endpoints (files, cases and projects) of GDC API
    - `get_response`: gets the response from the API using url and parameters provided
    - `get_json_data`: Extension of get_response that returns the json data
    - `download_file_by_id`: downloads a file from the API using the file_id
    - `query`: queries the API using the fields and filters provided
  
  - **Properties include:**
    - `<endpt>_url`: 
      - gives you the url for each of the 3 endpoints (files, cases and projects) of the GDC API
      - There are 3 such properties for each of the endpoint.
    - `endpt_fields`: gives a dictionary of fields available to be returned from a GDC API query for each of the 3 endpoints (files, cases and projects). 
  
  - **Purpose:**
    - Serves as the parent class for more specific endpoint classes.
    - Provides common API calls and properties for all the endpoint classes.
    - Provides a base structure for the endpoint classes to build on.
