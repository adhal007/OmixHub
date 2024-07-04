
   ## Source code documentation:
   This contains all the source code for different modules that contribute to 
   1. Backend data search, query and retrieval for Genomic data commons platform. (Connectors)
   2. Preprocessing for query data as well as before applying machine learning algorithms(PreProcess) 
   3. Logging(CustomLogger)
   4. Plotting(PlotUtils)
   5. Applications of custom bioinformatics and machine learning methods for disease classification or biomarker discovery. 

      #### Code structure:

         ├── ClassicML
         ├── CustomLogger   
         ├── Connectors
         ├── PlotUtils
         ├── PreProcess

      #### Documentation on Modules and Classes:
         | Module | Class Name | Description | Readme Reference |
         |--------|------------|-------------|-----------------|
         | Connectors | Base Class | Backend data search, query and retrieval for Genomic data commons platform. | [Base Class Readme](../docs/ConnectorsAPI/readme_base_endpt.md) |
         | Connectors | Filters Class | Creates Filters for Querying GDC API | [Filters Readme](../docs/ConnectorsAPI/readme_filters.md) |
         | Connectors | Parser Class | Parses queries to GDC API endpoints for returning metadata | [Parser Readme](../docs/ConnectorsAPI/readme_parsers.md) |
         | Engines | GDC Engine Class | Highest Level API for interacting with GDC and creating ML specific data matrix | [GDC Engine Readme](../docs/ConnectorsAPI/readme_gdc_engine.md) |


#### More Modules that will be coming

1. Deep learning 
2. Bayesian Methods 
3. NLP 