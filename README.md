## Main 

#### Overview
Here is a list of existing and upcoming projects for different omics datasets and application of different ML/AI methods and models for biomarker discovery and disease classification.  

#### High Level Objective: 
In NGS datasets for kidney cancer or other complex diseases, apply known or new ML models to identify patterns of gene expression to serve as a template for bio-informatics learning for aspiring scientists/researchers/students in the field.

#### Immediate/Short term focus:
- Focus on kidney cancer dataset (RNA-Seq) analysis and release a beta version of notebook and modules for conducting RNA-Seq data analysis and ML models 
- Strategize an Infrastructure that can be used for multi-omics analysis based on the Genomic Data Commons workflows to enable easy application to other disease datasets.
- Functionality to explore multiple NGS assays: Identify data structures/data types commonly used in the 6 types of NGS assays


#### Dataset download

- [Genome Data Commons(GDC)](https://portal.gdc.cancer.gov/projects?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22projects.summary.experimental_strategies.experimental_strategy%22%2C%22value%22%3A%5B%22RNA-Seq%22%5D%7D%7D%5D%7D)
- Add all chosen datasets to cart and then download it.
- Since this module is currently developed for bulk transcriptomics RNA-Seq data, in the GDC portal, select the "RNA-Seq" experimental strategy
- Here is an example of the same from a screen-shot 
  - ![example](Transciptomics/../Transcriptomics/images/GDC_portal_data_set_selection_RNA_SEQ.png)
- The data for RNA-Seq data for kidney cancer is used as an example here as it was provided as one of the datasets on the UCI ML repository. 

#### Codebase
```md
OmicsProjects
   ├── OmicsUtils
   │   ├── OutlierStatMethods
   │   │   ├── init.py
   │   │   ├── base_class.py
   │   │   ├── <some_child_method>.py
   │   │   └── README.md
   │   ├── DimRedMappers
   │   │   ├── init.py
   │   │   ├── <some_method_method>.py
   │   │   └── README.md
   │   ├── PlotUtils
   │   │   ├── init.py
   │   │   ├── <some_method_method>.py
   │   │   └── README.md
   │   ├── CustomLogger
   │   │   ├── init.py
   │   │   ├── custom_logger.py
   │   │   └── README.md      
   │   ├── init.py
   │   ├── preprocess_utils.py        
   │   └── README.md
   ├── Transcriptomics
   │   ├── init.py
   │   ├── analysis
   │   ├── images
   │   └── README.md
   └── README.md
   └── <jupyter_nb1.ipynb> 
   └── <jupyter_nb2.ipynb>
   ```
#### Method utilities development
- [Module for Outlier Statistic Methods](https://github.com/adhal007/OmicsProjects/tree/main/OmicsUtils/OutlierStatMethods/README.md)

- [Module for Omics data processing](https://github.com/adhal007/OmicsProjects/tree/main/OmicsUtils/README.md)

- [Module for Dimensionality Reduction Models](https://github.com/adhal007/OmicsProjects/tree/main/OmicUtils/DimRedMappers/README.md)  


#### Projects/Analysis (Application of Methods)

Analysis 1: [Application of Outlier Statistic Methods for differential gene expression](https://github.com/adhal007/OmicsProjects/tree/main/OmicsUtils/OutlierMethodsApplication/README.md) 


Analysis 2: [Application of Dimensionality reduction methods for stratification of Kidney-Cancer Subtypes](https://github.com/adhal007/OmicsProjects/blob/main/UmapApplication/README.md)        

Analysis 3: [Supervised ML models for classification of kidney cancer subtypes using bulk RNA-Seq Data](https://github.com/adhal007/OmicsProjects/blob/main/Transcriptomics/README.md)
