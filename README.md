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
  - ![example](./Transcriptomics/images/GDC_portal_data_set_selection_RNA_SEQ.png)
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
   ├── Applications  
   │   ├── OutlierStatMethodsApplication
   │   │   ├── analysis
   │   │   ├── images
   │   │   └── Summary.md
   │   │   └── Workflow.md
   │   ├── SupervisedLearningApplication
   │   │   ├── analysis
   │   │   ├── images
   │   │   └── Summary.md
   │   │   └── Workflow.md
   │   ├── UmapApplication
   │   │   ├── analysis
   │   │   ├── images
   │   │   └── Summary.md
   │   │   └── Workflow.md   
   └── README.md
   └── <jupyter_nb1.ipynb> 
   └── <jupyter_nb2.ipynb>
   ```
#### Modules developed:

- [Module for Outlier Statistic Methods](https://github.com/adhal007/OmixHub/blob/main/OmicsUtils/OutlierStatMethods/README.md)

- [Module for Omics data processing](https://github.com/adhal007/OmixHub/blob/main/OmicsUtils/README.md)

- [Module for Dimensionality Reduction Models](https://github.com/adhal007/OmixHub/blob/main/OmicsUtils/DimRedMappers/README.md)  


#### Projects/Analysis (Application of Methods)

Analysis 1: Supervised ML models for classification of kidney cancer subtypes using bulk RNA-Seq Data 
- [Summary](https://github.com/adhal007/OmixHub/blob/main/ProjectDocs/SupervisedLearningApplication/docs/summary.md)
- [Notebook workflow](https://github.com/adhal007/OmixHub/blob/main/ProjectDocs/SupervisedLearningApplication/docs/Workflow.md)

Analysis 2: Application of Outlier Statistic Methods for differential gene expression
- [Summary](https://github.com/adhal007/OmixHub/blob/main/ProjectDocs/OutlierMethodsApplication/docs/summary.md) 
- [Notebook workflow](https://github.com/adhal007/OmixHub/blob/main/ProjectDocs/OutlierMethodsApplication/docs/workflow.md)

Analysis 3: Bayesian optimized stratification of Kidney-Cancer Subtypes by dimensionality reduction and clustering
- [Summary](https://github.com/adhal007/OmixHub/blob/main/ProjectDocs/UmapApplication/docs/summary.md) 
- [Notebook workflow](https://github.com/adhal007/OmixHub/blob/main/ProjectDocs/UmapApplication/docs/workflow.md)
     

## Future work

- Develop a module for multi-omics data analysis
- Develop a module for single-cell RNA-Seq data analysis
- Develop a module for proteomics data analysis
- Develop a module for metabolomics data analysis
- Develop a module for epigenomics data analysis
- Develop a module for microbiomics data analysis
- Develop a module for clinical data analysis
- Develop a module for single-cell RNA-Seq data analysis
- Develop a module for WGAS data analysis
- Develop a module for WES data analysis
- Develop a module for WGS data analysis
- Develop a module for CHIP-Seq data analysis
- Develop a module for ATAC-Seq data analysis
- Develop a module for Hi-C data analysis
- Develop a module for Hi-Seq data analysis
