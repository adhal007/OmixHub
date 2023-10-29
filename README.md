## Main 

#### Overview
Here is a list of existing and upcoming projects for different omics datasets and application of different ML/AI methods and models for biomarker discovery and disease classification.  

#### High Level Objective: 
In NGS datasets for kidney cancer or other complex diseases, apply known or new ML models to identify patterns of gene expression to serve as a template for bio-informatics learning for aspiring scientists/researchers/students in the field.

#### Immediate/Short term focus:
- Focus on kidney cancer dataset (RNA-Seq) analysis and release a beta version of notebook and modules for conducting RNA-Seq data analysis and ML models 
- Strategize an Infrastructure that can be used for multi-omics analysis based on the Genomic Data Commons workflows to enable easy application to other disease datasets.
- Functionality to explore multiple NGS assays: Identify data structures/data types commonly used in the 6 types of NGS assays

#### Codebase
```md
OmicsProjects
   ├── CustomLogger
   │   ├── init.py
   │   ├── custom_logger.py
   ├── OutlierStatMethods
   │   ├── init.py
   │   ├── base_class.py
   │   ├── <some_child_method>.py
   │   └── README.md 
   ├── OmicsUtils
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
- [Module for Outlier Statistic Methods](https://github.com/adhal007/OmicsProjects/blob/main/OmicsUtils/README.md)
- [Module for Omics data processing](https://github.com/adhal007/OmicsProjects/blob/main/OmicsUtils/README.md)  


#### Projects (Application of Methods)
Project 1: [Application of Outlier Statistic Methods for differential gene expression](https://github.com/adhal007/OmicsProjects/blob/main/OutlierMethodsApplication/README.md) 


Project 2: [ML models for classification of kidney cancer subtypes using bulk RNA-Seq Data](https://github.com/adhal007/OmicsProjects/blob/main/Transcriptomics/README.md)



