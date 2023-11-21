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
  - ![example](/Transcriptomics/images/GDC_portal_data_set_selection_RNA_SEQ.png)
- The data for RNA-Seq data for kidney cancer is used as an example here as it was provided as one of the datasets on the UCI ML repository. 


#### Codebase
```md
OmicsProjects
   ├── src
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
   │   ├── base_preprocessor.py
   │   ├── pydeseq_utils.py
   │   ├── base_ml_models.py 
   │   └── README.md
   ├── Transcriptomics
   │   ├── images
   │   └── README.md
   ├── docs 
   │   ├── OutlierStatMethodsApplication
   │   │   ├── docs
   │   │   ├── images
   │   │   └── Summary.md
   │   │   └── Workflow.md
   │   ├── SupervisedLearningApplication
   │   │   ├── docs
   │   │   ├── images
   │   │   └── Summary.md
   │   │   └── Workflow.md
   │   ├── UmapApplication
   │   │   ├── docs
   │   │   ├── images
   │   │   └── Summary.md
   │   │   └── Workflow.md   
   └── README.md
   └── <jupyter_nb1.ipynb> 
   └── <jupyter_nb2.ipynb>
   ```
#### Modules developed:

- [Module for Outlier Statistic Methods](https://github.com/adhal007/OmixHub/blob/main/src/OutlierStatMethods/README.md)

- [Module for Omics data processing](https://github.com/adhal007/OmixHub/blob/main/src/README.md)
  - [Module for RNA-seq preprocessing](https://github.com/adhal007/OmixHub/blob/main/src/preprocess_utils.py)
  - [Module for base preprocessor class](https://github.com/adhal007/OmixHub/blob/main/src/quality_checker.py)
- [Module for Dimensionality Reduction Models](https://github.com/adhal007/OmixHub/blob/main/src/DimRedMappers/README.md)  
- [Module for ML classifier models (ensemble, multi_output)](https://github.com/adhal007/OmixHub/blob/main/src/base_ml_models.py)
- [Module for differential expression of RNA-Seq](https://github.com/adhal007/OmixHub/blob/main/src/pydeseq_utils.py)

  
#### Projects/Analysis (Application of Methods)
###### A. Characterization of kidney cancer using RNA-Seq transcriptome profiline
- Analysis 1: Supervised ML models for classification of kidney cancer subtypes using bulk RNA-Seq Data 
  - [Evaluation of multi-output classifier models(Jupyter NB)](/docs/SupervisedLearningApplication/docs/workflow.md)
  
  - [Evaluation of ensemble models for multi-label classification(Jupyter NB)](/docs/SupervisedLearningEnsembleApplication/docs/workflow.md)

- Analysis 2: Differential expression of kidney cancer subtypes by single factor and multi-factor methods
  <!-- - [Summary]() -->
  - [Notebook workflow](/docs/DeSeqApplication/docs/workflow.md)

- Analysis 3: Bayesian optimized stratification of Kidney-Cancer Subtypes by dimensionality reduction and clustering
    - [Notebook workflow](/docsUmapApplication/docs/workflow.md)

- Analysis 4: Application of Outlier Statistic Methods for differential gene expression
  (To be updated with workflow notebooks)
 

## Future work
**Development**:
- Develop a module for graph based machine learning models 
- Develop a module for deep learning models 
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

**Application work**:
- Autoimmune disease characterization by omics methods 
- Other cancers 
- Deep learning