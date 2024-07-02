## Main 

### Overview
We want to build an integrated platform for well known ML and DL based classification, feature selection and other bioinformatics for high dimensional NGS data. Here is a list of existing and upcoming projects for different omics datasets and modules of different ML/AI methods and models for biomarker discovery and disease classification.

#### High Level Objective:
In NGS datasets for kidney cancer or other complex diseases, apply known or new ML models to identify patterns of gene expression to serve as a template for bio-informatics learning for aspiring scientists/researchers/students in the field.

### Immediate/Short term focus for prototype:
**Dataset:**

Focus on kidney cancer dataset (RNA-Seq) analysis  as our input data (tabular form) 

N x P matrix where N - number of unique patients, P - number of features (genes from RNA-Seq expression)

This contain gene expression quantification data (definition for the gene expression, RNA-Seq data, etc) can be found here RNA-Seq - GDC Docs 

**Methods/Applications for UI:** Here is a list of what applications we expect to have for our initial release:

- **Cohort Selector:** Click button availability to select a cohort from GDC using queries for the following parameters:
  - primary_site: 60+ available sites eg: Lung, Kidney, Throat, etc
  - gender: Male, Female or Both
  - experimental_strategy: Only RNA-Seq for now. In the future we would add in the order of prioritization  
    - SNP (Genotyping data)
    - miRNA-seq 
    - WGS data
  - demography: White, latino, etc 

- **Modules for analysis:**
  - Classical ML: 
    - Supervised ML for disease vs control classification 
      - Multi-label multi-output classification 
      - Single-Label classification 
    - Biomarker identification using feature selection
      - Outlier Sum Statistics 
      - Differential Gene Expression using pydeseq wrapper 
      - Feature importances from different classifiers 
    - Unsupervised ML:
      - Bayesian optimized (Hyper-Opt) clustering (KNN) using UMap embeddings
    - Semi-supervised ML:
      - Iterative GMM + PCA for cohort stratification for niche disease-control applications 
  <!-- - Deep-learning:
    - Auto-encoders 
    - CNN’s
    - RNN’s for multi-timepoint data 
  - Bayesian ML
  - Knowledge Graphs and NLP  -->

#### Codebase
```md
OmicsProjects
   ├── src
   │   ├── ClassicML
   │   │   ├── init.py
   │   │   ├── DimRedMappers
   │   │   ├── DGE
   │   │   └── OutlierStatMethods
   │   │   ├── Supervised
   │   ├── CustomLogger   
   │   ├── Connectors
   │   ├── PlotUtils
   │   ├── PreProcess 
   └── README.md
   └── <jupyter_nb1.ipynb> 
   └── <jupyter_nb2.ipynb>
   ```
<!-- #### Modules developed:

- [Module for Genomic Data Commons API accession, querying, search and retrieval](https://github.com/adhal007/OmixHub/tree/main/src/Connectors)
- [Module for Outlier Statistic Methods](https://github.com/adhal007/OmixHub/blob/main/src/OutlierStatMethods/README.md)

- [Module for Omics data processing](https://github.com/adhal007/OmixHub/blob/main/src/README.md)
  - [Module for RNA-seq preprocessing](https://github.com/adhal007/OmixHub/blob/main/src/preprocess_utils.py)
  - [Module for base preprocessor class](https://github.com/adhal007/OmixHub/blob/main/src/base_preprocessor.py)
- [Module for Dimensionality Reduction Models](https://github.com/adhal007/OmixHub/blob/main/src/DimRedMappers/README.md)  
- [Module for ML classifier models (ensemble, multi_output)](https://github.com/adhal007/OmixHub/blob/main/src/base_ml_models.py)
- [Module for differential expression of RNA-Seq](https://github.com/adhal007/OmixHub/blob/main/src/pydeseq_utils.py) -->

  
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

- UI Prototype Release
- Semi-supervised ML:
    - Iterative GMM + PCA for cohort stratification for niche disease-control applications 
- Deep-learning:
  - Auto-encoders 
  - CNN’s
  - RNN’s for multi-timepoint data 
<!-- - Develop a module for graph based machine learning models 
- Develop a module for shotgun sequencing dataset
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
- Develop a module for Hi-Seq data analysis -->

**Application work**:
- Autoimmune disease characterization by omics methods 
- Other cancers 
- Deep learning

### ADDITIONAL CODE DOCS:
- [Methods in src.Connectors for GDC API search and retrieval using custom queries](./src/README.md)