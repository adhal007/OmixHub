
### Overview
<!-- We want to build an integrated platform for well known ML and DL based classification, feature selection and other bioinformatics for high dimensional NGS data. Here is a list of existing and upcoming projects for different omics datasets and modules of different ML/AI methods and models for biomarker discovery and disease classification. -->

OmixHub is a platform that interfaces with GDC using python to help users to apply ML based analysis on different sequencing data. Currently we **support only for RNA-Seq based datasets** from genomic data commons (GDC)

1. **Cohort Creation** of Bulk RNA Seq Tumor and Normal Samples from GDC. 
2. **Bioinformatics analysis:** 
   1. Application of PyDESeq2 and GSEA in a single pipeline.
  
3. **Classical ML analysis:** 
   1. Applying clustering, supervised ML and outlier sum statistics.

4. **Custom API Connections**:
   1. Search and retrieval of Cancer Data cohorts from GDC using complex json filters ([Methods in src.Connectors for GDC API search and retrieval using custom queries](./src/README.md))
   2. Interacting with MongoDB database in a pythonic manner (**DOCS coming soon**). 
   3. Interacting with Google cloud BigQuery in a pythonic manner (**DOCS coming soon**).  
    
<!-- #### High Level Objective:
In NGS datasets for kidney cancer or other complex diseases, apply known or new ML models to identify patterns of gene expression to serve as a template for bio-informatics learning for aspiring scientists/researchers/students in the field. -->

### GETTING STARTED:
1. Clone the repository `git clone https://github.com/adhal007/OmixHub.git` 
2. Create the correct conda enviroment for OmixHub: `conda env create -f environment.yaml`

#### Applications

1. **RNA Seq Cohort Creation of tumor and normal samples by primary site**
   1. [Example Jupyter Notebook](./tutorial_notebooks/cohort_creation_rna_seq.ipynb)
   2. **Code:**
  ```
  import grequests
  import src.Engines.gdc_engine as gdc_engine
  from importlib import reload
  reload(gdc_engine)

  ## Create Dataset for differential gene expression
  rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='DE')

  ## Create Dataset for machine learning analysis
  rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='ML') 
  ```

2. **Differential gene expression(DGE)  + Gene set enrichment analysis(GSEA) for tumor vs normal samples**  
   1. [Example jupyter notebook](./tutorial_notebooks/pydeseq_gsea.ipynb)
3. **Using GRADIO App for DGE + GSEA**:
   1. Currently this is restricted to Users. If you want to try this ou or contribute to this reach out to me via [adhalbiophysics@gmail.com](mailto:adhalbiophysics@gmail.com) with your interest.
   2. Running the app:
      1. After completing the steps in getting started, follow the next steps
      2. Run gradio app `python3 app_gradio.py`
      3. Check out the [app navigation documentation](./docs/UI%20Prototype/gradio_use.md).

  <!-- - Deep-learning:
    - Auto-encoders 
    - CNN’s
    - RNN’s for multi-timepoint data 
  - Bayesian ML
  - Knowledge Graphs and NLP  -->


<!-- #### Modules developed:

- [Module for Genomic Data Commons API accession, querying, search and retrieval](https://github.com/adhal007/OmixHub/tree/main/src/Connectors)
- [Module for Outlier Statistic Methods](https://github.com/adhal007/OmixHub/blob/main/src/OutlierStatMethods/README.md)

- [Module for Omics data processing](https://github.com/adhal007/OmixHub/blob/main/src/README.md)
  - [Module for RNA-seq preprocessing](https://github.com/adhal007/OmixHub/blob/main/src/preprocess_utils.py)
  - [Module for base preprocessor class](https://github.com/adhal007/OmixHub/blob/main/src/base_preprocessor.py)
- [Module for Dimensionality Reduction Models](https://github.com/adhal007/OmixHub/blob/main/src/DimRedMappers/README.md)  
- [Module for ML classifier models (ensemble, multi_output)](https://github.com/adhal007/OmixHub/blob/main/src/base_ml_models.py)
- [Module for differential expression of RNA-Seq](https://github.com/adhal007/OmixHub/blob/main/src/pydeseq_utils.py) -->

  
<!-- #### Projects/Analysis (Application of Methods)
###### A. Characterization of kidney cancer using RNA-Seq transcriptome profiline
- Analysis 1: Supervised ML models for classification of kidney cancer subtypes using bulk RNA-Seq Data 
  - [Evaluation of multi-output classifier models(Jupyter NB)](/docs/SupervisedLearningApplication/docs/workflow.md)
  
  - [Evaluation of ensemble models for multi-label classification(Jupyter NB)](/docs/SupervisedLearningEnsembleApplication/docs/workflow.md)

- Analysis 2: Differential expression of kidney cancer subtypes by single factor and multi-factor methods
  <!-- - [Summary]() -->
  <!-- - [Notebook workflow](/docs/DeSeqApplication/docs/workflow.md)

- Analysis 3: Bayesian optimized stratification of Kidney-Cancer Subtypes by dimensionality reduction and clustering
    - [Notebook workflow](/docsUmapApplication/docs/workflow.md)

- Analysis 4: Application of Outlier Statistic Methods for differential gene expression
  (To be updated with workflow notebooks) -->


<!-- ## Future work
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

<!-- **Application work**:
- Autoimmune disease characterization by omics methods 
- Other cancers 
- Deep learning --> 

### ADDITIONAL CODE DOCS:
- **Application Examples**
  - [Outlier sum statistics](./docs/OutlierMethodsApplication/docs/workflow.md)
  - [Supervised ML using Ensemble models on kidney cancer data](./docs/SuperviseLearningEnsembleApplication/workflow.md)
  - [Clustering using bayesian optimized parameters of kidney cancer sub-types from TCGA](./docs/UmapApplication/docs/workflow.md)
  - [Analysis of differentially expressed genes functionally]((./notebooks/pydeseq_gsea.ipynb))
- [Roadmap for future developments](./docs/UI%20Prototype/roadmap.md)
- [Methods in src.Connectors for GDC API search and retrieval using custom queries](./src/README.md)

### References:
1. [Characterizing tumor toxicity in Gene therapy targets from Bulk RNA-Sequencing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10028977/#S5)
2. [Bayesian Framework for identifying gene expression outliers in individual sample of RNA-Seq data](https://ascopubs.org/doi/10.1200/CCI.19.00095)