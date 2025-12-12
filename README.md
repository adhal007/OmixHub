
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

### API DOCUMENTATION LINK
https://omixhub.readthedocs.io/en/latest/getting_started.html
<!-- #### High Level Objective:
In NGS datasets for kidney cancer or other complex diseases, apply known or new ML models to identify patterns of gene expression to serve as a template for bio-informatics learning for aspiring scientists/researchers/students in the field. -->

### GETTING STARTED:
1. Clone the repository `git clone https://github.com/adhal007/OmixHub.git` 
2. Create the correct conda enviroment for OmixHub: `conda env create -f environment.yaml`

### INSTALLACTION
Please follow the instructions on


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