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
      - Differential Gene Expression using pydeseq wrapper (https://www.biostars.org/p/9536035/)
      - Gene set enrichment analysis using gseapy 
      - Feature importances from different classifiers 
    - Unsupervised ML:
      - Bayesian optimized (Hyper-Opt) clustering (KNN) using UMap embeddings
    - Semi-supervised ML:
      - Iterative GMM + PCA for cohort stratification for niche disease-control applications 

## Next 2 steps for the UI

1. **Matching Algorithm** 
   1. To find more normal tissue samples from a limited set of normal samples across many tissues.
   2. Methods:
      1. Optimal transport models
2. **Simulator** 
   1. to generate expression values for additional features.
3. **Functional Annotation**
   1. The goal here is to showcase if genes that are identified as potential drug targets will have some level of tumor toxicity and non-specificity.
   2. [Paper on BayesTS](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10028977/#S5):
      1. Describes a new method using 1) a normalized RNA count matrix, 2) tissue distribution profiles and 3) protein expression labels (Methods) to infer tumor toxicity or specificity.
   3. [Datasets for annotation of proteins of DGE](https://www.proteinatlas.org/about/download) 
   4. [Characterizing tumor toxicity in Gene therapy targets from Bulk RNA-Sequencing](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10028977/#S5)
   5. [Bayesian Framework for identifying gene expression outliers in individual sample of RNA-Seq data](https://ascopubs.org/doi/10.1200/CCI.19.00095)