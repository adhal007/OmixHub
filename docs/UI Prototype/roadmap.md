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