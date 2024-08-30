
Overview
========

OmixHub is a platform that interfaces with GDC using Python to help users apply ML-based analysis on different sequencing data. Currently, we **support only for RNA-Seq based datasets** from the genomic data commons (GDC).

1. **Cohort Creation** of Bulk RNA Seq Tumor and Normal Samples from GDC. 
2. **Bioinformatics analysis:** 
   1. Application of PyDESeq2 and GSEA in a single pipeline.
3. **Classical ML analysis:** 
   1. Applying clustering, supervised ML, and outlier sum statistics.
4. **Custom API Connections**:
   1. Search and retrieval of Cancer Data cohorts from GDC using complex JSON filters.
   2. Interacting with MongoDB in a Pythonic manner (DOCS coming soon).
   3. Interacting with Google Cloud BigQuery in a Pythonic manner (DOCS coming soon).
