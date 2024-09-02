
Applications
============

1. **RNA Seq Cohort Creation of tumor and normal samples by primary site**
   - [Example Jupyter Notebook](../tutorial_notebooks/cohort_creation_rna_seq.ipynb)
   
   **Code Example:**

   .. code-block:: python

      import grequests
      import src.Engines.gdc_engine as gdc_engine
      from importlib import reload
      reload(gdc_engine)

      # Create Dataset for differential gene expression
      rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='DE')

      # Create Dataset for machine learning analysis
      rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='ML')

2. **Differential gene expression(DGE)  + Gene set enrichment analysis(GSEA) for tumor vs normal samples**
   - [Example Jupyter Notebook](../tutorial_notebooks/pydeseq_gsea.ipynb)

3. **Using GRADIO App for DGE + GSEA**:
   - Currently restricted to users. To contribute or try it, contact [adhalbiophysics@gmail.com](mailto:adhalbiophysics@gmail.com).
   - Running the app:

   .. code-block:: bash

      python3 app_gradio.py

   App navigation documentation: 
   .. code-block:: bash

      cd ../tutorial_notebooks/docs/UI%20Prototype/gradio_use.md.
