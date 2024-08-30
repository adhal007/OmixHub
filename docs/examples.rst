Examples
=============

Installation/Usage:
*******************
As the package has not been published on PyPi yet, it CANNOT be install using pip.

For now, the suggested method is to clone the repository and view the example notebooks.


Example 1. Cohort Creation of Bulk RNA Seq Experiments from Genomic Data Commons (GDC)
**************************************************
.. code-block:: python

    """This example demonstrates how to create a data matrix for Differential gene expression (DE) or machine learning analysis.
    You can select the primary site of the samples and the downstream analysis you want to perform.
    """

      import grequests
      import src.Engines.gdc_engine as gdc_engine
      from importlib import reload
      reload(gdc_engine)

      # Create Dataset for differential gene expression
      rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='DE')

      # Create Dataset for machine learning analysis
      rna_seq_DGE_data = gdc_eng_inst.run_rna_seq_data_matrix_creation(primary_site='Kidney', downstream_analysis='ML')
