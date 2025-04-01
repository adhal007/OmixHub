```python
%cd '/Users/abhilashdhal/Projects/'
```

    /Users/abhilashdhal/Projects



```python
import pandas as pd
from importlib import reload
import src.Engines.analysis_engine as analysis_engine
import src.Connectors.gcp_bigquery_utils as gcp_bigquery_utils
reload(analysis_engine)
reload(gcp_bigquery_utils)
```




    <module 'src.Connectors.gcp_bigquery_utils' from '/Users/abhilashdhal/Projects/src/Connectors/gcp_bigquery_utils.py'>



## 1. Download Dataset from BigQuery for a given Primary Diagnosis By Primary Site and the Normal Tissue for the Primary site


```python
project_id = 'rnaseqml'
dataset_id = 'rnaseqexpression'
table_id = 'expr_clustered_08082024'
bq_queries = gcp_bigquery_utils.BigQueryQueries(project_id=project_id, 
                                              dataset_id=dataset_id,
                                              table_id=table_id)


```

    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1726382529.860340 1764338 config.cc:230] gRPC experiments enabled: call_status_override_on_cancellation, event_engine_dns, event_engine_listener, http2_stats_fix, monitoring_experiment, pick_first_new, trace_record_callops, work_serializer_clears_time_cache
    I0000 00:00:1726382529.873960 1764338 check_gcp_environment_no_op.cc:29] ALTS: Platforms other than Linux and Windows are not supported



```python
pr_site = 'Head and Neck'
pr_diag = 'Squamous cell carcinoma, NOS'
data_from_bq = bq_queries.get_df_for_pydeseq(primary_site=pr_site, primary_diagnosis=pr_diag)
```

    I0000 00:00:1726382538.487678 1764338 check_gcp_environment_no_op.cc:29] ALTS: Platforms other than Linux and Windows are not supported



```python
data_from_bq
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>case_id</th>
      <th>primary_site</th>
      <th>sample_type</th>
      <th>tissue_type</th>
      <th>tissue_or_organ_of_origin</th>
      <th>primary_diagnosis</th>
      <th>expr_unstr_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ae90972d-5bc2-4b53-b5ff-1b8c31f39342</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Border of tongue</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[1904, 2, 2880, 538, 690, 1118, 2461, 2083, 18...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15eb8d88-adb1-4f4f-a44d-087049db60ce</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Mandible</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[1633, 0, 2062, 465, 665, 207, 8759, 2220, 227...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17885905-280f-40fa-943e-346ed45403ea</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Supraglottis</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[790, 0, 459, 62, 72, 142, 545, 786, 1902, 420...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>91d44093-d8d9-442c-92ed-e659c45b4e0a</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Lower gum</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[2653, 1, 1109, 275, 226, 65, 907, 717, 5426, ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6a0490ea-d9c6-41cf-bec3-3257e98cc6ed</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Anterior floor of mouth</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[2842, 0, 1559, 602, 621, 282, 1967, 1609, 515...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>402</th>
      <td>dff4246a-57ea-46f2-a5bc-7aa82e494137</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Overlapping lesion of lip, oral cavity and pha...</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[1095, 0, 2223, 419, 510, 962, 2094, 1888, 419...</td>
    </tr>
    <tr>
      <th>403</th>
      <td>ff9db03d-f81c-4079-8b05-f5a8c05b8cfe</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Overlapping lesion of lip, oral cavity and pha...</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[786, 0, 1345, 342, 263, 838, 4669, 968, 1442,...</td>
    </tr>
    <tr>
      <th>404</th>
      <td>8ebe8c25-5ef9-42d4-9414-8313227b673f</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Overlapping lesion of lip, oral cavity and pha...</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[1426, 0, 2444, 912, 587, 643, 8505, 1906, 337...</td>
    </tr>
    <tr>
      <th>405</th>
      <td>aa10d6da-ba20-43e8-ab8f-a9b4b58738b4</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Overlapping lesion of lip, oral cavity and pha...</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[1087, 0, 2047, 581, 605, 1321, 6157, 2091, 25...</td>
    </tr>
    <tr>
      <th>406</th>
      <td>acd98e20-d2da-4256-99a5-13e261bc88e6</td>
      <td>Head and Neck</td>
      <td>Primary Tumor</td>
      <td>Tumor</td>
      <td>Overlapping lesion of lip, oral cavity and pha...</td>
      <td>Squamous cell carcinoma, NOS</td>
      <td>[2039, 0, 4867, 484, 552, 350, 1077, 655, 2617...</td>
    </tr>
  </tbody>
</table>
<p>407 rows × 7 columns</p>
</div>



## 2. Data Preprocessing for PyDeSeq and GSEA


```python
analysis_eng = analysis_engine.AnalysisEngine(data_from_bq, analysis_type='DE')
if not analysis_eng.check_tumor_normal_counts():
    raise ValueError("Tumor and Normal counts should be at least 10 each")
gene_ids_or_gene_cols_df = pd.read_csv('/Users/abhilashdhal/Projects/personal_docs/data/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')
gene_ids_or_gene_cols = list(gene_ids_or_gene_cols_df['gene_id'].to_numpy())
```


```python
exp_df = analysis_eng.expand_data_from_bq(data_from_bq, gene_ids_or_gene_cols=gene_ids_or_gene_cols, analysis_type='DE')
metadata = analysis_eng.metadata_for_pydeseq(exp_df=exp_df)
counts_for_de = analysis_eng.counts_from_bq_df(exp_df, gene_ids_or_gene_cols)
```

## 3. Run DESeq2 for the given Primary Diagnosis By Primary Site and the Normal Tissue for the Primary site


```python
res_pydeseq = analysis_eng.run_pydeseq(metadata=metadata, counts=counts_for_de)

```

    08/09//2024 01:14:1725738270 AM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2 for groups: {'group1': 'Tumor', 'group2': 'Normal'}
    08/09//2024 01:14:1725738270 AM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2  factor analysis with design factor: C and o
    08/09//2024 01:14:1725738270 AM - INFO - PyDeSeqWrapper.run_deseq: Statistical analysis of Tumor vs Normal in {'group1': 'Tumor', 'group2': 'Normal'}


    Fitting size factors...
    ... done in 0.39 seconds.
    
    I0000 00:00:1725738274.705281 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.829847 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.836386 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.842536 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.850395 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.858533 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.869203 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738274.986436 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738275.118370 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    I0000 00:00:1725738275.173707 1307865 work_stealing_thread_pool.cc:320] WorkStealingThreadPoolImpl::PrepareFork
    Fitting dispersions...
    ... done in 9.48 seconds.
    
    Fitting dispersion trend curve...
    ... done in 4.60 seconds.
    
    Fitting MAP dispersions...
    ... done in 10.55 seconds.
    
    Fitting LFCs...
    ... done in 6.71 seconds.
    
    Refitting 3818 outliers.
    
    Fitting size factors...
    ... done in 0.01 seconds.
    
    Fitting dispersions...
    ... done in 0.86 seconds.
    
    Fitting MAP dispersions...
    ... done in 1.09 seconds.
    
    Fitting LFCs...
    ... done in 1.05 seconds.
    
    Running Wald tests...
    ... done in 5.03 seconds.
    


    Log2 fold change & Wald test p-value: Condition Tumor vs Normal



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseMean</th>
      <th>log2FoldChange</th>
      <th>lfcSE</th>
      <th>stat</th>
      <th>pvalue</th>
      <th>padj</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ENSG00000258011.2</th>
      <td>2232.991406</td>
      <td>-1.416117</td>
      <td>0.179004</td>
      <td>-7.911108</td>
      <td>2.551073e-15</td>
      <td>1.366146e-13</td>
    </tr>
    <tr>
      <th>ENSG00000186792.17</th>
      <td>2.942486</td>
      <td>-3.730918</td>
      <td>0.730104</td>
      <td>-5.110118</td>
      <td>3.219568e-07</td>
      <td>3.327296e-06</td>
    </tr>
    <tr>
      <th>ENSG00000234551.2</th>
      <td>1982.284385</td>
      <td>0.129415</td>
      <td>0.111797</td>
      <td>1.157586</td>
      <td>2.470332e-01</td>
      <td>3.620330e-01</td>
    </tr>
    <tr>
      <th>ENSG00000270818.1</th>
      <td>518.247341</td>
      <td>-0.349234</td>
      <td>0.119929</td>
      <td>-2.912016</td>
      <td>3.591037e-03</td>
      <td>1.210372e-02</td>
    </tr>
    <tr>
      <th>ENSG00000008323.15</th>
      <td>471.753783</td>
      <td>1.068874</td>
      <td>0.164111</td>
      <td>6.513102</td>
      <td>7.361456e-11</td>
      <td>1.655386e-09</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ENSG00000267077.1</th>
      <td>0.360830</td>
      <td>-0.410850</td>
      <td>0.830174</td>
      <td>-0.494896</td>
      <td>6.206735e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ENSG00000109111.15</th>
      <td>190.771788</td>
      <td>-1.002799</td>
      <td>0.133398</td>
      <td>-7.517327</td>
      <td>5.590731e-14</td>
      <td>2.317959e-12</td>
    </tr>
    <tr>
      <th>ENSG00000253088.1</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ENSG00000255401.1</th>
      <td>2.472590</td>
      <td>-1.105194</td>
      <td>0.260667</td>
      <td>-4.239869</td>
      <td>2.236506e-05</td>
      <td>1.464306e-04</td>
    </tr>
    <tr>
      <th>ENSG00000270959.1</th>
      <td>22.169829</td>
      <td>0.450951</td>
      <td>0.222059</td>
      <td>2.030767</td>
      <td>4.227865e-02</td>
      <td>9.208805e-02</td>
    </tr>
  </tbody>
</table>
<p>60660 rows × 6 columns</p>
</div>



```python
res_pydeseq_with_gene_names = pd.merge(res_pydeseq, gene_ids_or_gene_cols_df, left_on='index', right_on='gene_id')
```

## 4. Run GSEA for the given Primary Diagnosis By Primary Site and the Normal Tissue for the Primary site using a gene set database


```python
from gseapy.plot import gseaplot
import gseapy as gp
from gseapy import dotplot
gsea_options = gp.get_library_name()
print(gsea_options)
```

    ['ARCHS4_Cell-lines', 'ARCHS4_IDG_Coexp', 'ARCHS4_Kinases_Coexp', 'ARCHS4_TFs_Coexp', 'ARCHS4_Tissues', 'Achilles_fitness_decrease', 'Achilles_fitness_increase', 'Aging_Perturbations_from_GEO_down', 'Aging_Perturbations_from_GEO_up', 'Allen_Brain_Atlas_10x_scRNA_2021', 'Allen_Brain_Atlas_down', 'Allen_Brain_Atlas_up', 'Azimuth_2023', 'Azimuth_Cell_Types_2021', 'BioCarta_2013', 'BioCarta_2015', 'BioCarta_2016', 'BioPlanet_2019', 'BioPlex_2017', 'CCLE_Proteomics_2020', 'CORUM', 'COVID-19_Related_Gene_Sets', 'COVID-19_Related_Gene_Sets_2021', 'Cancer_Cell_Line_Encyclopedia', 'CellMarker_2024', 'CellMarker_Augmented_2021', 'ChEA_2013', 'ChEA_2015', 'ChEA_2016', 'ChEA_2022', 'Chromosome_Location', 'Chromosome_Location_hg19', 'ClinVar_2019', 'DGIdb_Drug_Targets_2024', 'DSigDB', 'Data_Acquisition_Method_Most_Popular_Genes', 'DepMap_CRISPR_GeneDependency_CellLines_2023', 'DepMap_WG_CRISPR_Screens_Broad_CellLines_2019', 'DepMap_WG_CRISPR_Screens_Sanger_CellLines_2019', 'Descartes_Cell_Types_and_Tissue_2021', 'Diabetes_Perturbations_GEO_2022', 'DisGeNET', 'Disease_Perturbations_from_GEO_down', 'Disease_Perturbations_from_GEO_up', 'Disease_Signatures_from_GEO_down_2014', 'Disease_Signatures_from_GEO_up_2014', 'DrugMatrix', 'Drug_Perturbations_from_GEO_2014', 'Drug_Perturbations_from_GEO_down', 'Drug_Perturbations_from_GEO_up', 'ENCODE_Histone_Modifications_2013', 'ENCODE_Histone_Modifications_2015', 'ENCODE_TF_ChIP-seq_2014', 'ENCODE_TF_ChIP-seq_2015', 'ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X', 'ESCAPE', 'Elsevier_Pathway_Collection', 'Enrichr_Libraries_Most_Popular_Genes', 'Enrichr_Submissions_TF-Gene_Coocurrence', 'Enrichr_Users_Contributed_Lists_2020', 'Epigenomics_Roadmap_HM_ChIP-seq', 'FANTOM6_lncRNA_KD_DEGs', 'GO_Biological_Process_2013', 'GO_Biological_Process_2015', 'GO_Biological_Process_2017', 'GO_Biological_Process_2017b', 'GO_Biological_Process_2018', 'GO_Biological_Process_2021', 'GO_Biological_Process_2023', 'GO_Cellular_Component_2013', 'GO_Cellular_Component_2015', 'GO_Cellular_Component_2017', 'GO_Cellular_Component_2017b', 'GO_Cellular_Component_2018', 'GO_Cellular_Component_2021', 'GO_Cellular_Component_2023', 'GO_Molecular_Function_2013', 'GO_Molecular_Function_2015', 'GO_Molecular_Function_2017', 'GO_Molecular_Function_2017b', 'GO_Molecular_Function_2018', 'GO_Molecular_Function_2021', 'GO_Molecular_Function_2023', 'GTEx_Aging_Signatures_2021', 'GTEx_Tissue_Expression_Down', 'GTEx_Tissue_Expression_Up', 'GTEx_Tissues_V8_2023', 'GWAS_Catalog_2019', 'GWAS_Catalog_2023', 'GeDiPNet_2023', 'GeneSigDB', 'Gene_Perturbations_from_GEO_down', 'Gene_Perturbations_from_GEO_up', 'Genes_Associated_with_NIH_Grants', 'Genome_Browser_PWMs', 'GlyGen_Glycosylated_Proteins_2022', 'HDSigDB_Human_2021', 'HDSigDB_Mouse_2021', 'HMDB_Metabolites', 'HMS_LINCS_KinomeScan', 'HomoloGene', 'HuBMAP_ASCT_plus_B_augmented_w_RNAseq_Coexpression', 'HuBMAP_ASCTplusB_augmented_2022', 'HumanCyc_2015', 'HumanCyc_2016', 'Human_Gene_Atlas', 'Human_Phenotype_Ontology', 'IDG_Drug_Targets_2022', 'InterPro_Domains_2019', 'Jensen_COMPARTMENTS', 'Jensen_DISEASES', 'Jensen_TISSUES', 'KEA_2013', 'KEA_2015', 'KEGG_2013', 'KEGG_2015', 'KEGG_2016', 'KEGG_2019_Human', 'KEGG_2019_Mouse', 'KEGG_2021_Human', 'KOMP2_Mouse_Phenotypes_2022', 'Kinase_Perturbations_from_GEO_down', 'Kinase_Perturbations_from_GEO_up', 'L1000_Kinase_and_GPCR_Perturbations_down', 'L1000_Kinase_and_GPCR_Perturbations_up', 'LINCS_L1000_CRISPR_KO_Consensus_Sigs', 'LINCS_L1000_Chem_Pert_Consensus_Sigs', 'LINCS_L1000_Chem_Pert_down', 'LINCS_L1000_Chem_Pert_up', 'LINCS_L1000_Ligand_Perturbations_down', 'LINCS_L1000_Ligand_Perturbations_up', 'Ligand_Perturbations_from_GEO_down', 'Ligand_Perturbations_from_GEO_up', 'MAGMA_Drugs_and_Diseases', 'MAGNET_2023', 'MCF7_Perturbations_from_GEO_down', 'MCF7_Perturbations_from_GEO_up', 'MGI_Mammalian_Phenotype_2013', 'MGI_Mammalian_Phenotype_2017', 'MGI_Mammalian_Phenotype_Level_3', 'MGI_Mammalian_Phenotype_Level_4', 'MGI_Mammalian_Phenotype_Level_4_2019', 'MGI_Mammalian_Phenotype_Level_4_2021', 'MGI_Mammalian_Phenotype_Level_4_2024', 'MSigDB_Computational', 'MSigDB_Hallmark_2020', 'MSigDB_Oncogenic_Signatures', 'Metabolomics_Workbench_Metabolites_2022', 'Microbe_Perturbations_from_GEO_down', 'Microbe_Perturbations_from_GEO_up', 'MoTrPAC_2023', 'Mouse_Gene_Atlas', 'NCI-60_Cancer_Cell_Lines', 'NCI-Nature_2016', 'NIH_Funded_PIs_2017_AutoRIF_ARCHS4_Predictions', 'NIH_Funded_PIs_2017_GeneRIF_ARCHS4_Predictions', 'NIH_Funded_PIs_2017_Human_AutoRIF', 'NIH_Funded_PIs_2017_Human_GeneRIF', 'NURSA_Human_Endogenous_Complexome', 'OMIM_Disease', 'OMIM_Expanded', 'Old_CMAP_down', 'Old_CMAP_up', 'Orphanet_Augmented_2021', 'PFOCR_Pathways', 'PFOCR_Pathways_2023', 'PPI_Hub_Proteins', 'PanglaoDB_Augmented_2021', 'Panther_2015', 'Panther_2016', 'Pfam_Domains_2019', 'Pfam_InterPro_Domains', 'PheWeb_2019', 'PhenGenI_Association_2021', 'Phosphatase_Substrates_from_DEPOD', 'ProteomicsDB_2020', 'Proteomics_Drug_Atlas_2023', 'RNA-Seq_Disease_Gene_and_Drug_Signatures_from_GEO', 'RNAseq_Automatic_GEO_Signatures_Human_Down', 'RNAseq_Automatic_GEO_Signatures_Human_Up', 'RNAseq_Automatic_GEO_Signatures_Mouse_Down', 'RNAseq_Automatic_GEO_Signatures_Mouse_Up', 'Rare_Diseases_AutoRIF_ARCHS4_Predictions', 'Rare_Diseases_AutoRIF_Gene_Lists', 'Rare_Diseases_GeneRIF_ARCHS4_Predictions', 'Rare_Diseases_GeneRIF_Gene_Lists', 'Reactome_2013', 'Reactome_2015', 'Reactome_2016', 'Reactome_2022', 'Rummagene_kinases', 'Rummagene_signatures', 'Rummagene_transcription_factors', 'SILAC_Phosphoproteomics', 'SubCell_BarCode', 'SynGO_2022', 'SynGO_2024', 'SysMyo_Muscle_Gene_Sets', 'TF-LOF_Expression_from_GEO', 'TF_Perturbations_Followed_by_Expression', 'TG_GATES_2020', 'TRANSFAC_and_JASPAR_PWMs', 'TRRUST_Transcription_Factors_2019', 'Table_Mining_of_CRISPR_Studies', 'Tabula_Muris', 'Tabula_Sapiens', 'TargetScan_microRNA', 'TargetScan_microRNA_2017', 'The_Kinase_Library_2023', 'Tissue_Protein_Expression_from_Human_Proteome_Map', 'Tissue_Protein_Expression_from_ProteomicsDB', 'Transcription_Factor_PPIs', 'UK_Biobank_GWAS_v1', 'Virus-Host_PPI_P-HIPSTer_2020', 'VirusMINT', 'Virus_Perturbations_from_GEO_down', 'Virus_Perturbations_from_GEO_up', 'WikiPathway_2021_Human', 'WikiPathway_2023_Human', 'WikiPathways_2013', 'WikiPathways_2015', 'WikiPathways_2016', 'WikiPathways_2019_Human', 'WikiPathways_2019_Mouse', 'dbGaP', 'huMAP', 'lncHUB_lncRNA_Co-Expression', 'miRTarBase_2017']



```python
gene_set = 'KEGG_2021_Human'
result, plot, pre_res = analysis_eng.run_gsea(res_pydeseq_with_gene_names, gene_set)
```


    
![png](../bq_to_gsea_tutorial_files/bq_to_gsea_tutorial_14_0.png)
    

