```python
import OmicsUtils.pydeseq_utils as pydeseq_utils
import pandas as pd
```

## I. Process Counts data

## 1.1 Read in counts data and metadata


```python
count_data_kidney_cancer = pd.read_csv('./Transcriptomics/data/processed_data/stranded_first_data_with_labels.csv')
count_data_kidney_cancer
count_data_kidney_cancer[['Case ID', 'Project ID']]
kidney_cancer_count_data = count_data_kidney_cancer.iloc[:, :60660].T
counts = kidney_cancer_count_data.copy().reset_index()
counts = counts.set_index('index')
counts
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>1633</th>
      <th>1634</th>
      <th>1635</th>
      <th>1636</th>
      <th>1637</th>
      <th>1638</th>
      <th>1639</th>
      <th>1640</th>
      <th>1641</th>
      <th>1642</th>
    </tr>
    <tr>
      <th>index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ENSG00000000003.15</th>
      <td>4</td>
      <td>677</td>
      <td>1561</td>
      <td>2296</td>
      <td>4929</td>
      <td>6</td>
      <td>1340</td>
      <td>1</td>
      <td>2</td>
      <td>2383</td>
      <td>...</td>
      <td>1</td>
      <td>5554</td>
      <td>8</td>
      <td>2538</td>
      <td>2005</td>
      <td>0</td>
      <td>1774</td>
      <td>2</td>
      <td>1928</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ENSG00000000005.6</th>
      <td>12</td>
      <td>6</td>
      <td>1</td>
      <td>96</td>
      <td>8</td>
      <td>4</td>
      <td>0</td>
      <td>22</td>
      <td>26</td>
      <td>191</td>
      <td>...</td>
      <td>1</td>
      <td>18</td>
      <td>1</td>
      <td>3</td>
      <td>8</td>
      <td>1</td>
      <td>9</td>
      <td>8</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>ENSG00000000419.13</th>
      <td>9</td>
      <td>499</td>
      <td>693</td>
      <td>714</td>
      <td>936</td>
      <td>13</td>
      <td>904</td>
      <td>15</td>
      <td>8</td>
      <td>840</td>
      <td>...</td>
      <td>8</td>
      <td>948</td>
      <td>17</td>
      <td>984</td>
      <td>482</td>
      <td>8</td>
      <td>1122</td>
      <td>10</td>
      <td>991</td>
      <td>7</td>
    </tr>
    <tr>
      <th>ENSG00000000457.14</th>
      <td>294</td>
      <td>457</td>
      <td>175</td>
      <td>456</td>
      <td>841</td>
      <td>251</td>
      <td>394</td>
      <td>410</td>
      <td>961</td>
      <td>835</td>
      <td>...</td>
      <td>311</td>
      <td>798</td>
      <td>1112</td>
      <td>1325</td>
      <td>341</td>
      <td>300</td>
      <td>332</td>
      <td>142</td>
      <td>461</td>
      <td>185</td>
    </tr>
    <tr>
      <th>ENSG00000000460.17</th>
      <td>366</td>
      <td>242</td>
      <td>90</td>
      <td>185</td>
      <td>367</td>
      <td>262</td>
      <td>227</td>
      <td>371</td>
      <td>270</td>
      <td>489</td>
      <td>...</td>
      <td>317</td>
      <td>351</td>
      <td>883</td>
      <td>1062</td>
      <td>189</td>
      <td>291</td>
      <td>204</td>
      <td>322</td>
      <td>280</td>
      <td>181</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ENSG00000288669.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ENSG00000288670.1</th>
      <td>0</td>
      <td>168</td>
      <td>400</td>
      <td>455</td>
      <td>159</td>
      <td>4</td>
      <td>348</td>
      <td>1</td>
      <td>0</td>
      <td>377</td>
      <td>...</td>
      <td>0</td>
      <td>712</td>
      <td>7</td>
      <td>366</td>
      <td>172</td>
      <td>0</td>
      <td>258</td>
      <td>2</td>
      <td>288</td>
      <td>1</td>
    </tr>
    <tr>
      <th>ENSG00000288671.1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>ENSG00000288674.1</th>
      <td>4</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
    </tr>
    <tr>
      <th>ENSG00000288675.1</th>
      <td>16</td>
      <td>27</td>
      <td>30</td>
      <td>5</td>
      <td>27</td>
      <td>11</td>
      <td>36</td>
      <td>22</td>
      <td>72</td>
      <td>20</td>
      <td>...</td>
      <td>31</td>
      <td>10</td>
      <td>93</td>
      <td>56</td>
      <td>14</td>
      <td>22</td>
      <td>6</td>
      <td>8</td>
      <td>24</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
<p>60660 rows × 1643 columns</p>
</div>



## 1.2. Transpose the data for deseq2


```python
counts = counts.T
counts = pd.concat( [count_data_kidney_cancer[['Case ID']], counts],axis=1)
```


```python
counts.rename(columns={'Case ID':'Geneid'}, inplace=True)
counts.set_index('Geneid', inplace=True)
counts.head()
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
      <th>ENSG00000000003.15</th>
      <th>ENSG00000000005.6</th>
      <th>ENSG00000000419.13</th>
      <th>ENSG00000000457.14</th>
      <th>ENSG00000000460.17</th>
      <th>ENSG00000000938.13</th>
      <th>ENSG00000000971.16</th>
      <th>ENSG00000001036.14</th>
      <th>ENSG00000001084.13</th>
      <th>ENSG00000001167.14</th>
      <th>...</th>
      <th>ENSG00000288661.1</th>
      <th>ENSG00000288662.1</th>
      <th>ENSG00000288663.1</th>
      <th>ENSG00000288665.1</th>
      <th>ENSG00000288667.1</th>
      <th>ENSG00000288669.1</th>
      <th>ENSG00000288670.1</th>
      <th>ENSG00000288671.1</th>
      <th>ENSG00000288674.1</th>
      <th>ENSG00000288675.1</th>
    </tr>
    <tr>
      <th>Geneid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C3L-00966</th>
      <td>4</td>
      <td>12</td>
      <td>9</td>
      <td>294</td>
      <td>366</td>
      <td>1</td>
      <td>22</td>
      <td>87</td>
      <td>9</td>
      <td>11</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>16</td>
    </tr>
    <tr>
      <th>TCGA-MM-A563</th>
      <td>677</td>
      <td>6</td>
      <td>499</td>
      <td>457</td>
      <td>242</td>
      <td>1046</td>
      <td>2378</td>
      <td>1263</td>
      <td>616</td>
      <td>383</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>168</td>
      <td>0</td>
      <td>7</td>
      <td>27</td>
    </tr>
    <tr>
      <th>TCGA-GL-8500</th>
      <td>1561</td>
      <td>1</td>
      <td>693</td>
      <td>175</td>
      <td>90</td>
      <td>95</td>
      <td>53</td>
      <td>3168</td>
      <td>1540</td>
      <td>682</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>400</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
    </tr>
    <tr>
      <th>TCGA-BQ-5877</th>
      <td>2296</td>
      <td>96</td>
      <td>714</td>
      <td>456</td>
      <td>185</td>
      <td>195</td>
      <td>1120</td>
      <td>3311</td>
      <td>1722</td>
      <td>606</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>455</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>TCGA-KN-8423</th>
      <td>4929</td>
      <td>8</td>
      <td>936</td>
      <td>841</td>
      <td>367</td>
      <td>152</td>
      <td>271</td>
      <td>3019</td>
      <td>1440</td>
      <td>639</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>159</td>
      <td>0</td>
      <td>0</td>
      <td>27</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 60660 columns</p>
</div>



## II. Metadata pre-processing

## 2.1. Create a metadata dataframe with Case ID and Cancer subtype 

Only 3 cancers from TCGA (Renal Cell Carcinoma) are included in this


```python
metadata = count_data_kidney_cancer[['Case ID', 'Project ID']]
metadata.columns = ['Sample', 'Condition']
metadata = metadata.set_index(keys='Sample') 
TCGA_rcc_metadata = metadata[metadata['Condition'].isin(values=['TCGA-KIRC', 'TCGA-KIRP', 'TCGA-KICH'])]
TCGA_rcc_metadata
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
      <th>Condition</th>
    </tr>
    <tr>
      <th>Sample</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TCGA-MM-A563</th>
      <td>TCGA-KIRC</td>
    </tr>
    <tr>
      <th>TCGA-GL-8500</th>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>TCGA-BQ-5877</th>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>TCGA-KN-8423</th>
      <td>TCGA-KICH</td>
    </tr>
    <tr>
      <th>TCGA-5P-A9K0</th>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>TCGA-CW-6087</th>
      <td>TCGA-KIRC</td>
    </tr>
    <tr>
      <th>TCGA-BQ-5884</th>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>TCGA-AK-3456</th>
      <td>TCGA-KIRC</td>
    </tr>
    <tr>
      <th>TCGA-KN-8436</th>
      <td>TCGA-KICH</td>
    </tr>
    <tr>
      <th>TCGA-B0-5697</th>
      <td>TCGA-KIRC</td>
    </tr>
  </tbody>
</table>
<p>924 rows × 1 columns</p>
</div>



## 2.2. Subset the counts data to only include the 3 cancers


```python
counts_rcc = counts[counts.index.isin(TCGA_rcc_metadata.index.values)]
```

## III. Running Deseq2


```python
from importlib import reload
import OmicsUtils.pydeseq_utils as pydeseq_utils
reload(pydeseq_utils)
```




    <module 'OmicsUtils.pydeseq_utils' from '/Users/abhilashdhal/Projects/OmicsUtils/pydeseq_utils.py'>




```python
## Initialize the pydeseq_utils object
pydeseq_obj = pydeseq_utils.PyDeSeqWrapper(count_matrix=counts_rcc, metadata=TCGA_rcc_metadata, design_factors='Condition', groups = {'group1':'TCGA-KIRC', 'group2':'TCGA-KIRP'})
design_factor = 'Condition'
result = pydeseq_obj.run_deseq(design_factor=design_factor, group1 = 'TCGA-KIRC', group2 = 'TCGA-KIRP')
```

    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2 for groups: {'group1': 'TCGA-KIRC', 'group2': 'TCGA-KIRP'}
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2 for groups: {'group1': 'TCGA-KIRC', 'group2': 'TCGA-KIRP'}
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2 for groups: {'group1': 'TCGA-KIRC', 'group2': 'TCGA-KIRP'}
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2  factor analysis with design factor: C and o
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2  factor analysis with design factor: C and o
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Running DESeq2  factor analysis with design factor: C and o
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Statistical analysis of TCGA-KIRC vs TCGA-KIRP in {'group1': 'TCGA-KIRC', 'group2': 'TCGA-KIRP'}
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Statistical analysis of TCGA-KIRC vs TCGA-KIRP in {'group1': 'TCGA-KIRC', 'group2': 'TCGA-KIRP'}
    13/11//2023 05:59:1699878581 PM - INFO - PyDeSeqWrapper.run_deseq: Statistical analysis of TCGA-KIRC vs TCGA-KIRP in {'group1': 'TCGA-KIRC', 'group2': 'TCGA-KIRP'}


    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/anndata/_core/anndata.py:1897: UserWarning: Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
      utils.warn_names_duplicates("obs")
    Fitting size factors...
    ... done in 0.71 seconds.
    
    Fitting dispersions...
    ... done in 12.99 seconds.
    
    Fitting dispersion trend curve...
    ... done in 4.57 seconds.
    
    Fitting MAP dispersions...
    ... done in 14.78 seconds.
    
    Fitting LFCs...
    ... done in 30.33 seconds.
    
    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/anndata/_core/anndata.py:1897: UserWarning: Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
      utils.warn_names_duplicates("obs")
    Refitting 10827 outliers.
    
    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/anndata/_core/anndata.py:1897: UserWarning: Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
      utils.warn_names_duplicates("obs")
    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/anndata/_core/anndata.py:1897: UserWarning: Observation names are not unique. To make them unique, call `.obs_names_make_unique`.
      utils.warn_names_duplicates("obs")
    Fitting size factors...
    ... done in 0.08 seconds.
    
    Fitting dispersions...
    ... done in 2.75 seconds.
    
    Fitting MAP dispersions...
    ... done in 2.71 seconds.
    
    Fitting LFCs...
    ... done in 5.80 seconds.
    


## IV. Data Interpretation of the results

## 4.1. Create a PCA plot


```python
import scanpy as sc
```


```python
sc.tl.pca(pydeseq_obj.dds)
```


```python
sc.pl.pca(pydeseq_obj.dds, color = 'Condition', size = 200)
```

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/scanpy/plotting/_tools/scatterplots.py:394: UserWarning: No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored
      cax = scatter(



    
![png](/ProjectDocs/DeSeqApplication/images/TCGA_Deseq2_analysis_files/TCGA_Deseq2_analysis_19_1.png)
    


## 4.2. Create a volcano plot of the results


```python
import numpy as np
```


```python
result.summary()
```

    Running Wald tests...
    ... done in 26.52 seconds.
    


    Log2 fold change & Wald test p-value: Condition TCGA-KIRC vs TCGA-KIRP



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
      <th>ENSG00000000003.15</th>
      <td>1745.846969</td>
      <td>-0.229056</td>
      <td>0.054051</td>
      <td>-4.237788</td>
      <td>2.257328e-05</td>
      <td>4.716019e-05</td>
    </tr>
    <tr>
      <th>ENSG00000000005.6</th>
      <td>24.765175</td>
      <td>1.245152</td>
      <td>0.164656</td>
      <td>7.562141</td>
      <td>3.964869e-14</td>
      <td>1.391292e-13</td>
    </tr>
    <tr>
      <th>ENSG00000000419.13</th>
      <td>688.768193</td>
      <td>0.013235</td>
      <td>0.028567</td>
      <td>0.463276</td>
      <td>6.431665e-01</td>
      <td>7.231618e-01</td>
    </tr>
    <tr>
      <th>ENSG00000000457.14</th>
      <td>456.921769</td>
      <td>0.171048</td>
      <td>0.031631</td>
      <td>5.407658</td>
      <td>6.385407e-08</td>
      <td>1.593618e-07</td>
    </tr>
    <tr>
      <th>ENSG00000000460.17</th>
      <td>280.125529</td>
      <td>0.348439</td>
      <td>0.041999</td>
      <td>8.296374</td>
      <td>1.073368e-16</td>
      <td>4.215969e-16</td>
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
      <th>ENSG00000288669.1</th>
      <td>0.164879</td>
      <td>0.127490</td>
      <td>0.423766</td>
      <td>0.300851</td>
      <td>7.635283e-01</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ENSG00000288670.1</th>
      <td>263.675230</td>
      <td>-0.167515</td>
      <td>0.038957</td>
      <td>-4.300023</td>
      <td>1.707803e-05</td>
      <td>3.598875e-05</td>
    </tr>
    <tr>
      <th>ENSG00000288671.1</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>ENSG00000288674.1</th>
      <td>3.072331</td>
      <td>0.264358</td>
      <td>0.088761</td>
      <td>2.978326</td>
      <td>2.898276e-03</td>
      <td>4.982204e-03</td>
    </tr>
    <tr>
      <th>ENSG00000288675.1</th>
      <td>21.585309</td>
      <td>-0.656003</td>
      <td>0.066130</td>
      <td>-9.919931</td>
      <td>3.409917e-23</td>
      <td>1.742860e-22</td>
    </tr>
  </tbody>
</table>
<p>60660 rows × 6 columns</p>
</div>


## 4.2.1. Process results dataframe for plotting


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as n
import random
```

## 4.2.1.1 Read and merge dataframe of DESeq with gene annotation to display


```python
gene_annotation = pd.read_csv('./Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv')

results_df = result.results_df
results_df_filtered = results_df.dropna()
results_df_filtered = results_df_filtered.reset_index()
results_df_filtered['nlog10'] = -1*np.log10(results_df_filtered.padj)

results_df_filtered = results_df_filtered.merge(gene_annotation, left_on='index', right_on='gene_id')
results_df_filtered  = results_df_filtered .replace([np.inf, -np.inf], 300)
results_df_filtered.sort_values('padj', ascending=False)
```

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/pandas/core/arraylike.py:396: RuntimeWarning: divide by zero encountered in log10
      result = getattr(ufunc, method)(*inputs, **kwargs)





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
      <th>index</th>
      <th>baseMean</th>
      <th>log2FoldChange</th>
      <th>lfcSE</th>
      <th>stat</th>
      <th>pvalue</th>
      <th>padj</th>
      <th>nlog10</th>
      <th>gene_id</th>
      <th>gene_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4797</th>
      <td>ENSG00000118600.12</td>
      <td>497.937357</td>
      <td>-0.000004</td>
      <td>0.029879</td>
      <td>-0.000121</td>
      <td>0.999904</td>
      <td>0.999904</td>
      <td>0.000042</td>
      <td>ENSG00000118600.12</td>
      <td>RXYLT1</td>
    </tr>
    <tr>
      <th>26080</th>
      <td>ENSG00000235196.3</td>
      <td>0.204566</td>
      <td>0.000078</td>
      <td>0.364299</td>
      <td>0.000215</td>
      <td>0.999828</td>
      <td>0.999851</td>
      <td>0.000065</td>
      <td>ENSG00000235196.3</td>
      <td>Z68868.1</td>
    </tr>
    <tr>
      <th>1153</th>
      <td>ENSG00000071539.14</td>
      <td>133.506374</td>
      <td>-0.000023</td>
      <td>0.080789</td>
      <td>-0.000282</td>
      <td>0.999775</td>
      <td>0.999820</td>
      <td>0.000078</td>
      <td>ENSG00000071539.14</td>
      <td>TRIP13</td>
    </tr>
    <tr>
      <th>17434</th>
      <td>ENSG00000200247.1</td>
      <td>0.373922</td>
      <td>-0.000196</td>
      <td>0.604076</td>
      <td>-0.000324</td>
      <td>0.999741</td>
      <td>0.999810</td>
      <td>0.000083</td>
      <td>ENSG00000200247.1</td>
      <td>RNU6-254P</td>
    </tr>
    <tr>
      <th>30471</th>
      <td>ENSG00000251061.2</td>
      <td>1.063364</td>
      <td>0.000117</td>
      <td>0.314020</td>
      <td>0.000372</td>
      <td>0.999703</td>
      <td>0.999794</td>
      <td>0.000089</td>
      <td>ENSG00000251061.2</td>
      <td>LINC02512</td>
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
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>21165</th>
      <td>ENSG00000224490.5</td>
      <td>577.790849</td>
      <td>4.755576</td>
      <td>0.123465</td>
      <td>38.517561</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>ENSG00000224490.5</td>
      <td>TTC21B-AS1</td>
    </tr>
    <tr>
      <th>1744</th>
      <td>ENSG00000087494.16</td>
      <td>730.546390</td>
      <td>5.894148</td>
      <td>0.156149</td>
      <td>37.747042</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>ENSG00000087494.16</td>
      <td>PTHLH</td>
    </tr>
    <tr>
      <th>13974</th>
      <td>ENSG00000177464.5</td>
      <td>898.444596</td>
      <td>3.328664</td>
      <td>0.086959</td>
      <td>38.278497</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>ENSG00000177464.5</td>
      <td>GPR4</td>
    </tr>
    <tr>
      <th>8122</th>
      <td>ENSG00000142319.18</td>
      <td>2490.584653</td>
      <td>7.940965</td>
      <td>0.193696</td>
      <td>40.997152</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>ENSG00000142319.18</td>
      <td>SLC6A3</td>
    </tr>
    <tr>
      <th>8244</th>
      <td>ENSG00000143248.13</td>
      <td>35522.914658</td>
      <td>3.997299</td>
      <td>0.102497</td>
      <td>38.999202</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>300.000000</td>
      <td>ENSG00000143248.13</td>
      <td>RGS5</td>
    </tr>
  </tbody>
</table>
<p>44007 rows × 10 columns</p>
</div>



## 4.2.1.2 Set outlier threshold for log2FoldChange and nlog10 padj using tukey's fences 

Tukey's fence's outlier thresholds are as follows: 

General form: $Outlier_{x} = k \times (q_{75} - q_{25}) + q_{75}$

- Upper outlier threshold: k = 3.0
- middle outlier threshold: k = 1.5 
- lower outlier threshold: k = 1.0


```python
k = 3.0
log2_threshold = k*(np.percentile(results_df_filtered['log2FoldChange'], 75) - np.percentile(results_df_filtered['log2FoldChange'], 25)) + np.percentile(results_df_filtered['log2FoldChange'], 75)
log10_threshold = k*(np.percentile(results_df_filtered['nlog10'], 75) - np.percentile(results_df_filtered['nlog10'], 25)) + np.percentile(results_df_filtered['nlog10'], 75)
```


```python
print(f"The threshold for log2FoldChange is {log2_threshold}")
print(f"The threshold for nlog10 is {log10_threshold}")
```

    The threshold for log2FoldChange is 3.0146824737036013
    The threshold for nlog10 is 61.22039206963072



```python
#picked1 and picked2 simulate user lists of genes to label by color
picked1 = random.choices(population=results_df_filtered.gene_name.tolist(), weights = results_df_filtered.nlog10.tolist(), k = 250)
picked2 = random.choices(population=results_df_filtered.gene_name.tolist(), weights = results_df_filtered.nlog10.tolist(), k = 300)
picked2 = [x for x in picked2 if x not in picked1]

def map_color(a):
    log2FoldChange, gene_name, nlog10 = a
    
    if abs(log2FoldChange) < log2_threshold or nlog10 < log10_threshold:
        return 'Not significant'
    if gene_name in picked1:
        return 'picked1'
    if gene_name in picked2:
        return 'picked2'
    
    return 'Cherry picked'

results_df_filtered['color'] = results_df_filtered[['log2FoldChange', 'gene_name', 'nlog10']].apply(map_color, axis = 1)

#picked3 and picked24 simulate user lists of genes to label by shape
df = results_df_filtered.copy()
picked3 = random.choices(df.gene_name.tolist(), weights = df.nlog10.tolist(), k = 250)
picked4 = random.choices(df.gene_name.tolist(), weights = df.nlog10.tolist(), k = 300)
picked4 = [x for x in picked4 if x not in picked3]

def map_shape(symbol):
    if symbol in picked3:
        return 'picked3'
    if symbol in picked4:
        return 'picked4'
    
    return 'not_important'

df['shape'] = df.gene_name.map(map_shape)
df.head()
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
      <th>index</th>
      <th>baseMean</th>
      <th>log2FoldChange</th>
      <th>lfcSE</th>
      <th>stat</th>
      <th>pvalue</th>
      <th>padj</th>
      <th>nlog10</th>
      <th>gene_id</th>
      <th>gene_name</th>
      <th>color</th>
      <th>shape</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ENSG00000000003.15</td>
      <td>1745.846969</td>
      <td>-0.229056</td>
      <td>0.054051</td>
      <td>-4.237788</td>
      <td>2.257328e-05</td>
      <td>4.716019e-05</td>
      <td>4.326424</td>
      <td>ENSG00000000003.15</td>
      <td>TSPAN6</td>
      <td>Not significant</td>
      <td>not_important</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ENSG00000000005.6</td>
      <td>24.765175</td>
      <td>1.245152</td>
      <td>0.164656</td>
      <td>7.562141</td>
      <td>3.964869e-14</td>
      <td>1.391292e-13</td>
      <td>12.856582</td>
      <td>ENSG00000000005.6</td>
      <td>TNMD</td>
      <td>Not significant</td>
      <td>not_important</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ENSG00000000419.13</td>
      <td>688.768193</td>
      <td>0.013235</td>
      <td>0.028567</td>
      <td>0.463276</td>
      <td>6.431665e-01</td>
      <td>7.231618e-01</td>
      <td>0.140765</td>
      <td>ENSG00000000419.13</td>
      <td>DPM1</td>
      <td>Not significant</td>
      <td>not_important</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ENSG00000000457.14</td>
      <td>456.921769</td>
      <td>0.171048</td>
      <td>0.031631</td>
      <td>5.407658</td>
      <td>6.385407e-08</td>
      <td>1.593618e-07</td>
      <td>6.797616</td>
      <td>ENSG00000000457.14</td>
      <td>SCYL3</td>
      <td>Not significant</td>
      <td>not_important</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ENSG00000000460.17</td>
      <td>280.125529</td>
      <td>0.348439</td>
      <td>0.041999</td>
      <td>8.296374</td>
      <td>1.073368e-16</td>
      <td>4.215969e-16</td>
      <td>15.375103</td>
      <td>ENSG00000000460.17</td>
      <td>C1orf112</td>
      <td>Not significant</td>
      <td>not_important</td>
    </tr>
  </tbody>
</table>
</div>



## 4.2.2. Plot differentially expressed genes

## 4.2.2.1 FUll volcano plot


```python
ax1 = sns.scatterplot(data = df, x = 'log2FoldChange', y = 'nlog10',
                    hue = 'color', hue_order = ['Not significant', 'picked1', 'picked2', 'Cherry picked'],
                    palette = ['lightgrey', 'orange', 'purple', 'grey'],
                    style = 'shape', 
                    style_order = ['picked3', 'picked4', 'not_important'],
                    markers = ['^', 's', 'o'], 
                    size = 'baseMean', sizes = (40, 400),
                    )

```


    
![png](/ProjectDocs/DeSeqApplication/images/TCGA_Deseq2_analysis_files/TCGA_Deseq2_analysis_33_0.png)
    


## 4.2.2.2 Zoomed in and well annotated plot plot


```python
plt.figure(figsize = (6,6))
ax = sns.scatterplot(data = df, x = 'log2FoldChange', y = 'nlog10',
                    hue = 'color', hue_order = ['Not significant', 'picked1', 'picked2', 'Cherry picked'],
                    palette = ['lightgrey', 'orange', 'purple', 'grey'],
                    style = 'shape', 
                    style_order = ['picked3', 'picked4', 'not_important'],
                    markers = ['^', 's', 'o'], 
                    size = 'baseMean', sizes = (40, 400))

ax.axhline(log10_threshold, zorder = 0, c = 'k', lw = 2, ls = '--')
ax.axvline(log2_threshold, zorder = 0, c = 'k', lw = 2, ls = '--')
ax.axvline(-log2_threshold, zorder = 0, c = 'k', lw = 2, ls = '--')



## print texts for only very top genes
texts = []
for i in range(len(df)):
    if df.iloc[i].nlog10 > 150 and abs(df.iloc[i].log2FoldChange) > 6.5:
        texts.append(plt.text(x = df.iloc[i].log2FoldChange, y = df.iloc[i].nlog10, s = df.iloc[i].gene_name,
                             fontsize = 8, weight = 'bold'))
        
# adjust_text(texts, arrowprops = dict(arrowstyle = '-', color = 'k'))

plt.legend(loc = 1, bbox_to_anchor = (1.4,1), frameon = False, prop = {'weight':'bold'})

for axis in ['bottom', 'left']:
    ax.spines[axis].set_linewidth(2)
    
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.tick_params(width = 2)
ax.set_xlim(-10, 10)
ax.set_ylim(100, 400)
plt.xticks(size = 12, weight = 'bold')
plt.yticks(size = 12, weight = 'bold')

plt.xlabel("$log_{2}$ fold change", size = 15)
plt.ylabel("-$log_{10}$ FDR", size = 15)
plt.title('Differential gene expression in TCGA-KIRC vs TCGA-KIRP', weight='bold', size = 15)
plt.savefig('volcano.png', dpi = 300, bbox_inches = 'tight', facecolor = 'white')

plt.show()
```


    
![png](/ProjectDocs/DeSeqApplication/images/TCGA_Deseq2_analysis_files/TCGA_Deseq2_analysis_35_0.png)
    



```python

```
