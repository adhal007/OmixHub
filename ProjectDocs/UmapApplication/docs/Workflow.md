```python
import OmicsUtils.DimRedMappers.clusterer
import OmicsUtils.DimRedMappers.umap_embedders
from importlib import reload
reload(OmicsUtils.DimRedMappers.clusterer)
reload(OmicsUtils.DimRedMappers.umap_embedders)

import pandas as pd 
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
```

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm


## I. Data Preprocessing and transformation

## 1.1 Read and wrangle the data to get gene expression features and clinical data into a single dataframe


```python
## Read labels and gene expression data 
fpkm_unstr_df_with_labels = pd.read_csv('./Transcriptomics/data/processed_data/fpkm_unstr_data_with_labels.csv')
gene_cols = fpkm_unstr_df_with_labels.columns.to_numpy()[:60660]
exposure_tsv = pd.read_csv('./Transcriptomics/data/clinical.cart.2023-10-29/exposure.tsv', sep='\t')


## some columns have whitespace that needs to be removed for better processing
new_columns = fpkm_unstr_df_with_labels.columns.str.replace(' ', '_').to_numpy()
fpkm_unstr_df_with_labels.columns = new_columns 

## Merge exposure data to get labels for tumor type and kidney subtype
ge_kidney_cancer_data_with_tgca_labels = pd.merge(fpkm_unstr_df_with_labels,
                                                  exposure_tsv[['case_submitter_id', 'project_id']],
                                                  left_on='Case_ID',
                                                  right_on='case_submitter_id')
```

## 1.2. Data transformation

Here, given we have RNA-Seq Gene expression data we apply some commonly applied data transformations

1. **Log transformation**: Gene expression values have a very wide range from (0, 500000) usually in the form of a spike-slab distribution. Hence need to tighten the range with a log transform
2. **Subsetting**: For purposes of this analysis we only want to look at 3 sub-types of kidney cancer samples ('TCGA-KIRC', 'TCGA-KICH', 'TCGA-KIRP')
3. **Label conversion**: Converting categorical labels into numerical labels for compatibility with machine learning algorithms


```python
description_df = fpkm_unstr_df_with_labels.iloc[:, :60660].describe().reset_index()
showing_value_range = description_df[description_df['index'].isin(['min', 'mean', 'max'])].iloc[:,1:].values.flatten()

```


```python
ax = sns.distplot(showing_value_range)
ax.set_title("Distribution of min, mean, max of all gene expression values" )
```

    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/796282352.py:1: UserWarning: 
    
    `distplot` is a deprecated function and will be removed in seaborn v0.14.0.
    
    Please adapt your code to use either `displot` (a figure-level function with
    similar flexibility) or `histplot` (an axes-level function for histograms).
    
    For a guide to updating your code to use the new functions, please see
    https://gist.github.com/mwaskom/de44147ed2974457ad6372750bbe5751
    
      ax = sns.distplot(showing_value_range)





    Text(0.5, 1.0, 'Distribution of min, mean, max of all gene expression values')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_6_2.png)
    



```python

## Subset only 3 types of cancer sub-type from the dataset 
ge_kidney_cancer_data_correct_labels = (ge_kidney_cancer_data_with_tgca_labels[ge_kidney_cancer_data_with_tgca_labels['project_id'].isin(
    ['TCGA-KIRC', 'TCGA-KICH', 'TCGA-KIRP']
)]
)

## Apply log transformation
transformer = FunctionTransformer(np.log10)
ge_kidney_cancer_data_correct_labels[gene_cols] = ge_kidney_cancer_data_correct_labels[gene_cols] + 1
ge_kidney_cancer_data_correct_labels[gene_cols] = transformer.fit_transform(ge_kidney_cancer_data_correct_labels[gene_cols])

## Convert categorical labels to numerical labels 
columns_for_one_hot = ['project_id', 'Sample_Type']
ml_df = pd.get_dummies(ge_kidney_cancer_data_correct_labels, columns=['project_id', 'Sample_Type'], prefix=['cancer_subtype', 'tumor_subtype'])
```

    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/1726230766.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ge_kidney_cancer_data_correct_labels[gene_cols] = ge_kidney_cancer_data_correct_labels[gene_cols] + 1
    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/1726230766.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ge_kidney_cancer_data_correct_labels[gene_cols] = transformer.fit_transform(ge_kidney_cancer_data_correct_labels[gene_cols])



```python
columns_for_one_hot = ['project_id', 'Sample_Type']
ge_kidney_cancer_data_correct_labels['project_id_orig'] =ge_kidney_cancer_data_correct_labels['project_id']
ge_kidney_cancer_data_correct_labels['Sample_Type_orig'] = ge_kidney_cancer_data_correct_labels['Sample_Type']
ml_df = pd.get_dummies(ge_kidney_cancer_data_correct_labels, columns=['project_id', 'Sample_Type'], prefix=['cancer_subtype', 'tumor_subtype'])

```

    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/2156470527.py:2: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
      ge_kidney_cancer_data_correct_labels['project_id_orig'] =ge_kidney_cancer_data_correct_labels['project_id']
    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/2156470527.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ge_kidney_cancer_data_correct_labels['project_id_orig'] =ge_kidney_cancer_data_correct_labels['project_id']
    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/2156470527.py:3: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
      ge_kidney_cancer_data_correct_labels['Sample_Type_orig'] = ge_kidney_cancer_data_correct_labels['Sample_Type']
    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/2156470527.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ge_kidney_cancer_data_correct_labels['Sample_Type_orig'] = ge_kidney_cancer_data_correct_labels['Sample_Type']


## II. Exploratory Analysis

## 2.1 Check overall info of datatypes, null values


```python
ge_kidney_cancer_data_correct_labels.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 924 entries, 2 to 1293
    Columns: 60673 entries, ENSG00000000003.15 to Sample_Type_orig
    dtypes: float64(60660), object(13)
    memory usage: 427.7+ MB


## 2.2. Check Unique sample IDs


```python
print(f"The total patient ids are {ge_kidney_cancer_data_correct_labels['Case_ID'].count()}, from those the unique ids are {ge_kidney_cancer_data_correct_labels['Case_ID'].value_counts().shape[0]} ")
```

    The total patient ids are 924, from those the unique ids are 799 


## 2.3. Check number of data labels by project type and tumor type


```python
ge_kidney_cancer_data_correct_labels.groupby(['Sample_Type'])['Sample_Type'].count()
```




    Sample_Type
    Additional - New Primary      2
    Primary Tumor               800
    Solid Tissue Normal         122
    Name: Sample_Type, dtype: int64




```python
ax = sns.countplot(x='Sample_Type', data=ge_kidney_cancer_data_correct_labels)
ax.bar_label(ax.containers[0])

```




    [Text(0, 0, '800'), Text(0, 0, '122'), Text(0, 0, '2')]




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_16_1.png)
    



```python
ge_kidney_cancer_data_correct_labels.groupby(['project_id'])['project_id'].count()
```




    project_id
    TCGA-KICH     84
    TCGA-KIRC    539
    TCGA-KIRP    301
    Name: project_id, dtype: int64




```python
ax = sns.countplot(x='project_id', data=ge_kidney_cancer_data_correct_labels)
ax.bar_label(ax.containers[0])

```




    [Text(0, 0, '539'), Text(0, 0, '301'), Text(0, 0, '84')]




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_18_1.png)
    


## III. Data sampling, Augmentation and splitting into training, testing and validation sets for Supervised Clustering 

Given that we have imbalanced datasets for Tumor Types and Kidney Cancer subtypes (project_id), we should consider the following for an **unbiased analysis** 

1. **Patient overlap**: Given 924 total patients with 799 unique, 
   
   <br>

   1. We should keep same ID patients in only one set i.e (Training or Validation or Testing)
   2. Preferably, we should have have all overlapping patients in Testing so as to not bias our training 

<br>

2. **Set Sampling**: For minority classes, we sample atleast X% into our testing first followed by validation and then all the remaining in training.

3. Additionally for one of the minority classes (New Primary - we should group it into the same category as Primary)

4. **Data augmention**: Applying oversampling, undersampling techniques for minority or majority classes 

## 3.1. Label correction of Additional - New Primary tumor


```python
ge_kidney_cancer_data_correct_labels['Sample_Type'] = ge_kidney_cancer_data_correct_labels['Sample_Type'].apply(lambda x: 'Primary Tumor' if x == 'Additional - New Primary' else x)
```

    /var/folders/ng/bwk7d4ds7wz95l011dbvtc9r0000gn/T/ipykernel_2912/2849659451.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ge_kidney_cancer_data_correct_labels['Sample_Type'] = ge_kidney_cancer_data_correct_labels['Sample_Type'].apply(lambda x: 'Primary Tumor' if x == 'Additional - New Primary' else x)



```python
ge_kidney_cancer_data_correct_labels.groupby(['Sample_Type'])['Sample_Type'].count()
```




    Sample_Type
    Primary Tumor          802
    Solid Tissue Normal    122
    Name: Sample_Type, dtype: int64




```python
ax = sns.countplot(x='Sample_Type', data=ge_kidney_cancer_data_correct_labels)
ax.bar_label(ax.containers[0])

```




    [Text(0, 0, '802'), Text(0, 0, '122')]




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_23_1.png)
    


## 3.2 Separation of training, validation and testing data for unbiased analysis

Strategy
1. Find pairs of all duplicate sample IDs and randomly sample pairs 70% in training, 20% in validation and 10% in testing (with seed)
2. Out of the remaining samples For the 2 lowest groups KICH (10% - 84 samples) and Solid tissue(13% - 122 samples) put more than 25% samples into testing set first 
   1. Take union of 21 samples from KICH (25%) and 30 samples from Solid tissue (25%)  and put all those samples into testing set first
   2. Hold out this set till model training has occurred 
   3. Out of the remaining samples, again put more than 25% samples from minority classes into validation
   4. Put all the remaining into training 



```python
from sklearn.model_selection import train_test_split

```


```python
y_true = ml_df.iloc[:, -6:].drop(['tumor_subtype_Additional - New Primary'], axis=1).astype(int).to_numpy()
```


```python
# Count of each unique value
value_counts = ml_df['Case_ID'].value_counts()

# list of duplicate ids
duplicate_ids = value_counts[value_counts > 1].reset_index()['Case_ID'].unique()
```

## 3.2.1 Separating patient overlap samples


```python
# First split: Separate out a test set
train_val_ids, test_overlap_ids = train_test_split(duplicate_ids, test_size=0.1, random_state=42)

# Second split: Separate the remaining data into training and validation sets
train_overlap_ids, val_overlap_ids = train_test_split(train_val_ids, test_size=0.2, random_state=42) # 0.25 x 0.8 = 0.2

print(f"Number of Patient overlap Training IDs: {len(train_overlap_ids)}")
print(f"Number of Patient overlap Validation IDs: {len(val_overlap_ids)}")
print(f"Number of Patient overlap Testing IDs: {len(test_overlap_ids)}")
```

    Number of Patient overlap Training IDs: 87
    Number of Patient overlap Validation IDs: 22
    Number of Patient overlap Testing IDs: 13


## 3.2.2 Set sampling

Shows that majority of duplicate samples came from the Solid tissue class


```python
ml_df_without_overlap = ml_df[~ml_df['Case_ID'].isin(duplicate_ids)].reset_index(drop=True)
```


```python
ax1 = sns.countplot(x='Sample_Type', data=ge_kidney_cancer_data_correct_labels)
ax1.bar_label(ax1.containers[0])
ax1.set_title("Count of samples originally across tumors")
```




    Text(0.5, 1.0, 'Count of samples originally across tumors')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_32_1.png)
    



```python
ax = sns.countplot(x='Sample_Type_orig', data=ml_df_without_overlap )
ax.bar_label(ax.containers[0])
ax.set_title("Count of samples after separating patient overlap samples")

```




    Text(0.5, 1.0, 'Count of samples after separating patient overlap samples')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_33_1.png)
    



```python
ax = sns.countplot(x='project_id', data=ge_kidney_cancer_data_correct_labels)
ax.bar_label(ax.containers[0])
ax.set_title("Samples Count in Kidney cancer(KC) subtypes Originally")
```




    Text(0.5, 1.0, 'Samples Count in Kidney cancer(KC) subtypes Originally')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_34_1.png)
    



```python
ax = sns.countplot(x='project_id_orig', data=ml_df_without_overlap)
ax.bar_label(ax.containers[0])
ax.set_title("Samples count in KC subtypes after separating patient overlap")
```




    Text(0.5, 1.0, 'Samples count in KC subtypes after separating patient overlap')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_35_1.png)
    



```python
all_minority_samples1 = ml_df_without_overlap[(ml_df_without_overlap['Sample_Type_orig'] == 'Solid Tissue Normal')
| (ml_df_without_overlap['project_id_orig'] == 'TCGA-KICH')]['Case_ID'].to_numpy()

# First split: Separate out a test set with 50% 
train_val_set_sampl_ids1, test_set_samp_ids1 = train_test_split(all_minority_samples1, test_size=0.5, random_state=42)

# Second split: Separate the remaining data into training and validation sets
train_set_samp_ids1, val_set_samp_ids1 = train_test_split(train_val_set_sampl_ids1, test_size=0.5, random_state=42) 

print(f"Number of Minority set sampled Training IDs: {len(train_set_samp_ids1)}")
print(f"Number of Minority set sampled  Validation IDs: {len(val_set_samp_ids1)}")
print(f"Number of Minority set sampled  Testing IDs: {len(test_set_samp_ids1)}")
```

    Number of Minority set sampled Training IDs: 10
    Number of Minority set sampled  Validation IDs: 11
    Number of Minority set sampled  Testing IDs: 21



```python
rem_majority_samples = ml_df_without_overlap[~ml_df_without_overlap['Case_ID'].isin(all_minority_samples1)]['Case_ID'].to_numpy()

# First split: Separate out a test set
train_val_major_ids1, test_major_ids1 = train_test_split(rem_majority_samples, test_size=0.1, random_state=42)

# Second split: Separate the remaining data into training and validation sets
train_major_ids1, val_major_ids1 = train_test_split(train_val_major_ids1, test_size=0.2, random_state=42) # 0.25 x 0.8 = 0.2

print(f"Number of majority classes Training IDs: {len(train_major_ids1)}")
print(f"Number of majority classes Validation IDs: {len(val_major_ids1)}")
print(f"Number of majority classes Testing IDs: {len(test_major_ids1)}")
```

    Number of majority classes Training IDs: 456
    Number of majority classes Validation IDs: 115
    Number of majority classes Testing IDs: 64



```python
train_ids_df = pd.DataFrame({'Case_ID': list(train_overlap_ids) + list(train_major_ids1) + list(train_set_samp_ids1)}).assign(label='training')
val_ids_df = pd.DataFrame({'Case_ID': list(val_overlap_ids) + list(val_major_ids1) + list(val_set_samp_ids1)}).assign(label='validation')
test_ids_df = pd.DataFrame({'Case_ID': list(test_overlap_ids) + list(test_major_ids1) + list(test_set_samp_ids1)}).assign(label='testing') 
train_val_test_ids_df = pd.concat([train_ids_df, val_ids_df, test_ids_df], axis=0)
```


```python
train_val_test_ids_df = pd.merge(train_val_test_ids_df, ml_df[['Case_ID', 'project_id_orig', 'Sample_Type_orig']],
left_on='Case_ID',
right_on='Case_ID')
```


```python
train_val_test_ids_df 
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
      <th>Case_ID</th>
      <th>label</th>
      <th>project_id_orig</th>
      <th>Sample_Type_orig</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TCGA-CW-5585</td>
      <td>training</td>
      <td>TCGA-KIRC</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TCGA-CW-5585</td>
      <td>training</td>
      <td>TCGA-KIRC</td>
      <td>Solid Tissue Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TCGA-B0-5706</td>
      <td>training</td>
      <td>TCGA-KIRC</td>
      <td>Solid Tissue Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TCGA-B0-5706</td>
      <td>training</td>
      <td>TCGA-KIRC</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TCGA-DZ-6132</td>
      <td>training</td>
      <td>TCGA-KIRP</td>
      <td>Solid Tissue Normal</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>919</th>
      <td>TCGA-KN-8418</td>
      <td>testing</td>
      <td>TCGA-KICH</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>920</th>
      <td>TCGA-KO-8406</td>
      <td>testing</td>
      <td>TCGA-KICH</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>921</th>
      <td>TCGA-KO-8414</td>
      <td>testing</td>
      <td>TCGA-KICH</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>922</th>
      <td>TCGA-KL-8325</td>
      <td>testing</td>
      <td>TCGA-KICH</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>923</th>
      <td>TCGA-KM-8438</td>
      <td>testing</td>
      <td>TCGA-KICH</td>
      <td>Primary Tumor</td>
    </tr>
  </tbody>
</table>
<p>924 rows × 4 columns</p>
</div>




```python
train_val_test_ids_df[['label', 'project_id_orig']].value_counts()
```




    label       project_id_orig
    training    TCGA-KIRC          394
                TCGA-KIRP          215
    validation  TCGA-KIRC           96
                TCGA-KIRP           51
    testing     TCGA-KIRC           49
                TCGA-KIRP           35
    training    TCGA-KICH           34
    testing     TCGA-KICH           27
    validation  TCGA-KICH           23
    Name: count, dtype: int64




```python
train_val_test_ids_df[['label', 'Sample_Type_orig']].value_counts()
```




    label       Sample_Type_orig        
    training    Primary Tumor               556
    validation  Primary Tumor               148
    testing     Primary Tumor                96
    training    Solid Tissue Normal          86
    validation  Solid Tissue Normal          21
    testing     Solid Tissue Normal          15
    training    Additional - New Primary      1
    validation  Additional - New Primary      1
    Name: count, dtype: int64




```python
print(f"Final train-validation-test split dataframe has {train_val_test_ids_df.shape[0]} unique samples")
```

    Final train-validation-test split dataframe has 924 unique samples


## IV. Supervised Classification using optimized UMap embedddings and clustering

Using training set samples run bayesian optimizer to find the best parameters for 1) Umap (n_components, n_neighbors) 2) HdbScan (min_cluster_size) 


```python
clustering_optimizer = OmicsUtils.DimRedMappers.clusterer.ClusteringOptimizer(data=ge_kidney_cancer_data_correct_labels[gene_cols],
                                                                              )
```


```python
ge_kidney_cancer_data_correct_labels[gene_cols].to_numpy()
```




    array([[9.35477035e-01, 1.23067244e-01, 1.36078458e+00, ...,
            0.00000000e+00, 1.13589537e-02, 1.29818744e-01],
           [1.18591878e+00, 1.22465200e-02, 1.41848193e+00, ...,
            0.00000000e+00, 9.54398406e-04, 1.58332332e-01],
           [1.22948986e+00, 4.95363922e-01, 1.27373027e+00, ...,
            0.00000000e+00, 3.50361474e-03, 1.58625081e-02],
           ...,
           [1.15416521e+00, 4.16457433e-01, 1.25601981e+00, ...,
            0.00000000e+00, 3.93420617e-03, 3.61096671e-02],
           [9.14723857e-01, 5.86157970e-02, 1.51866263e+00, ...,
            0.00000000e+00, 7.15010537e-03, 3.39411687e-01],
           [1.32102645e+00, 8.96578816e-02, 1.28390927e+00, ...,
            0.00000000e+00, 2.03640226e-03, 7.48164406e-02]])




```python
X = ge_kidney_cancer_data_correct_labels[gene_cols]
label_lower=30
label_upper=100
max_evals = 100
best_param, best_clusters, trials = clustering_optimizer.bayesian_search(embeddings=ge_kidney_cancer_data_correct_labels[gene_cols].to_numpy(), 
                                     label_lower=label_lower,
                                     label_upper=label_upper,
                                     max_evals=max_evals)
```

      0%|          | 0/100 [00:00<?, ?trial/s, best loss=?]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      1%|          | 1/100 [00:27<44:58, 27.26s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      2%|▏         | 2/100 [00:52<42:42, 26.14s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      3%|▎         | 3/100 [01:19<42:42, 26.42s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      4%|▍         | 4/100 [01:45<41:55, 26.20s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      5%|▌         | 5/100 [02:12<41:53, 26.46s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      6%|▌         | 6/100 [02:37<41:03, 26.21s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      7%|▋         | 7/100 [03:03<40:24, 26.07s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      8%|▊         | 8/100 [03:28<39:36, 25.83s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


      9%|▉         | 9/100 [03:54<38:48, 25.59s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     10%|█         | 10/100 [04:18<38:03, 25.37s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     11%|█         | 11/100 [04:44<37:41, 25.41s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     12%|█▏        | 12/100 [05:09<37:11, 25.36s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     13%|█▎        | 13/100 [05:35<37:02, 25.55s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     14%|█▍        | 14/100 [06:01<36:47, 25.67s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     15%|█▌        | 15/100 [06:27<36:26, 25.72s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     16%|█▌        | 16/100 [06:52<35:53, 25.64s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     17%|█▋        | 17/100 [07:18<35:29, 25.66s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     18%|█▊        | 18/100 [07:44<35:04, 25.66s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     19%|█▉        | 19/100 [08:10<34:42, 25.71s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     20%|██        | 20/100 [08:35<34:19, 25.75s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     21%|██        | 21/100 [09:01<33:54, 25.75s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     22%|██▏       | 22/100 [09:27<33:31, 25.79s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     23%|██▎       | 23/100 [09:52<32:57, 25.68s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     24%|██▍       | 24/100 [10:18<32:27, 25.62s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     25%|██▌       | 25/100 [10:45<32:35, 26.07s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     26%|██▌       | 26/100 [11:11<32:09, 26.08s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     27%|██▋       | 27/100 [11:36<31:23, 25.80s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     28%|██▊       | 28/100 [12:01<30:36, 25.51s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     29%|██▉       | 29/100 [12:26<29:56, 25.31s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     30%|███       | 30/100 [12:52<29:45, 25.51s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     31%|███       | 31/100 [13:18<29:31, 25.67s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     32%|███▏      | 32/100 [13:44<29:21, 25.90s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     33%|███▎      | 33/100 [14:18<31:21, 28.08s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     34%|███▍      | 34/100 [14:53<33:22, 30.34s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     35%|███▌      | 35/100 [15:25<33:26, 30.88s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     36%|███▌      | 36/100 [15:56<32:54, 30.86s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     37%|███▋      | 37/100 [16:25<31:54, 30.39s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     38%|███▊      | 38/100 [16:54<30:44, 29.75s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     39%|███▉      | 39/100 [17:22<29:42, 29.22s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     40%|████      | 40/100 [17:49<28:45, 28.76s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     41%|████      | 41/100 [18:17<27:55, 28.39s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     42%|████▏     | 42/100 [18:44<27:07, 28.07s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     43%|████▎     | 43/100 [19:10<25:52, 27.24s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     44%|████▍     | 44/100 [19:37<25:26, 27.25s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     45%|████▌     | 45/100 [20:06<25:27, 27.77s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     46%|████▌     | 46/100 [20:35<25:21, 28.18s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     47%|████▋     | 47/100 [21:04<25:08, 28.47s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     48%|████▊     | 48/100 [21:33<24:40, 28.48s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     49%|████▉     | 49/100 [22:01<24:15, 28.54s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     50%|█████     | 50/100 [22:30<23:48, 28.56s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     51%|█████     | 51/100 [22:58<23:13, 28.44s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     52%|█████▏    | 52/100 [23:26<22:42, 28.39s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     53%|█████▎    | 53/100 [23:54<22:05, 28.20s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     54%|█████▍    | 54/100 [24:22<21:33, 28.13s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     55%|█████▌    | 55/100 [24:50<21:04, 28.11s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     56%|█████▌    | 56/100 [25:18<20:37, 28.13s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     57%|█████▋    | 57/100 [25:46<20:07, 28.09s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     58%|█████▊    | 58/100 [26:15<19:41, 28.14s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     59%|█████▉    | 59/100 [26:43<19:11, 28.09s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     60%|██████    | 60/100 [27:11<18:46, 28.17s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     61%|██████    | 61/100 [27:39<18:16, 28.13s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     62%|██████▏   | 62/100 [28:07<17:50, 28.18s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     63%|██████▎   | 63/100 [28:36<17:26, 28.28s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     64%|██████▍   | 64/100 [29:04<17:02, 28.41s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     65%|██████▌   | 65/100 [29:33<16:34, 28.42s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     66%|██████▌   | 66/100 [30:01<16:07, 28.44s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     67%|██████▋   | 67/100 [30:30<15:45, 28.64s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     68%|██████▊   | 68/100 [31:00<15:28, 29.03s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     69%|██████▉   | 69/100 [31:30<15:07, 29.28s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     70%|███████   | 70/100 [31:59<14:35, 29.19s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     71%|███████   | 71/100 [32:28<14:02, 29.06s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     72%|███████▏  | 72/100 [32:56<13:26, 28.81s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     73%|███████▎  | 73/100 [33:25<12:53, 28.65s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     74%|███████▍  | 74/100 [33:53<12:23, 28.58s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     75%|███████▌  | 75/100 [34:21<11:49, 28.40s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     76%|███████▌  | 76/100 [34:49<11:22, 28.43s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     77%|███████▋  | 77/100 [35:17<10:50, 28.26s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     78%|███████▊  | 78/100 [35:45<10:18, 28.13s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     79%|███████▉  | 79/100 [36:13<09:48, 28.00s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     80%|████████  | 80/100 [36:40<09:17, 27.90s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     81%|████████  | 81/100 [37:08<08:50, 27.93s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     82%|████████▏ | 82/100 [37:36<08:22, 27.93s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     83%|████████▎ | 83/100 [38:04<07:53, 27.88s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     84%|████████▍ | 84/100 [38:33<07:29, 28.08s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     85%|████████▌ | 85/100 [39:01<07:00, 28.01s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     86%|████████▌ | 86/100 [39:28<06:31, 27.98s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     87%|████████▋ | 87/100 [39:56<06:02, 27.91s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     88%|████████▊ | 88/100 [40:24<05:35, 27.96s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     89%|████████▉ | 89/100 [40:52<05:06, 27.87s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     90%|█████████ | 90/100 [41:20<04:40, 28.07s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     91%|█████████ | 91/100 [41:49<04:13, 28.16s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     92%|█████████▏| 92/100 [42:17<03:45, 28.15s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     93%|█████████▎| 93/100 [42:45<03:16, 28.12s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     94%|█████████▍| 94/100 [43:13<02:47, 27.98s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     95%|█████████▌| 95/100 [43:43<02:24, 28.81s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     96%|█████████▌| 96/100 [44:13<01:56, 29.09s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     97%|█████████▋| 97/100 [44:42<01:27, 29.16s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     98%|█████████▊| 98/100 [45:12<00:58, 29.15s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


     99%|█████████▉| 99/100 [45:41<00:29, 29.12s/trial, best loss: 0.05]

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")
    


    100%|██████████| 100/100 [46:12<00:00, 27.72s/trial, best loss: 0.05]
    best:
    {'min_cluster_size': 7, 'n_components': 6, 'n_neighbors': 5, 'random_state': 42}
    label count: 6


    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")


## Generate clusters using best params


```python
clusters, umap_embeddings = clustering_optimizer.generate_clusters(message_embeddings=ge_kidney_cancer_data_correct_labels[gene_cols].to_numpy(),
                        n_neighbors=5,
                        n_components=6,
                        clust_params={"min_cluster_size": 7},   
                        random_state = 42)
```

    /opt/homebrew/anaconda3/envs/umap-env/lib/python3.11/site-packages/umap/umap_.py:1943: UserWarning: n_jobs value -1 overridden to 1 by setting random_state. Use no seed for parallelism.
      warn(f"n_jobs value {self.n_jobs} overridden to 1 by setting random_state. Use no seed for parallelism.")



```python
ge_kidney_cancer_data_correct_labels
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
      <th>File ID</th>
      <th>File Name</th>
      <th>Data Category</th>
      <th>Data Type</th>
      <th>Project ID</th>
      <th>Case ID</th>
      <th>Sample ID</th>
      <th>Sample Type</th>
      <th>case_submitter_id</th>
      <th>project_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>0.935477</td>
      <td>0.123067</td>
      <td>1.360785</td>
      <td>0.452966</td>
      <td>0.211307</td>
      <td>1.220691</td>
      <td>1.203055</td>
      <td>1.257009</td>
      <td>0.630936</td>
      <td>0.760030</td>
      <td>...</td>
      <td>2f9b9698-edc6-4049-809c-9733f05f5c2b</td>
      <td>ed920009-6bce-4300-b928-14d0ec474ac6.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRC</td>
      <td>TCGA-MM-A563</td>
      <td>TCGA-MM-A563-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-MM-A563</td>
      <td>TCGA-KIRC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.185919</td>
      <td>0.012247</td>
      <td>1.418482</td>
      <td>0.222404</td>
      <td>0.065468</td>
      <td>0.344648</td>
      <td>0.111834</td>
      <td>1.503345</td>
      <td>0.875102</td>
      <td>0.890488</td>
      <td>...</td>
      <td>0a4e4402-13a5-4eea-b70c-b121bcf81156</td>
      <td>6abcbd13-ff11-46d6-9ce0-6204a6658c39.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRP</td>
      <td>TCGA-GL-8500</td>
      <td>TCGA-GL-8500-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-GL-8500</td>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.229490</td>
      <td>0.495364</td>
      <td>1.273730</td>
      <td>0.385338</td>
      <td>0.082606</td>
      <td>0.446848</td>
      <td>0.738614</td>
      <td>1.363431</td>
      <td>0.786524</td>
      <td>0.742269</td>
      <td>...</td>
      <td>091d9ed1-6127-4b37-b2d5-a463b28e2e9e</td>
      <td>c8d14d90-63a8-4576-ae80-ef8582792d16.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRP</td>
      <td>TCGA-BQ-5877</td>
      <td>TCGA-BQ-5877-11A</td>
      <td>Solid Tissue Normal</td>
      <td>TCGA-BQ-5877</td>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.326889</td>
      <td>0.000000</td>
      <td>1.241932</td>
      <td>0.471732</td>
      <td>0.225774</td>
      <td>0.794509</td>
      <td>1.370754</td>
      <td>1.592809</td>
      <td>0.618038</td>
      <td>1.189510</td>
      <td>...</td>
      <td>dcb89df3-3699-45fb-a636-9bff204ff377</td>
      <td>e47989a4-a2f7-46b7-9244-c91e0ae0ee0f.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRP</td>
      <td>TCGA-BQ-5877</td>
      <td>TCGA-BQ-5877-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-BQ-5877</td>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.490151</td>
      <td>0.052886</td>
      <td>1.323252</td>
      <td>0.518553</td>
      <td>0.125839</td>
      <td>0.342955</td>
      <td>0.294797</td>
      <td>1.262937</td>
      <td>0.675292</td>
      <td>0.695587</td>
      <td>...</td>
      <td>c405ed6b-d3db-4db2-9686-ff8c0a4089b7</td>
      <td>df99ecaa-143c-4dbf-91c5-8e3a76c60f74.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KICH</td>
      <td>TCGA-KN-8423</td>
      <td>TCGA-KN-8423-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-KN-8423</td>
      <td>TCGA-KICH</td>
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
      <th>1287</th>
      <td>1.236482</td>
      <td>0.123035</td>
      <td>1.469803</td>
      <td>0.588025</td>
      <td>0.263423</td>
      <td>0.967061</td>
      <td>0.726124</td>
      <td>1.423254</td>
      <td>0.807474</td>
      <td>1.037223</td>
      <td>...</td>
      <td>68b2e11e-37c9-4e2b-a532-fa8f91424694</td>
      <td>e045554d-52a6-4e83-8ca1-0cbb75a2a68b.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRP</td>
      <td>TCGA-B3-3925</td>
      <td>TCGA-B3-3925-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-B3-3925</td>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>1290</th>
      <td>1.267758</td>
      <td>0.074999</td>
      <td>1.257655</td>
      <td>0.242019</td>
      <td>0.086467</td>
      <td>0.450988</td>
      <td>0.394469</td>
      <td>1.383887</td>
      <td>0.795908</td>
      <td>0.582881</td>
      <td>...</td>
      <td>a4ac446e-4c70-4360-9e88-879641cff075</td>
      <td>dca406a5-9351-41b4-8cc6-40ed702103f6.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRP</td>
      <td>TCGA-A4-8518</td>
      <td>TCGA-A4-8518-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-A4-8518</td>
      <td>TCGA-KIRP</td>
    </tr>
    <tr>
      <th>1291</th>
      <td>1.154165</td>
      <td>0.416457</td>
      <td>1.256020</td>
      <td>0.440074</td>
      <td>0.208173</td>
      <td>0.667322</td>
      <td>1.144951</td>
      <td>1.280795</td>
      <td>0.678964</td>
      <td>0.964684</td>
      <td>...</td>
      <td>8c4bd061-8371-4313-bfdc-1b2213daea12</td>
      <td>d85f1c60-f3f3-4355-9f31-6c0e5d53360b.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRC</td>
      <td>TCGA-B8-5165</td>
      <td>TCGA-B8-5165-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-B8-5165</td>
      <td>TCGA-KIRC</td>
    </tr>
    <tr>
      <th>1292</th>
      <td>0.914724</td>
      <td>0.058616</td>
      <td>1.518663</td>
      <td>0.371622</td>
      <td>0.140225</td>
      <td>0.681187</td>
      <td>1.172004</td>
      <td>1.386880</td>
      <td>0.520064</td>
      <td>0.919235</td>
      <td>...</td>
      <td>ba209e1e-87f1-47a9-a858-d60953ef4528</td>
      <td>28089de8-830d-4774-937f-97d6dd7cfa0a.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRC</td>
      <td>TCGA-B0-4821</td>
      <td>TCGA-B0-4821-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-B0-4821</td>
      <td>TCGA-KIRC</td>
    </tr>
    <tr>
      <th>1293</th>
      <td>1.321026</td>
      <td>0.089658</td>
      <td>1.283909</td>
      <td>0.403275</td>
      <td>0.157880</td>
      <td>0.674815</td>
      <td>1.000130</td>
      <td>1.615682</td>
      <td>0.900159</td>
      <td>0.952187</td>
      <td>...</td>
      <td>4a2da9f0-b75d-466c-b9f0-2e910ee24365</td>
      <td>2cf32f8f-0698-453d-a96b-c66a48617d2d.rna_seq.a...</td>
      <td>Transcriptome Profiling</td>
      <td>Gene Expression Quantification</td>
      <td>TCGA-KIRC</td>
      <td>TCGA-AK-3456</td>
      <td>TCGA-AK-3456-01A</td>
      <td>Primary Tumor</td>
      <td>TCGA-AK-3456</td>
      <td>TCGA-KIRC</td>
    </tr>
  </tbody>
</table>
<p>924 rows × 60671 columns</p>
</div>



## Create dataframe of labels and cluster labels


```python
embed_x, embed_y = umap_embeddings[:, 0], umap_embeddings[:, 1]
umap_embedding_df =  pd.DataFrame({'embed_x': embed_x,
                                     'embed_y': embed_y}
                                     )

umap_embedding_df['project_labels'] = ge_kidney_cancer_data_correct_labels['project_id'].to_numpy()
umap_embedding_df['cluster_labels'] = clusters.labels_
umap_embedding_df['tumor_type_labels'] = ge_kidney_cancer_data_correct_labels['Sample Type'].to_numpy()
```


```python
umap_embedding_df
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
      <th>embed_x</th>
      <th>embed_y</th>
      <th>project_labels</th>
      <th>cluster_labels</th>
      <th>tumor_type_labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.298291</td>
      <td>6.447575</td>
      <td>TCGA-KIRC</td>
      <td>2</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.352697</td>
      <td>4.007873</td>
      <td>TCGA-KIRP</td>
      <td>3</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.777693</td>
      <td>0.079027</td>
      <td>TCGA-KIRP</td>
      <td>5</td>
      <td>Solid Tissue Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.620252</td>
      <td>4.031096</td>
      <td>TCGA-KIRP</td>
      <td>3</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.964766</td>
      <td>11.059839</td>
      <td>TCGA-KICH</td>
      <td>0</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>919</th>
      <td>9.150193</td>
      <td>3.324971</td>
      <td>TCGA-KIRP</td>
      <td>3</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>920</th>
      <td>10.274561</td>
      <td>3.604016</td>
      <td>TCGA-KIRP</td>
      <td>3</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>921</th>
      <td>10.978981</td>
      <td>6.363336</td>
      <td>TCGA-KIRC</td>
      <td>2</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>922</th>
      <td>9.980220</td>
      <td>6.798731</td>
      <td>TCGA-KIRC</td>
      <td>2</td>
      <td>Primary Tumor</td>
    </tr>
    <tr>
      <th>923</th>
      <td>11.133520</td>
      <td>3.426035</td>
      <td>TCGA-KIRC</td>
      <td>3</td>
      <td>Primary Tumor</td>
    </tr>
  </tbody>
</table>
<p>924 rows × 5 columns</p>
</div>



## Plot results


```python
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
ax3 = sns.scatterplot(data=umap_embedding_df,
x="embed_x", y="embed_y", hue="cluster_labels")
ax3.set_title("HDBSCAN Clustering with bayesian optimized clusters")


```




    Text(0.5, 1.0, 'HDBSCAN Clustering with bayesian optimized clusters')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_56_1.png)
    



```python
ax2 = sns.scatterplot(data = 
umap_embedding_df, x='embed_x', y='embed_y', hue='project_labels')
ax2.set_title("HDBSCAN Clustering with ground truth labels")
```




    Text(0.5, 1.0, 'HDBSCAN Clustering with ground truth labels')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_57_1.png)
    



```python
ax1 = sns.scatterplot(data = 
umap_embedding_df, x='embed_x', y='embed_y', hue='tumor_type_labels')
ax1.set_title("HDBSCAN Clustering with ground truth labels")

```




    Text(0.5, 1.0, 'HDBSCAN Clustering with ground truth labels')




    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_58_1.png)
    



```python
## combining plots


fig, axes = plt.subplots(3, 1, figsize=(10, 20))
 


sns.scatterplot(ax=axes[0], data=umap_embedding_df, x='embed_x', y='embed_y', hue='tumor_type_labels')
sns.scatterplot(ax=axes[1], data=umap_embedding_df, x='embed_x', y='embed_y', hue='cluster_labels')
sns.scatterplot(ax=axes[2], data=umap_embedding_df, x='embed_x', y='embed_y', hue='project_labels')
fig.suptitle('Umap(Cosine) + HdbScan on Kidney cancer GE data', fontsize = 20)
fig = axes[0].get_figure()
fig.tight_layout()
fig.subplots_adjust(top=0.95)

```


    
![png](/ProjectDocs/UmapApplication/images/TCGA_Semisupervised_clustering_files/TCGA_Semisupervised_clustering_59_0.png)
    



