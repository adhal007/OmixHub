## Template for creating RNA-Seq Data Matrix for different Primary sites


```python
%cd .. 

import grequests
import pandas as pd
import numpy as np
import src.Engines.gdc_engine as gdc_engine
import os
from importlib import reload
from flatten_json import flatten
from tqdm import tqdm 


reload(gdc_engine)
```

    /Users/abhilashdhal/Projects





    <module 'src.Engines.gdc_engine' from '/Users/abhilashdhal/Projects/src/Engines/gdc_engine.py'>



## 0.1. Get Metadata


```python

params = {
    'files.experimental_strategy': 'RNA-Seq', 
    'data_type': 'Gene Expression Quantification'
}

gdc_eng_inst = gdc_engine.GDCEngine(**params)
rna_seq_metadata = gdc_eng_inst._get_rna_seq_metadata()
meta = rna_seq_metadata['metadata']


```

    dict_keys(['files.experimental_strategy', 'data_type'])
    file_id,file_name,experimental_strategy,data_type,platform,cases.case_id,cases.diagnoses.last_known_disease_status,cases.diagnoses.primary_diagnosis,cases.diagnoses.tumor_stage,cases.diagnoses.tumor_grade,cases.diagnoses.treatments.treatment_or_therapy,cases.diagnoses.days_to_last_follow_up,cases.diagnoses.age_at_diagnosis,cases.diagnoses.days_to_death,cases.project.primary_site,analysis.workflow_type,cases.demographic.ethnicity,cases.demographic.gender,cases.demographic.race,cases.diagnoses.tissue_or_organ_of_origin,cases.exposures.bmi,cases.exposures.alcohol_history,cases.exposures.years_smoked,cases.samples.tissue_type


## 0.2. Print the count of primary_sites descending by most counts


```python
meta['primary_site'].value_counts()
```




    primary_site
    Blood             3564
    Kidney            1246
    Breast            1230
    Lung              1153
    Brain              703
    Colorectal         698
    Uterus             634
    Thyroid            572
    Head and Neck      566
    Prostate           554
    Skin               473
    Stomach            448
    Bladder            431
    Ovary              429
    Liver              424
    Lymph Nodes        398
    Cervix             309
    Adrenal Gland      266
    Soft Tissue        265
    Esophagus          198
    Pancreas           183
    Nervous System     162
    Bone Marrow        151
    Testis             139
    Thymus             122
    Bone                88
    Pleura              87
    Eye                 80
    Bile Duct           44
    Name: count, dtype: int64



## 0.3. Choose a primary site to create the data set 


```python
def create_data_matrix_for_DE(primary_site):
    lung_meta = meta[meta['primary_site'] == primary_site].reset_index(drop=True)

    chunks = lung_meta.shape[0]//50
    chunk_ls = []
    for chunk_i in tqdm(range(chunks)):
        lung_meta_i = lung_meta.iloc[chunk_i*50:(chunk_i*50+50), :].reset_index(drop=True)
        file_ids = lung_meta_i['file_id'].to_list()
        file_id_url_map =  gdc_eng_inst._make_file_id_url_map(file_ids)
        rawDataMap = gdc_eng_inst._get_urls_content(file_id_url_map)
        ids_with_none = [key for key in rawDataMap.keys() if rawDataMap[key] is None]
        rna_seq_data_matrix = gdc_eng_inst._make_rna_seq_data_matrix(rawDataMap, lung_meta_i, feature_col='unstranded')
        
        lung_meta_sub_i = lung_meta_i[~lung_meta_i['file_id'].isin(ids_with_none)]
        rna_seq_data_matrix['tissue_type'] = lung_meta_sub_i['tissue_type'].to_numpy()
        rna_seq_data_matrix['case_id'] = lung_meta_sub_i['case_id'].to_numpy()
        chunk_ls.append(rna_seq_data_matrix)
    df = pd.concat(chunk_ls)
    return df 
```
