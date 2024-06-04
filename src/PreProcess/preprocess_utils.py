import pandas as pd
import numpy as np
import os 
from shutil import move

from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from src import base_preprocessor as bp
import src.CustomLogger.custom_logger

logger = src.CustomLogger.custom_logger.CustomLogger()

class RNASeqPreProcessor(bp.BaseDataProcessor):
    def __init__(self, data: pd.DataFrame, x_cols: list[str], y_cols: list[str], unique_id_col: str) -> None:
        super().__init__(data=data, x_cols=x_cols, y_cols=y_cols, unique_id_col=unique_id_col)

        self.logger =  logger.custlogger(loglevel='DEBUG')
        self.logger.debug("Initialized PreProcessor Class for TCGA data")
        self.metric_types_rna_seq = ['fpkm_unstranded', 'fpkm_uq_unstranded', 'tpm_unstranded', 'stranded_second', 'stranded_first', 'unstranded']
        self.logger.debug("Initialized Omics PreProcessor class")

        ## property variables 
        self._duplicate_samples = None 
        self._non_overlap_samples = None         

    def remove_empty_folders(self, directory_path):
        for root, dirs, files in os.walk(directory_path, topdown=False):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if not os.listdir(folder_path):
                    os.rmdir(folder_path)

    def get_sample_ids_folder_names_file_names(self, main_dir):
        sample_ids = []
        folder_paths = []
        for root, dirs, files in os.walk(main_dir):
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                if os.listdir(folder_path):
                    sample_ids.append(folder)
                    folder_paths.append(folder_path)
        return sample_ids, folder_paths 
    
    def get_list_of_tsv_files(self, main_dir):
        file_names = []
        file_paths = []
        file_idx = []
        sample_ids, folder_paths = self.get_sample_ids_folder_names_file_names(main_dir)
        for folder in folder_paths:
            file_name = os.listdir(folder)[0]
            if 'tsv' in file_name:
                file_id = file_name.split('.')[0]
                file_names.append(file_name)
                file_path = os.path.join(folder, file_name)
                file_paths.append(file_path)
                file_idx.append(file_id)
        return file_paths, file_names, file_idx 

    ## Should give an option to change excessive logging of starting and finishing each file
    def make_data_matrix(self, file_path_ls, metric_type='fpkm_unstranded'):
        logger_child = self.logger.getChild(suffix='make_data_matrix')
        
        data_matrix_ls = []
        if metric_type not in self.metric_types_rna_seq:
            raise ValueError("Specify correct metric for data matrix generation")
        logger_child.info(msg=f"Preparing data matrix for {metric_type} values")

        for file_path in file_path_ls:
            if os.path.exists(path=file_path):
                file_name = file_path.split('/')[-1]
                logger_child.info(msg=f"Starting file {file_name} processing")
                file_idx = file_path.split('/')[-1].split('.')[0]
                df = pd.read_csv(filepath_or_buffer=file_path, skiprows=[0], sep='\t')
                df_t = df.T.reset_index()

                df_col_names = df_t[df_t['index'] == 'gene_id'].iloc[:, 5:]
                df_col_names_ls = list(np.concatenate(df_col_names.to_numpy(), axis=0)) 
                df_col_names_ls.append('file_identifier')
                df_t['file_identifier'] = file_idx
                df_row = df_t[df_t['index'] == metric_type].iloc[:, 5:]
                df_row.columns = df_col_names_ls
            
            data_matrix_ls.append(df_row)
            logger_child.info(f"Finished file {file_name} processing")
        logger_child.info("Finished making data matrix")
        return data_matrix_ls     



    @property
    def duplicate_samples(self):
        if self._duplicate_samples is None:
            if self.check_duplicate_data():
                self._duplicate_samples = self.data[self.unique_id_col].value_counts()[self.data[self.unique_id_col].value_counts() > 1].index.tolist()
            return self._duplicate_samples
        else:
            return self._duplicate_samples

    @property
    def non_overlap_samples(self):
        if self._non_overlap_samples is None:
            non_overlap_samples = self.data[~self.data[self.unique_id_col].isin(self.duplicate_samples)][self.unique_id_col].unique().tolist()    
            self._non_overlap_samples = non_overlap_samples
            return self._non_overlap_samples
        else:
            return self._non_overlap_samples
    
    def get_patient_overlap_train_test_split(self)
        patient_overlap_splits = self.split_data(self.duplicate_samples, self.unique_id_col)
        return patient_overlap_splits

    def get_non_overlap_count(self, target_columns:list[str]):
        df = self.data[~self.data[self.unique_id_col].isin(self.non_overlap_samples)]
        counts = df[target_columns].value_counts().reset_index
        counts.columns = ['unique_id', 'count']

## Add more functions for the following:
## 1. Checking if there is class imbalance in the data
## 2. Checking if there is any missing data
## 3. Checking if there is any duplicate data (for set sampling)
## 4. Checking if there is any outlier data 
## 5. Checking if there is any data leakage (between training, test and validation cohorts)
## 6. Checking if there is any data drift (between training, test and validation cohorts)
## 7. Checking if there is any data skew (between training, test and validation cohorts)
## 8. Checking if there is any data bias (between training, test and validation cohorts)
## 9. Checking if there is any data noise (between training, test and validation cohorts)
## 10. Splitter for generating training, test and validation cohorts (seeded or random)
## 11. Splitter for generating training, test and validation cohorts (patient overlap)
## 12. Splitter for generating training, test and validation cohorts (set sampling)
## 13. Oversampling for minority samples (SMOTE)
    
## now get minority samples across each class a
## For example if there are 3 classes, then there will be 2 minority classes 
## and 1 majority class.
## This function will return a dataframe with the minority samples

## Challenges
## 1. How do we handle if there are 2 types of target labels and we want to combine them for a multiclass problem
## Example: 
## 3 kidney cancer types, 2 of them are minority and 1 is majority
## 2 tumor types, 1 majority and 1 minority 

## Solution
## We find the union of all the minority samples across all the type of target labels
## then we perform sampling to raise the minority samples to the level of majority samples 
## Use this new dataset to define training, testing and validation splits 

    
## Steps for data splitting
## 1. Process duplicate samples uniquely into training, test and validation splits 
## 2. Oversample the remaining minority classes to the level of majority class
## 3. Split the balanced dataset into training, testing and validation splits 
