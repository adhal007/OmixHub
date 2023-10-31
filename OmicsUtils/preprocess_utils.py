import pandas as pd
import numpy as np
import os 
from shutil import move
import CustomLogger.custom_logger
logger = CustomLogger.custom_logger.CustomLogger()

class PreProcessor:
    def __init__(self):
        self.logger =  logger.custlogger(loglevel='DEBUG')
        self.logger.debug("Initialized PreProcessor Class for TCGA data")
        self.metric_types_rna_seq = ['fpkm_unstranded', 'fpkm_uq_unstranded', 'tpm_unstranded', 'stranded_second', 'stranded_first', 'unstranded']
        self.logger.debug("Initialized Omics PreProcessor class")

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
        logger_child = self.logger.getChild('make_data_matrix')
        
        data_matrix_ls = []
        if metric_type not in self.metric_types_rna_seq:
            raise ValueError("Specify correct metric for data matrix generation")
        logger_child.info(f"Preparing data matrix for {metric_type} values")

        for file_path in file_path_ls:
            if os.path.exists(file_path):
                file_name = file_path.split('/')[-1]
                logger_child.info(f"Starting file {file_name} processing")
                file_idx = file_path.split('/')[-1].split('.')[0]
                df = pd.read_csv(file_path, skiprows=[0], sep='\t')
                df_t = df.T.reset_index()

                df_col_names = df_t[df_t['index'] == 'gene_id'].iloc[:, 5:]
                df_col_names_ls = list(np.concatenate(df_col_names.to_numpy(), axis=0)) 
                df_col_names_ls.append('file_identifier')
                df_t['file_identifier'] = file_idx
                df_row = df_t[df_t['index'] == 'fpkm_unstranded'].iloc[:, 5:]
                df_row.columns = df_col_names_ls
            
            data_matrix_ls.append(df_row)
            logger_child.info(f"Finished file {file_name} processing")
        logger_child.info("Finished making data matrix")

        return data_matrix_ls     



# How to read Gene Expression Files
# pd.read_csv(os.path.join(folder_paths[0], os.listdir(folder_paths[0])[0]), skiprows=[0], sep='\t')
        