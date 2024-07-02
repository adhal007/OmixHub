import pandas as pd
import numpy as np
import os 
from shutil import move
import src.CustomLogger.custom_logger
import src.Connectors.gdc_parser as gdc_prs
import src.Connectors.gdc_files_endpt as gdc_files
import src.Connectors.gdc_cases_endpt as gdc_cases 
import src.Connectors.gdc_projects_endpt as gdc_projects
"""
Copyright (c) 2024 OmixHub.  All rights are reserved.
GDC Endpoint Preprocess Class for high-level API functions to create and preprocess data into MongoDB

@author: Abhilash Dhal
@date:  2024_07_01


"""
class EndptPreProcessor(gdc_prs.GDCParser):
    def __init__(self, 
                 gdc_files_sub:gdc_files.GDCFilesEndpt, 
                 gdc_cases_sub:gdc_cases.GDCCasesEndpt,
                 gdc_projs_sub:gdc_projects.GDCProjectsEndpt) -> None:
        
        super().__init__(gdc_files_sub=gdc_files_sub, gdc_cases_sub=gdc_cases_sub, gdc_projs_sub=gdc_projs_sub)
        
        self._gdc_files_sub = gdc_files_sub
        self._gdc_cases_sub = gdc_cases_sub
        self._gdc_proj_sub = gdc_projs_sub

        self._data = None
        self.logger = src.CustomLogger.custom_logger.get_logger(__name__)
        

    def _get_metadata(self,query):
        # gdc_files_edpt_inst.fetch_rna_seq_star_counts_data(ps_list=ps_list_1)
        json_data, filters = self._gdc_files_sub.fetch_rna_seq_star_counts_data(query)
        metadata = self.create_df_from_rna_star_count_q_op(json_data)
        return metadata
    
    # def create_data_matrix(self, file_path_ls, metric_type='fpkm_unstranded'):
    #     """
    #     Goal of this function is to create a data matrix given the metadata from query of GDC API
        
    #     Args:
    #     Returns: Dataframe of data matrix
    #     """
    #     logger_child = self.logger.getChild(suffix='make_data_matrix')
        
    #     data_matrix_ls = []
    #     if metric_type not in self.metric_types_rna_seq:
    #         raise ValueError("Specify correct metric for data matrix generation")
    #     logger_child.info(msg=f"Preparing data matrix for {metric_type} values")

    #     for file_path in file_path_ls:
    #         if os.path.exists(path=file_path):
    #             file_name = file_path.split('/')[-1]
    #             logger_child.info(msg=f"Starting file {file_name} processing")
    #             file_idx = file_path.split('/')[-1].split('.')[0]
    #             df = pd.read_csv(filepath_or_buffer=file_path, skiprows=[0], sep='\t')
    #             df_t = df.T.reset_index()

    #             df_col_names = df_t[df_t['index'] == 'gene_id'].iloc[:, 5:]
    #             df_col_names_ls = list(np.concatenate(df_col_names.to_numpy(), axis=0)) 
    #             df_col_names_ls.append('file_identifier')
    #             df_t['file_identifier'] = file_idx
    #             df_row = df_t[df_t['index'] == metric_type].iloc[:, 5:]
    #             df_row.columns = df_col_names_ls
            
    #         data_matrix_ls.append(df_row)
    #         logger_child.info(f"Finished file {file_name} processing")
    #     logger_child.info("Finished making data matrix")

    
    def get_patient_overlap(self,data):
        """
        Args: Dataframe of patient data

        Returns: Dictionary of Count of different groups of categortical variables that and their patient overlap distribution 

        pat_ovr = {"Sample type": pd.DataFrame, "Project Type": pd.DataFrame, "Sample type": pd.DataFrame}
        """
        categorical_variables = ['Sample Type','Project ID']
        pat_over = {}

        for category in categorical_variables:
            if category in data.columns:
                category_counts = data.groupby(category)['Case ID'].nunique().reset_index()
                category_counts.columns = [category, 'Unique_Patient_Count']
                pat_over[category] = category_counts
        
        return pat_over
    