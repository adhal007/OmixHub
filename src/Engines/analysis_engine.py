import numpy as np 
import pandas as pd 
import src.ClassicML.DGE.pydeseq_utils as pydeseq_utils

class Analysis:
    def __init__(self, data_from_bq:pd.DataFrame, analysis_type:str) -> None:
        self.data_from_bq = data_from_bq
        self.analysis_type = analysis_type 
    
    def expand_data_from_bq(self, data_from_bq, gene_ids_or_gene_cols, analysis_type):
        if analysis_type is None:
            raise Warning("No analysis type was specified")
            return None
        elif analysis_type == 'DE':
            # Expand 'expr_unstr_count' into separate columns using apply with pd.Series
            feature_col = 'expr_unstr_count'
        elif analysis_type == 'ML':
            feature_col = 'expr_unstr_tpm'

        expr_unstr_df = data_from_bq[].apply(pd.Series)

        # Optionally rename the new columns to something meaningful
        expr_unstr_df.columns = gene_ids_or_gene_cols

        # Concatenate the expanded columns back to the original dataframe
        exp_df = pd.concat([data_from_bq.drop(columns=['expr_unstr_count']), expr_unstr_df], axis=1)   
        return exp_df 

    def counts_from_bq_df(self, exp_df:pd.DataFrame, gene_ids_or_gene_cols: list):
        gene_case_cols = gene_ids_or_gene_cols.append('case_id') 
        counts = exp_df[gene_case_cols]
        counts.set_index('case_id', inplace=True)
        return counts 
        
    def metadata_for_pydeseq(self, exp_df:pd.DataFrame):
        """
        This function will take the expanded data from bigquery and then create a metadata for pydeseq
        """
        metadata = exp_df[['case_id', 'tissue_type']]
        metadata.columns = ['Sample', 'Condition']
        metadata = metadata.set_index(keys='Sample') 
        return metadata     
    
    def run_pydeseq(self, metadata, counts):
        pydeseq_obj = pydeseq_utils.PyDeSeqWrapper(count_matrix=counts, metadata=metadata, design_factors='Condition', groups = {'group1':'Tumor', 'group2':'Normal'})
        design_factor = 'Condition'
        result = pydeseq_obj.run_deseq(design_factor=design_factor, group1 = 'Tumor', group2 = 'Normal')
        

    def data_for_ml(self):
        raise NotImplementedError()
    