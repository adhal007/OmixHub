import numpy as np 
import pandas as pd 
import src.ClassicML.DGE.pydeseq_utils as pydeseq_utils
import pandas as pd
from gseapy.plot import gseaplot
import gseapy as gp
import numpy as np

class Analysis:
    """
    Analysis class to perform data analysis based on the specified analysis type.

    Attributes:
        data_from_bq (pd.DataFrame): The input data from BigQuery.
        analysis_type (str): The type of analysis to perform.
    """
    def __init__(self, data_from_bq:pd.DataFrame, analysis_type:str) -> None:
        """
        Initialize the Analysis class with the given data and analysis type.

        Args:
            data_from_bq (pd.DataFrame): The input data from BigQuery.
            analysis_type (str): The type of analysis to perform.
        """
        self.data_from_bq = data_from_bq
        self.analysis_type = analysis_type 
    
    def expand_data_from_bq(self, data_from_bq, gene_ids_or_gene_cols, analysis_type):
        """
        Expand the data from BigQuery by separating 'expr_unstr_count' or 'expr_unstr_tpm' into separate columns.

        Args:
            data_from_bq (pd.DataFrame): The input data from BigQuery.
            gene_ids_or_gene_cols (list): The list of gene IDs or gene column names.
            analysis_type (str): The type of analysis to perform. Should be either 'DE' or 'ML'.

        Raises:
            Warning: If no analysis type is specified.

        Returns:
            pd.DataFrame: The expanded DataFrame with separated columns.
        """
        if analysis_type is None:
            raise Warning("No analysis type was specified")
            return None
        elif analysis_type == 'DE':
            # Expand 'expr_unstr_count' into separate columns using apply with pd.Series
            feature_col = 'expr_unstr_count'
        elif analysis_type == 'ML':
            feature_col = 'expr_unstr_tpm'

        expr_unstr_df = data_from_bq[feature_col].apply(pd.Series)

        # Optionally rename the new columns to something meaningful
        expr_unstr_df.columns = gene_ids_or_gene_cols

        # Concatenate the expanded columns back to the original dataframe
        exp_df = pd.concat([data_from_bq.drop(columns=[feature_col]), expr_unstr_df], axis=1)   
        return exp_df 

    def counts_from_bq_df(self, exp_df:pd.DataFrame, gene_ids_or_gene_cols: list):
        """
        Expand the data from BigQuery by separating 'expr_unstr_count' or 'expr_unstr_tpm' into separate columns.

        Args:
            data_from_bq (pd.DataFrame): The input data from BigQuery.
            gene_ids_or_gene_cols (list): The list of gene IDs or gene column names.
            analysis_type (str): The type of analysis to perform. Should be either 'DE' or 'ML'.

        Raises:
            Warning: If no analysis type is specified.

        Returns:
            pd.DataFrame: The expanded DataFrame with separated columns.
        """
        gene_ids_or_gene_cols.append('case_id') 
        counts = exp_df[gene_ids_or_gene_cols]
        counts.set_index('case_id', inplace=True)
        return counts 
        
    def metadata_for_pydeseq(self, exp_df:pd.DataFrame):
        """
        Create metadata for PyDeSeq from the expanded DataFrame.

        Args:
            exp_df (pd.DataFrame): The expanded DataFrame containing expression data.

        Returns:
            pd.DataFrame: Metadata DataFrame with 'Sample' and 'Condition' columns.
        """
        metadata = exp_df[['case_id', 'tissue_type']]
        metadata.columns = ['Sample', 'Condition']
        metadata = metadata.set_index(keys='Sample') 
        return metadata     
    
    def run_pydeseq(self, metadata, counts):
        """
        Run Gene Set Enrichment Analysis (GSEA) on the given DataFrame.

        Args:
            df_de (pd.DataFrame): DataFrame containing differential expression results.
            gene_set (str): The gene set to use for GSEA.

        Returns:
            tuple: A tuple containing the GSEA results DataFrame and the plot axes.
        """
        pydeseq_obj = pydeseq_utils.PyDeSeqWrapper(count_matrix=counts, metadata=metadata, design_factors='Condition', groups = {'group1':'Tumor', 'group2':'Normal'})
        design_factor = 'Condition'
        result = pydeseq_obj.run_deseq(design_factor=design_factor, group1 = 'Tumor', group2 = 'Normal')
        result.summary()
        results_df = result.results_df
        results_df_filtered = results_df.dropna()
        results_df_filtered = results_df_filtered.reset_index()
        results_df_filtered['nlog10'] = -1*np.log10(results_df_filtered.padj)
        return results_df_filtered
    
    def run_gsea(self, df_de:pd.DataFrame, gene_set):
        df = df_de.copy()
        df['Rank'] = -np.log10(df.padj)*df.log2FoldChange
        df = df.sort_values('Rank', ascending = False).reset_index(drop = True)
        df = df.rename(columns = {'gene_name': 'Gene'})
        ranking = df[['Gene', 'Rank']]
        pre_res = gp.prerank(rnk = ranking, gene_sets = gene_set, seed = 6, permutation_num = 100)
        out = []

        for term in list(pre_res.results):
            out.append([term,
                    pre_res.results[term]['fdr'],
                    pre_res.results[term]['es'],
                    pre_res.results[term]['nes']])

        out_df = pd.DataFrame(out, columns = ['Term','fdr', 'es', 'nes']).sort_values('fdr').reset_index(drop = True)
        terms = pre_res.res2d.Term
        axs = pre_res.plot(terms=terms[1])
        return out_df, axs
     
    def data_for_ml(self):
        raise NotImplementedError()
    