from OutlierStatMethods import base_class
import pandas as pd 
import numpy as np
import scipy.stats as sp 

class OSPerm(base_class.OutlierStatMethod):
    def __init__(self, disease_data=None, control_data=None):
        ## baseclass initializations
        super().__init__()

        ## subclass initializations
        self.disease_data = disease_data
        self.control_data = control_data
        if self.disease_data is None or self.disease_data.shape[0] == 0 or self.disease_data.shape[1] == 0:
            raise ValueError("Input disease data is invalid")
        if self.control_data is None or self.control_data.shape[0] == 0 or self.control_data.shape[1] == 0:
            raise ValueError("Input disease data is invalid")

        ## properties        
        self._no_of_feats = None
        self._mad_norm_disease_df = None 
        self._mad_norm_control_df = None
        self._n_cases = None
        self._n_controls = None 

    @property
    def no_of_feats(self):
        if self._no_of_feats is None:
            feats = self.disease_data.shape[1]
        return feats 

    @property
    def mad_norm_disease_df(self):
        if self._mad_norm_disease_df is None:
            df = self.apply_mad_norm(self.disease_data)
        return df

    @property
    def mad_norm_control_df(self):
        if self._mad_norm_control_df is None:
            df = self.apply_mad_norm(self.control_data)
        return df
    
    @property
    def n_cases(self):
        if self._n_cases is None:
            return self.disease_data.shape[0]
    
    @property
    def n_controls(self):
        if self._n_controls is None:
            return self.control_data.shape[0]
        
    ## inherited functions 
    def get_mad(self, x):
        return super().get_mad(x)
    
    def get_median(self, x):
        return super().get_median(x)
    
    def get_all_mads(self, data: pd.DataFrame):
        return super().get_all_mads(data)
    
    def get_all_meds(self, data: pd.DataFrame):
        return super().get_all_meds(data)
    
    def apply_mad_norm(self, data: pd.DataFrame):
        return super().apply_mad_norm(data)

    def get_all_threshes(self, data:pd.DataFrame):
        return super().get_all_threshes(data)
    
    def multiprocess_os(self, data, n_features):
        return super().multiprocess_os(data, n_features)
    
    ## Method specific functions 
    def generate_null(self):
        iters = self.iters
        OS_null = []
        for i in range(iters):
            if self.n_controls - self.n_cases >= 50:
                sampled_df = self.mad_norm_control_df.sample(n=self.n_cases,replace=True)
            else:
                comb_df = pd.concat([self.mad_norm_control_df, self.mad_norm_disease_df], axis=0)
                sampled_df = comb_df.sample(n=self.n_cases, replace=True)
            OS_i = self.multiprocess_os(sampled_df, self.no_of_feats)
            OS_null.append(OS_i)
        OS_null_df = pd.DataFrame(OS_null)
        return OS_null_df
    
    def get_pvalue_for_feat(self, OS_null):
        OS_disease = self.multiprocess_os(self.disease_data, self.no_of_feats)
        OS_mean, OS_sd = np.mean(OS_null, axis=0), np.std(OS_null, axis=0)
        zscore = [(OS_disease[i] - OS_mean[i])/OS_sd[i] for i in range(self.no_of_feats)]
        pvalue = sp.norm.sf(zscore)
        return zscore, pvalue, OS_disease 
    
    def get_stats(self):
        ## initial stats from controls 
        meds = self.get_all_meds(self.mad_norm_control_df)
        mads = self.get_all_mads(self.mad_norm_control_df)
        threshes = self.get_all_threshes(self.mad_norm_control_df)

        ## OS_null distribution 
        OS_null_df = self.generate_null()
        zscores, pvalues, OS_disease = self.get_pvalue_for_feat(OS_null_df)
        
        ## final stats_dict 
        stat_dict = {'median': meds, 'mad': mads, 'iqr_threshold': threshes, 'zscores': zscores, 'pvalues': pvalues, 'OutlierSum': OS_disease}
        return stat_dict 