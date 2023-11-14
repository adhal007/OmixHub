from OutlierStatMethods import base_class
import pandas as pd 
import numpy as np
import scipy.stats as sp 
from scipy.integrate import quad
import logging
# logger = logging.getLogger(__name__)

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
        self.k_chen = 0.5 
        # self.logger = logger.getChild('OSPerm')
        # self.logger.setLevel(logging.INFO)
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
    
    def get_cdf(self,x):
        return sp.norm.cdf(x)

    def get_mu_x(self,x):
        mu_x = np.mean(x)
        return mu_x 
    
    def get_sigma_sq_x(self, x):
        sig_x = np.var(x)
        return sig_x 
    
    def get_q75_x(self, x):
        return sp.norm.ppf(0.75, loc=0, scale=1)
    
    def integrand_1_z(self, z):
        return z*self.get_cdf(z)
    
    def integrand_2_z(self, z):
        return (z**2)*self.get_cdf(z)
    
    def integration_1(self, z, x):
        self.get_sigma_sq_x()
    ## Method specific functions 
    def generate_null(self):
        raise NotImplementedError()
        # return OS_null_df
    
    # def get_pvalue_for_feat(self, OS_null):
    #     OS_disease = self.multiprocess_os(self.disease_data, self.no_of_feats)
    #     OS_mean, OS_sd = np.mean(OS_null, axis=0), np.std(OS_null, axis=0)
    #     zscore = [(OS_disease[i] - OS_mean[i])/OS_sd[i] for i in range(self.no_of_feats)]
    #     pvalue = sp.norm.sf(zscore)
    #     return zscore, pvalue, OS_disease 
    
    def get_stats(self):
        ## initial stats from controls 
        print("Calculating median, mad and applying mad normalization")
        meds = self.get_all_meds(self.mad_norm_control_df)
        mads = self.get_all_mads(self.mad_norm_control_df)
        threshes = self.get_all_threshes(self.mad_norm_control_df)

        print("Generating null distribution")
        ## OS_null distribution 
        OS_null_df = self.generate_null()
        zscores, pvalues, OS_disease = self.get_pvalue_for_feat(OS_null_df)
        
        print("Consolidating stats into dictionary")
        ## final stats_dict 
        stat_dict = {'median': meds, 'mad': mads, 'iqr_threshold': threshes, 'zscores': zscores, 'pvalues': pvalues, 'OutlierSum': OS_disease}
        
        print("Finished applying outlier stat methods")
        return stat_dict 