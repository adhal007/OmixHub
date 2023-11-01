import pandas as pd 
import numpy as np
import scipy.stats as sp
from multiprocessing.pool import ThreadPool
import os 


class OutlierStatMethod:
    def __init__(self):
        self.k = 1.0
        self.iters = 200
        self.method_name = 'perm'
    
    def get_median(self, x):
        return np.median(x)
    
    def get_mad(self, x):
        return sp.median_abs_deviation(x)
    
    def get_all_meds(self, data:pd.DataFrame):
        return [self.get_median(data.iloc[:, i].to_numpy()) for i in range(data.shape[1])]
    
    def get_all_mads(self, data:pd.DataFrame):
        return [self.get_mad(data.iloc[:, i].to_numpy()) for i  in range(data.shape[1])]
    
    def get_all_threshes(self, data:pd.DataFrame):
        threshes = []
        for i in range(data.shape[1]):
            z = data.iloc[:, i]
            q75 = np.percentile(z, 75)
            q25 = np.percentile(z, 25)
            iqr = q75 - q25
            thresh = q75 + self.k*iqr
            threshes.append(thresh)
        return threshes 
    
    def apply_mad_norm(self, data:pd.DataFrame):
        ## mad norm is (x - median)/mad
        all_meds = self.get_all_meds(data)
        all_mads = self.get_all_mads(data)
        data_norm = [(data.iloc[:, i].to_numpy() - all_meds[i])/all_mads[i] for i in range(data.shape[1])]
        data_norm_df = pd.DataFrame(data_norm)
        return data_norm_df.T
    
    def calc_os_single(self,data,i):
        z = data.iloc[:, i].to_numpy()
        q75 = np.percentile(z, 75)
        q25 = np.percentile(z, 25)
        iqr = q75 - q25
        thresh = q75 + self.k*iqr
        OS = sum(z[z > thresh])
        return OS
    
    def multiprocess_os(self, data, n_features):
        i_args = list(np.arange(0, n_features, 1))
        data_args = [data for i in range(n_features)]
        with ThreadPool(processes=2) as pool:
            results = pool.starmap(self.calc_os_single, zip(data_args, i_args))
        return results 
    
    def generate_null(self, data:pd.DataFrame):
        raise NotImplementedError("method will be implemented in the subclass")
    
    def get_stats(self):
        raise NotImplementedError("method will be implemented in the subclass")
