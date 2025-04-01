from src.ClassicML.OutlierStatMethods import base_class
import pandas as pd 
import numpy as np
import scipy.stats as sp 
from scipy.integrate import quad
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import logging

class OSApprox(base_class.OutlierStatMethod):
    def __init__(self, disease_data=None, control_data=None, k_chen=1):
        ## baseclass initializations
        super().__init__()

        ## subclass initializations
        self.disease_data = disease_data
        self.control_data = control_data
        if self.disease_data is None or self.disease_data.shape[0] == 0 or self.disease_data.shape[1] == 0:
            raise ValueError("Input disease data is invalid")
        if self.control_data is None or self.control_data.shape[0] == 0 or self.control_data.shape[1] == 0:
            raise ValueError("Input control data is invalid")
        self.k_chen = k_chen
        self.min_pvalue = 1e-300  # Minimum p-value to avoid exact zeros
        self.epsilon = 1e-8
        ## properties        
        self._no_of_feats = None
        self._mad_norm_disease_df = None 
        self._mad_norm_control_df = None
        self._n_cases = None
        self._n_controls = None 

    @property
    def no_of_feats(self):
        if self._no_of_feats is None:
            self._no_of_feats = self.disease_data.shape[1]
        return self._no_of_feats 

    @property
    def mad_norm_disease_df(self):
        if self._mad_norm_disease_df is None:
            self._mad_norm_disease_df = self.apply_mad_norm(self.disease_data)
        return self._mad_norm_disease_df

    @property
    def mad_norm_control_df(self):
        if self._mad_norm_control_df is None:
            self._mad_norm_control_df = self.apply_mad_norm(self.control_data)
        return self._mad_norm_control_df
    
    @property
    def n_cases(self):
        if self._n_cases is None:
            self._n_cases = self.disease_data.shape[0]
        return self._n_cases
    
    @property
    def n_controls(self):
        if self._n_controls is None:
            self._n_controls = self.control_data.shape[0]
        return self._n_controls
        
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
        print("Calculating median, mad and applying mad normalization")
        all_meds = self.get_all_meds(data)
        all_mads = self.get_all_mads(data)
        
        # Replace zero MADs with a small value to avoid division by zero
        all_mads = np.where(all_mads == 0, self.epsilon, all_mads)
        
        normalized_data = (data - all_meds) / all_mads
        
        # Replace inf values with a large number and -inf with a small number
        normalized_data = normalized_data.clip(-1e8, 1e8)
        
        return normalized_data

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
        return  self.get_sigma_sq_x(x)
    
    ## Method specific functions 
    def generate_null(self):
        control_mean = np.mean(self.control_data)
        control_std = np.std(self.control_data)
        k = self.k_chen
        threshold = control_mean + 3 * k * sp.norm.ppf(0.75) * control_std
        
        beta = 1 - sp.norm.cdf(3 * k * sp.norm.ppf(0.75))
        
        n_controls = self.n_controls
        n_cases = self.n_cases
        gamma = n_cases / n_controls
        
        b1 = np.sqrt(2 * np.pi) * beta * 3 * k * sp.norm.ppf(0.75) * control_std * sp.norm.pdf(3 * k * sp.norm.ppf(0.75)) * np.sqrt(gamma)
        b2 = b3 = 1.5 * k * b1 * sp.norm.pdf(sp.norm.ppf(0.75)) ** (-1) * np.sqrt(2 * np.pi)
        
        mu_l = control_mean + control_std / beta * quad(lambda z: z * sp.norm.pdf(z), 3 * k * sp.norm.ppf(0.75), np.inf)[0]
        
        v = (control_std ** 2) / (beta ** 2) * (
            quad(lambda z: z ** 2 * sp.norm.pdf(z), 3 * k * sp.norm.ppf(0.75), np.inf)[0] -
            quad(lambda z: z * sp.norm.pdf(z), 3 * k * sp.norm.ppf(0.75), np.inf)[0] ** 2
        )
        
        sigma_l_sq = (3/256) * (
            (2*b1 + b2 - 3*b3)**2 + (2*b1 + b2 + b3)**2 +
            (-2*b1 + b2 + b3)**2 + (-2*b1 - 3*b2 + b3)**2
        ) + v
        
        return mu_l, sigma_l_sq, threshold, beta

    def get_pvalue_for_feat(self, null_params, disease_data):
        mu_l, sigma_l_sq, threshold, beta = null_params
        
        L = np.sum(disease_data[disease_data > threshold], axis=0)
        n_cases = self.n_cases
        
        # Avoid division by zero
        denominator = np.sqrt(n_cases) * beta * np.sqrt(sigma_l_sq)
        Z_test = np.divide(L - n_cases * beta * mu_l, denominator, 
                           out=np.zeros_like(L, dtype=float), where=denominator != 0)
        
        pvalue = 1 - sp.norm.cdf(Z_test)
        pvalue = np.maximum(pvalue, self.min_pvalue)  # Ensure p-value is not smaller than min_pvalue
        return Z_test, pvalue, L

    def get_stats(self):
        print("Calculating median, mad and applying mad normalization")
        meds = self.get_all_meds(self.mad_norm_control_df)
        mads = self.get_all_mads(self.mad_norm_control_df)
        threshes = self.get_all_threshes(self.mad_norm_control_df)

        print("Generating null distribution")
        null_params = self.generate_null()
        
        print("Calculating Z-scores, p-values, and Outlier Sum")
        zscores, pvalues, OS_disease = self.get_pvalue_for_feat(null_params, self.disease_data)
        
        print("Applying FDR correction")
        _, fdr_pvalues = fdrcorrection(pvalues, method='indep')
        
        print("Calculating additional statistical metrics")
        
        # Convert DataFrames to numpy arrays for faster computation
        disease_array = self.disease_data.values
        control_array = self.control_data.values
        
        # Calculate means and standard deviations
        disease_means = np.mean(disease_array, axis=0)
        control_means = np.mean(control_array, axis=0)
        disease_stds = np.std(disease_array, axis=0, ddof=1)
        control_stds = np.std(control_array, axis=0, ddof=1)
        
        # Cohen's d effect size
        pooled_std = np.sqrt((disease_stds**2 + control_stds**2) / 2)
        cohens_d = (disease_means - control_means) / pooled_std
        
        # Fold change
        epsilon = 1e-10  # Small value to avoid division by zero
        fold_changes = (disease_means + epsilon) / (control_means + epsilon)
        
        # T-test (assuming equal variance)
        n1, n2 = disease_array.shape[0], control_array.shape[0]
        pooled_se = np.sqrt(pooled_std**2 * (1/n1 + 1/n2))
        t_statistics = (disease_means - control_means) / pooled_se
        df = n1 + n2 - 2
        t_pvalues = 2 * (1 - stats.t.cdf(np.abs(t_statistics), df))
        
        # Additional metrics
        z_scores = (disease_means - control_means) / control_stds
        percent_change = (disease_means - control_means) / control_means * 100
        hedges_g = cohens_d * (1 - (3 / (4 * (n1 + n2 - 2) - 1)))
        log10_ratio_of_means = np.log10((disease_means + epsilon) / (control_means + epsilon))
        difference_of_means = disease_means - control_means
        
        # Log odds ratio (assuming data is on a 0-1 scale, if not, this might need adjustment)
        disease_odds = np.clip(disease_means / (1 - disease_means), epsilon, 1/epsilon)
        control_odds = np.clip(control_means / (1 - control_means), epsilon, 1/epsilon)
        log_odds_ratio = np.log(disease_odds / control_odds)
        
        print("Consolidating stats into dictionary")
        stat_dict = {
            'median': meds,
            'mad': mads,
            'iqr_threshold': threshes,
            'zscores': zscores,
            'pvalues': pvalues,
            'fdr_pvalues': fdr_pvalues,
            'OutlierSum': OS_disease,
            'cohens_d': cohens_d,
            'fold_change': fold_changes,
            't_statistic': t_statistics,
            't_pvalue': t_pvalues,
            'z_score': z_scores,
            'percent_change': percent_change,
            'hedges_g': hedges_g,
            'log10_ratio_of_means': log10_ratio_of_means,
            'difference_of_means': difference_of_means,
            'log_odds_ratio': log_odds_ratio
        }
        
        print("Finished applying outlier stat methods")
        return stat_dict