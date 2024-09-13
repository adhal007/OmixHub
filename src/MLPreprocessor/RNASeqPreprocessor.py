import src.ClassicML.DataAug.simulators as simulators
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from typing import Union, Tuple

class RNASeqPP:
    def __init__(self, data_from_bq, gene_cols):
        self.data_from_bq = data_from_bq
        self.tumor_samples = None
        self.tumor_train = None
        self.tumor_val = None
        self.normal_samples = None
        self.simulated_normal_samples = None
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.gene_cols = gene_cols
        self.gene_id_to_name = self.load_gene_mapping()
        self.raw_counts = None
        self.normalized_counts = None
        self.size_factors = None
        self.logmeans = None
        self.filtered_genes = None
    
    def load_gene_mapping(self):
        mapping_file = '/Users/abhilashdhal/Projects/personal_docs/data/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv'
        mapping_df = pd.read_csv(mapping_file)
        return dict(zip(mapping_df['gene_id'], mapping_df['gene_name']))
    
    def get_gene_name(self, gene_id):
        # Remove version number from gene_id if present
        return self.gene_id_to_name[gene_id]

    def deseq2_norm_fit(self, counts: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        with np.errstate(divide="ignore"):
            log_counts = np.log(counts)
        logmeans = log_counts.mean(0)
        filtered_genes = ~np.isinf(logmeans)
        return logmeans, filtered_genes

    def deseq2_norm_transform(self, counts: Union[pd.DataFrame, np.ndarray],
                              logmeans: np.ndarray, filtered_genes: np.ndarray) -> Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.DataFrame, np.ndarray]]:
        with np.errstate(divide="ignore"):
            log_counts = np.log(counts)
        if isinstance(log_counts, pd.DataFrame):
            log_ratios = log_counts.loc[:, filtered_genes] - logmeans[filtered_genes]
        else:
            log_ratios = log_counts[:, filtered_genes] - logmeans[filtered_genes]
        log_medians = np.median(log_ratios, axis=1)
        size_factors = np.exp(log_medians)
        deseq2_counts = counts / size_factors[:, None]
        return deseq2_counts, size_factors

    def normalize_counts(self, data):
        """
        Normalize counts using DESeq2-like method.
        """
        counts = np.array(data['expr_unstr_count'].tolist())
        if self.logmeans is None or self.filtered_genes is None:
            self.logmeans, self.filtered_genes = self.deseq2_norm_fit(counts)
        normalized_counts, size_factors = self.deseq2_norm_transform(counts, self.logmeans, self.filtered_genes)
        return normalized_counts

    def norm_transform(self, data):
        """
        Perform log2 transformation: log2(n + 1).
        """
        normalized_counts = self.normalize_counts(data)
        return np.log2(normalized_counts + 1)

    def vst(self, data):
        """
        Variance stabilizing transformation.
        """
        normalized_counts = self.normalize_counts(data)
        # Apply a simple approximation of VST
        return np.sqrt(normalized_counts + 3/8)

    def rlog(self, data):
        """
        Regularized log transformation.
        """
        normalized_counts = self.normalize_counts(data)
        # Apply a simple approximation of rlog
        return np.log(normalized_counts + 1)


    def prepare_data(self, method='ML', num_samples_to_simulate=None, normalization='log2'):
        # Split tumor samples
        self.tumor_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Tumor']
        tumor_train, tumor_test = train_test_split(self.tumor_samples, test_size=0.2, random_state=42)
        self.tumor_train, self.tumor_val = train_test_split(tumor_train, test_size=0.2, random_state=42)

        # Get original normal samples
        self.normal_samples = self.data_from_bq[self.data_from_bq['tissue_type'] == 'Normal']

        # Simulate normal samples
        simulator = simulators.AutoencoderSimulator(self.data_from_bq)
        preprocessed_data = simulator.preprocess_data()
        simulator.train_autoencoder(preprocessed_data)
        if method == 'ML':
            num_samples_to_simulate = len(self.tumor_samples) - len(self.normal_samples)
            self.simulated_normal_samples = simulator.simulate_samples(num_samples_to_simulate)
        elif method == 'OS':
            if num_samples_to_simulate is None:
                raise ValueError("Number of Simulated Samples Cannot be None for Outlier Sum Statistics")
            elif num_samples_to_simulate < 2*len(self.tumor_samples):
                raise ValueError("InSufficient Number of Simulated Normal Samples: Must be atleast 2X of tumor samples size")
            self.simulated_normal_samples = simulator.simulate_samples(num_samples_to_simulate)            

        # Combine original and simulated normal samples
        all_normal_samples = pd.concat([self.normal_samples, self.simulated_normal_samples])

        # Split normal samples
        normal_train, normal_test = train_test_split(all_normal_samples, test_size=0.2, random_state=42)
        normal_train, normal_val = train_test_split(normal_train, test_size=0.2, random_state=42)

        # Combine tumor and normal samples for each split
        self.X_train = pd.concat([self.tumor_train, normal_train])
        self.X_test = pd.concat([tumor_test, normal_test])
        self.X_val = pd.concat([self.tumor_val, normal_val])

        # Apply the chosen normalization method
        expr_col = 'expr_unstr_count'
        if normalization == 'log2':
            norm_func = self.norm_transform
        elif normalization == 'vst':
            norm_func = self.vst
        elif normalization == 'rlog':
            norm_func = self.rlog
        else:
            norm_func = self.normalize_counts

        def safe_normalize(x):
            try:
                return norm_func(pd.DataFrame({expr_col: [x]}))[0]
            except Exception as e:
                print(f"Error in normalization: {e}")
                return x  # Return original data if normalization fails




        # Apply log1p transformation to expression data if method is 'DE'
        if method == 'DE':
            self.X_train[expr_col] = self.X_train[expr_col].apply(safe_normalize)
            self.X_test[expr_col] = self.X_test[expr_col].apply(safe_normalize)
            self.X_val[expr_col] = self.X_val[expr_col].apply(safe_normalize)

        def expand_expr_col(df):
            expr_data = np.array(df['expr_unstr_count'].tolist())
            expr_df = pd.DataFrame(expr_data, columns=[f'gene_{i}' for i in range(expr_data.shape[1])])
            return pd.concat([df.drop('expr_unstr_count', axis=1).reset_index(drop=True), expr_df], axis=1)

        self.X_train = expand_expr_col(self.X_train)
        self.X_test = expand_expr_col(self.X_test)
        self.X_val = expand_expr_col(self.X_val)

        # Keep only the gene expression features
        gene_columns = [col for col in self.X_train.columns if col.startswith('gene_')]
        self.X_train = self.X_train[gene_columns]
        self.X_test = self.X_test[gene_columns]
        self.X_val = self.X_val[gene_columns]

        # Now add the gene code labels to the columns
        self.X_train.columns = self.gene_cols
        self.X_test.columns = self.gene_cols
        self.X_val.columns = self.gene_cols
        
        # Prepare labels
        self.y_train = np.concatenate([np.ones(len(self.tumor_train)), np.zeros(len(normal_train))])
        self.y_test = np.concatenate([np.ones(len(tumor_test)), np.zeros(len(normal_test))])
        self.y_val = np.concatenate([np.ones(len(self.tumor_val)), np.zeros(len(normal_val))])

        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val