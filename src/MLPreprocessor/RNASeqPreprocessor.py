import src.ClassicML.DataAug.simulators as simulators
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


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
    
    def load_gene_mapping(self):
        mapping_file = '/Users/abhilashdhal/Projects/personal_docs/data/Transcriptomics/data/gene_annotation/gene_id_to_gene_name_mapping.csv'
        mapping_df = pd.read_csv(mapping_file)
        return dict(zip(mapping_df['gene_id'], mapping_df['gene_name']))
    
    def get_gene_name(self, gene_id):
        # Remove version number from gene_id if present
        return self.gene_id_to_name[gene_id]

    def prepare_data(self):
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
        num_samples_to_simulate = len(self.tumor_samples) - len(self.normal_samples)
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

        # Apply log1p transformation to expression data
        expr_col = 'expr_unstr_count'
        self.X_train[expr_col] = self.X_train[expr_col].apply(lambda x: np.log1p(np.array(x)))
        self.X_test[expr_col] = self.X_test[expr_col].apply(lambda x: np.log1p(np.array(x)))
        self.X_val[expr_col] = self.X_val[expr_col].apply(lambda x: np.log1p(np.array(x)))

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
        
        # # Now get the gene names from the gene ids
        # self.X_train.columns = [self.get_gene_name(col) for col in self.X_train.columns]
        # self.X_test.columns = [self.get_gene_name(col) for col in self.X_test.columns]
        # self.X_val.columns = [self.get_gene_name(col) for col in self.X_val.columns]
        # Prepare labels
        self.y_train = np.concatenate([np.ones(len(self.tumor_train)), np.zeros(len(normal_train))])
        self.y_test = np.concatenate([np.ones(len(tumor_test)), np.zeros(len(normal_test))])
        self.y_val = np.concatenate([np.ones(len(self.tumor_val)), np.zeros(len(normal_val))])
        return self.X_train, self.X_test, self.X_val, self.y_train, self.y_test, self.y_val