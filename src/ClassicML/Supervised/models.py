

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import rankdata
import src.ClassicML.DataAug.simulators as simulators
from sklearn.decomposition import PCA, SparsePCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
from hpsklearn import HyperoptEstimator, any_classifier
from hpsklearn.components import svc, random_forest_classifier, mlp_classifier
from hyperopt import hp
from hyperopt import tpe
import logging
# Custom Conditional Inference Random Forest (simplified version)
class ConditionalInferenceRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=None):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    
    def fit(self, X, y):
        self.rf.fit(X, y)
        return self
    
    def predict(self, X):
        return self.rf.predict(X)
    
    def predict_proba(self, X):
        return self.rf.predict_proba(X)

# Custom Weighted Subspace Random Forest (simplified version)
class WeightedSubspaceRandomForest(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, random_state=None):
        self.rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.feature_weights = None
    
    def fit(self, X, y):
        # Implement feature weighting here (simplified)
        self.feature_weights = np.random.rand(X.shape[1])
        self.rf.fit(X * self.feature_weights, y)
        return self
    
    def predict(self, X):
        return self.rf.predict(X * self.feature_weights)
    
    def predict_proba(self, X):
        return self.rf.predict_proba(X * self.feature_weights)

class ExperimentRunner:
    def __init__(self, data_from_bq, gene_cols):
        self.data_from_bq = data_from_bq
        self.tumor_samples = None
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

        # Prepare labels
        self.y_train = np.concatenate([np.ones(len(self.tumor_train)), np.zeros(len(normal_train))])
        self.y_test = np.concatenate([np.ones(len(tumor_test)), np.zeros(len(normal_test))])
        self.y_val = np.concatenate([np.ones(len(self.tumor_val)), np.zeros(len(normal_val))])

    def apply_dimensionality_reduction(self, method='PCA', n_components=20):
        expr_col = 'expr_unstr_count'
        X_train = np.vstack(self.X_train[expr_col].apply(pd.Series).values)
        X_test = np.vstack(self.X_test[expr_col].apply(pd.Series).values)
        X_val = np.vstack(self.X_val[expr_col].apply(pd.Series).values)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_val_scaled = scaler.transform(X_val)

        if method == 'PCA':
            reducer = PCA(n_components=n_components)
            X_train_reduced = reducer.fit_transform(X_train_scaled)
            X_test_reduced = reducer.transform(X_test_scaled)
            X_val_reduced = reducer.transform(X_val_scaled)
        elif method == 'TSNE':
            tsne_components = min(3, n_components)  # Ensure t-SNE uses at most 3 components
            reducer = TSNE(n_components=tsne_components, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train_scaled)
            # For t-SNE, we need to fit_transform for test and validation sets separately
            X_test_reduced = TSNE(n_components=tsne_components, random_state=42).fit_transform(X_test_scaled)
            X_val_reduced = TSNE(n_components=tsne_components, random_state=42).fit_transform(X_val_scaled)
        elif method == 'SparsePCA':
            reducer = SparsePCA(n_components=n_components, random_state=42)
            X_train_reduced = reducer.fit_transform(X_train_scaled)
            X_test_reduced = reducer.transform(X_test_scaled)
            X_val_reduced = reducer.transform(X_val_scaled)

        return X_train_reduced, X_test_reduced, X_val_reduced
    
    def evaluate_variable_importance(self, classifier, X_reduced, y, original_features, reducer, n_top_features=15):
        # Create a background dataset for SHAP
        background = shap.sample(X_reduced, 200)  # Use 100 samples as background

        # Create a SHAP explainer on the reduced feature space
        if isinstance(classifier, RandomForestClassifier) or isinstance(classifier, ConditionalInferenceRandomForest) or isinstance(classifier, WeightedSubspaceRandomForest):
            explainer = shap.TreeExplainer(classifier)
        elif isinstance(classifier, SVC):
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
        elif isinstance(classifier, MLPClassifier):
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
        else:
            raise ValueError("Unsupported classifier type for SHAP analysis")

        # Calculate SHAP values for the reduced features
        shap_values = explainer.shap_values(X_reduced)

        # For binary classification, we're interested in the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        if isinstance(reducer, PCA):
            # For PCA, we can use the components_ attribute
            components = reducer.components_
        elif isinstance(reducer, TSNE):
            # For t-SNE, we need to approximate the mapping
            # We'll use the correlation between original features and t-SNE components
            components = np.array([np.corrcoef(original_features[:, i], X_reduced[:, j])[0, 1] 
                                   for i in range(original_features.shape[1]) 
                                   for j in range(X_reduced.shape[1])]).reshape(original_features.shape[1], X_reduced.shape[1])
        else:
            raise ValueError("Unsupported dimensionality reduction method")

        # Calculate Low Rank Attributions (LRA)
        k = components.shape[0]  # number of components
        feature_importance = np.zeros(original_features.shape[1])
        for i in range(original_features.shape[1]):
            numerator = np.sum(shap_values * components[:, i])
            denominator = np.sqrt(np.mean(components[:, i]**2))
            feature_importance[i] = numerator / denominator if denominator != 0 else 0

        # Get indices of top features
        top_features = np.argsort(np.abs(feature_importance))[::-1][:n_top_features]

        return top_features, feature_importance[top_features], shap_values

    def plot_feature_importances(self, feature_importances, feature_ids, title):
        plt.figure(figsize=(10, 12))
        y_pos = np.arange(len(feature_ids))
        
        # Get gene names for the feature IDs
        feature_names = [f"{self.get_gene_name(id)} ({id})" for id in feature_ids]
        
        plt.barh(y_pos, feature_importances, align='center')
        plt.yticks(y_pos, feature_names)
        plt.xlabel('Feature Importance')
        plt.title(title)
        plt.tight_layout()
        plt.show()


    def train_and_evaluate(self, classifier, X_train, X_test, X_val, original_features, reducer):
        clf = classifier.fit(X_train, self.y_train)

        # ROC and AUC
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Sensitivity at 99% specificity
        specificity = 1 - fpr
        idx = np.argmin(np.abs(specificity - 0.99))
        sensitivity_at_99_spec = tpr[idx]

        # Precision and Recall
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)

        # Variable importance
        top_features, feature_importance, _ = self.evaluate_variable_importance(clf, X_test, self.y_test, original_features, reducer)

        return {
            'ROC_AUC': roc_auc,
            'Sensitivity_at_99_spec': sensitivity_at_99_spec,
            'Average_Precision': avg_precision,
            'Top_Features': top_features,
            'Feature_Importance': feature_importance
        }

    def optimize_hyperparameters(self, classifier_type, X_train, y_train):
        # Set hyperopt logger to ERROR level to suppress less critical messages
        logging.getLogger('hyperopt').setLevel(logging.ERROR)

        if classifier_type == 'SVM':
            classifier = svc('my_svc',
                             C=hp.loguniform('svm_C', np.log(1e-5), np.log(1e5)),
                             kernel=hp.choice('kernel', ['rbf', 'linear']),
                             gamma=hp.loguniform('gamma', np.log(1e-5), np.log(1e5)),
                             probability=True)
        elif classifier_type == 'Random Forest':
            classifier = random_forest_classifier('my_rf')
        elif classifier_type == 'Neural Network':
            classifier = mlp_classifier('my_mlp')
        else:
            raise ValueError("Unsupported classifier type for hyperparameter optimization")

        estim = HyperoptEstimator(
            classifier=classifier,
            preprocessing=[],
            algo=tpe.suggest,
            max_evals=100,
            trial_timeout=300,
            verbose=0  # Set to 0 to suppress HyperoptEstimator's own verbose output
        )
        
        try:
            estim.fit(X_train, y_train)
            return estim.best_model()['learner']
        except Exception as e:
            print(f"Error during hyperparameter optimization for {classifier_type}: {str(e)}")
            # Return a default classifier if optimization fails
            if classifier_type == 'SVM':
                return SVC(probability=True, random_state=42)
            elif classifier_type == 'Random Forest':
                return RandomForestClassifier(random_state=42)
            elif classifier_type == 'Neural Network':
                return MLPClassifier(random_state=42)
    
    def run_experiment(self):
        self.prepare_data()

        dimensionality_reduction_methods = [
            ('PCA', 50),
            ('TSNE', 2),
        ]
        classifiers = {
            'SVM': 'SVM',
            'Random Forest': 'Random Forest',
            'Neural Network': 'Neural Network',
        }

        classification_results = {}

        expr_col = 'expr_unstr_count'
        original_features_train = np.vstack(self.X_train[expr_col].apply(pd.Series).values)
        original_features_test = np.vstack(self.X_test[expr_col].apply(pd.Series).values)

        for dr_method, n_components in dimensionality_reduction_methods:
            X_train_reduced, X_test_reduced, X_val_reduced = self.apply_dimensionality_reduction(method=dr_method, n_components=n_components)

            # Get the reducer used
            if dr_method == 'PCA':
                reducer = PCA(n_components=n_components)
            elif dr_method == 'TSNE':
                reducer = TSNE(n_components=n_components, random_state=42)
            reducer.fit(original_features_train)

            for clf_name, clf_type in classifiers.items():
                # Optimize hyperparameters
                optimized_clf = self.optimize_hyperparameters(clf_type, X_train_reduced, self.y_train)
                
                # Train and evaluate with optimized classifier
                results = self.train_and_evaluate(optimized_clf, X_train_reduced, X_test_reduced, X_val_reduced, original_features_test, reducer)
                
                # Get top gene IDs
                top_gene_ids = [self.gene_cols[i] for i in results['Top_Features']]
                
                # Plot feature importances for top genes
                top_feature_importances = results['Feature_Importance']
                self.plot_feature_importances(top_feature_importances, top_gene_ids, 
                                              f"Feature Importances for Top Genes - {dr_method} {clf_name}")
                
                # Add gene information to the results
                results['Top_Genes'] = [f"{self.get_gene_name(id)} ({id})" for id in top_gene_ids]
                
                # Add best hyperparameters to the results
                results['Best_Hyperparameters'] = optimized_clf.get_params()
                
                classification_results[f"{dr_method}_{clf_name}"] = results

        return classification_results