import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class LRA:
    def __init__(self, classifier_model, dim_reduction_model):
        self.classifier_model = classifier_model
        self.dim_reduction_model = dim_reduction_model
        self.scaler = StandardScaler()

    def calculate_shap_values(self, X_test):
        # First, scale the data
        X_scaled = self.scaler.fit_transform(X_test)
        
        # Then, reduce the dimensionality of X_test
        X_reduced = self.dim_reduction_model.transform(X_scaled)

        def model_predict(X):
            return self.classifier_model.predict_proba(X)
        
        # Use X_reduced for background and SHAP calculations
        background = shap.sample(X_reduced, 100, random_state=42)
        
        explainer = shap.KernelExplainer(model_predict, background)
        shap_values = explainer.shap_values(X_reduced)
        
        # Handle binary classification case
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use values for positive class
        
        if hasattr(self.dim_reduction_model, 'components_'):
            lra_attributions = self.compute_lra_attributions(shap_values)
            return lra_attributions
        else:
            print(f"Back-projection not available for this dimensionality reduction method")
            return shap_values

    def compute_lra_attributions(self, shap_values_reduced):
        projection_matrix = self.dim_reduction_model.components_.T
        M = projection_matrix.shape[0]
        k = projection_matrix.shape[1]
        lra_attributions = np.zeros(M)

        for i in range(M):
            weighted_sum = np.sum([np.mean(shap_values_reduced[:, j]) * projection_matrix[i, j] for j in range(k)])
            normalization = np.sqrt(np.mean([projection_matrix[i, j] ** 2 for j in range(k)]))
            lra_attributions[i] = weighted_sum / normalization if normalization != 0 else 0
        
        return lra_attributions

    def run_single_method(self, X_test, y_test):
        lra_attributions = self.calculate_shap_values(X_test)
        return lra_attributions

    def plot_feature_importance(self, lra_attributions, feature_names=None, top_n=20):
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(lra_attributions))]
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(lra_attributions)
        }).sort_values('importance', ascending=False)

        top_features = feature_importance.head(top_n)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importances')
        plt.xlabel('Absolute Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()
        

####### ORIGINAL Working low rank attribution ##########################################################
# import shap
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import pandas as pd

# class LRA:
#     def __init__(self, model, n_components):
#         self.model = model
#         self.n_components = n_components
#         self.projection_matrix = None
#         self.scaler = StandardScaler()
#         self.dim_reduction = None

#     def _fit_transform(self, X, method, fit=True):
#         if method == 'pca':
#             if fit:
#                 self.dim_reduction = PCA(n_components=self.n_components)
#                 X_reduced = self.dim_reduction.fit_transform(X)
#                 self.projection_matrix = self.dim_reduction.components_.T  # M x k
#             else:
#                 X_reduced = self.dim_reduction.transform(X)
#         else:
#             raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
#         return X_reduced

#     def fit(self, X_train, y_train, method='pca'):
#         """
#         Fit the model on the reduced-dimensional space after applying PCA or other methods.
#         """
#         self.y = y_train  # Store y for LDA
#         X_scaled = self.scaler.fit_transform(X_train)
#         X_reduced = self._fit_transform(X_scaled, method, fit=True)
#         self.model.fit(X_reduced, y_train)

#     def calculate_shap_values(self, X_test, method='pca'):
#         """
#         Calculate SHAP values in the reduced-dimensional space.
#         """
#         X_scaled = self.scaler.transform(X_test)
#         X_reduced = self._fit_transform(X_scaled, method, fit=False)
        
#         # Create a wrapper function for the model
#         def model_predict(X):
#             return self.model.predict(X)
        
#         # Use KernelExplainer to calculate SHAP values
#         background = shap.sample(X_reduced, 100)  # Create background dataset
#         explainer = shap.KernelExplainer(model_predict, background)
#         shap_values = explainer.shap_values(X_reduced)
        
#         if self.projection_matrix is not None:
#             # Now compute LRA attributions based on SHAP values in reduced space
#             lra_attributions = self.compute_lra_attributions(shap_values)
#             return lra_attributions
#         else:
#             print(f"Back-projection not available for method: {method}")
#             return shap_values

#     def compute_lra_attributions(self, shap_values_reduced):
#         """
#         Compute the Low Rank Attribution (LRA) for each feature across all samples.
#         This will return an M x 1 vector, where M is the number of original features.
#         """
#         # shap_values_reduced: SHAP values in PCA component space (n_samples, n_components)
#         # projection_matrix: PCA loadings (M x k), where M is the number of original features, k is the number of components
        
#         M = self.projection_matrix.shape[0]  # Number of original features
#         k = self.projection_matrix.shape[1]  # Number of components
#         lra_attributions = np.zeros(M)

#         # Compute LRA for each feature i
#         for i in range(M):
#             # Numerator: Sum of SHAP values weighted by the loadings of feature i for each component
#             weighted_sum = np.sum([np.mean(shap_values_reduced[:, j]) * self.projection_matrix[i, j] for j in range(k)])
            
#             # Denominator: Normalization by RMS of the loadings for feature i across components
#             normalization = np.sqrt(np.mean([self.projection_matrix[i, j] ** 2 for j in range(k)]))
            
#             # Compute LRA for feature i
#             lra_attributions[i] = weighted_sum / normalization
        
#         return lra_attributions

#     def run_single_method(self, X_train, y_train, X_test, y_test, method):
#         """
#         Run the entire pipeline for a single dimensionality reduction method and return the LRA vector.
#         """
#         self.fit(X_train, y_train, method)
#         lra_attributions = self.calculate_shap_values(X_test, method)
#         return {method: lra_attributions}