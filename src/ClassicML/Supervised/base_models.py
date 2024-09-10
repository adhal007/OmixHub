import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA, FastICA, NMF, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from umap import UMAP
import logging
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.decomposition import PCA
import joblib

class BaseMLModel(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='single', random_state=42, prob_thresh=0.5, n_components=20):
        self.model_type = model_type
        self.random_state = random_state
        self.prob_thresh = prob_thresh
        self.n_components = n_components
        self.model = None
        self.dim_reducer = None
        self.chains = None
        self.num_labels = None
        self.scaler = StandardScaler()

    def fit(self, X, y):
        if self.model_type == 'ensemble':
            self.num_labels = y.shape[1] if len(y.shape) > 1 else 1
            self.chains = [ClassifierChain(base_estimator=self.model, order="random", random_state=i) 
                           for i in range(self.num_labels)]
            for chain in self.chains:
                chain.fit(X, y)
        else:
            self.model.fit(X, y)
        return self

    def predict(self, X):
        if self.model_type == 'ensemble':
            y_pred = np.array([chain.predict(X) for chain in self.chains])
            return y_pred.mean(axis=0) >= self.prob_thresh
        else:
            return self.model.predict(X)

    def predict_proba(self, X):
        if self.model_type == 'ensemble':
            y_pred = np.array([chain.predict_proba(X) for chain in self.chains])
            return y_pred.mean(axis=0)
        else:
            return self.model.predict_proba(X)

    def apply_dimensionality_reduction(self, X_train, X_test, X_val, method='PCA'):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_val_scaled = self.scaler.transform(X_val)

        method = method.upper()

        if method == 'PCA':
            self.dim_reducer = PCA(n_components=self.n_components)
        elif method == 'TSNE':
            self.dim_reducer = CustomTSNE(n_components=min(3, self.n_components), random_state=self.random_state)
        elif method == 'SPARSEPCA':
            self.dim_reducer = CustomSparsePCA(n_components=self.n_components, random_state=self.random_state)
        elif method == 'ICA':
            self.dim_reducer = CustomICA(n_components=self.n_components, random_state=self.random_state)
        elif method == 'NMF':
            self.dim_reducer = CustomNMF(n_components=self.n_components, random_state=self.random_state)
        elif method == 'LDA':
            self.dim_reducer = CustomLDA(n_components=self.n_components)
        elif method == 'KERNELPCA':
            self.dim_reducer = CustomKernelPCA(n_components=self.n_components, kernel='rbf', random_state=self.random_state)
        elif method == 'MDS':
            self.dim_reducer = CustomMDS(n_components=self.n_components, random_state=self.random_state)
        elif method == 'ISOMAP':
            self.dim_reducer = CustomIsomap(n_components=self.n_components)
        elif method == 'UMAP':
            self.dim_reducer = CustomUMAP(n_components=self.n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        X_train_reduced = self.dim_reducer.fit_transform(X_train_scaled)
        X_test_reduced = self.dim_reducer.transform(X_test_scaled)
        X_val_reduced = self.dim_reducer.transform(X_val_scaled)

        return X_train_reduced, X_test_reduced, X_val_reduced

    def run_model(self, X_train, X_test, y_train, y_test, reduce_dim=False, method='PCA', optimize_hyperparams=False):
        if reduce_dim:
            X_train_reduced, X_test_reduced, _ = self.apply_dimensionality_reduction(X_train, X_test, X_test, method=method)
        else:
            X_train_reduced, X_test_reduced = X_train, X_test

        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            self.optimize_hyperparameters(X_train_reduced, y_train)
            print("Hyperparameter optimization completed.")

        if isinstance(self, EnsembleModel):
            # For ensemble models, we need to reshape y to be 2D
            y_train_reshaped = y_train.reshape(-1, 1)
            self.fit(X_train_reduced, y_train_reshaped)
            y_pred = self.predict(X_test_reduced)
        else:
            self.fit(X_train_reduced, y_train)
            y_pred = self.predict(X_test_reduced)
        
        self.plot_performance(y_test, y_pred)
        
        # Calculate and print accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        return self, y_pred, accuracy

    def optimize_hyperparameters(self, X_train, y_train):
        # Implementation depends on the specific model
        raise NotImplementedError("Hyperparameter optimization not implemented for this model.")

    def calculate_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.model, X, y, cv=cv)
        return scores.mean(), scores.std()

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)
    
    def plot_performance(self, y_true, y_pred):
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

class SVMModel(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SVC(probability=True, random_state=self.random_state)

class RandomForestModel(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = RandomForestClassifier(random_state=self.random_state)

class NeuralNetworkModel(BaseMLModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = MLPClassifier(random_state=self.random_state)



class EnsembleModel:
    def __init__(self, base_estimators, voting='soft', n_components=10):
        self.base_estimators = base_estimators
        self.voting = voting
        self.n_components = n_components
        self.ensemble = None
        self.dim_reducer = None

    def run_model(self, X_train, X_test, y_train, y_test, reduce_dim=True, method='PCA'):
        if reduce_dim:
            self.dim_reducer = PCA(n_components=self.n_components)
            X_train_reduced = self.dim_reducer.fit_transform(X_train)
            X_test_reduced = self.dim_reducer.transform(X_test)
        else:
            X_train_reduced = X_train
            X_test_reduced = X_test

        self.ensemble = VotingClassifier(estimators=[(f'est{i}', est) for i, est in enumerate(self.base_estimators)], 
                                         voting=self.voting)
        
        self.ensemble.fit(X_train_reduced, y_train)
        y_pred = self.ensemble.predict(X_test_reduced)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall
        }

        return self, y_pred, metrics

    def cross_validate(self, X, y, cv=5):
        scores = cross_val_score(self.ensemble, X, y, cv=cv)
        return scores.mean(), scores.std()

    def save_model(self, filename):
        joblib.dump(self, filename)

    @classmethod
    def load_model(cls, filename):
        return joblib.load(filename)

class CustomTSNE:
    def __init__(self, n_components, random_state):
        self.n_components = n_components
        self.random_state = random_state
        self.tsne = TSNE(n_components=n_components, random_state=random_state)
        self.fit_data = None

    def fit(self, X):
        self.fit_data = X
        return self

    def transform(self, X):
        if self.fit_data is None:
            raise ValueError("TSNE must be fit before transform")
        combined = np.vstack((self.fit_data, X))
        embedded = self.tsne.fit_transform(combined)
        return embedded[len(self.fit_data):]

    def fit_transform(self, X):
        self.fit_data = X
        return self.tsne.fit_transform(X)

class CustomSparsePCA:
    def __init__(self, n_components, random_state):
        self.sparsepca = SparsePCA(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.sparsepca.fit(X)
        return self

    def transform(self, X):
        return self.sparsepca.transform(X)

    def fit_transform(self, X):
        return self.sparsepca.fit_transform(X)

class CustomICA:
    def __init__(self, n_components, random_state):
        self.ica = FastICA(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.ica.fit(X)
        return self

    def transform(self, X):
        return self.ica.transform(X)

    def fit_transform(self, X):
        return self.ica.fit_transform(X)

class CustomNMF:
    def __init__(self, n_components, random_state):
        self.nmf = NMF(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.nmf.fit(X)
        return self

    def transform(self, X):
        return self.nmf.transform(X)

    def fit_transform(self, X):
        return self.nmf.fit_transform(X)

class CustomLDA:
    def __init__(self, n_components):
        self.lda = LDA(n_components=n_components)

    def fit(self, X, y):
        self.lda.fit(X, y)
        return self

    def transform(self, X):
        return self.lda.transform(X)

    def fit_transform(self, X, y):
        return self.lda.fit_transform(X, y)

class CustomKernelPCA:
    def __init__(self, n_components, kernel, random_state):
        self.kpca = KernelPCA(n_components=n_components, kernel=kernel, random_state=random_state)

    def fit(self, X):
        self.kpca.fit(X)
        return self

    def transform(self, X):
        return self.kpca.transform(X)

    def fit_transform(self, X):
        return self.kpca.fit_transform(X)

class CustomMDS:
    def __init__(self, n_components, random_state):
        self.mds = MDS(n_components=n_components, random_state=random_state)
        self.fit_data = None

    def fit(self, X):
        self.fit_data = X
        return self

    def transform(self, X):
        if self.fit_data is None:
            raise ValueError("MDS must be fit before transform")
        combined = np.vstack((self.fit_data, X))
        embedded = self.mds.fit_transform(combined)
        return embedded[len(self.fit_data):]

    def fit_transform(self, X):
        self.fit_data = X
        return self.mds.fit_transform(X)

class CustomIsomap:
    def __init__(self, n_components):
        self.isomap = Isomap(n_components=n_components)

    def fit(self, X):
        self.isomap.fit(X)
        return self

    def transform(self, X):
        return self.isomap.transform(X)

    def fit_transform(self, X):
        return self.isomap.fit_transform(X)

class CustomUMAP:
    def __init__(self, n_components, random_state):
        self.umap = UMAP(n_components=n_components, random_state=random_state)

    def fit(self, X):
        self.umap.fit(X)
        return self

    def transform(self, X):
        return self.umap.transform(X)

    def fit_transform(self, X):
        return self.umap.fit_transform(X)