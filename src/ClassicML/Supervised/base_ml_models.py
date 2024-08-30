from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain

from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from multiprocessing import Pool
from functools import partial
class BaseEnsembleClf:
    def __init__(self, model_clfs: dict, num_labels: int = None, random_state: int = 42, prob_thresh: float = None) -> None:
        
        self._model_clf = model_clfs
        self._model_names = None
        self.prob_thresh = prob_thresh

        if self.prob_thresh is None:
            raise ValueError("prob_thresh must be specified for jacard score")
    
        self.num_labels = num_labels
        self.random_state = random_state
        self.check_models()

    @property
    def model_names(self):
        if self._model_names is None:
            self._model_names = tuple([f"Chain {i}" for i in range(1, self.num_labels+1)])
            return self._model_names
        
    @property
    def model_clf(self):
        if self._model_clf is None:
            self._model_clf = {"clf_lr": LogisticRegression(solver='liblinear', penalty='l1', random_state=self.random_state)}
            return self._model_clf
        else:
            return self._model_clf

    def check_models(self):
        for clf_name, clf in self.model_clf.items():
            if not isinstance(clf, ClassifierMixin):
                raise ValueError(f"{clf_name} is not a valid scikit-learn classifier.")

    def train_clf(self, X_train, y_train, X_val, y_val):
        """
        Train a multi-label classifier using given sklearn model.
        
        This code is adapted from the scikit-learn documentation:
        https://scikit-learn.org/stable/auto_examples/linear_model/plot_sparse_logistic_regression_mnist.html#sphx-glr-auto-examples-linear-model-plot-sparse-logistic-regression-mnist-py

        The code has been expanded to be more generalizable to other sklearn models with any number of features.

            X_train: training data
            y_train: training labels
            X_val: validation data
            y_val: validation labels
        Returns: dictionary of model scores, model names, chain jaccard scores, and chains
        """
        ## can probabaly use multiprocessing.Pool here to parallelize the training of each model
        ovr_jaccard_scores = {}
        model_outs = {}
        for clf_name, clf in self.model_clf.items():
            ovr = OneVsRestClassifier(estimator=clf)
            ovr.fit(X_train, y=y_train)
            Y_val_ovr = ovr.predict(X=X_val)
            ovr_jaccard_score = jaccard_score(y_true=y_val, y_pred=Y_val_ovr, average="samples")
            ovr_jaccard_scores[clf_name] = ovr_jaccard_score

            # Fit an ensemble of model classifier chains and take the
            # take the average prediction of all the chains.
            chains = [ClassifierChain(base_estimator=clf, order="random", random_state=i) for i in range(self.num_labels)]
            for chain in chains:
                chain.fit(X=X_train, Y=y_train)

            Y_val_chains = np.array(object=[chain.predict(X=X_val) for chain in chains])
            chain_jaccard_scores = [
                jaccard_score(y_true=y_val, y_pred=Y_val_chain >= self.prob_thresh, average="samples")
                for Y_val_chain in Y_val_chains
            ]

            Y_val_ensemble = Y_val_chains.mean(axis=0)
            ensemble_jaccard_score = jaccard_score(
                y_true=y_val, y_pred=Y_val_ensemble >= self.prob_thresh, average="samples"
            )

            model_scores = [ovr_jaccard_score] + chain_jaccard_scores
            model_scores.append(ensemble_jaccard_score)


            model_names = tuple( ["Independent"] + [f"Chain {i}" for i in range(1, self.num_labels+1)] + ["Ensemble"])
            out_dict = {'model_scores': model_scores,
                        'model_names': model_names,
                        'chain_jaccard_scores': chain_jaccard_scores,
                        'chains': chains}
            model_outs[clf_name] = out_dict
        return model_outs
    
    def train_multiple_cls(self):
        raise NotImplementedError("Evaluation of multiple models not implemented yet")
       
    def test_clf(self, X_test, y_test, model_outs):
        """
        Test a multi-label classifier using Logistic Regression.
            X_test: test data (pandas dataframe of samples x features)
            y_test: test labels (pandas dataframe of samples x labels)
            chains: list of classifier chains (output of train_clf)
        return: ensemble_jaccard_score
        """
        testing_scores = {}
        for clf_name, clf in self.model_clf.items():
            chain_i = model_outs[clf_name]['chains']
            Y_test_chains = np.array(object=[chain.predict(X=X_test) for chain in chain_i])
            Y_test_ensemble = Y_test_chains.mean(axis=0)
            ensemble_jaccard_score = jaccard_score(
                y_true=y_test, y_pred=Y_test_ensemble >= self.prob_thresh, average="samples"
            )
            testing_scores[clf_name] = {'ensemble_jacard_score': ensemble_jaccard_score, 'y_pred_ensemble': Y_test_ensemble, 'y_test': y_test}
        return testing_scores
    
    def test_multiple_cls(self):
        raise NotImplementedError("Evaluation of multiple models not implemented yet")
    

    def plot_performance(self, model_scores, model_names, chain_jaccard_scores):
        """
        Plot the performance of the ensemble vs. independent classifier chains
            model_scores: list of scores for each model
            model_names: list of names for each model
            chain_jaccard_scores: list of scores for each chain
        """
        x_pos = np.arange(len(model_names))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.grid(True)
        ax.set_title("Classifier Chain Ensemble Performance Comparison")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation="vertical")
        ax.set_ylabel("Jaccard Similarity Score")
        ax.set_ylim([min(model_scores) * 0.9, max(model_scores) * 1.1])
        colors = ["r"] + ["b"] * len(chain_jaccard_scores) + ["g"]
        ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
        plt.tight_layout()
        plt.show()
        return 

    
    def eval_feats_single(self):
        raise NotImplementedError("Feature importance evaluation not implemented yet")
    

    def eval_feats_multiple(self):
        raise NotImplementedError("Feature importance evaluation not implemented yet")
    
    def ROC_AUC(self, Y_train, Y_val, X_val, model_outs):
        for clf_name, clf in self.model_clf.items():
            chains_i = model_outs[clf_name]['chains']
            n_classes = Y_train.shape[1]
            y_score = np.zeros(Y_val.shape)

            for i, chain in enumerate(chains_i):
                y_score += chain.predict_proba(X_val)

            y_score /= len(chains_i)

            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            specificity = dict()
            sensitivity = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(Y_val[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                
                tn, fp, fn, tp = confusion_matrix(Y_val[:, i], (y_score[:, i] >= 0.5)).ravel()
                specificity[i] = tn / (tn + fp)
                sensitivity[i] = tp / (tp + fn)
            sens_spec_df = pd.DataFrame({'sensitivity': sensitivity, 'specificity': specificity})
    
        return roc_auc, sens_spec_df
        
    def swarm_plot(self, n_classes, roc_auc, specificity, sensitivity):
        # Convert specificity and sensitivity to DataFrames
        specificity_df = pd.DataFrame({'Label': list(range(n_classes)), 'Specificity': specificity.values()})
        sensitivity_df = pd.DataFrame({'Label': list(range(n_classes)), 'Sensitivity': sensitivity.values()})

        # Create Swarm plots
        plt.figure(figsize=(12, 6))
        sns.swarmplot(x='Label', y='Specificity', data=specificity_df)
        plt.xlabel('Label')
        plt.ylabel('Specificity')
        plt.title('Specificity Swarm Plot')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.swarmplot(x='Label', y='Sensitivity', data=sensitivity_df)
        plt.xlabel('Label')
        plt.ylabel('Sensitivity')
        plt.title('Sensitivity Swarm Plot')
        plt.show()

    def plot_roc(self, n_classes, roc_auc, fpr, tpr):
        plt.figure(figsize=(10, 6))
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'ROC curve (area = {roc_auc[i]:.2f}) for label {i}')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    # def shap_value_calculation(self, X_train, Y_train, X_val):
    #     # raise NotImplementedError("SHAP value calculation not implemented yet"

    #     # Create an ensemble of logistic regression classifier chains
    #     chains = [ClassifierChain(LogisticRegression(), order="random", random_state=i) for i in range(10)]
    #     for chain in chains:
    #         chain.fit(X_train, Y_train)

    #     # Create a Shap explainer for the ensemble
    #     explainer = shap.Explainer(chains, X_train)

    #     # Calculate Shapley values for a specific sample or multiple samples (e.g., X_test)
    #     num_samples_to_explain = 10
    #     shap_values = explainer.shap_values(X_val[:num_samples_to_explain])

    #     # shap_values now contains the Shapley values for the ensemble model

    #     # Optionally, you can visualize the Shapley values for a specific sample
    #     shap.summary_plot(shap_values, X_val[:num_samples_to_explain], feature_names=X_train.columns)
    #     return shap, shap_values
    

## very slow implementation with multiprocessing
# def _train_single_clf(self, clf_name, clf, X_train, y_train, X_val, y_val):
#     """
#     Train a single classifier and return its outputs.
#     """
#     ovr = OneVsRestClassifier(estimator=clf)
#     ovr.fit(X_train, y=y_train)
#     Y_val_ovr = ovr.predict(X_val)
#     ovr_jaccard_score = jaccard_score(y_true=y_val, y_pred=Y_val_ovr, average="samples")

#     # Fit an ensemble of model classifier chains and take the average prediction
#     chains = [ClassifierChain(base_estimator=clf, order="random", random_state=i) for i in range(self.num_labels)]
#     for chain in chains:
#         chain.fit(X=X_train, Y=y_train)

#     Y_val_chains = np.array([chain.predict(X_val) for chain in chains])
#     chain_jaccard_scores = [
#         jaccard_score(y_true=y_val, y_pred=Y_val_chain >= self.prob_thresh, average="samples")
#         for Y_val_chain in Y_val_chains
#     ]

#     Y_val_ensemble = Y_val_chains.mean(axis=0)
#     ensemble_jaccard_score = jaccard_score(
#         y_true=y_val, y_pred=Y_val_ensemble >= self.prob_thresh, average="samples"
#     )

#     model_scores = [ovr_jaccard_score] + chain_jaccard_scores + [ensemble_jaccard_score]
#     model_names = ["Independent"] + [f"Chain {i}" for i in range(1, self.num_labels+1)] + ["Ensemble"]

#     return {
#         clf_name: {
#             'model_scores': model_scores,
#             'model_names': model_names,
#             'chain_jaccard_scores': chain_jaccard_scores,
#             'chains': chains
#         }
#     }

# def train_clf(self, X_train, y_train, X_val, y_val, num_processes=None):
#     """
#     Train multiple classifiers in parallel using multiprocessing.

#         X_train: training data
#         y_train: training labels
#         X_val: validation data
#         y_val: validation labels
#         num_processes: number of processes to use in parallel, defaults to None (all available cores)
#     Returns: dictionary of model scores, model names, chain jaccard scores, and chains
#     """
#     train_func = partial(self._train_single_clf, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)
    
#     with Pool(processes=num_processes) as pool:
#         results = pool.starmap(train_func, self.model_clf.items())

#     # Combine results from all classifiers
#     model_outs = {key: value for result in results for key, value in result.items()}
#     return model_outs 