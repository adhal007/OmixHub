
## sklearn packages 
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
## optimizer packages 
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial, space_eval
## python imports  
import hdbscan
import numpy as np
import random 
import umap
import pandas as pd 
## omics utils imports 
import OmicsUtils.DimRedMappers.clusterer
import OmicsUtils.DimRedMappers.umap_embedders

## example usage of getter and setter class 
# class Geeks: 
#      def __init__(self): 
#           self._age = 0
    
#      # using property decorator 
#      # a getter function 
#      @property
#      def age(self): 
#          print("getter method called") 
#          return self._age 
    
#      # a setter function 
#      @age.setter 
#      def age(self, a): 
#          if(a < 18): 
#             raise ValueError("Sorry you age is below eligibility criteria") 
#          print("setter method called") 
#          self._age = a 

# mark = Geeks() 

# mark.age = 19

# print(mark.age)

class DRClusterer:
    """Base/Parent class for clustering models : Contains all the functions/methods generalizable across clustering models
    """
    def __init__(self, 
                 data, 
                 clust_model_name = None,
                 clust_params=None
                ):
        
        ## clustering params 
        self.eps = None
        self.min_samples = None

        ## embedding params 
        self.data = data

        ## cluster model params 
        self.clust_allowed_models = ['knn', 'gmm', 'hdbscan']
        self.clust_params = clust_params 
        
        self._clust_model_name = clust_model_name


    @property
    def clust_model_name(self): 
        if self._clust_model_name is None:
            return 'hdbscan'
        else:
            return self._clust_model_name 
            
    def clust_model(self, params):
        if self.clust_model_name  == 'hdbscan':
            clust_model = hdbscan.HDBSCAN(min_cluster_size = params['min_cluster_size'],
                                    metric='euclidean', 
                                    cluster_selection_method='eom')
        elif self.clust_model_name == 'knn':
            clust_model = KMeans(n_clusters = params["num_clusters"], init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)  
        return clust_model 


    def calculate_ARI(self, label_group, cluster):
        """_summary_

        Args:
            label_group (_type_): _description_
            cluster (_type_): _description_

        Returns:
            _type_: _description_
        """
        scores = adjusted_rand_score(label_group, cluster)
        return scores 
    
    def calculate_NMI(self, label_group, cluster):
        """_summary_

        Args:
            label_group (_type_): _description_
            cluster (_type_): _description_

        Returns:
            _type_: _description_
        """
        scores = normalized_mutual_info_score(label_group, cluster)
        return scores 

    def prediction(self, model, embedding_data):
        """_summary_

        Args:
            model (_type_): _description_
            embedding_data (_type_): _description_

        Returns:
            _type_: _description_
        """
        return model.fit_predict(embedding_data)
        
    def get_clustering_performance(self, label_group, cluster):
        """_summary_

        Args:
            label_group (_type_): _description_
            cluster (): _description_

        Returns:
            _type_: _description_
        """
        ari_scores = self.calculate_ARI(label_group, cluster)
        nmi_scores = self.calculate_NMI(label_group, cluster)
        clust_performance_dict = {'ARI': ari_scores, 'NMI': nmi_scores}
        return clust_performance_dict



###############################################
###############################################
class ClusteringOptimizer(DRClusterer):
    def __init__(self, data, clust_model_name=None, clust_params=None, search_space=None):
        super().__init__(data=data,clust_model_name=clust_model_name, clust_params=clust_params)

        self.umap_default_params = {"n_neighbors":None}

        if search_space is None:
            self.search_space = {
                "n_neighbors": range(12,16),
                "n_components": range(3,16),
                "min_cluster_size":range(2,16),
                # "min_ncluster": range(2,10),
                "random_state": 42
            } 

            self.hp_search_space = {"n_neighbors": hp.choice('n_neighbors', range(3, 16)),
                                   "n_components": hp.choice('n_components', range(3, 16)),
                                   "min_cluster_size": hp.choice('min_cluster_size', range(2, 16)),
                                   "random_state": 42}

        self.data = data
     
    def score_clusters(self, clusters, prob_threshold = 0.05):
        """
        Returns the label count and cost of a given cluster supplied from running hdbscan
        """
        
        cluster_labels = clusters.labels_
        label_count = len(np.unique(cluster_labels))
        total_num = len(clusters.labels_)
        cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)/total_num)
        
        return label_count, cost



    def generate_clusters(self, message_embeddings,
                        n_neighbors,
                        n_components,
                        clust_params,   
                        random_state = None):
        """
        Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
        """

        ## can substitute different umap embeddings         
        umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors, 
                                    n_components=n_components, 
                                    metric='cosine', 
                                    random_state=random_state)
                                .fit_transform(X=message_embeddings))


        clusters =  self.clust_model(params=clust_params).fit(umap_embeddings)
        return clusters, umap_embeddings
    
    def random_search(self, embeddings, space, num_evals):
        """
        Randomly search hyperparameter space and limited number of times 
        and return a summary of the results
        """
        
        results = []
        
        for i in range(num_evals):
            n_neighbors = random.choice(seq=space['n_neighbors'])
            n_components = random.choice(seq=space['n_components'])
            min_cluster_size = random.choice(seq=space['min_cluster_size'])
            clust_params = {"min_cluster_size": min_cluster_size}
            
            clusters = self.generate_clusters(message_embeddings=embeddings, 
                                        n_neighbors = n_neighbors, 
                                        n_components = n_components, 
                                        clust_params=clust_params, 
                                        random_state = 42)
        
            label_count, cost = self.score_clusters(clusters=clusters, prob_threshold = 0.05)
                    
            results.append([i, n_neighbors, n_components, min_cluster_size, 
                            label_count, cost])
        
        result_df = pd.DataFrame(data=results, columns=['run_id', 'n_neighbors', 'n_components', 
                                                'min_cluster_size', 'label_count', 'cost'])
        
        return result_df.sort_values(by='cost')

    def objective(self, params, embeddings, label_lower, label_upper):
        """
        Objective function for hyperopt to minimize, which incorporates constraints
        on the number of clusters we want to identify
        """
        
        clusters = self.generate_clusters(message_embeddings=embeddings, 
                                    n_neighbors = params['n_neighbors'], 
                                    n_components = params['n_components'], 
                                    clust_params= params, 
                                    random_state = params['random_state'])
        
        label_count, cost = self.score_clusters(clusters=clusters, prob_threshold = 0.1)
        
        #15% penalty on the cost function if outside the desired range of groups
        if (label_count < label_lower) | (label_count > label_upper):
            penalty = 0.05 
        else:
            penalty = 0
        
        loss = cost + penalty
        
        return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}
    

    def bayesian_search(self, embeddings, label_lower, label_upper, max_evals=100):
        """
        Perform bayseian search on hyperopt hyperparameter space to minimize objective function
        """
        
        trials = Trials()
        fmin_objective = partial(fn=self.objective, embeddings=embeddings, label_lower=label_lower, label_upper=label_upper)
        best = fmin(fn=fmin_objective, 
                    space = self.hp_search_space, 
                    algo=tpe.suggest,
                    max_evals=max_evals, 
                    trials=trials)

        best_params = space_eval(space=self.hp_search_space, hp_assignment=best)
        print ('best:')
        print (best_params)
        print (f"label count: {trials.best_trial['result']['label_count']}")
        
        best_clusters = self.generate_clusters(message_embeddings=embeddings, 
                                        n_neighbors = best_params['n_neighbors'], 
                                        n_components = best_params['n_components'], 
                                        clust_params= best_params,
                                        random_state = best_params['random_state'])
        
        return best_params, best_clusters, trials
    

    def optimized_results(self):
        raise NotImplementedError()
# class Hierarchical(DRClustererBase):
#     def __init__(self):
#         super().__init__()
    
#     @property
#     def model(self):
#         model = AgglomerativeClustering(n_clusters = self.num_clusters)
#         return model 


# class SpectralClust(OmicsUtils.DimRedMappers.clusterer.DRClustererBase):
#     """Clustering by Spectral method

#     Args:
#         DRClustererBase (_type_): child class of Clusterer base class for reduced embeddings 
#     """
#     def __init__(self, num_clusters, data:pd.DataFrame)->None:
#         super().__init__(data=data)
#         self.num_clusters = num_clusters
#         self._model = None 

#     @property
#     def model(self):
#         """_summary_

#         Returns:
#             _type_: _description_
#         """
#         if self._model is None:
#             model_x = SpectralClustering(n_clusters = self.num_clusters, assign_labels="discretize", random_state=0)
#         return model_x 
    

# class GMM(OmicsUtils.DimRedMappers.clusterer.DRClustererBase):
#     def __init__(self, num_clusters, data:pd.DataFrame)->None:
#         super().__init__(data=data)
#         self._model = None 
#         self.num_clusters = num_clusters

#     @property
#     def model(self):
#         if self._model is None:
#             model_x = mixture.GaussianMixture(n_components = self.num_clusters, covariance_type='full')
#         return model_x 

# class KNN(OmicsUtils.DimRedMappers.clusterer.DRClustererBase):
#     def __init__(self, num_clusters, data:pd.DataFrame)->None:
#         super().__init__(data=data)
#         self.num_clusters = num_clusters

#     @property
#     def model(self):
#         model = KMeans(n_clusters = self.num_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
#         return model