from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import pandas as pd

class DRClustererBase:
    def __init__(self):
        self.eps = 1.0
        self.min_samples = 10
        self.num_clusters = 2
        # self.dbscan_eps = 0.6
        # self.dbscan_min_samples = 10
        # self.optics_eps = 0.9
        # self.optics_min_samples = 50 
        # self._dbscan_model = None
        # self._optics_model = None
        #   
    
    def calculate_ARI(self, label_group, cluster):
        scores = adjusted_rand_score(label_group, cluster)
        return scores 
    
    def calculate_NMI(self, label_group, cluster):
        scores = normalized_mutual_info_score(label_group, cluster)
        return scores 

    def prediction(self, model, embedding_data):
        return model.fit_predict(embedding_data)
        
    def get_clustering_performance(self):
        raise NotImplementedError()

    def make_scatter_plot(self):
        raise NotImplementedError()

## Question: How can we add customizations to these Clustering models? 
# class DBScanner(DRClustererBase):
#     def __init__(self):
#         super().__init__()

#     def model(self):
#         raise NotImplementedError()
    
class SpectralClust(DRClustererBase):
    def __init__(self):
        super().__init__()
    
    @property
    def model(self):
        model = SpectralClustering(n_clusters = self.num_clusters, assign_labels="discretize", random_state=0)
        return model 
    

class GMM(DRClustererBase):
    def __init__(self):
        super().__init__()

    @property
    def model(self):
        model = mixture.GaussianMixture(n_components = self.num_clusters, covariance_type='full')
        return model 


    
class KNN(DRClustererBase):
    def __init__(self):
        super().__init__()

    @property
    def model(self):
        model = KMeans(n_clusters = self.num_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        return model 
    
        
class Hierarchical(DRClustererBase):
    def __init__(self):
        super().__init__()
    
    @property
    def model(self):
        model = AgglomerativeClustering(n_clusters = self.num_clusters)
        return model 


        
    ## Next - Neural Network embeddings in parametric UMAP  


    # @numba.njit(fastmath=True)
    # def torus_euclidean_grad(x, y, torus_dimensions=(2*np.pi,2*np.pi)):
    #     """Standard euclidean distance.

    #     ..math::
    #         D(x, y) = \sqrt{\sum_i (x_i - y_i)^2}
    #     """
    #     distance_sqr = 0.0
    #     g = np.zeros_like(x)
    #     for i in range(x.shape[0]):
    #         a = abs(x[i] - y[i])
    #         if 2*a < torus_dimensions[i]:
    #             distance_sqr += a ** 2
    #             g[i] = (x[i] - y[i])
    #         else:
    #             distance_sqr += (torus_dimensions[i]-a) ** 2
    #             g[i] = (x[i] - y[i]) * (a - torus_dimensions[i]) / a
    #     distance = np.sqrt(distance_sqr)
    #     return distance, g/(1e-6 + distance)