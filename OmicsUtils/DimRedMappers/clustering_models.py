from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn import mixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans

import OmicsUtils.DimRedMappers.clusterer
import pandas as pd 

class SpectralClust(OmicsUtils.DimRedMappers.clusterer.DRClustererBase):
    """Clustering by Spectral method

    Args:
        DRClustererBase (_type_): child class of Clusterer base class for reduced embeddings 
    """
    def __init__(self, num_clusters, data:pd.DataFrame)->None:
        super().__init__(data=data)
        self.num_clusters = num_clusters
        self._model = None 

    @property
    def model(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self._model is None:
            model_x = SpectralClustering(n_clusters = self.num_clusters, assign_labels="discretize", random_state=0)
        return model_x 
    

class GMM(OmicsUtils.DimRedMappers.clusterer.DRClustererBase):
    def __init__(self, num_clusters, data:pd.DataFrame)->None:
        super().__init__(data=data)
        self._model = None 
        self.num_clusters = num_clusters

    @property
    def model(self):
        if self._model is None:
            model_x = mixture.GaussianMixture(n_components = self.num_clusters, covariance_type='full')
        return model_x 

class KNN(OmicsUtils.DimRedMappers.clusterer.DRClustererBase):
    def __init__(self, num_clusters, data:pd.DataFrame)->None:
        super().__init__(data=data)
        self.num_clusters = num_clusters

    @property
    def model(self):
        model = KMeans(n_clusters = self.num_clusters, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        return model 