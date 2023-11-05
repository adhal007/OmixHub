
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import hdbscan

import OmicsUtils.DimRedMappers.umap_embedders
import pandas as pd

class DRClustererBase:
    """Base/Parent class for clustering models 
    """
    def __init__(self, data, embedding_type='optimized'):
        
        ## clustering params 
        self.eps = None
        self.min_samples = None
        self.num_clusters = None

        ## embedding params 
        self.data = data
        self.embedding_type=embedding_type
        self._umap_mapper_embd = None 

    @property
    def umap_mapper_embd(self):
        """Get the embedding for optimized embedding

        Returns:
            _type_: _description_
        """
        if self._umap_mapper_embd is None:
            embd = OmicsUtils.DimRedMappers.umap_embedders.umap_embedder(self.data, embedding_type=self.embedding_type)
        return embd 
    
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

    def generate_clusters(self, 
        min_cluster_size):
        """
        Generate cluster object from umap embeddings
        """
        umap_embeddings = self.umap_mapper_embd
        clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                                metric='euclidean', 
                                cluster_selection_method='eom').fit(umap_embeddings)

        return clusters


    
        
# class Hierarchical(DRClustererBase):
#     def __init__(self):
#         super().__init__()
    
#     @property
#     def model(self):
#         model = AgglomerativeClustering(n_clusters = self.num_clusters)
#         return model 

