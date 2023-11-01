import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import umap


class umap_embedder:
    def __init__(self, data):
        self.data = data
        self.mapper = 'plane'
        self.mapper_list = ['plane', 'sphere', 'custom', 'hyperbolic']
        self.dims = ['2D', '3D']

    def mappers(self, mapper_type='plane'):
        if mapper_type not in self.mapper_list:
            raise ValueError("Incorrect mapper used for Umap")
        elif mapper_type == 'plane':
            mapper = umap.UMAP(random_state=42).fit(self.data)
        elif mapper_type == 'sphere':
            mapper = umap.UMAP(output_metric='haversine', random_state=42).fit(self.data)
        elif mapper_type == 'hyperbolic':
            mapper = umap.UMAP(output_metric='hyperboloid',random_state=42).fit(self.data)
        return mapper

    def create_embedded_df(self,  mapper:umap.UMAP, dim='2D'):
        if dim not in self.dims:
            raise ValueError("Incorrect number of dimensions provided for creating a dataframe")
        if dim == '2D':
            embed_x, embed_y = mapper.embedding_.T[0], mapper.embedding_.T[1]
            embed_df = pd.DataFrame({'embed_x': embed_x,
                                     'embed_y': embed_y}
                                     )
        else:
            embed_x = np.sin(mapper.embedding_[:, 0]) * np.cos(mapper.embedding_[:, 1])
            embed_y = np.sin(mapper.embedding_[:, 0]) * np.sin(mapper.embedding_[:, 1])
            embed_z = np.cos(mapper.embedding_[:, 0])
            embed_df = pd.DataFrame({'embed_x': embed_x,
                                     'embed_y': embed_y, 
                                     'embed_z': embed_z
                                     }
                                     )
        return embed_df 
    

class DRClusterer:
    def __init__(self, clust_method='knn'):
        self.clust_method = clust_method

    def perform_clustering(self):
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