import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer



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
            mapper = umap.UMAP(n_components = 2, random_state=143, n_jobs=-1, negative_gradient_method='bh').fit(self.data)
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
    
    def log10_transformer(self, list_of_cols, df:pd.DataFrame):
        df[gene_cols] = df[gene_cols] + 1
        transformer = FunctionTransformer(np.log10)
        df[list_of_cols] = transformer.fit_transform(df[list_of_cols])
        return df 
    
    

