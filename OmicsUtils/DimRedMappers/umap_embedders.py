import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer



import pandas as pd
import umap


class umap_embedder:
    def __init__(self, 
                 data, 
                 embedding_type = None):
        self.data = data
        self.mapper = 'plane'
        self.mapper_list = ['plane', 'sphere', 'custom', 'hyperbolic']
        self.dims = ['2D', '3D']
        self.allowed_embedding_types = ['optimized']
        self.embedding_type = embedding_type
        self.n_neighbors = None 
        self.n_components = None 
        self.random_state = None 
        self._embedding = None 

    @property
    def embedding(self):
        if self._embedding is None:
            if self.embedding_type is None:
                embedding = self.default_embedding()
            elif self.embedding_type == 'optimized':
                embedding = self.optimized_embedding()
        return embedding
    
    def default_embedding(self, mapper_type='plane'):
        if mapper_type not in self.mapper_list:
            raise ValueError("Incorrect mapper used for Umap")
        elif mapper_type == 'plane':
            mapper = umap.UMAP(n_components = 2, random_state=143, n_jobs=-1, negative_gradient_method='bh').fit(self.data)
        elif mapper_type == 'sphere':
            mapper = umap.UMAP(output_metric='haversine', random_state=42).fit(self.data)
        elif mapper_type == 'hyperbolic':
            mapper = umap.UMAP(output_metric='hyperboloid',random_state=42).fit(self.data)
        return mapper

    def optimized_embedding(self):

        umap_embeddings = umap.UMAP(negative_gradient_method='bh', 
                                    n_neighbors=n_neighbors, 
                                    n_components=n_components, 
                                    random_state=random_state).fit_transform(self.data)
        return umap_embeddings

    def create_embedded_df(self,  mapper:umap.UMAP, dim='2D', embedding=None):
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
    
    

