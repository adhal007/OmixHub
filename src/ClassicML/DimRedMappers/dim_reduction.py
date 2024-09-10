from sklearn.decomposition import PCA, SparsePCA, FastICA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap
from umap import UMAP

class BaseDimReduction:
    def __init__(self, n_components, random_state=None):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None

    def fit(self, X, y=None):
        if y is not None and self.model.fit.__code__.co_argcount > 2:
            self.model.fit(X, y)
        else:
            self.model.fit(X)
        return self

    def transform(self, X):
        return self.model.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

class PCAReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(n_components, random_state)
        self.model = PCA(n_components=self.n_components, random_state=self.random_state)

class TSNEReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(min(3, n_components), random_state)
        self.model = TSNE(n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y=None):
        # TSNE doesn't have a separate fit method
        return self

    def transform(self, X):
        # TSNE doesn't have a separate transform method, so we use fit_transform
        return self.model.fit_transform(X)

class SparsePCAReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(n_components, random_state)
        self.model = SparsePCA(n_components=self.n_components, random_state=self.random_state)

class ICAReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(n_components, random_state)
        self.model = FastICA(n_components=self.n_components, random_state=self.random_state)

class LDAReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(n_components, random_state)
        self.model = LDA(n_components=self.n_components)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def fit_transform(self, X, y):
        return self.model.fit_transform(X, y)

class KernelPCAReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(n_components, random_state)
        self.model = KernelPCA(n_components=self.n_components, kernel='rbf', random_state=self.random_state)

class IsomapReduction(BaseDimReduction):
    def __init__(self, n_components):
        super().__init__(n_components)
        self.model = Isomap(n_components=self.n_components)

class UMAPReduction(BaseDimReduction):
    def __init__(self, n_components, random_state=None):
        super().__init__(n_components, random_state)
        self.model = UMAP(n_components=self.n_components, random_state=self.random_state)

def get_dim_reduction(method, n_components, random_state=None):
    dim_reduction_methods = {
        'PCA': PCAReduction,
        'TSNE': TSNEReduction,
        'SPARSEPCA': SparsePCAReduction,
        'ICA': ICAReduction,
        'LDA': LDAReduction,
        'KERNELPCA': KernelPCAReduction,
        'ISOMAP': IsomapReduction,
        'UMAP': UMAPReduction
    }
    
    if method.upper() not in dim_reduction_methods:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    reduction_class = dim_reduction_methods[method.upper()]
    
    # Check if the reduction class uses random_state
    if 'random_state' in reduction_class.__init__.__code__.co_varnames:
        return reduction_class(n_components, random_state)
    else:
        return reduction_class(n_components)