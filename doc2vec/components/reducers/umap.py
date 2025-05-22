import numpy as np
import umap

from typing import Callable

from doc2vec.components.reducer import DimReducer



class UMAP(DimReducer):
    """
    Uniform Manifold Approximation and Projection (UMAP) dimensionality reducer.

    This class wraps the `umap-learn` library's UMAP implementation in a consistent interface.

    See Also
    --------
    - UMAP: https://umap-learn.readthedocs.io/en/latest.
    """
    
    def __init__(self, n_neighbors: float = 15, n_components: int = 2, metric: str|Callable = 'cosine', min_dist: float = 0.1, **kwargs):
        """
        Parameters
        ----------
        n_neighbors : float, optional
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller values
            result in more local data being preserved. Recommended range is `[2, 100]`.
            Default is `15`.
        n_components : int, optional
            The dimensionality of the reduced space. Typically `2` for visualization.
            Default is `2`.
        metric : str | Callable,  optional
            The metric, by default `'cosine'`, to use to compute distances in high dimensional space.
            If a string is passed it must match a valid predefined metric, check the official documentation
            for supported distances.
        min_dist : float, optional
            The effective minimum distance between embedded points. Smaller values
            produce more clustered embeddings; larger values result in a more even
            spread. Default is `0.1`.
        **kwargs : dict
            Additional keyword arguments passed to `umap.UMAP`.
        """
        super().__init__()
        
        self.dimred = umap.UMAP(
            n_neighbors = n_neighbors,
            n_components = n_components,
            metric = metric,
            min_dist = min_dist,
            **kwargs
        )
        return


    def fit(self, x: np.ndarray) -> None:
        self.dimred = self.dimred.fit(x)
        return None


    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.dimred.fit_transform(x)


    def transform(self, x: np.ndarray) -> np.ndarray:
        return self.dimred.transform(x)
