import numpy as np
import umap

from abc import ABC, abstractmethod
from typing import Callable



class DimReducer(ABC):
    """
    Abstract class, it represents a generic dimensionality reducer.
    """

    @abstractmethod
    def fit(self, x :np.ndarray) -> None:
        """
        Fit ``x`` into an embedded space.

        Parameters
        ----------
        x : numpy.array of shape \(M, N)
            One sample per row.
        """
        pass


    @abstractmethod
    def fit_transform(self, x :np.ndarray) -> np.ndarray:
        """
        Fit ``x`` into an embedded space and return that transformed output.

        Parameters
        ----------
        x : numpy.array of shape \(M, N)
            One sample per row.

        Returns
        -------
        x_new : numpy.array of shape \(M, K)
            Embedding of the training data in low-dimensional space.
        """
        pass


    @abstractmethod
    def transform(self, x :np.ndarray) -> np.ndarray:
        """
        Transform ``x`` into the existing embedded space and
        return that transformed output.
    
        Parameters
        ----------
        x : numpy.array of shape \(M, N)
            New data to be transformed.
        
        Returns
        -------
        x_new : numpy.array of shape \(M, K)
            Embedding of the new data in low-dimensional space.
        """
        pass



class UMAP(DimReducer):
    """
    Dimensional reduction with UMAP.

    See Also
    --------
    - UMAP: https://umap-learn.readthedocs.io/en/latest/
    """

    def __init__(self, n_neighbors :float=15, n_components :int=2, metric :str|Callable='cosine', min_dist :float=0.1, **kwargs):
        """
        Parameters
        ----------
        n_neighbors : float, optional
            The size of local neighborhood (in terms of number of neighboring
            sample points) used for manifold approximation. Larger values
            result in more global views of the manifold, while smaller
            values result in more local data being preserved. In general
            values should be in the range ``2`` to ``100``, by default ``15``.
        n_components : int, optional
            The dimension of the space to embed into. This defaults to ``2`` to
            provide easy visualization, but can reasonably be set to any
            integer value in the range ``2`` to ``100``.
        metric : str | Callable,  optional
            The metric, by default ``'cosine'``, to use to compute distances in high dimensional space.
            If a string is passed it must match a valid predefined metric, check the official documentation
            for supported distances.
        min_dist : float, optional
            The effective minimum distance between embedded points. Smaller values
            will result in a more clustered/clumped embedding where nearby points
            on the manifold are drawn closer together, while larger values will
            result on a more even dispersal of points. The value should be set
            relative to the ``spread`` value, which determines the scale at which
            embedded points will be spread out. By default ``0.1``.
        **kwargs :
            All remaining parameters supported by ``umap.UMAP(...)``.
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


    def fit(self, x :np.ndarray) -> None:
        self.dimred = self.dimred.fit(x)
        return None


    def fit_transform(self, x :np.ndarray) -> np.ndarray:
        return self.dimred.fit_transform(x)


    def transform(self, x :np.ndarray) -> np.ndarray:
        return self.dimred.transform(x)