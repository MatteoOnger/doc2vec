import numpy as np

from abc import ABC, abstractmethod



class Cluster(ABC):
    """
    Abstract base class for clustering algorithms.

    This class defines the interface for all clustering implementations.
    """

    @abstractmethod
    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """
        Perform clustering on `x` and return cluster labels.

        Parameters
        ----------
        x : ndarray of shape (M, N)
            Input data to be clustered, where M is the number of samples and N is the number of features.

        Returns
        -------
        y : ndarray of shape (M,)
            Cluster labels assigned to each sample.
        """
        pass
