import numpy as np

from abc import ABC, abstractmethod



class DimReducer(ABC):
    """
    Abstract base class for dimensionality reduction methods.

    This class defines the interface for all dimensionality reducers.
    """

    @abstractmethod
    def fit(self, x: np.ndarray) -> None:
        """
        Fit the model with input data.

        Parameters
        ----------
        x : numpy.ndarray of shape (M, N)
            Training data, where M is the number of samples and N is the number of features.
        """
        pass


    @abstractmethod
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        """
        Fit the model with input data and apply the dimensionality reduction on it.

        Parameters
        ----------
        x : ndarray of shape (M, N)
            Training data, where M is the number of samples and N is the number of features.

        Returns
        -------
        x_new : ndarray of shape (M, K)
            Reduced representation of the input data, where K is the number of output dimensions.
        """
        pass


    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply the dimensionality reduction to new data.

        Parameters
        ----------
        x : ndarray of shape (M, N)
            New data to transform, where M is the number of samples and N is the number of features.

        Returns
        -------
        x_new : ndarray of shape (M, K)
            Transformed data in the reduced-dimensional space.
        """
        pass
