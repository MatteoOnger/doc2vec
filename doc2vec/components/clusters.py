import numpy as np
import sklearn

from abc import ABC, abstractmethod
from typing import Callable, Literal

import sklearn.cluster



class Cluster(ABC):
    """
    Abstract class, it represents a generic clustering algorithm.
    """

    @abstractmethod
    def fit_predict(self, x :np.ndarray) -> np.ndarray:
        """
        Cluster ``x`` and return the associated cluster labels.

        Parameters
        ----------
        x : numpy.array of shape \(M, N)
            One sample per row.
        
        Returns
        -------
        y : numpy.array of shape \(M,)
            Cluster labels.
        """
        pass



class HDBSCAN(Cluster):
    """
    HDBSCAN clustering algorithm.

    See also
    --------
    - Original paper: https://doi.org/10.1007/978-3-642-37456-2_14
    - Scikit-learn HDBSCAN: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#hdbscan
    """
    
    def __init__(
        self,
        min_cluster_size :int=5,
        min_samples :int|None=None,
        cluster_selection_epsilon :float=0.0,
        metric :str|Callable='cosine',
        store_centers :Literal['both', 'centroid', 'medoid']|None='centroid',
        **kwargs
    ):
        """
        Parameters
        ----------
        min_cluster_size : int, optional
            The minimum number of samples in a group for that group to be considered a cluster;
            groupings smaller than this size will be left as noise. By default ``5``.
        min_samples : int | None, optional
            The parameter k used to calculate the distance between a point x_p and its k-th nearest neighbor.
            When ``None``, defaults to ``min_cluster_size``.
        cluster_selection_epsilon : float, optional
            A distance threshold. Clusters below this value will be merged, by default ``0.0``.
        metric : str | Callable, optional
            The metric to use when calculating distance between instances in a feature array.
            If a string is passed it must match a valid predefined metric, check the official documentation
            for supported distances.
        store_centersstr : Literal['both', 'centroid', 'medoid'] | None, optional
            Which, if any, cluster centers to compute and store. The options are:
            - ``None`` which does not compute nor store any centers.
            - ``'centroid'`` which calculates the center by taking the weighted average of their positions.
            - ``'medoid'`` which calculates the center by taking the point in the fitted data which minimizes
            the distance to all other points in the cluster. Slower than ``'centroid'``.
            - ``'both'`` which computes and stores both forms of centers.
        """
        super().__init__()

        if "n_jobs" not in kwargs.keys():
            kwargs["n_jobs"] = -1

        print(sklearn.__version__)

        self.hdbscan = sklearn.cluster.HDBSCAN(
            min_cluster_size = min_cluster_size,
            min_samples = min_samples,
            cluster_selection_epsilon = cluster_selection_epsilon,
            metric = metric,
            store_centers = store_centers,
            **kwargs
        )

        self.isfitted = False
        """
        ``True`` if the object has been fitted at least once.
        """
        return


    def get_centroids(self) -> np.ndarray:
        """
        Get a collection containing the centroid of each cluster
        calculated under the standard euclidean metric. 

        Returns
        -------
        : numpy.array of shape \(n_clusters, N)
            Centroids.
        
        Raises
        ------
        ValueError
            If the object has not been fitted.
        """
        if not self.isfitted:
            raise ValueError("Please fit the model before calling this method")
        return self.hdbscan.centroids_


    def get_medoids(self) -> np.ndarray:
        """
        A collection containing the medoid of each cluster calculated
        under the whichever metric was passed to the metric parameter.

        Returns
        -------
        : numpy.array of shape \(n_clusters, N)
            Medoids.
        
        Raises
        ------
        ValueError
            If the object has not been fitted.
        """
        if not self.isfitted:
            raise ValueError("Please fit the model before calling this method")
        return self.hdbscan.medoids_


    def get_n_clusters(self) -> int:
        """
        Get the number of clusters without considering outliers.

        Returns
        -------
        : int
            Number of clusters.
        
        Raises
        ------
        ValueError
            If the object has not been fitted.
        """
        if not self.isfitted:
            raise ValueError("Please fit the model before calling this method")
        return len(np.unique(self.hdbscan.labels_)) - 1


    def get_probabilities(self) -> np.ndarray:
        """
        Get the strength with which each sample is a member of its assigned cluster.

        Returns
        -------
        : numpy.array of shape \(M,)
            Probability with which each sample is a member of its assigned cluster.
        
        Raises
        ------
        ValueError
            If the object has not been fitted.
        """
        if not self.isfitted:
            raise ValueError("Please fit the model before calling this method")
        return self.hdbscan.probabilities_


    def fit_predict(self, x :np.ndarray) -> np.ndarray:
        self.isfitted =  True
        return self.hdbscan.fit_predict(x)