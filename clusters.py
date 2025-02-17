import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Literal


class KMeansHelper():
    """
    This class is based on the module ``sklearn.cluster.KMeans`` and
    provides some heuristics to estimate the best number of clusters.
    """

    def __init__(
        self,
        init :Literal['k-means++', 'random']="k-means++",
        n_init :int|str="auto",
        max_iter :int=300,
        random_state :int|None=None
    ):
        """
        Parameters
        ----------
        init : Literal['k-means++', 'random'], optional
            Method for initialization, by default ``'k-means++'``:
            - ``'k-means++'``: selects initial cluster centroids using sampling based on an empirical probability distribution
            of the points' contribution to the overall inertia. This technique speeds up convergence.
            The algorithm implemented is greedy k-means++. It differs from the vanilla k-means++ by making several trials
            at each sampling step and choosing the best centroid among them.
            - ``'random'``: choose n_clusters observations at random from data for the initial centroids.
        n_init : int | str, optional
            Number of times the k-means algorithm is run with different centroid seeds.
            By default ``n_init='auto'``, the number of runs depends on the value of init:
            10 if using ``init='random'``; 1 if using ``init='k-means++'``.
        max_iter : int, optional
            Maximum number of iterations of the k-means algorithm for a single run, by default ``300``.
        random_state : int | None, optional
            Determines random number generation for centroid initialization. Use an int to make the randomness deterministic,
            by default ``None``.
        """
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        return


    def elbow_method(self, X :np.ndarray, min_K :int, max_K :int) -> dict[int, int]:
        """
        Returns and draws the inertia as the number of clusters changes.
        This heuristic is used to determine the number of clusters in a data set ``X``.

        Parameters
        ----------
        X : np.ndarray
            New data to transform.
        min_K : int
            Minimum number of clusters.
        max_K : int
            maximum number of clusters.

        Returns
        -------
        :dict[int, int]
            A dictionary containing the SSE for every possible number of clusters
            between ``min_K`` and ``max_K``.
        """
        sse = {}
        for k in range(min_K, max_K):
            kmeans = KMeans(n_clusters=k, init=self.init, n_init=self.n_init, max_iter=self.max_iter, random_state=self.random_state).fit(X)
            sse[k] = kmeans.inertia_

        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()), marker='*')
        plt.title("Elbow method")
        plt.xlabel("Number of cluster (K)")
        plt.ylabel("SSE")
        plt.show()
        return sse


    def fit_predict(self, X :np.ndarray, K :int) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : np.ndarray
            New data to transform.
        K : int
            Number of clusters.

        Returns
        -------
        :tuple[np.ndarray, np.ndarray]
            Index of the cluster each sample belongs to and the centroids.
        """
        kmeans = KMeans(n_clusters=K, init=self.init, n_init=self.n_init, max_iter=self.max_iter, random_state=self.random_state)
        clusters = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        return clusters, centroids


    def silhouette_score(self, X :np.ndarray, min_K :int, max_K :int) -> dict[int, int]:
        """
        Returns the silhouette score [-1, 1] as the number of clusters changes.
        This heuristic is used to determine the number of clusters in the dataset ``X``.

        Parameters
        ----------
        X : np.ndarray
            New data to transform.
        min_K : int
            Minimum number of clusters.
        max_K : int
            maximum number of clusters.

        Returns
        -------
        :dict[int, int]
            A dictionary containing the silhouette score for every possible number of clusters
            between ``min_K`` and ``max_K``.
        """
        sc = {}
        print("Number of clusters:")
        for k in range(min_K, max_K):
            kmeans = KMeans(n_clusters=k, init=self.init, n_init=self.n_init, max_iter=self.max_iter, random_state=self.random_state).fit(X)
            sc[k] = silhouette_score(X, kmeans.labels_, metric='euclidean')

            s = f" - K:{k} => Silhouette coeff.: {sc[k]}"
            if (k != min_K) and (sc[k] < sc[k-1]):
                s += " *"
            print(s)
        return sc