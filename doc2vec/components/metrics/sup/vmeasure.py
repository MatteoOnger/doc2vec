import numpy as np

from sklearn.metrics import v_measure_score

from doc2vec.components.metric import SupervisedMetric



class VMeasureScore(SupervisedMetric):
    """
    Computes the V-measure score for clustering performance evaluation.

    The V-measure is the harmonic mean between homogeneity and completeness:
    - Homogeneity: each cluster contains only members of a single class.
    - Completeness: all members of a given class are assigned to the same cluster.

    The weighting between these two metrics can be adjusted using the `beta` parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from doc2vec.components.v_measure import VMeasureScore
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0])
    >>> metric = VMeasureScore(beta=1.0)
    >>> metric.evaluate(y_true, y_pred)
    1.0
    """

    def __init__(self, beta: float = 1.0):
        """
        Parameters
        ----------
        beta : float, optional
            Weight of homogeneity relative to completeness:
            - `beta` > 1.0 favors homogeneity.
            - `beta` < 1.0 favors completeness.
            - `beta` = 1.0 gives equal importance (by default).
        """
        super().__init__()
        self.beta = beta
        return


    def evaluate(self, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        return v_measure_score(labels_true, labels_pred, beta=self.beta)
