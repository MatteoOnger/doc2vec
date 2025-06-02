import numpy as np

from sklearn.metrics import completeness_score

from doc2vec.components.metric import SupervisedMetric



class CompletenessScore(SupervisedMetric):
    """
    Computes the completeness score for clustering performance evaluation.

    The completeness score measures whether all members of a given class are assigned to the same cluster.
    It is a value between 0.0 and 1.0, where 1.0 indicates perfectly complete labeling.

    Examples
    --------
    >>> import numpy as np
    >>> from doc2vec.components.completeness import CompletenessScore
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([0, 0, 1, 1])
    >>> metric = CompletenessScore()
    >>> metric.evaluate(y_true, y_pred)
    1.0
    """

    def __init__(self):
        """
        No additional parameters are required.
        """
        super().__init__()
        return


    def evaluate(self, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        return completeness_score(labels_true, labels_pred)
