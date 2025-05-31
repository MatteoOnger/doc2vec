import numpy as np

from sklearn.metrics import rand_score

from doc2vec.components.metric import SupervisedMetric



class RandScore(SupervisedMetric):
    """
    Rand Index metric for evaluating clustering similarity.

    The Rand Index computes a similarity measure between two clusterings
    by considering all pairs of samples and counting pairs that are assigned
    in the same or different clusters in the predicted and true clusterings.

    RI = (number of agreeing pairs) / (number of pairs)
    """

    def __init__(self):
        """
        No additional parameters are required.
        """
        super().__init__()


    def evaluate(self, labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
        return rand_score(labels_pred, labels_true)
