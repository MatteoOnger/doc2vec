import numpy as np

from sklearn.metrics import homogeneity_score

from doc2vec.components.metric import SupervisedMetric



class HomogeneityScore(SupervisedMetric):
    """
    Computes the homogeneity score for clustering performance evaluation.

    The homogeneity score measures whether each cluster contains only members of a single class. 
    It is a value between 0.0 and 1.0, where 1.0 stands for perfectly homogeneous labeling.

    Examples
    --------
    >>> import numpy as np
    >>> from doc2vec.components.homogeneity import HomogeneityScore
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_pred = np.array([1, 1, 0, 0])
    >>> metric = HomogeneityScore()
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
        return homogeneity_score(labels_true, labels_pred)
