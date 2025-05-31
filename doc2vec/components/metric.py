import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple



class SupervisedMetric(ABC):
    """
    Abstract base class for supervised evaluation metrics of topic models.

    This class is intended to be extended by specific metric implementations
    that require ground truth labels to assess the quality of topics.
    """
    
    @abstractmethod
    def evaluate(
        self,
        labels_true: np.ndarray,
        labels_pred: np.ndarray
    ) -> float:
        """
        Evaluate the alignment between predicted and true labels.

        This method computes a scalar score that quantifies how well the 
        predicted labels match the ground truth labels.

        Parameters
        ----------
        labels_true : numpy.ndarray
            An array of ground truth topic labels for each document.
        labels_pred : numpy.ndarray
            An array of predicted topic labels for each document.

        Returns
        -------
        : float
            A single float value representing the evaluation score.
        """
        pass



class UnsupervisedMetric(ABC):
    """
    Abstract base class for unsupervised evaluation metrics of topic models.

    These metrics evaluate topic coherence or quality using only the discovered
    topics and the tokenized input corpus, without requiring ground truth labels.
    """

    @abstractmethod
    def evaluate(
        self,
        topics: Dict[int|str, List[Tuple[str, float]]],
        tokenized_corpus: List[List[str]]
    ) -> float:
        """
        Evaluate the overall quality of topics using the tokenized corpus.

        Parameters
        ----------
        topics : Dict[int | str, List[Tuple[str, float]]]
            A dictionary mapping topic identifiers to lists of `(word, weight)` tuples.
            Each list contains the most representative words for that topic.
        tokenized_corpus : List[List[str]]
            The tokenized documents that the topics were derived from.

        Returns
        -------
        : float
            A single float value representing the overall topic quality.
        """
        pass
    
    
    @abstractmethod
    def evaluate_per_topic(
        self,
        topics: Dict[int|str, List[Tuple[str, float]]],
        tokenized_corpus: List[List[str]]
    ) -> np.ndarray:
        """
        Evaluate topic quality on a per-topic basis.

        Parameters
        ----------
        topics : Dict[int | str, List[Tuple[str, float]]]
            A dictionary mapping topic identifiers to lists of `(word, weight)` tuples.
            Each list contains the most representative words for that topic.
        tokenized_corpus : List[List[str]]
            The tokenized documents from which the topics were generated.

        Returns
        -------
        : numpy.ndarray
            An array of shape (n_topics,) containing the evaluation score for each topic.
        """
        pass
