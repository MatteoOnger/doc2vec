import numpy as np

from gensim import corpora
from gensim.models import CoherenceModel
from typing import Dict, List, Literal, Tuple

from doc2vec.components.metric import UnsupervisedMetric



class Coherence(UnsupervisedMetric):
    """
    Compute the coherence of topics based on their most representative words
    and the tokenized input corpus.

    This class wraps Gensim's `CoherenceModel` to support various coherence
    measures, including `u_mass`, `c_v`, `c_uci`, and `c_npmi`.

    See Also
    --------
    - Gensim KeyedVectors : https://radimrehurek.com/gensim/models/coherencemodel.html.
    """
    
    def __init__(
        self,
        window_size: int|None = None,
        coherence: Literal['u_mass', 'c_v', 'c_uci', 'c_npmi'] = 'c_v',
        processes: int = -1
    ):
        """
        Parameters
        ----------
        window_size : int | None, optional
            Size of the sliding window for co-occurrence statistics.
            Only used for certain coherence types (`c_v`, `c_uci`, `c_npmi`).
        coherence : {'u_mass', 'c_v', 'c_uci', 'c_npmi'}, optional
            The type of coherence metric to compute. Default is `'c_v'`.
        processes : int, optional
            Number of processes to use for computation. Default is `-1` (use all).
        """
        self.window_size = window_size
        self.coherence = coherence
        self.processes = processes
        return
    

    def evaluate(
        self,
        topics: Dict[int|str, List[Tuple[str, float]]],
        tokenized_corpus: List[List[str]]
    ) -> float:
        self._compute(topics, tokenized_corpus)
        return self.model.get_coherence()
    
    
    def evaluate_per_topic(
        self,
        topics: Dict[int|str, List[Tuple[str, float]]],
        tokenized_corpus: List[List[str]]
    ) -> np.ndarray:
        self._compute(topics, tokenized_corpus)
        return np.array(self.model.get_coherence_per_topic())


    def _compute(self, topics: Dict[int|str, List[Tuple[str, float]]], tokenized_corpus: List[List[str]]) -> None:
        """
        Internal method to initialize the Gensim CoherenceModel with the provided data.

        Parameters
        ----------
        topics : Dict[int | str, List[Tuple[str, float]]]
            A dictionary mapping topic identifiers to lists of `(word, weight)` tuples.
            Each list contains the most representative words for that topic.
        tokenized_corpus : List[List[str]]
            The tokenized documents from which the topics were generated.
        """
        topics = [[token[0] for token in topic] for topic in topics.values()]
        dictionary = corpora.Dictionary(tokenized_corpus)

        self.model = CoherenceModel(
            topics = topics,
            texts = tokenized_corpus,
            dictionary = dictionary,
            window_size = self.window_size,
            coherence = self.coherence,
            processes = self.processes
        )
        return
