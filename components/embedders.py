import numpy as np

from abc import ABC, abstractmethod
from gensim.models import KeyedVectors
from typing import List



class Embedder(ABC):
    """
    Abstract class, it represents a generic embedder for tokens.
    """

    def __init__(self, vocab :set, vector_size :int):
        """
        """
        super().__init__()
        self.vocab = vocab
        self.vector_size = vector_size
        return


    @abstractmethod
    def get_vector(self, word :str) -> np.ndarray:
        """
        """
        pass


    @abstractmethod
    def get_vectors(self, words :List[str]) -> np.ndarray:
        """
        Raises
        ------
        """
        pass


    def is_in(self, words :List[str]) -> np.ndarray:
        """
        """
        return np.array([word in self.vocab for word in words])



class GensimEmbedder(Embedder):
    """
    """

    def __init__(self, word_vectors :KeyedVectors):
        """
        """
        super().__init__(
            set(word_vectors.index_to_key),
            word_vectors.vector_size
        )
        self.wv = word_vectors
        return


    def get_vector(self, word :str) -> np.ndarray:
        return self.wv[word]


    def get_vectors(self, words :List[str]) -> np.ndarray:
        return self.wv[words]