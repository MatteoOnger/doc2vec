import numpy as np

from abc import ABC, abstractmethod
from gensim.models import KeyedVectors
from typing import List, Set



class Embedder(ABC):
    """
    Abstract class, it represents a generic embedder for tokens.
    """

    def __init__(self, vocab :Set[str], vector_size :int):
        """
        Parameters
        ----------
        vocab : Set[str]
            Embedder vocabulary, i.e. words for which a vector representation is known.
        vector_size : int
            Size of the vectors.
        """
        super().__init__()
        self.vocab = vocab
        self.vector_size = vector_size
        return


    @abstractmethod
    def get_vector(self, word :str) -> np.ndarray:
        """
        Get the vector representation of a word.

        Parameters
        ----------
        word : str
            Word to be embedded.
        
        Returns
        -------
        : numpy.ndarray of shape \(vector_size,)
            Vector representation of the word.

        Raises
        ------
        KeyError
            If no vector is known for the token ``word``.
        """
        pass


    @abstractmethod
    def get_vectors(self, words :List[str]) -> np.ndarray:
        """
        Get the vector representation of each token in ``words``.

        Parameters
        ----------
        words : List[str]
            List of words to be embedded.
        
        Returns
        -------
        : np.ndarray of shape \(len(words), vector_size)
            Vector representation of the words.
        
        Raises
        ------
        KeyError
            If no vector is known for one of the token in ``words``.
        """
        pass


    def is_in(self, words :List[str]) -> np.ndarray:
        """
        Check if a word is in the vocabulary.

        Parameters
        ----------
        words : List[str]
            List of words to be checked.
        
        Returns
        -------
        : np.ndarray of shape \(len(words),)
            Boolean array indicating if the word is in the vocabulary.
        """
        return np.array([word in self.vocab for word in words])



class GensimEmbedder(Embedder):
    """
    This class implements a word embedder.
    It is based on Gensim's KeyedVectors.

    See Also
    --------
    - Gensim KeyedVectors: https://radimrehurek.com/gensim/models/keyedvectors.html
    """

    def __init__(self, word_vectors :KeyedVectors):
        """
        Parameters
        ----------
        word_vectors : KeyedVectors
            Word vectors.
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