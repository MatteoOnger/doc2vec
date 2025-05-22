import numpy as np

from abc import ABC, abstractmethod
from typing import List, Set, Tuple



class Embedder(ABC):
    """
    Abstract class representing a generic embedder for tokens.

    This class defines the interface for embedding textual tokens into a
    continuous vector space and for querying similarities.
    """

    def __init__(self, vocab: Set[str], vector_size: int):
        """
        Parameters
        ----------
        vocab : Set[str]
            Embedder vocabulary, i.e. words for which a vector representation is known.
        vector_size : int
            Dimensionality of the embedding space.
        """
        super().__init__()
        self.vocab = vocab
        self.vector_size = vector_size
        return


    @abstractmethod
    def get_top_words(self, vector: np.ndarray|None = None, word: str|None = None, topk: int|None = 10) -> List[Tuple[str, float]]:
        """
        Return the `topk` most representative words for the given word or vector.

        Parameters
        ----------
        vector : numpy.ndarray of shape (vector_size,), optional
            Word vector to query by similarity, by default `None`.
        word : str, optional
            Word to query by similarity, by default `None`.
        topk : int, optional
            Number of top representative words to return, by default `10`.

        Returns
        -------
        : List[Tuple[str, float]]
            List of top `topk` most similar words and their similarity scores.

        Raises
        ------
        KeyError
            - If `word` is not in the vocabulary.
        ValueError
            - If both `vector` and `word` are `None` or both are provided.
            - If `vector` does not match the expected vector size.
        """
        pass


    @abstractmethod
    def get_vector(self, word: str) -> np.ndarray:
        """
        Get the vector representation of a word.

        Parameters
        ----------
        word : str
            Word to embed.

        Returns
        -------
        vector : numpy.ndarray of shape (vector_size,)
            Vector representation of the word.

        Raises
        ------
        KeyError
            - If the word is not in the vocabulary.
        """
        pass


    @abstractmethod
    def get_vectors(self, words: List[str]) -> np.ndarray:
        """
        Get the vector representations of a list of words.

        Parameters
        ----------
        words : List[str]
            List of words to embed.

        Returns
        -------
        vectors : numpy.ndarray of shape (len(words), vector_size)
            Matrix of vector representations for the input words.

        Raises
        ------
        KeyError
            - If any of the words are not in the vocabulary.
        """
        pass


    def is_in(self, words: List[str]) -> np.ndarray:
        """
        Check if each word in a list is present in the vocabulary.

        Parameters
        ----------
        words : List[str]
            List of words to check.

        Returns
        -------
        res : numpy.ndarray of shape (len(words),)
            Boolean array indicating for each word whether it is in the vocabulary.
        """
        return np.array([word in self.vocab for word in words])
