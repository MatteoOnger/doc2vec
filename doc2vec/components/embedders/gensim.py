import numpy as np

from gensim.models import KeyedVectors
from typing import List, Tuple

from doc2vec.components.embedder import Embedder



class GensimEmbedder(Embedder):
    """
    Word embedder implementation using Gensim's KeyedVectors.

    This class wraps around Gensim KeyedVectors and uses cosine similarity
    to find the most similar words to a given vector or word.

    See Also
    --------
    - Gensim KeyedVectors : https://radimrehurek.com/gensim/models/keyedvectors.html.
    """

    def __init__(self, word_vectors: KeyedVectors):
        """
        Parameters
        ----------
        word_vectors : KeyedVectors
            Pre-trained word vectors from Gensim.
        """
        super().__init__(
            set(word_vectors.index_to_key),
            word_vectors.vector_size
        )
        self.wv = word_vectors
        return


    def get_top_words(self, vector: np.ndarray|None = None, word: str|None = None, topk: int|None = 10) -> List[Tuple[str, float]]:
        if vector is None and word is None:
            raise ValueError("Both <vector> and <word> are None")
        if vector is not None and word is not None:
            raise ValueError("Both <vector> and <word> are not None")

        if word is not None:
            vector = self.get_vector(word)
        
        if len(vector) != self.vector_size:
            raise ValueError("<vector> size does not match embedding space")
        return self.wv.similar_by_vector(vector, topk)


    def get_vector(self, word: str) -> np.ndarray:
        return self.wv[word]


    def get_vectors(self, words: List[str]) -> np.ndarray:
        return self.wv[words]
