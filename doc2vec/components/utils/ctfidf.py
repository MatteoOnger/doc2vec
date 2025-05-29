import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix, diags
from typing import List



class CtfidfVectorizer():
    """
    Class-based implementation of the Class-based TF-IDF (C-TF-IDF) Vectorizer.

    The c-TF-IDF algorithm is an adaptation of the traditional TF-IDF metric for use in
    scenarios where each document belongs to a class or topic (label). Instead of computing
    term frequencies per document, it aggregates token counts by class and computes
    a class-based representation of term importance.
    """

    def __init__(
        self,
        max_features: int|None = None,
        max_df: float|int = 1.0,
        min_df: float|int = 1,
        vocabulary: List[str]|None = None
    ):
        """
        Parameters
        ----------
        max_features : int | None, optional
            If not `None`, build a vocabulary that only considers the top `max_features`
            ordered by term frequency across the corpus. By default `None`.
            This parameter is ignored if vocabulary is not `None`.
        max_df : float | int, optional
            When building the vocabulary, ignore terms that have a document frequency
            strictly higher than the given threshold. By default `1.0`.
            This parameter is ignored if vocabulary is not `None`.
        min_df : float | int, optional
            When building the vocabulary, ignore terms that have a document frequency
            strictly lower than the given threshold. By default `1`.
            This parameter is ignored if vocabulary is not `None`.
        vocabulary : List[str] | None, optional
            Optional fixed vocabulary. If provided, it will not be learned from the data.
        """
        self.counter = CountVectorizer(
            lowercase = False,
            tokenizer = lambda x: x,
            token_pattern = None,
            max_features = max_features,
            max_df = max_df,
            min_df = min_df,
            vocabulary = vocabulary
        )

        self._label_names = None
        return


    def get_feature_names_out(self) -> np.ndarray:
        """
        Get output feature names (tokens) after fitting.

        Returns
        -------
        : numpy.ndarray
            Array of feature names (tokens).

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self._label_names is None:
            raise NotFittedError("Please fit the model before calling this method")
        return self.counter.get_feature_names_out()


    def get_label_names_out(self) -> np.ndarray:
        """
        Get the output label names.

        Returns
        -------
        : numpy.ndarray
            Array of label names.

        Raises
        ------
        NotFittedError
            If the model has not been fitted.
        """
        if self._label_names is None:
            raise NotFittedError("Please fit the model before calling this method")
        return self._label_names


    def fit_transform(self, tokenized_corpus: List[List[str]], labels: np.ndarray) -> csr_matrix:
        """
        Fit the C-TF-IDF model on the provided tokenized corpus and labels,
        and return the class-based TF-IDF matrix.

        Parameters
        ----------
        tokenized_corpus : List[List[str]]
            A corpus of tokenized documents, where each document is represented
            as a list of tokens.
        labels : numpy.ndarray
            A 1D array of labels corresponding to each document in the corpus.

        Returns
        -------
        ctfidf_corpus : scipy.csr_matrix of shape (n_classes, n_tokens)
            A sparse matrix  representing the class-based TF-IDF scores.
        """
        self._label_names, label_inverse = np.unique(labels, return_inverse=True)

        # compute matrxi G s.t. G[i, j] == 1 if doc j has label i
        n_rows = len(self._label_names)     # number of labels
        n_cols = len(tokenized_corpus)      # number of documents

        row_indices = label_inverse         # label's index of each doc
        col_indices = np.arange(n_cols)     # index of each doc
        data = np.ones(n_cols)

        G = csr_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))

        # abs freq matrix document/token
        X =  self.counter.fit_transform(tokenized_corpus)

        # abs freq matrix cluster/token
        Y = G @ X

        # abs freq of each token
        tokens_freq = np.squeeze(np.asarray(Y.sum(axis=0)))
        # clusters avg lengh in number of tokens
        avg_n_tokens = Y.sum(axis=1).mean()

        # tokens IDF as a sparse diagonal matrix
        idf = diags(
            np.log(1 + (avg_n_tokens / tokens_freq)),
            shape=(X.shape[1], X.shape[1]),
            format="csr"
        )

        # tokens TF per cluster
        Y = normalize(Y, axis=1, norm="l1", copy=False)
        
        # tokens CTFIDF
        ctfidf_corpus = Y @ idf
        return ctfidf_corpus
