import gensim
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import Any, Literal


def remove(lt :list, value :Any) -> list[Any]:
    """
    Returns a copy of the given list without the given value.

    Parameters
    ----------
    lt : list
        List to modify.
    value : Any
        Value to remove.

    Returns
    -------
    :list
        A new list without ``value``.
    """
    lt_copy = list(lt)
    lt_copy.remove(value)
    return lt_copy


class myDoc2vec():
    """
    This class maps a document to a vector using a weighted average of the vectors
    associated with the tokens that constitute the document under consideration.
    """

    def __init__(self,
        w2v :gensim.models.KeyedVectors,
        tfidf_vectorizer :TfidfVectorizer,
        exp_a :float=1.0,
        exp_b :float=1.0,
        eps_type :Literal['abs', 'pc']="abs",
        eps :float=0.0
    ):
        """
        Parameters
        ----------
        w2v : gensim.models.KeyedVectors
            Word vectors to use.
        tfidf_vectorizer : TfidfVectorizer
            Vectorizer to use to compute the TF-IDF scores.
        exp_a : float, optional
            Exponent of the TF-IDF term, by default ``1.0``.
        exp_b : float, optional
            Exponent of the similarity term, by default ``1.0``.
        eps_type : Literal['abs', 'pc'], optional
            Type of threshold:
            - ``'abs'``: set weights less than ``eps`` to zero.
            - ``'pc'``: keep only the largest n weights to cover at least ``eps`` (%) of the total,
            set the others to zero.
        eps : float, optional
            Threshold, by default ``0.0``.
        """
        self.w2v = w2v
        self.tfidf_vectorizer = tfidf_vectorizer
        self.exp_a = exp_a
        self.exp_b = exp_b
        self.eps_type = eps_type
        self.eps = eps
        return


    def transform(self, tokenized_corpus :list[list[str]], save :bool=False) -> np.ndarray:
        """
        For each document in the corpus, the function computes a vector representing it.
        The vectors produced lie in the same space defined by the word vectors provided
        during the object initialization.

        Parameters
        ----------
        tokenized_corpus : list[list[str]]
            Corpus, already tokenized, to be analyzed.
        save : bool, optional
            For debugging, if ``True``,
            the weights of each token in each document are saved in ``self._weights``, by default ``False``.

        Returns
        -------
        :np.ndarray
            A vector for each document.
        """
        corpus_size = len(tokenized_corpus)
        corpus_weights = self.get_weights(tokenized_corpus)

        if save:
            self._weights = corpus_weights
        else:
            self._weights = None

        res = np.zeros((corpus_size, self.w2v.vector_size))
        for i in range(corpus_size):
            # list of unique tokens in i-th doc
            # duplicated tokens are considered only once
            tokens = list(corpus_weights[i].keys())

            # weight of each token
            weights = list(corpus_weights[i].values())

            # weighted mean
            if len(tokens) != 0:
                res[i] = self.w2v.get_mean_vector(tokens, weights=weights)
        return res


    def get_weights(self, tokenized_corpus :list[list[str]]) -> list[dict[str, float]]:
        """
        Computes the weight of each token in each document.
        The weight of the token tk in document doc is computed as follows:
        weight := tfidf(tk, doc)**exp_a * average similarity(tk, other tokens in doc)**exp_b.
        The weights of a document are then filtered using the threshold and normalised to sum to one.

        Parameters
        ----------
        tokenized_corpus : list[list[str]]
            Corpus, already tokenized, to be analyzed.

        Returns
        -------
        :list[dict[str, float]]
            The weight of each token in each document.
        """
        corpus_size = len(tokenized_corpus)

        tfidf_corpus = self.tfidf_vectorizer.fit_transform(tokenized_corpus)
        tf_corpus = normalize(tfidf_corpus / self.tfidf_vectorizer.idf_, norm=self.tfidf_vectorizer.norm)

        corpus_weights = list()
        for i in range(corpus_size):
            # term frequency of tokens in i-th document
            tf_doc = tf_corpus[i].data

            # list of unique tokens in i-th document and their position in the tfidf matrix
            # duplicated tokens are considered only once
            token_idxs = tf_corpus[i].nonzero()[1]
            tokens = list(
                self.tfidf_vectorizer.get_feature_names_out()[token_idxs]
            )

            # compute the weight of each token in i-th document
            if len(tokens) == 0:
                corpus_weights.append({})
            elif len(tokens) == 1:
                corpus_weights.append({tokens[0]:1.0})
            else:
                weights = np.zeros(len(tokens))

                for j, (idx, tk) in enumerate(zip(token_idxs, tokens)):
                    weights[j] = (
                        tfidf_corpus[i, idx]**self.exp_a
                    ) * (
                        np.average(1 - self.w2v.distances(tk, remove(tokens, tk)) / 2, weights=np.delete(tf_doc, j))**self.exp_b
                    )

                    if self.eps_type == "abs" and weights[j] < self.eps:
                        weights[j] = 0.0

                if self.eps_type == "pc":
                    weights = weights / np.sum(weights)
                    sorted_weight_idxs = np.argsort(weights)[::-1]
                    cumsum_weights = np.cumsum(weights[sorted_weight_idxs])

                    for j in range(1, weights.size):
                        if cumsum_weights[j-1] >= self.eps:
                            weights[sorted_weight_idxs[j]] = 0.0

                corpus_weights.append(
                    dict(zip(tokens, normalize([weights], norm="l1")[0]))
                )
        return corpus_weights


    def get_most_similar(self, tokenized_corpus :list[list[str]], topn :int=10) -> list[list[tuple[str, float]]]:
        """
        Equivalent to calling the function ``self.w2v.most_similar`` on the vectors
        returned by ``self.transform``.

        Parameters
        ----------
        tokenized_corpus : list[list[str]]
            Corpus, already tokenized, to be analyzed.
        topn : int, optional
            Number of most similar keys to return, by default ``10``.

        Returns
        -------
        :list[list[tuple[str, float]]]
            The ``top-n`` most similar keys found for each document.

        Notes
        -----
        - Finding the most similar keys for each vector is a slow operation;
        if you are only interested in a few documents, it is better to call ``self.transform``
        and then call ``self.w2v.most_similar`` only on the documents of interest.
        """
        corpus_size = len(tokenized_corpus)
        vec_corpus = self.transform(tokenized_corpus)

        res = list()
        for i in range(corpus_size):
            res.append(
                self.w2v.most_similar(vec_corpus[i], topn=topn)
            )
        return res
     