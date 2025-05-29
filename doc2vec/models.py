import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import Any, Dict, List, Literal, Tuple, Type

from doc2vec.components.cluster import Cluster
from doc2vec.components.reducer import DimReducer
from doc2vec.components.embedder import Embedder
from doc2vec.components.preprocessor import Preprocessor
from doc2vec.components.utils.ctfidf import CtfidfVectorizer



logger = logging.getLogger(__name__)



class Doc2vec():
    """
    Maps documents to vectors using a weighted average of word embeddings
    combined with TF-IDF statistics and intra-document token similarity.

    This implementation allows optional preprocessing, dimensionality reduction,
    and clustering of the resulting document embeddings.

    Attributes
    ----------
    tfidfer : sklearn.feature_extraction.text.TfidfVectorizer
        TF-IDF vectorizer used to calculate term weights.
    vocab_ : numpy.ndarray or None
        Vocabulary of the most recently processed corpus.
    weights_ : list of dict or None
        Token weights for each document in the most recent transformation.
    vector_size : int
        Dimensionality of the resulting document vectors.
    """

    def __init__(
        self,
        preproc: Preprocessor|None,
        embedder: Embedder,
        dimreducer: DimReducer|None,
        cluster: Cluster|None,
        max_features: int|None = None,
        max_df: float|int = 1.0,
        min_df: float|int = 1,
        norm: Literal['l1','l2']|None = 'l1',
        eps_type: Literal['abs', 'pc'] = 'abs',
        eps: float = 0.0,
        exp_a: float = 1.0,
        exp_b: float = 1.0,
    ):
        """
        Parameters
        ----------
        preproc : Preprocessor | None
            Preprocessor used to tokenize the corpus.
            If `None`, a tokenized corpus must be supplied during transformation.
        embedder : Embedder
            Word embedding model for converting tokens to vectors.
        dimreducer : DimReducer | None
            Dimensionality reducer. If `None`, dimensionality is not reduced.
        cluster : Cluster | None
            Clustering model. If `None`, documents are not clustered.
        max_features : int | None, optional
            Build a vocabulary that only consider the top `max_features` ordered by 
            term frequency across the corpus. If `None`, all features are used.
        max_df : float | int, optional
            When building the vocabulary ignore terms that have a document frequency strictly higher
            than the given threshold. By default `1.0`.
        min_df : float | int, optional
            When building the vocabulary ignore terms that have a document frequency strictly lower
            than the given threshold. By default `1`.
        norm : Literal['l1', 'l2'] | None, optional
            Norm used to normalize term vectors, by default `'l1'`.
            If `None`, no normalization is performed.
        eps_type : Literal['abs', 'pc'], optional
            Type of threshold:
            - `'abs'` sets weights less than `eps` to zero (default choice).
            - `'pc'` keeps only the largest N weights to cover at least `eps` (%) of the total, set the others to zero.
        eps : float, optional
            Threshold value. By default `0.0`.
        exp_a : float, optional
            Exponent applied to the TF-IDF term, by default `1.0`.
        exp_b : float, optional
            Exponent applied to the cosine similarity term, by default `1.0`.
        """
        self.preproc = preproc
        self.embedder = embedder
        self.dimreducer = dimreducer
        self.cluster = cluster

        self.tfidfer = TfidfVectorizer(
            lowercase = False,
            tokenizer = lambda x: x,
            token_pattern = None,
            max_df = max_df,
            min_df = min_df,
            max_features = max_features,
            norm = norm,
            use_idf = True 
        )

        self.eps_type =eps_type
        self.eps = eps
        self.exp_a = exp_a
        self.exp_b = exp_b

        self.vocab_ = None
        """
        Vocabulary of the last corpus processed.
        """
        self.weights_ = None
        """
        Weights of the last corpus processed.
        """
        self.vector_size = self.embedder.vector_size
        """
        Size of the document vectors.
        """
        return


    def get_closest_words(self, vector: np.ndarray, topk: int|None = 10) -> List[Tuple[str, float]]:
        """
        Return the `topk` closest words for the given to vector.

        Parameters
        ----------
        vector : numpy.ndarray of shape (vector_size,)
            Vector to query by similarity.
        topk : int, optional
            Number of closest words to return, by default `10`.

        Returns
        -------
        : List[Tuple[str, float]]
            Closest words and their similarity scores.

        Raises
        ------
        ValueError
            - If `vector` does not match the expected vector size.
        """
        return self.embedder.get_top_words(vector, topk=topk)


    def get_top_words_per_cluster(
        self,
        tokenized_corpus: List[List[str]],
        labels: np.ndarray,
        topk: int|None = 10,
        max_df: float|int = 1.0,
        min_df: float|int = 1,
        max_features: int|None = None,
    ) -> Dict[int|str, List[Tuple[str, float]]]:
        """
        """
        if self.vocab_ is None:
            logger.warning("vocabolary is empty, computing the vocabulary using default parameters")

        ctfidfer = CtfidfVectorizer(
            max_df = max_df,
            min_df = min_df,
            max_features = max_features,
            vocabulary = self.vocab_
        )

        ctfidf_corpus = ctfidfer.fit_transform(tokenized_corpus, labels).sorted_indices()
        unique_labels = ctfidfer.get_label_names_out()
        voc = ctfidfer.get_feature_names_out()

        res = {}
        for i in range(len(unique_labels)):
            row = ctfidf_corpus.getrow(i)

            top_n_idx_in_data = np.argsort(row.data)[-topk:][::-1]
            top_n_col_indices = row.indices[top_n_idx_in_data]
            
            res[unique_labels[i]] = [(voc[v], row.data[s]) for v, s in zip(top_n_col_indices, top_n_idx_in_data)]
        return res


    def transform(
        self, 
        corpus: List[str]|None = None,
        tokenized_corpus: List[List[str]]|None = None,
        save_vocab: bool = False,
        save_weights: bool = False
    ) -> np.ndarray|Tuple[np.ndarray, np.ndarray]:
        """
        For each document in the corpus, the function computes a vector representing it.
        The vectors produced lie in the same space defined by the word vectors provided
        during the object initialization.

        Parameters
        ----------
        corpus : List[str] | None
            List of raw documents. Will be tokenized if a preprocessor is provided.
        tokenized_corpus : List[List[str]] | None
            Tokenized documents. If given, bypasses preprocessing.
        save_vocab : bool, optional
            Whether to store the vocabulary in `self.vocab_` for inspection/debugging. Default is `False`.
        save_weights : bool, optional
            Whether to store the token weights in `self.weights_`. Default is `False`.

        Returns
        -------
        doc_vects : numpy.ndarray of shape (n_documents, vector_size)
            Matrix where each row is a vector representation of a document.
        labels : numpy.ndarray of shape (n_documents,)
            Cluster assignments if `self.cluster` is not `None`. Only returned if clustering is performed.

        Raises
        ------
        ValueError
            - If both `corpus` and `tokenized_corpus` are provided.
            - If `corpus` is provided without a preprocessor.
        NotImplementedError
            - If `eps_type` is not one of `['abs', 'pc']`.
        """
        if corpus is not None and tokenized_corpus is not None:
            raise ValueError("Only one between <corpus> and <tokenized_corpus> can be set")
        elif corpus is not None:
            if self.preproc is not None:
                logger.info("preprocessing start")
                tokenized_corpus = self.preproc.preprocess_corpus(corpus)
                logger.info("preprocessing done")
            else:    
                raise ValueError("Please providing a preprocessor during object initialization or a tokenised corpus")
        
        logger.info("embedding start")
        corpus_size = len(tokenized_corpus)

        # tfidf and tf each token in each document 
        tfidf_corpus = self.tfidfer.fit_transform(tokenized_corpus).sorted_indices()

        # skip tokens with no vector representation
        cols_to_keep = self.embedder.is_in(self.tfidfer.get_feature_names_out())
        vocab = self.tfidfer.get_feature_names_out()[cols_to_keep]

        tfidf_corpus = normalize(tfidf_corpus[:, cols_to_keep], norm=self.tfidfer.norm)
        tf_corpus = normalize(tfidf_corpus / self.tfidfer.idf_[cols_to_keep], norm=self.tfidfer.norm)

        # for debugging, save the weights and/or vocabulary
        self.weights_ = list() if save_weights else None
        self.vocab_ = vocab if save_vocab else None

        # document vectors
        doc_vecs = np.zeros((corpus_size, self.vector_size))        
        for i in range(corpus_size):
            # list of unique tokens in i-th document
            # duplicated tokens are considered only once
            tokens = np.array(
                vocab[tf_corpus[i].nonzero()[1]]
            )

            if tokens.size == 0:
                logger.warning(f"document {i} is empty")
                continue
            elif tokens.size == 1:
                doc_vecs[i] = self.embedder.get_vector(tokens[0])
                self.weights_.append({tokens[0]:1})
                continue

            # word vectors and their norms
            vecs = self.embedder.get_vectors(tokens)
            vecs_norm = np.linalg.norm(vecs, axis=1).reshape(tokens.size,1)

            # cosine similarity between each pair of tokens in i-th document
            cos_sim = ((vecs @ vecs.T) / (vecs_norm @ vecs_norm.T) + 1) / 2
            np.fill_diagonal(cos_sim, 1)

            # compute weight of each token for the i-th document
            weights = tfidf_corpus[i].data**self.exp_a * np.average(cos_sim, axis=1, weights=tf_corpus[i].data)**self.exp_b

            if self.eps_type == "abs":
                weights[(weights <= self.eps) & (weights != weights.max())] = 0.0
            elif self.eps_type == "pc":
                sorted_weights_idxs = np.argsort(weights / np.sum(weights))[::-1]
                cumsum_weights = np.cumsum(weights[sorted_weights_idxs])
                weights[sorted_weights_idxs[1:][cumsum_weights[1:] > self.eps]] = 0.0
            else:
                raise NotImplementedError(f"'{self.eps_type}' not implemented")

            weights = normalize(weights.reshape(1, -1), norm="l1")[0]

            if save_weights:
                self.weights_.append({tokens[i]:weights[i] for i in range(tokens.size)})

            # i-th document vector
            doc_vecs[i] = np.average(vecs, axis=0, weights=weights)
        logger.info("embedding done")

        if self.dimreducer is not None:
            logger.info("dim reduction start")
            doc_vecs = self.dimreducer.fit_transform(doc_vecs)
            logger.info("dim reduction done")
        
        if self.cluster is not None:
            logger.info("clustering start")
            labels = self.cluster.fit_predict(doc_vecs)
            logger.info("clustering done")
            return doc_vecs, labels
        return doc_vecs
