import logging
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import List, Literal, Tuple

from .components.clusters import Cluster
from .components.dimreducers import DimReducer
from .components.embedders import Embedder
from .components.preprocessors import Preprocessor



logger = logging.getLogger(__name__)



class Doc2vec():
    """
    This class maps a document to a vector using a weighted average of the vectors
    associated with the tokens that constitute the document under consideration.
    """

    def __init__(
        self,
        preproc :Preprocessor|None,
        embedder :Embedder,
        dimreducer :DimReducer|None,
        cluster :Cluster|None,
        max_features :int|None=None,
        max_df :float|int=1.0,
        min_df :float|int=1,
        norm :Literal['l1','l2']|None='l1',
        eps_type :Literal['abs', 'pc']='abs',
        eps :float=0.0,
        exp_a :float=1.0,
        exp_b :float=1.0,
    ):
        """
        Parameters
        ----------
        preproc : Preprocessor | None
            Preprocessor used to tokenize the corpus.
            If ``None``, a corpus already tokenized must be provided.
        embedder : Embedder
            Embedder used to embed tokens, i.e. get the vector representation of the words.
        dimreducer : DimReducer | None
            Dimensionality reducer used to reduce the number of dimensions of document vectors.
            If ``None``, document vectors are kept as they are.
        cluster : Cluster | None
            Clustering algorithm used to cluster the document vectors.
            If ``None``, documents are not grouped.
        max_features : int | None, optional
            If not ``None``, build a vocabulary that only consider the top ``max_features`` ordered by term frequency across the corpus.
            Otherwise, all features are used, by default ``None``.
        max_df : float | int, optional
            When building the vocabulary ignore terms that have a document frequency strictly higher
            than the given threshold, by default ``1.0``.
        min_df : float | int, optional
            When building the vocabulary ignore terms that have a document frequency strictly lower
            than the given threshold, by default ``1``.
        norm : Literal['l1', 'l2'] | None, optional
            Each output row will have unit norm, either:
            - ``'l2'`` sums the squares of vector elements is ``1``.
            - ``'l1'`` sums the absolute values of vector elements is ``1`` (default choice).
            - ``None`` means no normalization.
        eps_type : Literal['abs', 'pc'], optional
            Type of threshold:
            - ``'abs'`` sets weights less than ``eps`` to zero (default choice).
            - ``'pc'`` keeps only the largest n weights to cover at least ``eps`` (%) of the total, set the others to zero.
        eps : float, optional
            Threshold, by default ``0.0``.
        exp_a : float, optional
            Exponent of the TF-IDF term, by default ``1.0``.
        exp_b : float, optional
            Exponent of the similarity term, by default ``1.0``.
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
    

    def transform(
        self, 
        corpus :List[str]|None=None,
        tokenized_corpus :List[List[str]]|None=None,
        save_vocab :bool=False,
        save_weights :bool=False
    ) -> np.ndarray|Tuple[np.ndarray, np.ndarray]:
        """
        For each document in the corpus, the function computes a vector representing it.
        The vectors produced lie in the same space defined by the word vectors provided
        during the object initialization.

        Parameters
        ----------
        corpus : List[str] | None
            Corpus to be analyzed, it will be tokenized using 
            the preprocessor provided during the object initialization.
            Only ``corpus`` or ``tokenized_corpus`` can be set, by default ``None``.
        tokenized_corpus : List[List[str]] | None
            Corpus, already tokenized, to be analyzed.
            Only ``corpus`` or ``tokenized_corpus`` can be set, by default ``None``.
        save_vocab : bool, optional
            For debugging, if ``True``,
            the vocabulary for this corpus is saved in ``self.vocab_``, by default ``False``.
        save_weights : bool, optional
            For debugging, if ``True``,
            the weights of each token in each document are saved in ``self.weights_``, by default ``False``.

        Returns
        -------
        doc_vects : numpy.ndarray of shape\(len(tokenized_corpus), self.vector_size)
            A vector for each document.
        labels : numpy.ndarray of shape\(len(tokenized_corpus),)
            The cluster to which each document has been assigned if ``self.cluster`` is not ``None``.
        
        Raises
        ------
        ValueError
            - If both parameters ``corpus`` and ``tokenized_corpus`` are provided.
            - If a non-tokenised corpus is supplied to ``self`` initialized without a preprocessor.
        NotImplementedError
            - If ``self.eps_type`` is a non-implemented threshold type.
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