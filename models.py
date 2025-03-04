import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from typing import List, Literal

from components.embedders import Embedder
from components.preprocessors import Preprocessor



class Doc2vec():
    """
    """

    def __init__(
        self,
        preproc :Preprocessor,
        embedder :Embedder,
        max_df :float|int=1.0,
        min_df :float|int=1,
        max_features :int|None=None,
        norm :Literal['l1','l2']|None='l1',
        eps_type :Literal['abs', 'pc']='abs',
        eps :float=0.0,
        exp_a :float=1.0,
        exp_b :float=1.0,
    ):
        self.preproc = preproc
        self.embedder = embedder
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
        self.weights_ = None
        self.vector_size = self.embedder.vector_size
        return
    

    def transform(self, tokenized_corpus :List[List[str]], save_vocab :bool=False, save_weights :bool=False) -> np.ndarray:
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
                self.tfidf_vectorizer.get_feature_names_out()[tf_corpus[i].nonzero()[1]]
            )

            if tokens.size == 0:
                continue
            elif tokens.size == 1:
                doc_vecs[i] = self.embedder.get_vector(tokens[0])

            # word vectors and their norms
            vecs = self.embedder.get_vectors(tokens)
            vecs_norm = np.linalg.norm(vecs, axis=1).reshape(tokens.size,1)

            # cosine similarity between each pair of tokens in i-th document
            cos_sim = ((vecs @ vecs.T) / (vecs_norm @ vecs_norm.T) + 1) / 2
            np.fill_diagonal(cos_sim, 1)

            # compute weight of each token for the i-th document
            weights = tfidf_corpus[i].data**self.exp_a * np.average(cos_sim, axis=1, weights=tf_corpus[i].data)**self.exp_b

            if self.eps_type == "abs":
                weights[weights <= self.eps] = 0.0
            elif self.eps_type == "pc":
                sorted_weights_idxs = np.argsort(weights / np.sum(weights))[::-1]
                cumsum_weights = np.cumsum(weights[sorted_weights_idxs])
                weights[sorted_weights_idxs[cumsum_weights >= self.eps]] = 0.0
            else:
                raise NotImplementedError(f"'{self.eps_type}' not implemented")

            weights = normalize(weights.reshape(1, -1), norm="l1")[0]

            if save_weights:
                self.weights_.append({tokens[i]:weights[i] for i in range(tokens.size)})

            # i-th document vector
            doc_vecs[i] = np.average(vecs, axis=0, weights=weights)
        return doc_vecs