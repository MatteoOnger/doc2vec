from components.embedders import Embedder
from components.preprocessors import Preprocessor



class Doc2vec():
    def __init__(self, preproc :Preprocessor, embedder :Embedder):
        self.preproc = preproc
        self.embedder = embedder
        return