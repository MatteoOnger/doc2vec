import spacy

from abc import ABC, abstractmethod
from typing import List, Set
from spacy.language import Language



class Preprocessor(ABC):
    """
    """
    @abstractmethod
    def preprocess_corpus(self, corpus :List[str], n_process :int, batch_size :int) -> List[List[str]]:
        """
        Applies the preprocessing procedure to the given corpus.

        Parameters
        ----------
        corpus : str
            Corpus to process.
        n_process : int, optional
            Multiprocessing, maximum number of processes, by default ``1``.
            Use as many processes as CPUs if set to ``-1``.
        batch_size : int, optional
            Number of documents per batch, by default ``1000``.

        Returns
        -------
        :List[List[str]]
            One of list of tokens for each document in the corpus.
        """
        pass


    @abstractmethod
    def preprocess_doc(self, doc :str) -> List[str]:
        """
        Applies the preprocessing procedure to the given document.

        Parameters
        ----------
        document : str
            Document to process.

        Returns
        -------
        :List[str]
            List of tokens representing the document.
        """
        pass



class SpacyPreprocessor(Preprocessor):
    """
    This class implements a simple preprocessor to split documents into lists of tokens.
    It is mainly based on SpaCy.

    See Also
    --------
    - SpaCy: https://spacy.io
    """

    @Language.component("lower_case_lemmas")
    def _lower_case_lemmas(doc :spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Changes the capitalization of the lemmas to lowercase.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            Doc to modify.

        Returns
        -------
        :spacy.tokens.Doc
            Doc modified.
        """
        for token in doc :
            token.lemma_ = token.lemma_.lower()
        return doc


    def __init__(
        self,
        lemmatize :bool=False,
        case_sensitive :bool=False,
        keep_stopwords :bool=False,
        extend_stopwords :Set[str]|None=None,
        pos_to_keep :set[str]|None=None,
        pos_to_rm :set[str]|None=None,
        pipeline :str='en_core_web_sm',
    ):
        """
        Parameters
        ----------
        lemmatize : bool, optional
            If ``True``, tokens are lemmatized, by default ``False``.
        case_sensitive : bool, optional
            If ``False``, all returned tokens are lowercase, by default ``False``.
        keep_stopwords : bool, optional
            If ``True``, stop words are kept, by default ``False``.
        extend_stopwords : Set[str] | None, optional
            List of stop words to add, by default ``None``.
        pos_to_keep : set[str] | None, optional
            Only tokens marked with one of these POS tags are kept,
            by default all tokens are retained.
        pos_to_rm : set[str] | None, optional
            Tokens marked with one of these POS tags are removed,
            by default all tokens are retained.
        pipeline : str, optional
            SpaCy pipeline used, by default ``'en_core_web_sm'``.
        """
        super().__init__()
        self.nlp = spacy.load(pipeline, disable=["parser", "ner"])
        
        # update nlp pipeline
        if lemmatize and not case_sensitive:
            self.nlp.add_pipe("lower_case_lemmas", name="lower_case_lemmas")
        if not lemmatize:
            self.nlp.remove_pipe("lemmatizer")
            if pos_to_keep is None and pos_to_rm is None:
                self.nlp.remove_pipe("tagger")
                self.nlp.remove_pipe("attribute_ruler")
        if extend_stopwords is not None:
            for w in extend_stopwords:
                self.nlp.vocab[w].is_stop = True
        
        # save fields
        self.lemmatize = lemmatize
        self.case_sensitive = case_sensitive
        self.keep_stopwords = keep_stopwords
        self.extend_stopwords = extend_stopwords
        self.pos_to_keep = pos_to_keep
        self.pos_to_rm = pos_to_rm
        self.pipeline = pipeline

        # assemble the condition that tokens must satisfy
        self.conditions = list()
        if not keep_stopwords:
            self.conditions.append(lambda x: not x.is_stop)
        if pos_to_keep is not None:
            self.conditions.append(lambda x: x.pos_ in pos_to_keep)
        if pos_to_rm is not None:
            self.conditions.append(lambda x: x.pos_ not in pos_to_rm)
        return


    def preprocess_corpus(self, corpus :List[str], n_process :int=1, batch_size :int=1000)  -> List[List[str]]:
        docs = self.nlp.pipe(corpus, n_process=n_process, batch_size=batch_size)
        if self.lemmatize:
            tokens = [[tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
        else:
            if self.case_sensitive:
                tokens = [[tk.text for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
            else:
                tokens = [[tk.text.lower_ for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
        return tokens


    def preprocess_doc(self, document :str) -> List[str]:
        doc = self.nlp(document)
        if self.lemmatize:
            tokens = [tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)]
        else:
            if self.case_sensitive:
                tokens = [tk.text for tk in doc if all(f(tk) for f in self.conditions)]
            else:
                tokens = [tk.text.lower_ for tk in doc if all(f(tk) for f in self.conditions)]
        return tokens