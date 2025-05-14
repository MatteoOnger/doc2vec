import re
import spacy

from abc import ABC, abstractmethod
from enum import Enum
from spacy.language import Language
from typing import List, Literal, Set



class Preprocessor(ABC):
    """
    Abstract class, it represents a generic preprocessor.
    """

    @abstractmethod
    def preprocess_corpus(self, corpus :List[str]) -> List[List[str]]:
        """
        Apply the preprocessing procedure to the given corpus.

        Parameters
        ----------
        corpus : List[str]
            Corpus to process.

        Returns
        -------
        : List[List[str]]
            One list of tokens for each document in the corpus.
        """
        pass


    @abstractmethod
    def preprocess_doc(self, doc :str) -> List[str]:
        """
        Apply the preprocessing procedure to the given document.

        Parameters
        ----------
        document : str
            Document to process.

        Returns
        -------
        : List[str]
            List of tokens representing the document.
        """
        pass



class SpaCyPreprocessor(Preprocessor):
    """
    This class implements a simple preprocessor to split documents into lists of tokens.
    It is mainly based on SpaCy.

    See Also
    --------
    - SpaCy: https://spacy.io
    """

    class CREIT(Enum):
        """
        Common regex of invalid tokens.
        """
        ALPHA3 = r"^(?:.*[^a-z]{1,}.*|.{,2})$"
        """
        Only tokens of three or more characters consisting only of alphabetic characters are retained.
        """


    @Language.component("lower_case_lemmas")
    def _lower_case_lemmas(doc :spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        Change the capitalization of the lemmas to lowercase.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            Doc to modify.

        Returns
        -------
        : spacy.tokens.Doc
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
        regex_invalid_tokens :'str|SpaCyPreprocessor.CREIT|None'=None,
        email :Literal['KP', 'RM']='KP',
        numb :Literal['KP', 'RM']='KP',
        punc :Literal['KP', 'RM']='KP',
        url :Literal['KP', 'RM']='KP',
        pipeline :str='en_core_web_sm',
        n_process :int=-1,
        batch_size :int=1000
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
        pos_to_keep : Set[str] | None, optional
            Only tokens marked with one of these POS tags are kept,
            by default all tokens are retained.
        pos_to_rm : Set[str] | None, optional
            Tokens marked with one of these POS tags are removed,
            by default all tokens are retained.
        regex_invalid_tokens : str | SpaCyPreprocessor.CREIT | None, optional
            Regex to mark tokens that must be excluded, by default ``None``.
        email : Literal['KP', 'RM'], optional
            Keep (``'KP'``) or remove (``'RM'``) tokens that represent emails, by default ``'KP'``.
        numb : Literal['KP', 'RM'], optional
            Keep (``'KP'``) or remove (``'RM'``) tokens that represent numbers, by default ``'KP'``.
        punc : Literal['KP', 'RM'], optional
            Keep (``'KP'``) or remove (``'RM'``) tokens that represent punctuation, by default ``'KP'``.
        url : Literal['KP', 'RM'], optional
            Keep (``'KP'``) or remove (``'RM'``) tokens that represent URLs, by default ``'KP'``.
        pipeline : str, optional
            SpaCy pipeline used, by default ``'en_core_web_sm'``.
        n_process : int, optional
            Multiprocessing, maximum number of processes, by default ``-1``.
            Use as many processes as CPUs if set to ``-1``.
        batch_size : int, optional
            Number of documents per batch, by default ``1000``.
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
        self.regex_invalid_tokens = regex_invalid_tokens.value if isinstance(regex_invalid_tokens, SpaCyPreprocessor.CREIT) else regex_invalid_tokens
        self.email = email
        self.numb = numb
        self.punc = punc
        self.url = url
        self.pipeline = pipeline
        self.n_process = n_process
        self.batch_size = batch_size

        # assemble the condition that tokens must satisfy
        self.conditions = list()
        if not keep_stopwords:
            self.conditions.append(lambda x: not x.is_stop)
        if pos_to_keep is not None:
            self.conditions.append(lambda x: x.pos_ in pos_to_keep)
        if pos_to_rm is not None:
            self.conditions.append(lambda x: x.pos_ not in pos_to_rm)
        if regex_invalid_tokens is not None:
            self.pattern_invalid_tokens = re.compile(self.regex_invalid_tokens, 0 if case_sensitive else re.I)
            self.conditions.append(lambda x: not self.pattern_invalid_tokens.match(x.text))
        if email == "RM":
            self.conditions.append(lambda x: not x.like_email)
        if numb == "RM":
            self.conditions.append(lambda x: not x.like_num)
        if punc == "RM":
            self.conditions.append(lambda x: not x.is_punct)
        if url == "RM":
            self.conditions.append(lambda x: not x.like_url)     
        return


    def preprocess_corpus(self, corpus :List[str])  -> List[List[str]]:
        docs = self.nlp.pipe(corpus, n_process=self.n_process, batch_size=self.batch_size)
        if self.lemmatize:
            tokens = [[tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
        else:
            if self.case_sensitive:
                tokens = [[tk.text for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
            else:
                tokens = [[tk.text.lower() for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
        return tokens


    def preprocess_doc(self, document :str) -> List[str]:
        doc = self.nlp(document)
        if self.lemmatize:
            tokens = [tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)]
        else:
            if self.case_sensitive:
                tokens = [tk.text for tk in doc if all(f(tk) for f in self.conditions)]
            else:
                tokens = [tk.text.lower() for tk in doc if all(f(tk) for f in self.conditions)]
        return tokens