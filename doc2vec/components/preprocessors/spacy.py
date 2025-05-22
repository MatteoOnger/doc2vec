import re
import spacy

from enum import Enum
from spacy.language import Language
from typing import List, Literal, Set

from doc2vec.components.preprocessor import Preprocessor



class SpaCyPreprocessor(Preprocessor):
    """
    A spaCy-based implementation of the Preprocessor abstract class.
    
    This preprocessor supports lemmatization, case control, stopword handling,
    POS tag filtering, and token exclusion based on regex or token type 
    (e.g., email, number, punctuation, URL).

    See Also
    --------
    - SpaCy website: https://spacy.io.
    """

    class CREIT(Enum):
        """
        Enum for common regular expressions to identify invalid tokens.

        Attributes
        ----------
        ALPHA3 : str
            Excludes tokens with fewer than 3 characters or that are not alphabetic.
        """
        ALPHA3 = r"^(?:.*[^a-z]{1,}.*|.{,2})$"
        """
        Excludes tokens with fewer than 3 characters or that are not alphabetic.
        """


    @Language.component("lower_case_lemmas")
    def _lower_case_lemmas(doc: spacy.tokens.Doc) -> spacy.tokens.Doc:
        """
        SpaCy pipeline component that lowercases all lemmas in a document.

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
        lemmatize: bool = False,
        case_sensitive: bool = False,
        keep_stopwords: bool = False,
        extend_stopwords: Set[str]|None = None,
        pos_to_keep: Set[str]|None = None,
        pos_to_rm: Set[str]|None = None,
        regex_invalid_tokens: 'str|SpaCyPreprocessor.CREIT|None' = None,
        email: Literal['KP', 'RM'] = 'KP',
        numb: Literal['KP', 'RM'] = 'KP',
        punc: Literal['KP', 'RM'] = 'KP',
        url: Literal['KP', 'RM'] = 'KP',
        pipeline: str = 'en_core_web_sm',
        n_process: int = -1,
        batch_size: int = 1000
    ):
        """
        Parameters
        ----------
        lemmatize : bool, optional
            Whether to lemmatize tokens. Default is `False`.
        case_sensitive : bool, optional
            Whether to preserve original casing. Default is `False`.
        keep_stopwords : bool, optional
            Whether to retain stopwords. Default is `False`.
        extend_stopwords : set of str, optional
            Additional stopwords to include. Default is `None`.
        pos_to_keep : set of str, optional
            POS tags to retain. Default is `None` (keep all).
        pos_to_rm : set of str, optional
            POS tags to remove. Default is `None` (remove none).
        regex_invalid_tokens : str or CREIT, optional
            Regex pattern or CREIT enum to filter out unwanted tokens. Default is `None`.
        email : {'KP', 'RM'}, optional
            Whether to keep (`'KP'`) or remove (`'RM'`) email tokens. Default is `'KP'`.
        numb : {'KP', 'RM'}, optional
            Whether to keep or remove numerical tokens. Default is `'KP'`.
        punc : {'KP', 'RM'}, optional
            Whether to keep or remove punctuation tokens. Default is `'KP'`.
        url : {'KP', 'RM'}, optional
            Whether to keep or remove URL tokens. Default is `'KP'`.
        pipeline : str, optional
            spaCy model to load. Default is `'en_core_web_sm'`.
        n_process : int, optional
            Number of processes to use. Set to `-1` to use all available CPUs. Default is `-1`.
        batch_size : int, optional
            Number of documents per processing batch. Default is `1000`.
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


    def preprocess_corpus(self, corpus: List[str])  -> List[List[str]]:
        docs = self.nlp.pipe(corpus, n_process=self.n_process, batch_size=self.batch_size)
        if self.lemmatize:
            tokens = [[tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
        else:
            if self.case_sensitive:
                tokens = [[tk.text for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
            else:
                tokens = [[tk.text.lower() for tk in doc if all(f(tk) for f in self.conditions)] for doc in docs]
        return tokens


    def preprocess_doc(self, document: str) -> List[str]:
        doc = self.nlp(document)
        if self.lemmatize:
            tokens = [tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)]
        else:
            if self.case_sensitive:
                tokens = [tk.text for tk in doc if all(f(tk) for f in self.conditions)]
            else:
                tokens = [tk.text.lower() for tk in doc if all(f(tk) for f in self.conditions)]
        return tokens
