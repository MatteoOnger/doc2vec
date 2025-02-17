import re
import spacy

from enum import Enum
from spacy.language import Language


class Preprocessor():
    """
    This class implements a simple preprocessor to split documents into lists of tokens.
    It is mainly based on spaCy.
    """

    class CREIL(Enum):
        """
        Common regex of invalid lines.
        """
        HEADER = r"^\s*(?:-{5,}|.*-{5,}|from:|to:|cc:|ccn:|sent by:|subject:).*$"
        """
        Removes (forwarded, etc.) email header and blank lines.
        """


    class CREIT(Enum):
        """
        Common regex of invalid tokens.
        """
        ALPHA2 = r"^(?:.*[^a-z]{1,}.*|.)$"
        """
        Only tokens of two or more characters consisting only of alphabetic characters are retained.
        """
        ALPHA_2 = r"^(?:.*[^a-z_-]{1,}.*|.)$"
        """
        Only tokens of two or more characters consisting only of alphabetic characters, and high or low dashes, are retained.
        """
        ALPHANUM = r".*(?:\W|_).*"
        """
        Removes tokens that contain non-alphanumeric characters, '_' included.
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
        case_sensitive :bool=True,
        lemmatize :bool=False,
        noun_chunks :bool=False,
        keep_stopwords :bool=True,
        extend_stopwords :set[str]|None=None,
        regex_flags :int=0,
        regex_invalid_line :'str|Preprocessor.CREIL|None'=None,
        regex_invalid_tokens :'str|Preprocessor.CREIT|None'=None,
        ent_to_keep :set[str]|None=None,
        ent_to_rm :set[str]|None=None,
        pos_to_keep :set[str]|None=None,
        pos_to_rm :set[str]|None=None,
    ):
        """
        Parameters
        ----------
        case_sensitive : bool, optional
            If ``False``, all returned tokens are lowercase, by default ``True``.
        lemmatize : bool, optional
            If ``True``, tokens are lemmatized, by default ``False``.
        noun_chunks : bool, optional
            If ``True``, chunk nouns are merged, by default ``False``.
        keep_stopwords : bool, optional
            If ``True``, stop words are kept, by default ``True``.
        extend_stopwords : set[str] | None, optional
            List of stop words to add, by default ``None``.
        regex_flags : int, optional
            Flags of the regular expressions, by default ``0``.
        regex_invalid_line : str | Preprocessor.CREIL | None, optional
            Regex to mark lines that must be excluded, by default ``None``.
        regex_invalid_tokens : str | Preprocessor.CREIT | None, optional
            Regex to mark tokens that must be excluded, by default ``None``.
        ent_to_keep : set[str] | None, optional
            Only tokens marked with one of these entity tags are kept,
            by default all tokens are retained.
        ent_to_rm : set[str] | None, optional
            Tokens marked with one of these entity tags are removed,
            by default all tokens are retained.
        pos_to_keep : set[str] | None, optional
            Only tokens marked with one of these POS tags are kept,
            by default all tokens are retained.
        pos_to_rm : set[str] | None, optional
            Tokens marked with one of these POS tags are removed,
            by default all tokens are retained.
        """
        self.nlp = spacy.load('en_core_web_sm')

        # update nlp piepline
        if not lemmatize and not noun_chunks:
            self.nlp.remove_pipe("lemmatizer")
            if ent_to_keep is None and ent_to_rm is None and pos_to_keep is None and pos_to_rm is None:
                self.nlp.remove_pipe("tagger")
                self.nlp.remove_pipe("parser")
                self.nlp.remove_pipe("attribute_ruler")
                self.nlp.remove_pipe("ner")
        if noun_chunks:
            self.nlp.add_pipe("merge_noun_chunks")
        if lemmatize and not case_sensitive:
            self.nlp.add_pipe("lower_case_lemmas", name="lower_case_lemmas")

        # standard regex
        self.regex_empty_line = r"^\s*$"
        self.regex_endline_1 = r"(=20\n)"
        self.regex_endline_2 = r"(=\n)"

        self.pattern_empty_line = re.compile(self.regex_empty_line, regex_flags)
        self.pattern_endline_1 = re.compile(self.regex_endline_1, regex_flags)
        self.pattern_endline_2 = re.compile(self.regex_endline_2, regex_flags)

        # save fields
        self.case_sensitive = case_sensitive
        self.noun_chunks = noun_chunks
        self.lemmatize = lemmatize
        self.keep_stopwords = keep_stopwords
        self.extend_stopwords = extend_stopwords
        self.regex_flags = regex_flags
        self.regex_invalid_line = regex_invalid_line.value if isinstance(regex_invalid_line, Preprocessor.CREIL) else regex_invalid_line
        self.regex_invalid_tokens = regex_invalid_tokens.value if isinstance(regex_invalid_tokens,  Preprocessor.CREIT) else regex_invalid_tokens
        self.ent_to_keep = ent_to_keep
        self.ent_to_rm = ent_to_rm
        self.pos_to_keep = pos_to_keep
        self.pos_to_rm = pos_to_rm

        # assemble the condition that tokens/lines must satisfy
        self.conditions = list()
        if not keep_stopwords:
            self.conditions.append(lambda x: not x.is_stop)
            if extend_stopwords is not None:
                if lemmatize:
                    self.conditions.append(lambda x: x.lemma_ not in extend_stopwords)
                else:
                    self.conditions.append(lambda x: x.text not in extend_stopwords if case_sensitive else x.text.lower() not in extend_stopwords)
        if ent_to_keep is not None:
            self.conditions.append(lambda x: x.ent_type_ in ent_to_keep)
        if ent_to_rm is not None:
            self.conditions.append(lambda x: x.ent_type_ not in ent_to_rm)
        if pos_to_keep is not None:
            self.conditions.append(lambda x: x.pos_ in pos_to_keep)
        if pos_to_rm is not None:
            self.conditions.append(lambda x: x.pos_ not in pos_to_rm)

        if regex_invalid_line is not None:
            self.pattern_invalid_line = re.compile(self.regex_invalid_line, regex_flags)
        if regex_invalid_tokens is not None:
            self.pattern_invalid_tokens = re.compile(self.regex_invalid_tokens, regex_flags)
            self.conditions.append(lambda x: not self.pattern_invalid_tokens.match(x.text))
        return


    def preprocess(self, document :str) -> list[str]:
        """
        Applies the preprocessing procedure to the given document.

        Parameters
        ----------
        document : str
            Document to process.

        Returns
        -------
        :list[str]
            List of tokens representing the document.
        """
        # MIME protocol for base64, the max line length in the encoded data is 76 characters
        # an '=' sign at the end of an encoded line is used to tell the decoder that the line is to be continued,
        # while an '=20' sign is used to tell the line ends
        document = self.pattern_endline_1.sub("\n", document)
        document = self.pattern_endline_2.sub("", document)

        # read the document line by line and remove the invalid ones that are:
        # - blank lines
        # - lines that match the given regex 
        # - lines preceded by invalid non-white lines
        prev = True
        filtered_document = list()
        for line in document.splitlines():
            if self.pattern_empty_line.match(line):
                prev = True
            elif prev and (self.regex_invalid_line is None or not self.pattern_invalid_line.match(line)):
                filtered_document.append(line)
            else:
                prev = False

        # replace all withespaces with a single whitespace
        document = re.sub(r"\s{1,}", " ", "\n".join(filtered_document))

        # tokenize, etc. using spaCy
        doc = self.nlp(document)
        if self.lemmatize:
            tokens = [tk.lemma_ for tk in doc if all(f(tk) for f in self.conditions)]
        else:
            if self.case_sensitive:
                tokens = [tk.text for tk in doc if all(f(tk) for f in self.conditions)]
            else:
                tokens = [tk.text.lower() for tk in doc if all(f(tk) for f in self.conditions)]
        return tokens
