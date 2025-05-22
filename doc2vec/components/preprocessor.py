from abc import ABC, abstractmethod
from typing import List



class Preprocessor(ABC):
    """
    Abstract base class for a generic text preprocessor.
    """

    @abstractmethod
    def preprocess_corpus(self, corpus: List[str]) -> List[List[str]]:
        """
        Preprocess an entire corpus of documents.

        Parameters
        ----------
        corpus : List[str]
            List of documents to preprocess.

        Returns
        -------
        : List[List[str]]
            A list where each sublist contains the tokens of one document.
        """
        pass


    @abstractmethod
    def preprocess_doc(self, doc: str) -> List[str]:
        """
        Preprocess a single document.

        Parameters
        ----------
        doc : str
            A single document to preprocess.

        Returns
        -------
        : List[str]
            A list of tokens from the document.
        """
        pass
