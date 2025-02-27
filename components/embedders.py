import numpy as np

from abc import ABC, abstractmethod



class Embedder(ABC):
    """
    """
    @abstractmethod
    def get_vector(self, word :str) -> np.array:
        pass