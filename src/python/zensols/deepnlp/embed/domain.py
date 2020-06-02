"""Interface file for embedings.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WordEmbedModel(ABC):
    """This is an abstract base class that represents a set of word vectors
    (i.e. GloVe).

    :param path: the path to the model file(s)
    :param cache: if ``True`` globally cache all data strucures, which should
                  be ``False`` if more than one embedding across a model type
                  is used.

    """
    UNKNOWN = '<unk>'
    ZERO = UNKNOWN
    CACHE = {}

    name: str
    path: Path
    cache: bool = field(default=True)
    lowercase: bool = field(default=False)

    @abstractmethod
    def _create_data(self) -> Tuple[np.ndarray, List[str],
                                    Dict[str, int], Dict[str, np.ndarray]]:
        """Return the vector data from the model in the form:

            (vectors, word2vec, words, word2idx)

        where:
            - ``vectors`` are the word vectors
            - ``word2vec`` is the word to word vector mapping
            - ``words`` are the string vocabulary
            - ``word2idx`` is the word to word vector index mapping

        """
        pass

    def clear_cache(self):
        self.CACHE.clear()

    def _data(self) -> Tuple[np.ndarray, List[str],
                             Dict[str, int], Dict[str, np.ndarray]]:
        if not hasattr(self, '_data_inst'):
            self._data_inst = self.CACHE.get(self.name)
            if self._data_inst is None:
                self._data_inst = self._create_data()
                self.CACHE[self.name] = self._data_inst
        return self._data_inst

    def __getstate__(self):
        state = dict(self.__dict__())
        state.pop('_data_inst', None)
        return state

    @property
    def matrix(self) -> np.ndarray:
        """Return the word vector matrix.

        """
        return self._data()[0]

    @property
    def vectors(self) -> Dict[str, np.ndarray]:
        """Return all word vectors with the string words as keys.

        """
        return self._data()[1]

    @property
    def vector_dimension(self) -> int:
        """Return the dimension of the word vectors.

        """
        return self.matrix.shape[1]

    def word2idx_or_unk(self, word: str) -> int:
        if self.lowercase:
            word = word.lower()
        word2idx = self._data()[3]
        idx = word2idx.get(word)
        if idx is None:
            idx = word2idx.get(self.UNKNOWN)
        return idx

    def get(self, key: str, default: np.ndarray = None) -> np.ndarray:
        """Just like a ``dict.get()``, but but return the vector for a word.

        :param key: the word to get the vector

        :param default: what to return if ``key`` doesn't exist in the dict

        :return: the word vector
        """
        if self.lowercase:
            key = key.lower()
        if key not in self.vectors:
            key = self.UNKNOWN
        return self.vectors.get(key, default)

    def __getitem__(self, key: str):
        if self.lowercase:
            key = key.lower()
        return self.vectors[key]

    def __contains__(self, key: str):
        if self.lowercase:
            key = key.lower()
        return key in self.vectors

    def __len__(self):
        return self.matrix.shape[0]

    def __str__(self):
        return (f'glove: num words: {len(self)}, ' +
                f'vector dim: {self.vector_dimension}')
