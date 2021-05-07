"""Interface file for embedings.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
from pathlib import Path
import logging
import numpy as np
import torch
from zensols.config import Deallocatable
from zensols.deeplearn import TorchConfig

logger = logging.getLogger(__name__)


@dataclass
class WordVectorModel(object):
    """Vector data from the model

    """
    vectors: np.ndarray = field()
    """The word vectors."""

    word2vec: List[str] = field()
    """The word to word vector mapping."""

    words: Dict[str, int] = field()
    """The string vocabulary."""

    word2idx: Dict[str, np.ndarray] = field()
    """The word to word vector index mapping."""

    def __post_init__(self):
        self.tensors = {}

    def to_matrix(self, torch_config: TorchConfig) -> torch.Tensor:
        dev = torch_config.device
        if dev in self.tensors:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'reusing already cached from {torch_config}')
            vecs = self.tensors[dev]
        else:
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'created tensor vectory matrix on {torch_config}')
            vecs = torch_config.from_numpy(self.vectors)
            self.tensors[dev] = vecs
        return vecs


@dataclass
class WordEmbedModel(Deallocatable, metaclass=ABCMeta):
    """This is an abstract base class that represents a set of word vectors
    (i.e. GloVe).

    """
    UNKNOWN = '<unk>'
    """The unknown symbol used for out of vocabulary words."""

    ZERO = UNKNOWN
    """The zero vector symbol used for padding vectors."""

    CACHE = {}
    """Contains cached embedding model that point to the same source."""

    name: str = field()
    """The name of the model given by the configuration and must be unique
    across word vector type and dimension.

    """

    path: Path = field()
    """The path to the model file(s)."""

    cache: bool = field(default=True)
    """If ``True`` globally cache all data strucures, which should be ``False``
    if more than one embedding across a model type is used.

    """

    lowercase: bool = field(default=False)
    """If ``True``, downcase each word for all methods that take a word as
    input.

    """

    @abstractmethod
    def _get_model_id(self) -> str:
        """Return a string that uniquely identifies this instance of the embedding
        model.  This should have the type, size and dimension of the embedding.

        :see: :obj:`model_id`

        """
        pass

    @abstractmethod
    def _create_data(self) -> WordVectorModel:
        """Return the vector data from the model in the form:

            (vectors, word2vec, words, word2idx)

        where:

        """
        pass

    def clear_cache(self):
        for model in self.CACHE.values():
            self._try_deallocate(model)
        self.CACHE.clear()

    def deallocate(self):
        self.clear_cache()
        super().deallocate()

    @property
    def model_id(self) -> str:
        """Return a string that uniquely identifies this instance of the embedding
        model.  This should have the type, size and dimension of the embedding.

        This string is used to cache models in both CPU and GPU memory so the
        layers can have the benefit of reusing the same in memeory word
        embedding matrix.

        """
        return self._get_model_id()

    def _data(self) -> Tuple[np.ndarray, List[str],
                             Dict[str, int], Dict[str, np.ndarray]]:
        if not hasattr(self, '_data_inst'):
            model_id = self.model_id
            self._data_inst = self.CACHE.get(model_id)
            if self._data_inst is None:
                self._data_inst = self._create_data()
                self.CACHE[model_id] = self._data_inst
        return self._data_inst

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop('_data_inst', None)
        return state

    @property
    def matrix(self) -> np.ndarray:
        """Return the word vector matrix.

        """
        return self._data().vectors

    def to_matrix(self, torch_config: TorchConfig) -> torch.Tensor:
        """Return a matrix the represents the entire vector embedding as a tensor.

        :param torch_config: indicates where to load the new tensor

        """
        return self._data().to_matrix(torch_config)

    @property
    def vectors(self) -> Dict[str, np.ndarray]:
        """Return all word vectors with the string words as keys.

        """
        return self._data().word2vec

    @property
    def vector_dimension(self) -> int:
        """Return the dimension of the word vectors.

        """
        return self.matrix.shape[1]

    def word2idx_or_unk(self, word: str) -> int:
        """Return the index of ``word`` or :obj:UNKONWN if not indexed.

        """
        if self.lowercase:
            word = word.lower()
        word2idx = self._data().word2idx
        idx = word2idx.get(word)
        if idx is None:
            idx = word2idx.get(self.UNKNOWN)
        return idx

    def prime(self):
        pass

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
        if not hasattr(self, '_data_inst'):
            return self.model_id
        else:
            return (f'word embed model: num words: {len(self)}, ' +
                    f'vector dim: {self.vector_dimension}')
