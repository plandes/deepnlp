"""Interface file for word vectors, aka non-contextual word embeddings.

"""
__author__ = 'Paul Landes'

from typing import List, Dict, Tuple, Iterable
from dataclasses import dataclass, field
from abc import ABCMeta, abstractmethod
import logging
import numpy as np
import torch
from torch import Tensor
import gensim
from gensim.models.keyedvectors import Word2VecKeyedVectors, KeyedVectors
from zensols.persist import persisted, PersistableContainer, PersistedWork
from zensols.deeplearn import TorchConfig, DeepLearnError

logger = logging.getLogger(__name__)


class WordEmbedError(DeepLearnError):
    """Raised for any errors pertaining to word vectors."""


@dataclass
class WordVectorModel(object):
    """Vector data from the model

    """
    vectors: np.ndarray = field()
    """The word vectors."""

    word2vec: Dict[str, np.ndarray] = field()
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
class _WordEmbedVocabAdapter(object):
    """Adapts a :class:`.WordEmbedModel` to a gensim :class:`.KeyedVectors`, which
    is used in :meth:`.WordEmbedModel._create_keyed_vectors`.

    """
    model: WordVectorModel

    def __post_init__(self):
        self._index = -1

    @property
    def index(self):
        return self._index

    def __iter__(self):
        words: List[str] = self.model.words
        return iter(words)

    def get(self, word: int, default: str):
        self._index = self.model.word2idx.get(word, default)

    def __getitem__(self, word: str):
        self._index = self.model.word2idx[word]
        return self


@dataclass
class WordEmbedModel(PersistableContainer, metaclass=ABCMeta):
    """This is an abstract base class that represents a set of word vectors
    (i.e. GloVe).

    """
    UNKNOWN = '<unk>'
    """The unknown symbol used for out of vocabulary words."""

    ZERO = UNKNOWN
    """The zero vector symbol used for padding vectors."""

    _CACHE = {}
    """Contains cached embedding model that point to the same source."""

    name: str = field()
    """The name of the model given by the configuration and must be unique
    across word vector type and dimension.

    """

    cache: bool = field(default=True)
    """If ``True`` globally cache all data strucures, which should be ``False``
    if more than one embedding across a model type is used.

    """

    lowercase: bool = field(default=False)
    """If ``True``, downcase each word for all methods that take a word as
    input.

    """
    def __post_init__(self):
        super().__init__()
        self._data_inst = PersistedWork('_data_inst', self, transient=True)

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
        for model in self._CACHE.values():
            self._try_deallocate(model)
        self._CACHE.clear()

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

    @persisted('_data_inst', transient=True)
    def _data(self) -> WordVectorModel:
        model_id = self.model_id
        wv_model = self._CACHE.get(model_id)
        if wv_model is None:
            wv_model = self._create_data()
            self._CACHE[model_id] = wv_model
        return wv_model

    @property
    def matrix(self) -> np.ndarray:
        """The word vector matrix."""
        return self._data().vectors

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the word vector :obj"`matrix`."""
        return self.matrix.shape

    def to_matrix(self, torch_config: TorchConfig) -> Tensor:
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

    def keys(self) -> Iterable[str]:
        """Return the keys, which are the word2vec words.

        """
        return self.vectors.keys()

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

    @property
    @persisted('_keyed_vectors', transient=True)
    def keyed_vectors(self) -> KeyedVectors:
        return self._create_keyed_vectors()

    def _create_keyed_vectors(self) -> KeyedVectors:
        kv = Word2VecKeyedVectors(vector_size=self.vector_dimension)
        if gensim.__version__[0] >= '4':
            kv.key_to_index = self._data().word2idx
        else:
            kv.vocab = _WordEmbedVocabAdapter(self._data())
        kv.vectors = self.matrix
        kv.index2entity = list(self._data().words)
        return kv

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
        if self._data_inst.is_set():
            return (f'word embed model: num words: {len(self)}, ' +
                    f'vector dim: {self.vector_dimension}')
        else:
            return self.model_id
