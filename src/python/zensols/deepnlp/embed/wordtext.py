"""Contains an abstract class that makes it easier to implement load word
vectors from text files.

"""
__author__ = 'Paul Landes'

from typing import List, Dict
from dataclasses import dataclass, field
from abc import abstractmethod, ABCMeta
import logging
from pathlib import Path
import pickle
import numpy as np
import h5py
from h5py import Dataset
from zensols.util import time
from zensols.persist import Primeable
from zensols.deepnlp.embed import WordVectorModel, WordEmbedModel

logger = logging.getLogger(__name__)


@dataclass
class TextWordModelMetadata(object):
    """Describes a text based :class:`.WordEmbedModel`.  This information in this
    class is used to construct paths both text source vector file and all
    generated binary files

    """
    name: str = field()
    """The name of the word vector set (i.e. glove)."""

    desc: str = field()
    """A descriptor about this particular word vector set (i.e. 6B)."""

    dimension: int = field()
    """The dimension of the word vectors."""

    n_vocab: int = field()
    """The number of words in the vocabulary."""

    source_path: Path = field()
    """The path to the text file."""

    def __post_init__(self):
        sub_dir = Path('bin', f'{self.desc}.{self.dimension}')
        self.bin_dir = self.source_path.parent / sub_dir
        self.bin_file = self.bin_dir / 'vec.dat'
        self.words_file = self.bin_dir / 'words.dat'
        self.idx_file = self.bin_dir / 'idx.dat'


@dataclass
class TextWordEmbedModel(WordEmbedModel, Primeable, metaclass=ABCMeta):
    """Extensions of this class read a text vectors file and compile, then write a
    binary representation for fast loading.

    """
    DATASET_NAME = 'vec'
    """Name of the dataset in the HD5F file."""

    @abstractmethod
    def _get_metadata(self) -> TextWordModelMetadata:
        """Create the metadata used to construct paths both text source vector file and
        all generated binary files.

        """
        pass

    @property
    def metadata(self):
        """Return the metadata used to construct paths both text source vector file and
        all generated binary files.

        """
        if not hasattr(self, '_metadata'):
            self._metadata = self._get_metadata()
        return self._metadata

    def _get_model_id(self) -> str:
        """Return a string used to uniquely identify this model.

        """
        meta = self.metadata
        return f'{meta.name}: description={meta.desc}, dim={meta.dimension}'

    def _populate_vec_lines(self, words: List[str], word2idx: Dict[str, int],
                            ds: Dataset):
        """Add word vectors to the h5py dataset, vocab and vocab index.

        :param words: the list of vocabulary words

        :param word2idx: dictionary of word to word vector index (row)

        :param ds: the h5py data structure to add the word vectors

        """
        meta = self.metadata
        idx = 0
        lc = 0
        with open(meta.source_path, 'rb') as f:
            for rix, ln in enumerate(f):
                lc += 1
                line = ln.decode().split(' ')
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                try:
                    ds[rix, :] = line[1:]
                except Exception as e:
                    logger.error(f'could not parse line {lc} ' +
                                 f'(word: {word}): {e}; line: {ln}')
                    raise e

    def _write_vecs(self) -> np.ndarray:
        """Write the bcolz binary files.  Only when they do not exist on the files
        system already are they calculated and written.

        """
        meta = self.metadata
        meta.bin_dir.mkdir(parents=True, exist_ok=True)
        words = []
        word2idx = {}
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'writing binary vectors {meta.source_path} ' +
                        f'-> {meta.bin_dir}')
        shape = (meta.n_vocab, meta.dimension)
        with time(f'wrote h5py to {meta.bin_file}'):
            with h5py.File(meta.bin_file, 'w') as f:
                dset: Dataset = f.create_dataset(
                    self.DATASET_NAME, shape, dtype='float64')
                self._populate_vec_lines(words, word2idx, dset)
        with open(meta.words_file, 'wb') as f:
            pickle.dump(words[:], f)
        with open(meta.idx_file, 'wb') as f:
            pickle.dump(word2idx, f)

    def _assert_binary_vecs(self):
        meta = self.metadata
        if not meta.bin_file.exists():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'wriging binary vectors to: {meta.bin_file}')
            self._write_vecs()

    def prime(self):
        self._assert_binary_vecs()

    def _create_data(self) -> WordVectorModel:
        """Read the binary bcolz, vocabulary and index files from disk.

        """
        self._assert_binary_vecs()
        meta = self.metadata
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'reading binary vector file: {meta.bin_file}')
        with time('loaded {cnt} vectors'):
            with h5py.File(meta.bin_file, 'r') as f:
                ds: Dataset = f[self.DATASET_NAME]
                vectors = ds[:]
            with open(meta.words_file, 'rb') as f:
                words = pickle.load(f)
            with open(meta.idx_file, 'rb') as f:
                word2idx = pickle.load(f)
            cnt = len(word2idx)
        with time('prepared vectors'):
            unknown_vec = np.expand_dims(np.zeros(self.dimension), axis=0)
            vectors = np.concatenate((vectors, unknown_vec))
            word2idx[self.UNKNOWN] = len(words)
            words.append(self.UNKNOWN)
            word2vec = {w: vectors[word2idx[w]] for w in words}
        return WordVectorModel(vectors, word2vec, words, word2idx)
