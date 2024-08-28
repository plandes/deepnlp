"""Contains an abstract class that makes it easier to implement load word
vectors from text files.

"""
__author__ = 'Paul Landes'

from typing import List, Dict
from dataclasses import dataclass, field, InitVar
from abc import abstractmethod, ABCMeta
import logging
from pathlib import Path
import pickle
import numpy as np
import h5py
from h5py import Dataset
from zensols.util import time
from zensols.config import Dictable
from zensols.persist import Primeable
from zensols.install import Installer, Resource
from zensols.deepnlp.embed import WordVectorModel, WordEmbedModel
from . import WordEmbedError

logger = logging.getLogger(__name__)


@dataclass
class TextWordModelMetadata(Dictable):
    """Describes a text based :class:`.WordEmbedModel`.  This information in
    this class is used to construct paths both text source vector file and all
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

    sub_directory: InitVar[Path] = field(default=None)
    """The subdirectory to be appended to :obj:`self.bin_dir`, which defaults to
    the directory ``bin/<description>.<dimension>``.

    """
    def __post_init__(self, sub_directory: Path):
        if sub_directory is None:
            fname: str = f'{self.name}.{self.desc}.{self.dimension}'
            sub_directory = Path('bin', fname)
        self.bin_dir = self.source_path.parent / sub_directory
        self.bin_file = self.bin_dir / 'vec.dat'
        self.words_file = self.bin_dir / 'words.dat'
        self.idx_file = self.bin_dir / 'idx.dat'


@dataclass
class TextWordEmbedModel(WordEmbedModel, Primeable, metaclass=ABCMeta):
    """Extensions of this class read a text vectors file and compile, then write
    a binary representation for fast loading.

    """
    DATASET_NAME = 'vec'
    """Name of the dataset in the HD5F file."""

    path: Path = field(default=None)
    """The path to the model file(s)."""

    installer: Installer = field(default=None)
    """The installer used to for the text vector zip file."""

    resource: Resource = field(default=None)
    """The zip resource used to find the path to the model files."""

    @abstractmethod
    def _get_metadata(self) -> TextWordModelMetadata:
        """Create the metadata used to construct paths both text source vector
        file and all generated binary files.

        """
        pass

    def _install(self) -> Path:
        """Install any missing word vector models."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'install resource for {self.name}: {self.resource}')
        self.installer()
        return self.installer[self.resource]

    @property
    def metadata(self):
        """Return the metadata used to construct paths both text source vector
        file and all generated binary files.

        """
        if not hasattr(self, '_metadata'):
            if self.path is None and self.installer is None:
                raise WordEmbedError('No path is not set')
            if self.installer is not None and self.resource is None:
                raise WordEmbedError("Installer given but not 'resource''")
            if self.installer is not None:
                self.path = self._install()
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
                line = ln.decode().strip().split(' ')
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                try:
                    ds[rix, :] = line[1:]
                except Exception as e:
                    raise WordEmbedError(
                        f'Could not parse line {lc} (word: {word}): ' +
                        f'{e}; line: {ln}') from e

    def _write_vecs(self) -> np.ndarray:
        """Write the h5py binary files.  Only when they do not exist on the
        files system already are they calculated and written.

        """
        meta = self.metadata
        meta.bin_dir.mkdir(parents=True, exist_ok=True)
        words = []
        word2idx = {}
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'writing binary vectors {meta.source_path} ' +
                        f'-> {meta.bin_dir}')
        shape = (meta.n_vocab, meta.dimension)
        if logger.isEnabledFor(logging.INFO):
            logger.info(f'creating h5py binary vec files with shape {shape}:')
            meta.write_to_log(logger, logging.INFO, 1)
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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f'{meta.bin_file} exists: {meta.bin_file.exists()}')
        if not meta.bin_file.exists():
            if logger.isEnabledFor(logging.INFO):
                logger.info(f'writing binary vectors to: {meta.bin_file}')
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
                vectors: np.ndarray = ds[:]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f'word embedding type: {vectors.dtype}')
            with open(meta.words_file, 'rb') as f:
                words = pickle.load(f)
            with open(meta.idx_file, 'rb') as f:
                word2idx = pickle.load(f)
            cnt = len(word2idx)
        with time('prepared vectors'):
            unknown_vec: np.ndarray = np.expand_dims(
                np.zeros(self.dimension), axis=0)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f'unknown type: {unknown_vec.dtype}')
            vectors: np.ndarray = np.concatenate((vectors, unknown_vec))
            word2idx[self.UNKNOWN] = len(words)
            words.append(self.UNKNOWN)
            word2vec = {w: vectors[word2idx[w]] for w in words}
        return WordVectorModel(vectors, word2vec, words, word2idx)


@dataclass
class DefaultTextWordEmbedModel(TextWordEmbedModel):
    """This class uses the Stanford pretrained GloVE embeddings as a ``dict``
    like Python object.  It loads the glove vectors from a text file and then
    creates a binary file that's quick to load on subsequent uses.

    An example configuration would be::

        [glove_embedding]
        class_name = zensols.deepnlp.embed.GloveWordEmbedModel
        path = path: ${default:corpus_dir}/glove
        desc = 6B
        dimension = 50

    """
    name: str = field(default='unknown_name')
    """The name of the word vector set (i.e. glove)."""

    desc: str = field(default='unknown_desc')
    """The size description (i.e. 6B for the six billion word trained vectors).

    """
    dimension: int = field(default=50)
    """The word vector dimension."""

    vocab_size: int = field(default=0)
    """Vocabulary size."""

    file_name_pattern: str = field(default='{name}.{desc}.{dimension}d.txt')
    """The format of the file to create."""

    @property
    def file_name(self) -> str:
        return self.file_name_pattern.format(
            name=self.name,
            desc=self.desc,
            dimension=self.dimension)

    def _get_metadata(self) -> TextWordModelMetadata:
        name: str = self.name
        dim: int = self.dimension
        desc: str = self.desc
        path: Path = self.path / self.file_name
        return TextWordModelMetadata(name, desc, dim, self.vocab_size, path)
