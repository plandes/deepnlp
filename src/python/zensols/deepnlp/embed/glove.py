"""This library contains the definition of a class that operates like a dict to
retrieve GloVE word embeddings.  It also creates, stores and reads a binary
representation for quick loading on start up.

"""
__author__ = 'Paul Landes'

import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np
import bcolz
from zensols.deepnlp.embed import WordEmbedModel


logger = logging.getLogger(__name__)


@dataclass
class GloveWordEmbedModel(WordEmbedModel):
    """This class uses the Stanford pretrained GloVE embeddings as a ``dict`` like
    Python object.  It loads the glove vectors from a text file and then
    creates a binary file that's quick to load on subsequent uses.

    An example configuration would be:
    ``
        [glove_embedding]
        class_name = zensols.deepnlp.embed.GloveWordEmbedModel
        path = path: ${default:corpus_dir}/glove
        desc = 6B
        dimension = 50
    ``

    :param path: the directory path to where the unziped GloVE
                 embedding text vector files and binary representation
                 files subdirectory lives
    :param desc: the size description (i.e. 6B for the six billion word trained
                 vectors)
    :param dimension: the word vector dimension
    :param vocab_size: vocabulary size

    """

    desc: str = field(default='6B')
    dimension: str = field(default=300)
    vocab_size: int = field(default=400000)

    def _vec_paths(self) -> List[Path]:
        """Return a tuple of

        - the path to the text file
        - binary vectors directory
        - the binary vector file for the current configuration
        - the binary vocabulary file
        - the binary vocabulary to vector index mapping file

        """
        dim = self.dimension
        desc = self.desc
        vec_txt_path = self.path / f'glove.{desc}.{dim}d.txt'
        vec_bin_dir = self.path / 'bin'
        vec_bin_file = vec_bin_dir / f'{desc}.{dim}.dat'
        vec_words_file = vec_bin_dir / f'{desc}.{dim}_words.pkl'
        vec_idx_file = vec_bin_dir / f'{desc}.{dim}_idx.pkl'
        logger.debug(f'creating paths to {self.path}')
        return vec_txt_path, vec_bin_dir, vec_bin_file, vec_words_file, vec_idx_file

    def _write_vecs(self) -> np.ndarray:
        """Write the bcolz binary files.  Only when they do not exist on the files
        system already are they calculated and written.

        """
        vec_txt_path, vec_bin_dir, vec_bin_file, vec_words_file, vec_idx_file = self._vec_paths()
        vec_bin_dir.mkdir(parents=True, exist_ok=True)
        words = []
        idx = 0
        word2idx = {}
        logger.info(f'writing text {vec_txt_path} -> {vec_bin_dir}')
        vectors = bcolz.carray(np.zeros(1), rootdir=vec_bin_file, mode='w')
        with open(vec_txt_path, 'rb') as f:
            lc = 0
            for l in f:
                lc += 1
                line = l.decode().split(' ')
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                try:
                    vect = np.array(line[1:]).astype(np.float)
                except Exception as e:
                    logger.error(f"couldn't parse line {lc} (word: {word}): {e}; line: {l}")
                    raise e
                vectors.append(vect)
        vectors = bcolz.carray(
            vectors[1:].reshape((self.vocab_size, self.dimension)),
            rootdir=vec_bin_file,
            mode='w')
        vectors.flush()
        pickle.dump(words[:], open(vec_words_file, 'wb'))
        pickle.dump(word2idx, open(vec_idx_file, 'wb'))

    def _create_data(self) -> Tuple[np.ndarray, List[str],
                                    Dict[str, int], Dict[str, np.ndarray]]:
        """Read the binary bcolz, vocabulary and index files from disk.

        """
        vec_txt_path, vec_bin_dir, vec_bin_file, vec_words_file, vec_idx_file = self._vec_paths()
        if not vec_bin_file.exists():
            logger.info(f'wriging binary vectors to: {vec_bin_file}')
            self._write_vecs()
        logger.info(f'reading binary vector file: {vec_bin_file}')
        vectors = bcolz.open(vec_bin_file)[:]
        with open(vec_words_file, 'rb') as f:
            words = pickle.load(f)
        with open(vec_idx_file, 'rb') as f:
            word2idx = pickle.load(f)
        unknown_vec = np.expand_dims(np.zeros(self.dimension), axis=0)
        vectors = np.concatenate((vectors, unknown_vec))
        word2idx[self.UNKNOWN] = len(words)
        words.append(self.UNKNOWN)
        word2vec = {w: vectors[word2idx[w]] for w in words}
        return vectors, word2vec, words, word2idx
