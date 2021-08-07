"""Fast text word vector implementation.

"""
___author__ = 'Paul Landes'

from typing import List, Dict
from dataclasses import dataclass, field
import logging
from h5py import Dataset
from zensols.deepnlp.embed import TextWordEmbedModel, TextWordModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class FastTextEmbedModel(TextWordEmbedModel):
    """This class reads the FastText word vector text data format and provides an
    instances of a :class:`.WordEmbedModel`.  Files that have the format that
    look like ``crawl-300d-2M.vec`` can be downloaded with the link below.

    :see: `FastText word vectors <https://fasttext.cc/docs/en/english-vectors.html>`_

    """
    desc: str = field(default='2M')
    corpus: str = field(default='crawl')
    dimension: str = field(default=300)

    def _get_metadata(self) -> TextWordModelMetadata:
        name = 'fasttext'
        # crawl-300d-2M.vec
        path = self.path / f'{self.corpus}-{self.dimension}d-{self.desc}.vec'
        desc = f'{self.corpus}-{self.desc}'
        with open(path, encoding='utf-8',
                  newline='\n', errors='ignore') as f:
            vocab_size, dim = map(int, f.readline().split())
        return TextWordModelMetadata(name, desc, dim, vocab_size, path)

    def _populate_vec_lines(self, words: List[str], word2idx: Dict[str, int],
                            ds: Dataset):
        meta = self.metadata
        idx = 0
        lc = 0
        with open(meta.source_path, encoding='utf-8',
                  newline='\n', errors='ignore') as f:
            n_vocab, dim = map(int, f.readline().split())
            for rix, ln in enumerate(f):
                lc += 1
                line = ln.rstrip().split(' ')
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
