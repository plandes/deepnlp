"""Convenience Gensim glue code for word embeddings/vectors.

"""
__author__ = 'Paul Landes'


from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import logging
import numpy as np
from gensim.models import (
    KeyedVectors,
    Word2Vec,
)
from zensols.util import time
from zensols.deepnlp.embed import WordEmbedModel

logger = logging.getLogger(__name__)


@dataclass
class Word2VecModel(WordEmbedModel):
    """Load keyed or non-keyed Gensim models.

    """
    dimension: int = field(default=300)
    model_type: str = field(default='keyed')

    def __post_init__(self):
        super().__post_init__()
        self.zero_arr = np.zeros((1, int(self.dimension)),)

    def _get_model(self):
        """The word2vec model.

        """
        with time('loaded word2vec model'):
            if self.model_type == 'keyed':
                model = self._get_keyed_model()
            else:
                model = self._get_trained_model()
            return model

    def _get_keyed_model(self):
        """Load a model from a pretrained word2vec model.

        """
        logger.info(f'loading keyed file: {self.path}')
        fname = str(self.path.absolute())
        return KeyedVectors.load_word2vec_format(fname, binary=True)

    def _get_trained_model(self):
        """Load a model trained with gensim.

        """
        path = self.path
        if path.exists():
            logger.info('loading trained file: {}'.format(path))
            model = Word2Vec.load(str(path.absolute()))
        else:
            model = self._train()
            logger.info('saving trained vectors to: {}'.format(path))
            model.save(str(path.absolute()))
        return model

    def _create_data(self) -> Tuple[np.ndarray, List[str],
                                    Dict[str, int], Dict[str, np.ndarray]]:
        logger.info(f'reading binary vector file')
        wv = self._get_model().wv
        words = wv.index2entity
        word2vec = wv
        word2idx = {}
        vectors = []
        with time('created data structures'):
            for i, word in enumerate(words):
                word2idx[word] = i
                vectors.append(wv[word])
            vectors = np.array(vectors)
        return vectors, word2vec, words, word2idx
