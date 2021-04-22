"""Convenience Gensim glue code for word embeddings/vectors.

"""
__author__ = 'Paul Landes'


from dataclasses import dataclass, field
import logging
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from zensols.util import time
from zensols.deepnlp.embed import WordVectorModel, WordEmbedModel

logger = logging.getLogger(__name__)


@dataclass
class Word2VecModel(WordEmbedModel):
    """Load keyed or non-keyed Gensim models.

    """
    dimension: int = field(default=300)
    """The dimension of the word embedding."""

    model_type: str = field(default='keyed')
    """The type of the embeddings, which is either ``keyed`` or ``gensim``."""

    def _get_model_id(self) -> str:
        return f'word2vec: type={self.model_type}, dim={self.dimension}'

    @property
    def model(self) -> KeyedVectors:
        return self._data()[1]

    def _get_model(self):
        """The word2vec model.

        """
        with time('loaded word2vec model'):
            if self.model_type == 'keyed':
                model = self._get_keyed_model()
            else:
                model = self._get_trained_model()
            return model

    def _get_keyed_model(self) -> KeyedVectors:
        """Load a model from a pretrained word2vec model.

        """
        logger.info(f'loading keyed file: {self.path}')
        fname = str(self.path.absolute())
        with time(f'loaded key model from {fname}'):
            return KeyedVectors.load_word2vec_format(fname, binary=True)

    def _get_trained_model(self) -> Word2Vec:
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

    def _create_data(self) -> WordVectorModel:
        logger.info('reading binary vector file')
        # https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
        #wv = self._get_model().wv
        wv = self._get_model()
        #words = wv.index2entity
        words = wv.index_to_key
        word2vec = {}
        word2idx = {}
        vectors = []
        with time('created data structures'):
            for i, word in enumerate(words):
                word2idx[word] = i
                vec = wv[word]
                vectors.append(vec)
                word2vec[word] = vec
            vectors = np.array(vectors)
        unknown_vec = np.expand_dims(np.zeros(self.dimension), axis=0)
        vectors = np.concatenate((vectors, unknown_vec))
        word2idx[self.UNKNOWN] = len(words)
        words.append(self.UNKNOWN)
        word2vec[self.UNKNOWN] = unknown_vec
        return WordVectorModel(vectors, word2vec, words, word2idx)
