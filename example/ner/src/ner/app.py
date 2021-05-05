"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from dataclasses import dataclass
import logging
import itertools as it
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp import FeatureDocument
from zensols.persist import dealloc

logger = logging.getLogger(__name__)


@dataclass
class NERFacadeApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    CLASS_INSPECTOR = {}

    def stats(self):
        """Print out the corpus statistics.

       """
        with dealloc(self._create_facade()) as facade:
            facade.write_corpus_stats()

    def _test_transform(self):
        text = "I'm Paul Landes.  I live in the United States."
        with dealloc(self._create_facade()) as facade:
            model = facade.transformer_embedding_model
            sents = facade.doc_parser.parse(text)
            from zensols.util.log import loglevel
            with loglevel(['zensols.deepnlp.transformer'], logging.INFO):
                for sent in sents[0:1]:
                    tsent = model.tokenize(sent)
                    tsent.write()
                    print(model.transform(tsent).shape)

    def _test_decode(self):
        with dealloc(self._create_facade()) as facade:
            sents = tuple(it.islice(facade.feature_stash.values(), 3))
            doc = FeatureDocument(sents)
            vec = facade.language_vectorizer_manager.vectorizers['syn']
            from zensols.util.log import loglevel
            with loglevel('zensols.deepnlp'):
                vec.encode(doc)

    def _test_batch_write(self):
        if 0:
            self.sent_batch_stash.clear()
        import itertools as it
        with dealloc(self._create_facade()) as facade:
            for id, batch in it.islice(facade.batch_stash, 5):
                batch.write()

    def _write_max_word_piece_token_length(self):
        logger.info('calculatating word piece length on data set...')
        with dealloc(self._create_facade()) as facade:
            mlen = facade.get_max_word_piece_len()
            print(f'max word piece token length: {mlen}')

    def tmp(self):
        #self._test_transform()
        #self._test_decode()
        self._test_batch_write()
        #self._write_max_word_piece_token_length()
