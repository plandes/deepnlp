"""Example application to demonstrate Bert sequence classification.

"""
__author__ = 'plandes'

from dataclasses import dataclass
import logging
from zensols.deeplearn.cli import FacadeApplication
from zensols.persist import dealloc

logger = logging.getLogger(__name__)


@dataclass
class NERFacadeApplication(FacadeApplication):
    """Example application to demonstrate Bert sequence classification.

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
            model = facade.bert_embedding_model
            sents = facade.doc_parser.parse(text)
            from zensols.util.log import loglevel
            with loglevel(['zensols.deepnlp.bert'], logging.INFO):
                for sent in sents:
                    tsent = model.tokenize(sent)
                    print(tsent.input_ids.shape)
                    print(tsent.attention_mask.shape)
                    print(tsent.position_ids.shape)
                    print('-' * 20)
                    #out.write()
                    #print(model.transform(tsent).shape)

    def _test_batch_write(self):
        if 0:
            self.sent_batch_stash.clear()
        import itertools as it
        from zensols.util.log import loglevel
        with loglevel(['zensols.deepnlp.bert',
                       'zensols.deepnlp.vectorize.layer'], logging.DEBUG):
            for id, batch in it.islice(self.sent_batch_stash, 1):
                batch.write()

    def tmp(self):
        self._test_transform()
