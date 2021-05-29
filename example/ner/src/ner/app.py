"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from dataclasses import dataclass
import logging
import itertools as it
from zensols.persist import dealloc
from zensols.util.log import loglevel
from zensols.deeplearn.batch import BatchStash
from zensols.deeplearn.cli import FacadeApplication
from zensols.deepnlp import FeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class NERFacadeApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    CLASS_INSPECTOR = {}

    def __post_init__(self):
        super().__post_init__()
        self.sent = "I'm Paul Landes.  I live in the United States."
        self.sent2 = 'West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings and 39 runs in two days to take over at the head of the county championship.'

    def stats(self):
        """Print out the corpus statistics.

       """
        with dealloc(self._create_facade()) as facade:
            facade.write_corpus_stats()

    def _test_transform(self):
        with dealloc(self._create_facade()) as facade:
            model = facade.transformer_embedding_model
            sents = facade.doc_parser.parse(self.sent)
            # from zensols.util.log import loglevel
            # with loglevel(['zensols.deepnlp.transformer'], logging.INFO):
            for sent in sents[0:1]:
                tsent = model.tokenize(sent)
                tsent.write()
                print(model.transform(tsent).shape)

    def _test_decode(self):
        with dealloc(self._create_facade()) as facade:
            sents = tuple(it.islice(facade.feature_stash.values(), 3))
            doc = FeatureDocument(sents)
            vec = facade.language_vectorizer_manager['syn']
            from zensols.util.log import loglevel
            with loglevel('zensols.deepnlp'):
                vec.encode(doc)

    def _test_trans_label(self):
        with dealloc(self._create_facade()) as facade:
            facade.write()
            return
            batches = tuple(it.islice(facade.batch_stash.values(), 3))
            batch = batches[0]
            dps = batch.get_data_points()
            doc = FeatureDocument.combine_documents(map(lambda dp: dp.doc, dps))
            vec = facade.language_vectorizer_manager['entlabel_trans']
            with loglevel('zensols.deepnlp.transformer.vectorize'):
                arr = vec.transform(doc)
            #print(arr.squeeze(-1))

    def _test_batch_write(self, clear: bool = False):
        if clear:
            self.sent_batch_stash.clear()
        with dealloc(self._create_facade()) as facade:
            if 0:
                for id, batch in it.islice(facade.batch_stash, 2):
                    batch.write()
            facade.write_predictions()

    def _batch_sample(self):
        with dealloc(self._create_facade()) as facade:
            stash: BatchStash = facade.batch_stash
            for batch in it.islice(stash.values(), 1):
                batch.write()
                #print(batch.get_label_classes())
                print(batch.has_labels)
                print(batch.get_labels())
                for dp in batch.get_data_points():
                    if len(dp.doc) > 1:
                        print(dp.doc.polarity)
                        for s in dp.doc:
                            print(s)
                        print('-' * 30)

    def _write_max_word_piece_token_length(self):
        logger.info('calculatating word piece length on data set...')
        with dealloc(self._create_facade()) as facade:
            mlen = facade.get_max_word_piece_len()
            print(f'max word piece token length: {mlen}')

    def _test_preds(self):
        with dealloc(self._create_facade()) as facade:
            #with loglevel('zensols.deeplearn.batch.domain'):
            preds = facade.predict([self.sent, self.sent2])
            for i in preds:
                print(i)

    def all(self):
        self._test_transform()
        self._test_decode()
        self._test_batch_write()
        self._write_max_word_piece_token_length()

    def proto(self):
        if 0:
            self._batch_sample()
        else:
            self._test_preds()
