"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from typing import Tuple, List
from dataclasses import dataclass
import logging
import itertools as it
from zensols.persist import dealloc
from zensols.util.log import loglevel
from zensols.persist import persisted
from zensols.deeplearn.cli import FacadeApplication
from zensols.deeplearn.batch import Batch, BatchStash
from zensols.deepnlp import FeatureDocument

logger = logging.getLogger(__name__)


@dataclass
class ReviewApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    CLASS_INSPECTOR = {}

    def __post_init__(self):
        super().__post_init__()
        self.sent = "I'm Paul Landes.  I live in the United States."

    def stats(self):
        """Print out the corpus statistics.

       """
        with dealloc(self._create_facade()) as facade:
            facade.write()

    def _batch_sample(self):
        with dealloc(self._create_facade()) as facade:
            stash: BatchStash = facade.batch_stash
            for batch in it.islice(stash.values(), 1):
                batch.write()
                print(batch.get_label_classes())
                print(batch.has_labels)
                for dp in batch.get_data_points():
                    if len(dp.doc) > 1:
                        print(dp.doc.polarity)
                        for s in dp.doc:
                            print(s)
                        print('-' * 30)

    def _create_batch(self, sents: Tuple[str]):
        with dealloc(self._create_facade()) as facade:
            print(facade.predict(sents))

    def proto(self):
        sents = ["If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .",
                 'There are a few stabs at absurdist comedy ... but mostly the humor is of the sweet , gentle and occasionally cloying kind that has become an Iranian specialty .',
                 'Terrible',
                 'Great movie',
                 ]
        if 0:
            self._batch_sample()
        else:
            self._create_batch(sents)
