"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from typing import Tuple
from dataclasses import dataclass
import logging
import itertools as it
from zensols.persist import dealloc
from zensols.deeplearn.cli import FacadeApplication
from zensols.deeplearn.batch import Batch, BatchStash
from . import Review

logger = logging.getLogger(__name__)


@dataclass
class ReviewApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    CLASS_INSPECTOR = {}

    def stats(self):
        """Print out the corpus statistics.

       """
        with dealloc(self.create_facade()) as facade:
            facade.write()

    def batch_sample(self):
        """Print what's contained in this app specific batch.

        """
        import numpy as np
        with dealloc(self.create_facade()) as facade:
            stash: BatchStash = facade.batch_stash
            batch: Batch
            for batch in it.islice(stash.values(), 3):
                classes = batch.get_label_classes()
                uks = np.unique(np.array(classes))
                if len(uks) > 1 or True:
                    print(batch.split_name)
                    batch.write()
                    print(classes)
                    print(batch.has_labels)
                    for dp in batch.get_data_points():
                        if len(dp.doc) > 1:
                            print(dp.doc.polarity)
                            for s in dp.doc:
                                print(s)
                            print('-' * 30)

    def predict(self, sentence: str):
        """Predict several movie review test sentences.

        :param sentence: the sentence to classify

        """
        if sentence is None:
            sents = ["If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .",
                     'There are a few stabs at absurdist comedy ... but mostly the humor is of the sweet , gentle and occasionally cloying kind that has become an Iranian specialty .',
                     'Terrible',
                     'Great movie',
                     'Wonderful, great, awesome, 100%',
                     'Terrible, aweful, worst movie']
        else:
            sents = [sentence]
        with dealloc(self.create_facade()) as facade:
            docs: Tuple[Review] = facade.predict(sents)
            for doc in docs:
                doc.write()

    def proto(self):
        """Testing method."""
        self.predict(None)
