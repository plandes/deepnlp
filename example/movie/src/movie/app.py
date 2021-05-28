"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'plandes'

from dataclasses import dataclass
import logging
from zensols.persist import dealloc
from zensols.util.time import time
from zensols.deeplearn.cli import FacadeApplication
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

    def _create_batch(self, sent: str):
        from zensols.deeplearn.batch import BatchStash

        with dealloc(self._create_facade()) as facade:
            stash: BatchStash = facade.batch_stash
            with time('created batch'):
                batch = stash.create_nascent(sent)
            batch.write()
            print(batch['glove_50_embedding'])

    def proto(self):
        s = "If you sometimes like to go to the movies to have fun , Wasabi is a good place to start .",
        s = 'There are a few stabs at absurdist comedy ... but mostly the humor is of the sweet , gentle and occasionally cloying kind that has become an Iranian specialty .'
        self._create_batch(s)
