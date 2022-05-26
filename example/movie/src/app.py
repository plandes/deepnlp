"""Example application to demonstrate Transformer sequence classification.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass, field
import logging
import itertools as it
from zensols.config import ConfigFactory
from zensols.deeplearn.cli import FacadeApplication

logger = logging.getLogger(__name__)


@dataclass
class MovieReviewApplication(FacadeApplication):
    """Example application to demonstrate Transformer sequence classification.

    """
    config_factory: ConfigFactory = field(default=None)

    def proto(self):
        """Testing method."""
        #inst = self.config_factory('mr_installer')
        #fac = self.config_factory('dataset_factory')
        #print(fac.dataset)
        if 0:
            self.config_factory('feature_split_key_container').write()
            stash = self.config_factory('feature_stash')
            stash.write()
            for k, v in it.islice(stash, 1):
                v.write()
        stash = self.config_factory('batch_stash')
        stash.write()
        for k, v in it.islice(stash, 1):
            v.write()
