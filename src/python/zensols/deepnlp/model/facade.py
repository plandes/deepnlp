"""Client entry point to the model.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import logging
from zensols.deeplearn.model import ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelFacade(ModelFacade):
    def _configure_debug_logging(self):
        logging.getLogger(__name__).setLevel(logging.DEBUG)
        logging.getLogger('zensols.deeplearn.layer.linear').setLevel(logging.DEBUG)
