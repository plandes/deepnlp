"""Client entry point to the model.

"""
__author__ = 'Paul Landes'

from dataclasses import dataclass
import logging
from zensols.deeplearn.batch import BatchMetadata
from zensols.deeplearn.model import ModelFacade

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingModelFacade(ModelFacade):
    def _configure_debug_logging(self):
        lg = logging.getLogger(__name__ + '.module')
        lg.setLevel(logging.DEBUG)

    @property
    def batch_metadata(self) -> BatchMetadata:
        """Return the batch metadata used on the executor.  This will only work if
        there is an attribute set called ``batch_metadata_factory`` set on
        :py:attrib:~`executor.net_settings` (i.e. ``EmbeddingNetworkSettings``
        in the ``zensols.deepnlp`` package).

        :see: :class:`zensols.deepnlp.model.module.EmbeddingNetworkSettings`

        """
        return self.executor.net_settings.batch_metadata_factory()
